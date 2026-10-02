"""Protein--lipid affinity active learning.

Input CSV columns: protein_sequence, lipid_smiles, affinity.
Additional metadata columns are preserved but do not affect the model.
Affinity must be a *larger-is-better* quantity (e.g. pKd/pKi or -DeltaG).
Use --minimize for a lower-is-better experimental endpoint.

The default product kernel follows the pairwise-kernel idea used for molecular
binding data: K((P,L),(P',L')) = K_protein(P,P') * K_lipid(L,L').  It is a
particularly appropriate baseline when a protein and a lipid may recur in
different pairs. This script evaluates pool-based AL offline using known labels
as an oracle. It does not implement a live experimental campaign.
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

EPS = 1e-8

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_GROUPS = {"hydrophobic":"AVILMFWY", "positive":"KRH", "negative":"DE", "polar":"STNQCY"}

@lru_cache(maxsize=20000)
def protein_features(sequence: str) -> np.ndarray:
    """Interpretable amino-acid composition + length/charge/hydropathy proxies."""
    s = ''.join(c for c in str(sequence).upper() if c in AA)
    if not s: raise ValueError("protein_sequence has no standard amino acids")
    n = len(s); comp = [s.count(a)/n for a in AA]
    groups = [sum(s.count(a) for a in chars)/n for chars in AA_GROUPS.values()]
    charge = (s.count('K')+s.count('R')+0.1*s.count('H')-s.count('D')-s.count('E'))/n
    return np.asarray(comp + groups + [np.log1p(n), charge], float)

@lru_cache(maxsize=20000)
def lipid_features(smiles: str, n_bits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: raise ValueError(f"Invalid lipid_smiles: {smiles!r}")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits, includeChirality=True).GetFingerprint(mol)
    bits = np.zeros(n_bits, dtype=float); DataStructs.ConvertToNumpyArray(fp, bits)
    # Appended continuous descriptors make RBF/Matérn lipid kernels meaningful too.
    desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                     Descriptors.NumRotatableBonds(mol), Descriptors.HeavyAtomCount(mol)], float)
    return np.r_[bits, desc]

def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum((a*a).sum(1)[:,None] + (b*b).sum(1)[None,:] - 2*a@b.T, 0)

def continuous_kernel(a, b, name: str, lengthscale: float = 1.0):
    d = np.sqrt(_sqdist(a, b)) / lengthscale
    if name == "rbf": return np.exp(-0.5*d*d)
    if name == "matern32": return (1 + np.sqrt(3)*d)*np.exp(-np.sqrt(3)*d)
    if name == "matern52": return (1 + np.sqrt(5)*d + 5*d*d/3)*np.exp(-np.sqrt(5)*d)
    raise ValueError(f"Unknown continuous kernel: {name}")

def tanimoto(a, b):
    dot = a @ b.T
    return dot / (a.sum(1)[:,None] + b.sum(1)[None,:] - dot + EPS)

@dataclass
class ProductKernelGP:
    protein_kernel: str = "matern52"
    lipid_kernel: str = "tanimoto"
    noise: float = 1e-3
    minimize: bool = False

    def _k(self, p, l, q, m):
        kp = continuous_kernel(p, q, self.protein_kernel)
        # Tanimoto is best suited to the binary Morgan portion; RBF/Matérn use all scaled lipid features.
        kl = tanimoto(l[:, :1024], m[:, :1024]) if self.lipid_kernel == "tanimoto" else continuous_kernel(l, m, self.lipid_kernel)
        return kp * kl

    def fit(self, p, l, y):
        self.ps = StandardScaler().fit(p); self.ls = StandardScaler().fit(l[:,1024:])
        self.p = self.ps.transform(p)
        self.l = l.copy(); self.l[:,1024:] = self.ls.transform(l[:,1024:])
        self.y_mean, self.y_std = float(np.mean(y)), max(float(np.std(y)), 1e-6)
        self.y = (np.asarray(y)-self.y_mean)/self.y_std
        K = self._k(self.p,self.l,self.p,self.l) + (self.noise+1e-6)*np.eye(len(y))
        self.cf = cho_factor(K, lower=True, check_finite=False)
        self.alpha = cho_solve(self.cf, self.y, check_finite=False)
        return self

    def _transform(self, p,l):
        pp=self.ps.transform(p); ll=l.copy(); ll[:,1024:]=self.ls.transform(l[:,1024:]); return pp,ll

    def predict(self,p,l,full_cov=False):
        p,l=self._transform(p,l); Ks=self._k(p,l,self.p,self.l)
        mean=Ks@self.alpha; v=cho_solve(self.cf,Ks.T,check_finite=False)
        if not full_cov:
            # All implemented kernels have unit self-covariance (nonempty fingerprints).
            var=np.maximum(1.0-np.einsum('ij,ji->i',Ks,v),EPS)
            return mean*self.y_std+self.y_mean, np.sqrt(var)*self.y_std
        cov=self._k(p,l,p,l)-Ks@v
        cov=(cov+cov.T)/2
        if full_cov: return mean*self.y_std+self.y_mean, cov*self.y_std**2
        return mean*self.y_std+self.y_mean, np.sqrt(np.maximum(np.diag(cov),EPS))*self.y_std

    def sample_posterior(self,p,l,n_samples,rng):
        mu,cov=self.predict(p,l,full_cov=True)
        cov += 1e-7*np.eye(len(mu))
        return rng.multivariate_normal(mu,cov,size=n_samples,method='cholesky')

def acquisition(model, p_pool, l_pool, y_best, method, beta, rng):
    if method == 'random': return rng.random(len(p_pool))
    mu, sd = model.predict(p_pool,l_pool)
    sign = -1 if model.minimize else 1
    objective_mu, best = sign*mu, sign*y_best
    if method == "greedy": return objective_mu
    if method == "uncertainty": return sd
    if method == "ucb": return objective_mu + beta*sd
    if method == "pi": return ndtr((objective_mu-best)/(sd+EPS))
    if method == "ei":
        z=(objective_mu-best)/(sd+EPS); return (objective_mu-best)*ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    if method == "thompson": return sign*model.sample_posterior(p_pool,l_pool,1,rng)[0]
    raise ValueError("method must be greedy, uncertainty, ucb, pi, ei, or thompson")

def diverse_batch(scores, p, l, batch_size, diversity=0.20):
    """Greedy MMR: a lightweight batch-diverse TS/UCB/EI implementation."""
    selected=[]; remaining=list(range(len(scores)))
    z=(scores-scores.mean())/(scores.std()+EPS)
    x=np.c_[StandardScaler().fit_transform(p), StandardScaler().fit_transform(l[:,1024:])]
    while remaining and len(selected)<batch_size:
        if not selected: pick=max(remaining,key=lambda i:z[i])
        else:
            sim=(x[remaining]@x[selected].T)/(np.linalg.norm(x[remaining],axis=1)[:,None]*np.linalg.norm(x[selected],axis=1)[None,:]+EPS)
            pick=remaining[int(np.argmax(z[remaining]-diversity*sim.max(1)))]
        selected.append(pick); remaining.remove(pick)
    return np.asarray(selected)

def feature_attribution(model, p, l, max_rows=300, seed=0):
    """Model-agnostic grouped permutation explanation; positive = affects predicted affinity."""
    rng=np.random.default_rng(seed); take=rng.choice(len(p),min(max_rows,len(p)),replace=False)
    base=model.predict(p[take],l[take])[0]; groups={}
    for name, cols in {"protein_amino_acids":slice(0,20),"protein_groups":slice(20,24),"protein_length_charge":slice(24,26)}.items():
        q=p[take].copy(); q[:,cols]=q[rng.permutation(len(q)),cols]; groups[name]=float(np.mean(abs(base-model.predict(q,l[take])[0])))
    q=l[take].copy(); q[:,:1024]=q[rng.permutation(len(q)),:1024]; groups["lipid_Morgan_substructures"]=float(np.mean(abs(base-model.predict(p[take],q)[0])))
    q=l[take].copy(); q[:,1024:]=q[rng.permutation(len(q)),1024:]; groups["lipid_physchem"]=float(np.mean(abs(base-model.predict(p[take],q)[0])))
    return pd.Series(groups,name="mean_absolute_prediction_change").sort_values(ascending=False)

def run_active_learning(df, initial=32, batch_size=8, rounds=12, acquisition_name="thompson", protein_kernel="matern52", lipid_kernel="tanimoto", seed=7, minimize=False, diversity=0.0):
    required={"protein_sequence","lipid_smiles","affinity"}; missing=required-set(df.columns)
    if missing: raise ValueError(f"CSV missing columns: {sorted(missing)}")
    if len(df)<10 or initial<2 or batch_size<1 or rounds<0: raise ValueError('Need >=10 rows, initial>=2, batch>=1, rounds>=0')
    if df[list(required)].isna().any().any(): raise ValueError('Missing model inputs or affinity')
    if not np.isfinite(df.affinity.to_numpy(float)).all(): raise ValueError('Nonfinite affinity')
    if df.duplicated(['protein_sequence','lipid_smiles']).any(): raise ValueError('Deduplicate protein-lipid pairs before evaluation')
    p=np.vstack(df.protein_sequence.map(protein_features)); l=np.vstack(df.lipid_smiles.map(lipid_features)); y=df.affinity.to_numpy(float)
    rng=np.random.default_rng(seed); all_idx=np.arange(len(df)); rng.shuffle(all_idx)
    # A frozen 20% test set makes the AL curve honest; only the remaining pool is acquired.
    n_test=max(1,round(.2*len(df))); test, pool=all_idx[:n_test],all_idx[n_test:]
    if initial>=len(pool): raise ValueError('Initial labels exhaust acquisition pool')
    labelled=pool[:min(initial,len(pool))]; pool=pool[len(labelled):]; candidate_universe=np.r_[labelled,pool]; history=[]; selections=[]
    for step in range(rounds+1):
        gp=ProductKernelGP(protein_kernel,lipid_kernel,minimize=minimize).fit(p[labelled],l[labelled],y[labelled])
        pred,_=gp.predict(p[test],l[test]); top=max(1,round(.1*len(test)))
        # Hit recovery is measured against the initially unlabelled candidate campaign,
        # while MAE/R² use a disjoint frozen test set.
        top=max(1,round(.1*len(candidate_universe)))
        c_order=np.argsort(y[candidate_universe])
        truth_top=set(candidate_universe[c_order[:top] if minimize else c_order[-top:]])
        found=len(truth_top & set(labelled)); history.append({"round":step,"labelled":len(labelled),"test_mae":mean_absolute_error(y[test],pred),"test_r2":r2_score(y[test],pred),"top10pct_recall":found/len(truth_top)})
        if step==rounds or not len(pool): break
        score=acquisition(gp,p[pool],l[pool],y[labelled].min() if minimize else y[labelled].max(),acquisition_name,2.0,rng)
        choose=(diverse_batch(score,p[pool],l[pool],min(batch_size,len(pool)),diversity)
                if diversity>0 and acquisition_name!='random'
                else np.argsort(-score,kind='stable')[:min(batch_size,len(pool))])
        queried=pool[choose]; selections.append(df.iloc[queried].assign(round=step+1, acquisition=acquisition_name, acquisition_score=score[choose]))
        labelled=np.r_[labelled,queried]; pool=np.delete(pool,choose)
    return pd.DataFrame(history),pd.concat(selections,ignore_index=True) if selections else pd.DataFrame(),feature_attribution(gp,p[labelled],l[labelled])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv",type=Path); ap.add_argument("--out",type=Path,default=Path("al_results")); ap.add_argument("--acquisition",choices=["greedy","uncertainty","ucb","pi","ei","thompson"],default="thompson"); ap.add_argument("--protein-kernel",choices=["rbf","matern32","matern52"],default="matern52"); ap.add_argument("--lipid-kernel",choices=["tanimoto","rbf","matern32","matern52"],default="tanimoto"); ap.add_argument("--initial",type=int,default=32); ap.add_argument("--batch",type=int,default=8); ap.add_argument("--rounds",type=int,default=12); ap.add_argument("--seed",type=int,default=7); ap.add_argument("--minimize",action="store_true"); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(args.csv)
    history,selected,importance=run_active_learning(df,args.initial,args.batch,args.rounds,args.acquisition,args.protein_kernel,args.lipid_kernel,args.seed,args.minimize)
    history.to_csv(args.out/"learning_curve.csv",index=False); selected.to_csv(args.out/"selected_batches.csv",index=False); importance.to_csv(args.out/"grouped_feature_attribution.csv")
    print(history.to_string(index=False)); print("\nNext assay batch saved to",args.out/"selected_batches.csv")
if __name__ == "__main__": main()
