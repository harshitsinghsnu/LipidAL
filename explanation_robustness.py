"""Collision audit and individual-bit nonlinear Shapley/background diagnostics.

Uses four predeclared historical ECFP/Thompson models, not new-grid winners.
No retraining; fixed test query rows; acquired-only backgrounds; protein fixed.
"""
from pathlib import Path
import json,itertools,hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from threadpoolctl import threadpool_limits
from explain_grid import SavedPredictor
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/explanation_robustness'

def collisions():
    features=np.load(ROOT/'data/features/representations.npz');smiles=features['smiles'];gen=rdFingerprintGenerator.GetMorganGenerator(radius=4,fpSize=4096,includeChirality=False)
    environments={};per_lipid=[]
    for i,s in enumerate(smiles):
        mol=Chem.MolFromSmiles(str(s));ao=rdFingerprintGenerator.AdditionalOutput();ao.AllocateBitInfoMap()
        fp=gen.GetSparseCountFingerprint(mol,additionalOutput=ao);info=ao.GetBitInfoMap()
        assert {int(x)%4096 for x in fp.GetNonzeroElements()}==set(gen.GetFingerprint(mol).GetOnBits())
        ids={}
        for raw,occurrences in info.items():
            bit=int(raw)%4096;ids.setdefault(bit,set()).add(int(raw))
            for center,radius in occurrences:
                bonds=list(Chem.FindAtomEnvironmentOfRadiusN(mol,radius,center));atoms={int(center)}
                for bond in bonds:atoms.update([mol.GetBondWithIdx(bond).GetBeginAtomIdx(),mol.GetBondWithIdx(bond).GetEndAtomIdx()])
                fragment=Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(atoms),bondsToUse=bonds,canonical=True,isomericSmiles=False)
                environments.setdefault(bit,{}).setdefault(int(raw),set()).add(fragment)
        per_lipid.append(ids)
    rows=[dict(bit=bit,unfolded_identifiers=len(ids),folding_collision=len(ids)>1,
        distinct_fragment_strings=len(set().union(*ids.values())),unfolded_ids=';'.join(map(str,sorted(ids)))) for bit,ids in sorted(environments.items())]
    pd.DataFrame(rows).to_csv(OUT/'fingerprint_collisions.csv',index=False)
    lookup={s:i for i,s in enumerate(smiles)};strata=[]
    for endpoint in ('Kd','Ki'):
        df=pd.read_csv(ROOT/'data/processed'/f'biodolphin_{endpoint}.csv');median=df.affinity.median()
        high={};low={}
        for _,r in df.iterrows():
            target=high if r.affinity>=median else low
            for bit,ids in per_lipid[lookup[r.lipid_smiles]].items():target.setdefault(bit,set()).update(ids)
        for bit in sorted(set(high)|set(low)):
            a=high.get(bit,set());b=low.get(bit,set());union=a|b
            strata.append(dict(dataset='biodolphin_'+endpoint,bit=bit,median_pK=float(median),high_unfolded_ids=len(a),low_unfolded_ids=len(b),
                high_low_environment_jaccard=len(a&b)/len(union),different_unfolded_ids_in_bit=len(union)>1))
    pd.DataFrame(strata).to_csv(OUT/'collision_affinity_strata.csv',index=False)
    return dict(observed_bits=len(rows),bits_with_multiple_unfolded_ids=sum(r['folding_collision'] for r in rows),lipids=len(smiles),
        definition='Distinct sparse Morgan identifiers folding to one 4096-bit position; distinct 32-bit identifiers are not a proof of unique chemistry',
        affinity_strata='Retrospective median split using all measured labels; not model selection or independent SAR validation')

def individual_values(model,row,background,seed):
    x=model.x[row];bx=model.x[background];active=np.flatnonzero(np.any(bx!=x,axis=0));m=len(active)
    # Sufficient statistics evaluate each coalition exactly for the saved nonlinear GP.
    base_dot=bx@model.tx.T;base_norm=(bx*bx).sum(1)
    target_pd=np.maximum(((model.ps[row]-model.tp)**2).sum(1),0)[None,:]
    delta_dot=(x[active][None,:]-bx[:,active]).T[:,:,None]*model.tx[:,active].T[:,None,:]
    delta_norm=(x[active]**2-bx[:,active]**2).T
    base=float(model.predict(bx,np.repeat(model.p[[row]],len(background),axis=0)).mean())
    target=float(model.predict(model.x[[row]],model.p[[row]])[0]);value=np.zeros(4096)
    rng=np.random.default_rng(seed);permutations=[]
    for _ in range(4):
        order=rng.permutation(m);permutations.extend([order,order[::-1]])
    for order in permutations:
        dot=base_dot[None]+np.cumsum(delta_dot[order],axis=0)
        norm=base_norm[None]+np.cumsum(delta_norm[order],axis=0)
        k=model.kernel_stats(dot,norm,target_pd[None])
        predictions=model.center+model.scale*(model.par['mean']+(k@model.alpha).mean(axis=1))
        contributions=np.diff(np.r_[base,predictions]);value[active[order]]+=contributions/len(permutations)
        assert abs(predictions[-1]-target)<1e-7
    assert abs(base+value.sum()-target)<1e-7
    return value,dict(row_index=int(row),active_bits=m,permutations=len(permutations),base_pK=base,predicted_pK=target,additivity_error=abs(base+value.sum()-target))

def main():
    OUT.mkdir(parents=True,exist_ok=True);collision_summary=collisions();features=np.load(ROOT/'data/features/representations.npz');lookup={s:i for i,s in enumerate(features['smiles'])}
    records=[];stability=[]
    with threadpool_limits(limits=1):
        for endpoint,kernel in itertools.product(('Kd','Ki'),('rbf','matern32')):
            name='biodolphin_'+endpoint;df=pd.read_csv(ROOT/'data/processed'/f'{name}.csv');path=ROOT/'results/full_grid/runs'/name/'ecfp'/kernel/'thompson/seed7.json'
            record=json.loads(path.read_text());model=SavedPredictor(record,df,features['ecfp'][[lookup[s] for s in df.lipid_smiles]])
            queries=model.test[:2];importance={}
            for bgseed,repeat in itertools.product((7,19,42),(0,1)):
                background=np.random.default_rng(bgseed).choice(model.train,8,replace=False);values=[]
                for row in queries:
                    phi,audit=individual_values(model,int(row),background,100+repeat)
                    values.append(phi);records.append(dict(dataset=name,kernel=kernel,background_seed=bgseed,repeat=repeat,
                        background_indices=background.tolist(),source_run_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**audit))
                importance[bgseed,repeat]=np.mean(abs(np.vstack(values)),axis=0)
                pd.DataFrame(dict(bit=np.arange(4096),mean_abs_shap=importance[bgseed,repeat])).to_csv(OUT/f'{name}_{kernel}_bg{bgseed}_repeat{repeat}.csv',index=False)
            for a,b in itertools.combinations(importance,2):
                if a[0]!=b[0] and a[1]!=b[1]:continue
                ia=importance[a];ib=importance[b];ta=set(np.argsort(-ia,kind='stable')[:10]);tb=set(np.argsort(-ib,kind='stable')[:10]);active=(ia+ib)>1e-14
                rho=float(spearmanr(ia[active],ib[active]).statistic) if active.sum()>1 and np.std(ia[active])>0 and np.std(ib[active])>0 else None
                stability.append(dict(dataset=name,kernel=kernel,comparison='background' if a[0]!=b[0] else 'permutation_repeat',
                    first=str(a),second=str(b),top10_jaccard=len(ta&tb)/len(ta|tb),active_bit_spearman=rho))
            print('Nonlinear individual-bit analysis complete',name,kernel,flush=True)
    pd.DataFrame(stability).to_csv(OUT/'background_stability.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(collisions=collision_summary,explanations=len(records),records=records,
        scope='Four historical ECFP nonlinear models; two fixed held-out queries each; 3 acquired-only backgrounds; 2 permutation repeats. No physical contacts or causal effects inferred.',
        method='Individual marginal-game Shapley estimates; 8 antithetic permutations per explanation. Dummy bits exactly zero. Additivity is not attribution convergence.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(collision_summary,flush=True)

if __name__=='__main__':main()
