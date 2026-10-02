"""Frozen random, accession/sequence-clustered and scaffold-disjoint partitions.

Cold groups are connected components of shared accessions OR >40% identity
under global NW BLOSUM62 (gap-open 10, extend 1), matches/alignment length.
This is an operational global-identity rule, not curated protein families.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/features'
SEEDS=(7,19,42,73,101,137)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def identities(sequences):
    sys.path.append(str(ROOT/'study_tools'))  # do not shadow environment NumPy
    import parasail
    n=len(sequences);matrix=np.eye(n,dtype=np.float32);started=time.time()
    for i,a in enumerate(sequences):
        for j in range(i):
            b=sequences[j]
            # At most min(lengths) global matches across >=max(lengths) columns.
            # Still compute exact identity for every pair so audits report maxima.
            result=parasail.nw_stats_striped_16(a,b,10,1,parasail.blosum62)
            if result.saturated:raise ValueError('Alignment saturation')
            matrix[i,j]=matrix[j,i]=result.matches/max(result.length,1)
        if i%100==0:print('Global sequence alignments',i,'/',n,'seconds',round(time.time()-started,1),flush=True)
    return matrix

def groups(df,mode,sequences,identity):
    if mode=='random_pair':return np.arange(len(df))
    if mode=='scaffold':
        # All acyclic compounds share the empty scaffold; do not split them by SMILES.
        return np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),includeChirality=False) or 'ACYCLIC' for s in df.lipid_smiles])
    if mode!='cold_sequence40':raise ValueError(mode)
    lookup={s:i for i,s in enumerate(sequences)};idx=np.array([lookup[s] for s in df.protein_sequence])
    adjacency=identity[np.ix_(idx,idx)]>.4
    for accession,part in df.dropna(subset=['protein_id']).groupby('protein_id'):
        positions=part.index.to_numpy();adjacency[np.ix_(positions,positions)]=True
    return connected_components(csr_matrix(adjacency),directed=False)[1]

def holdout(indices,group,seed,fraction):
    rng=np.random.default_rng(seed);unique=np.unique(group[indices]);rng.shuffle(unique)
    desired=max(2,round(len(indices)*fraction));chosen=[];count=0
    for g in unique:
        members=indices[group[indices]==g]
        if not chosen or abs(count+len(members)-desired)<abs(count-desired):chosen.extend(members.tolist());count+=len(members)
    test=np.array(sorted(chosen),dtype=int);train=indices[~np.isin(indices,test)]
    if not len(train) or len(test)<2:raise ValueError('Insufficient groups for holdout')
    return train,test

def make_split(df,mode,seed,sequences,identity):
    group=groups(df,mode,sequences,identity);allidx=np.arange(len(df))
    trainval,test=holdout(allidx,group,seed,.2)
    train,validation=holdout(trainval,group,seed+1009,.2)
    if len(train)<120:raise ValueError(f'{mode} has only {len(train)} acquisition candidates; cannot match 120-label budget')
    rng=np.random.default_rng(seed+2027);order=rng.permutation(train)
    if mode!='random_pair':
        assert not set(group[train])&set(group[test]) and not set(group[train])&set(group[validation])
        assert not set(group[test])&set(group[validation])
    lookup={s:i for i,s in enumerate(sequences)};si=np.array([lookup[s] for s in df.protein_sequence])
    audit=dict(n_train=len(train),n_validation=len(validation),n_test=len(test),n_groups=len(set(group)),
        max_train_test_global_identity=float(identity[np.ix_(si[train],si[test])].max()),
        exact_sequence_overlap=len(set(df.iloc[train].protein_sequence)&set(df.iloc[test].protein_sequence)),
        accession_overlap=len(set(df.iloc[train].protein_id.dropna())&set(df.iloc[test].protein_id.dropna())),
        validation_labels_charged=len(validation))
    if mode=='cold_sequence40':assert audit['max_train_test_global_identity']<=.4+1e-7 and audit['accession_overlap']==0
    return dict(initial=order[:24].tolist(),pool=order[24:].tolist(),validation=validation.tolist(),test=test.tolist(),audit=audit)

def main():
    frames={n:pd.read_csv(ROOT/'data/processed'/f'{n}.csv') for n in ('biodolphin_Kd','biodolphin_Ki')}
    sequences=sorted(set(pd.concat(frames.values()).protein_sequence));cache=OUT/'sequence_identity.npz'
    if cache.exists():
        old=np.load(cache);assert np.array_equal(old['sequences'],sequences);identity=old['identity']
    else:
        identity=identities(sequences);np.savez_compressed(cache,sequences=np.array(sequences),identity=identity)
    splits={}
    for name,df in frames.items():
        for mode in ('random_pair','cold_sequence40','scaffold'):
            for seed in SEEDS:
                key=f'{name}/{mode}/seed{seed}';splits[key]=make_split(df,mode,seed,sequences,identity)
                print(key,splits[key]['audit'],flush=True)
    (OUT/'robust_splits.json').write_text(json.dumps(dict(splits=splits,source_sha256=sha(__file__),identity_sha256=sha(cache),
        data_sha256={n:sha(ROOT/'data/processed'/f'{n}.csv') for n in frames},seeds=SEEDS,
        identity_definition='Global NW BLOSUM62, gap open 10 / extend 1, matches/alignment length; >40% graph edges plus shared accessions',
        initial=24,batch=24,cycles=4,final=120),indent=2))

if __name__=='__main__':main()
