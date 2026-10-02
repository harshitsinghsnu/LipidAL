"""Resume the unchanged study in disjoint process shards, never concurrent writers."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import itertools,json,time,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from grid_gp import Geometry
from run_robust_study import ROOT,OUT,CACHE,PROTEIN_REPS,LIPID_REPS,KERNELS,POLICIES,MODES,SEEDS,run_one,aggregate
from robust_splits import sha

def worker(task):
    dataset,mode,seed,signature=task
    partitions=json.loads((CACHE/'robust_splits.json').read_text());part=partitions['splits'][f'{dataset}/{mode}/seed{seed}']
    proteins=np.load(CACHE/'protein_representations.npz');base=np.load(CACHE/'representations.npz');extra=np.load(CACHE/'extended_representations.npz')
    df=pd.read_csv(ROOT/'data/processed'/f'{dataset}.csv');plook={s:i for i,s in enumerate(proteins['sequences'])};llook={s:i for i,s in enumerate(base['smiles'])}
    pr=[plook[s] for s in df.protein_sequence];lr=[llook[s] for s in df.lipid_smiles];paths=[]
    with threadpool_limits(limits=1):
        for protein_rep,lipid_rep,kernel in itertools.product(PROTEIN_REPS,LIPID_REPS,KERNELS):
            geom=Geometry((base if lipid_rep in base.files else extra)[lipid_rep][lr],proteins[protein_rep][pr].astype(np.float64),np.array(part['initial']))
            for policy in POLICIES:
                conf=dict(dataset=dataset,split=mode,seed=seed,protein_representation=protein_rep,representation=lipid_rep,kernel=kernel,protocol=policy)
                dest=OUT/'runs'/dataset/mode/protein_rep/lipid_rep/kernel/policy/f'seed{seed}.json';paths.append(str(dest))
                if dest.exists():assert json.loads(dest.read_text())['signature']==signature
                else:
                    begin=time.time();r=run_one(df,geom,kernel,policy,part,seed);r.update(config=conf,signature=signature,runtime_seconds=time.time()-begin)
                    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(r));tmp.replace(dest)
    return paths

def main():
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    for name,digest in manifest['source_hashes'].items():assert sha(ROOT/name)==digest
    for name,digest in manifest['cache_hashes'].items():assert sha(CACHE/name)==digest
    for name,digest in manifest['data_hashes'].items():assert sha(ROOT/'data/processed'/f'{name}.csv')==digest
    signature=manifest['signature'];started=time.time();paths=[]
    (OUT/'parallel_execution.json').write_text(json.dumps(dict(source_sha256=sha(__file__),workers=4,experiment_signature=signature),indent=2))
    tasks=[(dataset,mode,seed,signature) for dataset,mode,seed in itertools.product(manifest['datasets'],MODES,SEEDS)]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(worker,task) for task in tasks]
        for future in as_completed(futures):
            paths.extend(Path(p) for p in future.result())
            progress=dict(completed=len(paths),expected=manifest['expected_runs'],elapsed_seconds=time.time()-started,unit='completed disjoint shards; in-progress shard files not yet counted')
            (OUT/'progress.json').write_text(json.dumps(progress));print(progress,flush=True)
    assert len(paths)==manifest['expected_runs'] and len(set(paths))==len(paths)
    aggregate(paths);print('COMPLETE',len(paths),'matched campaigns',flush=True)

if __name__=='__main__':main()
