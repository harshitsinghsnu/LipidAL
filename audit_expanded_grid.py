"""Fail closed unless the entire expanded real-data experiment is complete."""
import itertools,json
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from run_full_grid import datasets,split,budget,sha,SEEDS
from extended_features import REPRESENTATIONS,CACHE
from extended_methods import KERNELS,PROTOCOLS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/expanded_grid'

def main():
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    assert sha(CACHE/'extended_representations.npz')==manifest['feature_sha256']
    assert sha(CACHE/'extended_representation_manifest.json')==manifest['feature_manifest_sha256']
    files=datasets();frames={n:pd.read_csv(p) for n,p in files.items()}
    for name,path in files.items(): assert sha(path)==manifest['data_sha256'][name]
    index=json.loads((OUT/'run_index.json').read_text())
    keys=('dataset','representation','kernel','protocol','seed')
    expected=set(itertools.product(files,REPRESENTATIONS,KERNELS,PROTOCOLS,SEEDS))
    found=[tuple(e['config'][k] for k in keys) for e in index]
    assert len(found)==len(set(found)) and set(found)==expected
    total_fits=0;nonconverged=0
    for entry in index:
        path=(ROOT/entry['path']).resolve();assert path.is_relative_to(ROOT/'results')
        assert sha(path)==entry['sha256'],str(path)
        r=json.loads(path.read_text());conf=r['config'];df=frames[conf['dataset']]
        initial,batches=budget(len(df));test,labelled,pool=split(len(df),conf['seed'],initial)
        assert r['test_indices']==test.tolist() and r['initial_indices']==labelled.tolist()
        assert r['batches']==batches and len(r['history'])==11 and len(r['acquired'])==10
        for h,a,b in zip(r['history'][1:],r['acquired'],batches):
            q=a['indices'];assert len(q)==b and len(q)==len(set(q)) and set(q)<=set(pool)
            labelled=np.r_[labelled,q].astype(int);pool=pool[~np.isin(pool,q)]
            assert not set(labelled)&set(test) and h['labelled']==len(labelled)
            assert all(np.isfinite(v) for v in h.values() if isinstance(v,(int,float)))
        assert len(r['test_predictions'])==len(test) and np.isfinite(r['test_predictions']).all()
        total_fits+=len(r['fits']);nonconverged+=sum(not f['success'] for f in r['fits'])
    summary=pd.read_csv(OUT/'summary.csv')
    assert len(summary)==len(expected)//3 and (summary.n_seeds==3).all()
    comparison=summary.query("kernel=='matern32' and protocol=='thompson'").sort_values(['dataset','mae_mean'])
    comparison.to_csv(OUT/'representation_thompson_matern32.csv',index=False)
    audit=dict(status='PASS',runs=len(index),configurations=len(summary),representations=list(REPRESENTATIONS),
        kernels=list(KERNELS),protocols=list(PROTOCOLS),panels=list(files),seeds=list(SEEDS),
        total_gp_fits=total_fits,nonconverged_gp_fits=nonconverged,no_test_acquisition_overlap=True,
        software={n:version(n) for n in ['numpy','scipy','pandas','torch','transformers','rdkit','safetensors']},
        original_splits_and_budgets=True,source_and_data_hashes_verified=True,unique_complete_cartesian_product=True)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
    print(comparison.query("dataset in ['biodolphin_Kd','biodolphin_Ki']")[['dataset','representation','mae_mean','r2_mean','recall_mean']].to_string(index=False))

if __name__=='__main__':main()
