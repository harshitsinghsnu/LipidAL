"""Matched extension of the immutable original grid; checkpointed real data only."""
from pathlib import Path
import argparse, hashlib, itertools, json, time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from grid_gp import Geometry, ExactPairGP, PROTOCOLS as OLD_PROTOCOLS
from extended_methods import KERNELS, PROTOCOLS, SCHEDULES, select, XI, DIVERSITY_WEIGHT
from extended_features import REPRESENTATIONS, CACHE
from run_full_grid import datasets, budget, split, evaluate, sha, SEEDS
from protein_lipid_active_learning import protein_features

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/expanded_grid'
OLD=ROOT/'results/full_grid'

def run_one(df,geom,kernel,protocol,seed):
    initial,batches=budget(len(df));test,labelled,pool=split(len(df),seed,initial)
    initial_indices=labelled.copy();campaign=np.r_[labelled,pool]
    rng=np.random.default_rng(seed+104729);y=df.affinity.to_numpy(float)
    model=ExactPairGP(geom,kernel,maxiter=60);history=[];acquired=[];fits=[]
    for step in range(11):
        if step==0 or batches[step-1]>0:
            model.fit(labelled,y[labelled]);fits.append(dict(round=step,**model.diagnostics))
        metrics,pred,sd=evaluate(model,test,labelled,campaign,y)
        metrics.update(round=step,phase='initial' if step==0 else SCHEDULES[protocol][step-1]);history.append(metrics)
        if step==10: break
        q=select(model,pool,SCHEDULES[protocol][step],batches[step],rng,incumbent=float(y[labelled].max()))
        acquired.append(dict(round=step+1,phase=SCHEDULES[protocol][step],indices=q.tolist()))
        labelled=np.r_[labelled,q];pool=pool[~np.isin(pool,q)]
    assert not set(test)&set(labelled) and len(set(labelled))==len(labelled)
    return dict(history=history,acquired=acquired,test_indices=test.tolist(),initial_indices=initial_indices.tolist(),
                fits=fits,test_predictions=pred.tolist(),test_sd=sd.tolist(),initial=initial,batches=batches)

def aggregate(paths):
    finals=[];curves=[];index=[]
    for path in paths:
        r=json.loads(path.read_text());conf=r['config'];hist=r['history']
        labels=np.array([h['labelled'] for h in hist]);recall=np.array([h['recall_top10'] for h in hist])
        finals.append({**conf,**hist[-1], 'recall_auc':float(np.trapz(recall,labels)/(labels[-1]-labels[0])),
                       'fit_converged_fraction':float(np.mean([f['success'] for f in r['fits']]))})
        curves.extend({**conf,**h} for h in hist)
        index.append(dict(config=conf,path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
    final=pd.DataFrame(finals);final.to_csv(OUT/'final_by_seed.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'all_learning_curves.csv',index=False)
    summary=final.groupby(['dataset','representation','kernel','protocol']).agg(
        n_seeds=('seed','nunique'),mae_mean=('test_mae','mean'),mae_sd=('test_mae','std'),
        r2_mean=('test_r2','mean'),recall_mean=('recall_top10','mean'),recall_auc_mean=('recall_auc','mean'),
        convergence_fraction=('fit_converged_fraction','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    (OUT/'run_index.json').write_text(json.dumps(index))
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    files=datasets();old=json.loads((OLD/'experiment_manifest.json').read_text())
    assert old['engine_sha256']==sha(ROOT/'grid_gp.py')
    assert old['feature_sha256']==sha(CACHE/'representations.npz')
    assert old['data_sha256']=={n:sha(p) for n,p in files.items()}
    config=dict(version=1,representations=REPRESENTATIONS,kernels=KERNELS,protocols=PROTOCOLS,seeds=SEEDS,
        schedules=SCHEDULES,xi=XI,diversity_weight=DIVERSITY_WEIGHT,maxiter=60,
        original_manifest_sha256=sha(OLD/'experiment_manifest.json'),
        source_sha256={n:sha(ROOT/n) for n in ['grid_gp.py','extended_methods.py','extended_features.py','run_expanded_grid.py','run_full_grid.py']},
        feature_sha256=sha(CACHE/'extended_representations.npz'),feature_manifest_sha256=sha(CACHE/'extended_representation_manifest.json'),
        data_sha256={n:sha(p) for n,p in files.items()},expected_runs=len(files)*len(REPRESENTATIONS)*len(KERNELS)*len(PROTOCOLS)*len(SEEDS),
        description='Frozen molecular encoders; unchanged protein composition. Original matched runs reused, not refit. Descriptive test rankings, not nested model selection.')
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest=OUT/'experiment_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature: raise ValueError('Changed experiment: use a new output directory')
    manifest.write_text(json.dumps(dict(signature=signature,**config),indent=2))
    base=np.load(CACHE/'representations.npz',allow_pickle=False);extra=np.load(CACHE/'extended_representations.npz',allow_pickle=False)
    assert np.array_equal(base['smiles'],extra['smiles'])
    lookup={s:i for i,s in enumerate(base['smiles'])};paths=[];started=time.time();new_count=0
    with threadpool_limits(limits=1):
        for name,path in files.items():
            df=pd.read_csv(path);p=np.vstack([protein_features(s) for s in df.protein_sequence]);rows=[lookup[s] for s in df.lipid_smiles]
            for rep in REPRESENTATIONS:
                x=(base if rep in base.files else extra)[rep][rows]
                for seed in SEEDS:
                    _,initial,_=split(len(df),seed,budget(len(df))[0]);geom=Geometry(x,p,initial)
                    for kernel,protocol in itertools.product(KERNELS,PROTOCOLS):
                        relative=Path('runs')/name/rep/kernel/protocol/f'seed{seed}.json'
                        reused=rep in ('ecfp','maccs','chemberta') and protocol in OLD_PROTOCOLS
                        dest=(OLD if reused else OUT)/relative
                        if dest.exists():
                            assert json.loads(dest.read_text())['signature']==(old['signature'] if reused else signature)
                        else:
                            if reused: raise ValueError('Missing original run: '+str(dest))
                            begin=time.time();result=run_one(df,geom,kernel,protocol,seed)
                            result.update(config=dict(dataset=name,representation=rep,kernel=kernel,protocol=protocol,seed=seed),signature=signature,runtime_seconds=time.time()-begin)
                            dest.parent.mkdir(parents=True,exist_ok=True)
                            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(dest);new_count+=1
                        paths.append(dest)
                        if len(paths)%65==0:
                            progress=dict(completed=len(paths),expected=config['expected_runs'],new_this_process=new_count,elapsed_seconds=time.time()-started)
                            (OUT/'progress.json').write_text(json.dumps(progress));print(progress,flush=True)
                        if args.limit and new_count>=args.limit:
                            aggregate(paths);return
    summary=aggregate(paths)
    print('COMPLETE',len(paths),'runs;',len(summary),'three-seed configurations',flush=True)

if __name__=='__main__':main()
