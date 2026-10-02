"""Run every paper representation/kernel/protocol plus Thompson, with checkpoints.

Pooled BioDolphin: 60 initial + 10 x 30 measured pairs, 20% frozen test.
Tiny targets: 2 initial + 6 acquisitions distributed over the same ten phases;
zero-budget phases are explicit and are not counted as new measurements.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import time
import platform
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from threadpoolctl import threadpool_limits
from protein_lipid_active_learning import protein_features
from grid_gp import Geometry,ExactPairGP,KERNELS,PROTOCOLS,SCHEDULES,select

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'processed'
OUT=ROOT/'results'/'full_grid'
REPS=('ecfp','maccs','chemberta')
SEEDS=(7,19,42)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def datasets():
    paths={n:DATA/(n+'.csv') for n in ['biodolphin_Kd','biodolphin_Ki','traak_a_Kd1','traak_b_Kd1']}
    for _,r in pd.read_csv(DATA/'per_target_manifest.csv').query('n_pairs>=10').iterrows():paths[r.dataset]=DATA/r.file
    return paths

def budget(n):
    if n>=450:return 60,[30]*10
    # Preserve all ten phase positions but explicitly reduce small-panel budgets.
    initial=2;total=min(8,n-max(2,round(.2*n)))
    return initial,np.diff(np.floor(np.linspace(0,total-initial,11)).astype(int)).tolist()

def split(n,seed,initial):
    idx=np.random.default_rng(seed).permutation(n);nt=max(2,round(.2*n))
    return idx[:nt],idx[nt:nt+initial],idx[nt+initial:]

def evaluate(model,test,labelled,campaign,y):
    pred,_,sd=model.predict(test);truth=y[test]
    out=dict(labelled=len(labelled),test_mae=float(mean_absolute_error(truth,pred)),
             test_rmse=float(np.sqrt(mean_squared_error(truth,pred))),test_r2=float(r2_score(truth,pred)),
             test_coverage90=float(np.mean(abs(truth-pred)<=1.64485362695*sd)))
    for pct in [2,5,10]:
        k=max(1,int(np.ceil(len(campaign)*pct/100)))
        # Include ties at the percentile threshold rather than arbitrary row-order tie-breaking.
        threshold=np.sort(y[campaign])[-k];top=campaign[y[campaign]>=threshold]
        out[f'recall_top{pct}']=float(np.isin(top,labelled).mean())
    out['best_observed']=float(y[labelled].max())
    out['simple_regret']=float(y[campaign].max()-y[labelled].max())
    return out,pred,sd

def run_one(df,geom,kernel,protocol,seed,maxiter):
    initial,batches=budget(len(df));test,labelled,pool=split(len(df),seed,initial)
    test=test.copy();labelled=labelled.copy();pool=pool.copy();campaign=np.r_[labelled,pool]
    # Separate acquisition randomness from fixed partition generation.
    rng=np.random.default_rng(seed+104729)
    y=df.affinity.to_numpy(float);model=ExactPairGP(geom,kernel,maxiter=maxiter)
    history=[];acquired=[];fits=[]
    for step in range(11):
        if step==0 or batches[step-1]>0:
            model.fit(labelled,y[labelled]);fits.append(dict(round=step,**model.diagnostics))
        metrics,pred,sd=evaluate(model,test,labelled,campaign,y)
        metrics.update(round=step,phase='initial' if step==0 else SCHEDULES[protocol][step-1])
        history.append(metrics)
        if step==10:break
        q=select(model,pool,SCHEDULES[protocol][step],batches[step],rng)
        acquired.append(dict(round=step+1,phase=SCHEDULES[protocol][step],indices=q.tolist()))
        labelled=np.r_[labelled,q];pool=pool[~np.isin(pool,q)]
    assert not set(test)&set(labelled)
    assert len(set(labelled))==len(labelled)
    return dict(history=history,acquired=acquired,test_indices=test.tolist(),
                initial_indices=split(len(df),seed,initial)[1].tolist(),fits=fits,
                test_predictions=pred.tolist(),test_sd=sd.tolist(),initial=initial,batches=batches)

def aggregate(paths):
    finals=[];curves=[];failures=[]
    for p in paths:
        r=json.loads(p.read_text());conf=r['config'];hist=r['history']
        row={**conf,**hist[-1], 'runtime_seconds':r['runtime_seconds']}
        labels=np.array([x['labelled'] for x in hist]);recall=np.array([x['recall_top10'] for x in hist])
        row['recall_auc']=float(np.trapz(recall,labels)/(labels[-1]-labels[0]))
        row['fit_converged_fraction']=float(np.mean([f['success'] for f in r['fits']]))
        finals.append(row)
        for h in hist:curves.append({**conf,**h})
        for f in r['fits']:
            if not f['success']:failures.append({**conf,**f})
    final=pd.DataFrame(finals);curve=pd.DataFrame(curves)
    final.to_csv(OUT/'final_by_seed.csv',index=False);curve.to_csv(OUT/'all_learning_curves.csv',index=False)
    summary=final.groupby(['dataset','representation','kernel','protocol']).agg(
        n_seeds=('seed','nunique'),labelled=('labelled','first'),mae_mean=('test_mae','mean'),mae_sd=('test_mae','std'),
        r2_mean=('test_r2','mean'),recall_mean=('recall_top10','mean'),recall_sd=('recall_top10','std'),
        recall_auc_mean=('recall_auc','mean'),coverage90_mean=('test_coverage90','mean'),
        convergence_fraction=('fit_converged_fraction','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    # Descriptive rankings; never use held-out scores to refit or select future labels.
    summary.sort_values('mae_mean').groupby('dataset',sort=False).head(5).to_csv(OUT/'top5_by_test_mae.csv',index=False)
    summary.sort_values('recall_auc_mean',ascending=False).groupby('dataset',sort=False).head(5).to_csv(OUT/'top5_by_recall_auc.csv',index=False)
    (OUT/'optimizer_nonconvergence.json').write_text(json.dumps(failures,indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--maxiter',type=int,default=60)
    ap.add_argument('--limit',type=int,default=0,help='diagnostic number of configurations; zero means full grid')
    ap.add_argument('--aggregate-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    files=datasets();feature_file=ROOT/'data'/'features'/'representations.npz'
    config=dict(version=1,seeds=SEEDS,representations=REPS,kernels=KERNELS,protocols=PROTOCOLS,
                schedules=SCHEDULES,maxiter=args.maxiter,optimizer='L-BFGS-B analytic exact MLL, warm starts',
                data_sha256={n:sha(p) for n,p in files.items()},feature_sha256=sha(feature_file),
                engine_sha256=sha(ROOT/'grid_gp.py'),
                runner_sha256=sha(Path(__file__)),
                python=sys.version,numpy=np.__version__,platform=platform.platform(),
                protein_factor='Matern32 learned length; initial-label standardized composition',
                test_fraction=.2,small_budget='2 initial + 6 queries over 10 scheduled phases',
                expected_runs=len(files)*len(REPS)*len(KERNELS)*len(PROTOCOLS)*len(SEEDS))
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest=OUT/'experiment_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:
        raise RuntimeError('Config/data/engine changed. Use a new output directory to preserve the previous grid.')
    manifest.write_text(json.dumps(dict(signature=signature,**config),indent=2))
    paths=[];count=0;started=time.time()
    if args.aggregate_only:
        paths=list((OUT/'runs').glob('*/*/*/*/seed*.json'));aggregate(paths);return
    features=np.load(feature_file,allow_pickle=False);lookup={s:i for i,s in enumerate(features['smiles'])}
    with threadpool_limits(limits=1):
        for name,path in files.items():
            df=pd.read_csv(path);assert not df.duplicated(['protein_sequence','lipid_smiles']).any()
            p=np.vstack([protein_features(s) for s in df.protein_sequence]);rows=[lookup[s] for s in df.lipid_smiles]
            for rep in REPS:
                x=features[rep][rows]
                for seed in SEEDS:
                    _,initial,_=split(len(df),seed,budget(len(df))[0]);geom=Geometry(x,p,initial)
                    for kernel,protocol in itertools.product(KERNELS,PROTOCOLS):
                        dest=OUT/'runs'/name/rep/kernel/protocol/f'seed{seed}.json';paths.append(dest)
                        if not dest.exists():
                            begin=time.time();result=run_one(df,geom,kernel,protocol,seed,args.maxiter)
                            result.update(config=dict(dataset=name,representation=rep,kernel=kernel,protocol=protocol,seed=seed),
                                          signature=signature,runtime_seconds=time.time()-begin)
                            dest.parent.mkdir(parents=True,exist_ok=True)
                            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(dest)
                        else:
                            assert json.loads(dest.read_text())['signature']==signature
                        count+=1
                        (OUT/'progress.json').write_text(json.dumps(dict(completed=count,expected=config['expected_runs'],
                            last=str(dest.relative_to(OUT)),elapsed_seconds=time.time()-started)))
                        if count%8==0:print(f'{count}/{config["expected_runs"]} {name} {rep} seed={seed} {kernel} elapsed={time.time()-started:.0f}s',flush=True)
                        if args.limit and count>=args.limit:
                            aggregate(paths);print('Diagnostic limit reached; full grid incomplete',flush=True);return
    summary=aggregate(paths)
    print('COMPLETE',len(paths),'runs;',len(summary),'three-seed configurations',flush=True)

if __name__=='__main__':main()
