"""Reduced nested-holdout study: six repeats, 120 acquisition labels + validation cost.

Original 5,850-run experiment remains immutable. Selection uses inner validation
MAE only. Outer metrics are not used to select representations, kernels or weights.
"""
from pathlib import Path
import argparse,itertools,json,hashlib,time,platform
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error,r2_score
from threadpoolctl import threadpool_limits
from grid_gp import Geometry,ExactPairGP
from prediction_aware_acquisition import POLICIES,select,LAMBDA
from robust_splits import SEEDS,sha
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/robust_study_v2';CACHE=ROOT/'data/features'
PROTEIN_REPS=('composition','esm2_8m','esm2_35m')
LIPID_REPS=('ecfp','maccs','chemberta','chemberta_mlm','molformer')
KERNELS=('linear','rbf','matern32')
MODES=('random_pair','cold_sequence40','scaffold')

def metric(model,idx,y):
    pred,_,sd=model.predict(idx);truth=y[idx]
    return dict(mae=float(mean_absolute_error(truth,pred)),r2=float(r2_score(truth,pred)),
                coverage90=float(np.mean(abs(truth-pred)<=1.64485362695*sd))),pred

def run_one(df,geom,kernel,policy,partition,seed):
    initial=np.array(partition['initial']);labelled=initial.copy();pool=np.array(partition['pool'])
    validation=np.array(partition['validation']);test=np.array(partition['test']);campaign=np.r_[labelled,pool]
    proxy_rng=np.random.default_rng(seed+9981);proxy=proxy_rng.choice(validation,min(64,len(validation)),replace=False)
    rng=np.random.default_rng(seed+104729);y=df.affinity.to_numpy(float);model=ExactPairGP(geom,kernel,maxiter=60)
    hist=[];queries=[];fits=[];threshold=np.sort(y[campaign])[-int(np.ceil(.1*len(campaign)))];top=campaign[y[campaign]>=threshold]
    for step in range(5):
        model.fit(labelled,y[labelled]);fits.append(dict(round=step,**model.diagnostics))
        # Evaluation is read-only; acquisition never receives this dictionary or y.
        val,_=metric(model,validation,y);outer,pred=metric(model,test,y)
        hist.append(dict(round=step,labelled=len(labelled),validation_mae=val['mae'],test_mae=outer['mae'],test_r2=outer['r2'],
                         test_coverage90=outer['coverage90'],recall_top10=float(np.isin(top,labelled).mean())))
        if step==4:break
        q,audit=select(model,pool,proxy,policy,24,rng)
        queries.append(dict(round=step+1,indices=q.tolist(),variance_audit=audit))
        labelled=np.r_[labelled,q];pool=pool[~np.isin(pool,q)]
    assert not set(labelled)&(set(test)|set(validation)) and len(set(labelled))==120
    return dict(history=hist,queries=queries,fits=fits,initial_indices=initial.tolist(),validation_indices=validation.tolist(),
                test_indices=test.tolist(),proxy_indices=proxy.tolist(),test_predictions=pred.tolist(),
                total_label_cost=120+len(validation),split_audit=partition['audit'])

def aggregate(paths):
    finals=[];curves=[]
    for path in paths:
        r=json.loads(path.read_text());h=r['history'];c=r['config']
        auc=float(np.trapz([x['recall_top10'] for x in h],[x['labelled'] for x in h])/96)
        finals.append({**c,**h[-1], 'recall_auc':auc,'nonconverged_fits':sum(not f['success'] for f in r['fits']),
                       'total_label_cost':r['total_label_cost'],'run_path':path.relative_to(ROOT).as_posix()})
        curves.extend({**c,**x} for x in h)
    frame=pd.DataFrame(finals);frame.to_csv(OUT/'final_by_seed.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'all_learning_curves.csv',index=False)
    keys=['dataset','split','protein_representation','representation','kernel','protocol']
    frame.groupby(keys).agg(n_seeds=('seed','nunique'),mae_mean=('test_mae','mean'),mae_sd=('test_mae','std'),r2_mean=('test_r2','mean'),
        recall_mean=('recall_top10','mean'),recall_auc_mean=('recall_auc','mean'),validation_mae_mean=('validation_mae','mean')).reset_index().to_csv(OUT/'summary.csv',index=False)
    # Deterministic tie-break uses configuration names, never outer-test scores.
    ordered=frame.sort_values(['validation_mae','protein_representation','representation','kernel'],kind='stable')
    selected=ordered.groupby(['dataset','split','seed','protocol'],sort=False).head(1).copy()
    selected.to_csv(OUT/'validation_selected_by_seed.csv',index=False)
    selected.groupby(['dataset','split','protocol']).agg(n_seeds=('seed','nunique'),mae_mean=('test_mae','mean'),mae_sd=('test_mae','std'),
        r2_mean=('test_r2','mean'),recall_mean=('recall_top10','mean'),recall_auc_mean=('recall_auc','mean')).reset_index().to_csv(OUT/'validation_selected_summary.csv',index=False)
    # Keep protein representation fixed to quantify its effect after inner selection
    # of lipid representation/kernel. No best-test representation ranking is selected.
    fixed=ordered.groupby(['dataset','split','seed','protocol','protein_representation'],sort=False).head(1)
    fixed.to_csv(OUT/'protein_validation_selected_by_seed.csv',index=False)
    return frame,selected

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    partitions=json.loads((CACHE/'robust_splits.json').read_text());proteins=np.load(CACHE/'protein_representations.npz')
    base=np.load(CACHE/'representations.npz');extra=np.load(CACHE/'extended_representations.npz');assert np.array_equal(base['smiles'],extra['smiles'])
    config=dict(version=2,protein_geometry_dtype='float64 before initial standardization and distance calculation',seeds=SEEDS,protein_representations=PROTEIN_REPS,lipid_representations=LIPID_REPS,kernels=KERNELS,
        policies=POLICIES,splits=MODES,datasets=['biodolphin_Kd','biodolphin_Ki'],initial=24,batch=24,cycles=4,
        prediction_weight=LAMBDA,proxy='fixed <=64 inner-validation inputs; no proxy labels used in acquisition',
        selector='inner validation MAE only; deterministic lexical ties; no outer-test selection',
        validation_cost='all inner validation labels counted separately from 120 acquisition labels',
        source_hashes={n:sha(ROOT/n) for n in ['run_robust_study.py','prediction_aware_acquisition.py','robust_splits.py','prepare_protein_encoders.py','grid_gp.py']},
        cache_hashes={n:sha(CACHE/n) for n in ['robust_splits.json','protein_representations.npz','protein_representation_manifest.json','representations.npz','extended_representations.npz']},
        data_hashes={n:sha(ROOT/'data/processed'/f'{n}.csv') for n in ['biodolphin_Kd','biodolphin_Ki']},
        expected_runs=2*len(MODES)*len(SEEDS)*len(PROTEIN_REPS)*len(LIPID_REPS)*len(KERNELS)*len(POLICIES),
        python=platform.python_version(),numpy=np.__version__)
    for n,digest in partitions['data_sha256'].items():assert config['data_hashes'][n]==digest
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();manifest=OUT/'experiment_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:raise ValueError('Study changed; preserve results in a different output directory')
    manifest.write_text(json.dumps(dict(signature=signature,**config),indent=2));paths=[];started=time.time();new=0
    plook={s:i for i,s in enumerate(proteins['sequences'])};llook={s:i for i,s in enumerate(base['smiles'])}
    with threadpool_limits(limits=1):
        for dataset in config['datasets']:
            df=pd.read_csv(ROOT/'data/processed'/f'{dataset}.csv');pr=[plook[s] for s in df.protein_sequence];lr=[llook[s] for s in df.lipid_smiles]
            for mode,seed in itertools.product(MODES,SEEDS):
                part=partitions['splits'][f'{dataset}/{mode}/seed{seed}']
                for protein_rep,lipid_rep,kernel in itertools.product(PROTEIN_REPS,LIPID_REPS,KERNELS):
                    geom=Geometry((base if lipid_rep in base.files else extra)[lipid_rep][lr],proteins[protein_rep][pr].astype(np.float64),np.array(part['initial']))
                    for policy in POLICIES:
                        conf=dict(dataset=dataset,split=mode,seed=seed,protein_representation=protein_rep,representation=lipid_rep,kernel=kernel,protocol=policy)
                        dest=OUT/'runs'/dataset/mode/protein_rep/lipid_rep/kernel/policy/f'seed{seed}.json';paths.append(dest)
                        if dest.exists():assert json.loads(dest.read_text())['signature']==signature
                        else:
                            begin=time.time();r=run_one(df,geom,kernel,policy,part,seed);r.update(config=conf,signature=signature,runtime_seconds=time.time()-begin)
                            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(r));tmp.replace(dest);new+=1
                        if len(paths)%30==0:
                            progress=dict(completed=len(paths),expected=config['expected_runs'],elapsed_seconds=time.time()-started,last=conf)
                            (OUT/'progress.json').write_text(json.dumps(progress));print(progress,flush=True)
                        if args.limit and new>=args.limit:aggregate(paths);return
    aggregate(paths);print('COMPLETE',len(paths),'matched campaigns',flush=True)

if __name__=='__main__':main()
