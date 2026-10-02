"""Validate grid coverage and create descriptive comparison tables/heatmaps."""
from pathlib import Path
import itertools
import json
import hashlib
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'full_grid'

def main():
    (OUT/'package_versions.json').write_text(json.dumps({p:version(p) for p in
        ['numpy','scipy','pandas','scikit-learn','rdkit','torch','transformers','huggingface_hub','matplotlib']},indent=2))
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    expected=set(itertools.product(manifest['data_sha256'],manifest['representations'],manifest['kernels'],
                                   manifest['protocols'],manifest['seeds']))
    runs=list((OUT/'runs').glob('*/*/*/*/seed*.json'));seen=set();fits=[];budgets={}
    for path in runs:
        r=json.loads(path.read_text());c=r['config']
        key=tuple(c[k] for k in ['dataset','representation','kernel','protocol','seed'])
        assert key not in seen;seen.add(key)
        assert r['signature']==manifest['signature']
        test=set(r['test_indices']);initial=set(r['initial_indices'])
        acquired=[idx for batch in r['acquired'] for idx in batch['indices']]
        assert not test&initial and not test&set(acquired) and not initial&set(acquired)
        assert len(acquired)==len(set(acquired))
        assert len(r['history'])==11
        for step,(acq,batch) in enumerate(zip(r['acquired'],r['batches'])):
            assert acq['phase']==manifest['schedules'][c['protocol']][step]
            assert len(acq['indices'])==batch
        assert len(initial)+len(acquired)==r['history'][-1]['labelled']
        for f in r['fits']:
            assert np.isfinite(f['nll']) and f['nll']<=f['initial_nll']+1e-6
        assert np.isfinite(r['test_predictions']).all()
        assert len(r['test_predictions'])==len(test)
        seedkey=(c['dataset'],c['seed'])
        pattern=(tuple(r['test_indices']),tuple(r['initial_indices']),tuple(r['batches']))
        if seedkey in budgets:assert budgets[seedkey]==pattern
        else:budgets[seedkey]=pattern
        fits+=r['fits']
    missing=expected-seen;extra=seen-expected
    audit=dict(expected_runs=len(expected),completed_runs=len(seen),missing=list(missing),extra=list(extra),
               fitted_models=len(fits),optimizer_converged=sum(f['success'] for f in fits),
               max_iterations=manifest['maxiter'],split_overlap_checks='passed',
               matching_seed_initialization='passed',schedule_and_budget_checks='passed')
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2))
    if missing or extra:raise RuntimeError(f'Incomplete grid: {len(missing)} missing, {len(extra)} extra')
    final=pd.read_csv(OUT/'final_by_seed.csv')
    keys=['dataset','representation','kernel','seed']
    random=final[final.protocol=='random'][keys+['test_mae','recall_top10','recall_auc']]
    paired=final.merge(random,on=keys,suffixes=('','_random'),validate='many_to_one')
    paired['mae_improvement_vs_random']=paired.test_mae_random-paired.test_mae
    paired['recall_gain_vs_random']=paired.recall_top10-paired.recall_top10_random
    paired['recall_auc_gain_vs_random']=paired.recall_auc-paired.recall_auc_random
    paired.to_csv(OUT/'paired_vs_random_by_seed.csv',index=False)
    paired.groupby(['dataset','representation','kernel','protocol']).agg(
        mae_gain_mean=('mae_improvement_vs_random','mean'),mae_gain_sd=('mae_improvement_vs_random','std'),
        recall_gain_mean=('recall_gain_vs_random','mean'),recall_gain_sd=('recall_gain_vs_random','std'),
        auc_gain_mean=('recall_auc_gain_vs_random','mean'),auc_gain_sd=('recall_auc_gain_vs_random','std')
        ).reset_index().to_csv(OUT/'paired_vs_random_summary.csv',index=False)
    summary=pd.read_csv(OUT/'summary.csv')
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    order=[(r,k) for r in manifest['representations'] for k in manifest['kernels']]
    for dataset,g in summary.groupby('dataset'):
        fig,axes=plt.subplots(1,2,figsize=(16,9))
        for ax,metric,title in zip(axes,['mae_mean','recall_mean'],['Test MAE (pK), lower is better','Top-10% recall, higher is better']):
            pivot=g.pivot(index=['representation','kernel'],columns='protocol',values=metric).reindex(index=order,columns=manifest['protocols'])
            vals=pivot.to_numpy();im=ax.imshow(vals,aspect='auto',cmap='viridis_r' if metric=='mae_mean' else 'viridis')
            ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=45,ha='right',fontsize=8)
            ax.set_yticks(range(len(order)),[f'{r} / {k}' for r,k in order],fontsize=8)
            ax.set_title(title)
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    norm=im.norm(vals[i,j]);color='white' if (norm>.6 if metric=='mae_mean' else norm<.4) else 'black'
                    ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',fontsize=7,color=color)
            fig.colorbar(im,ax=ax,shrink=.65)
        fig.suptitle(f'{dataset}: three-seed means; exploratory rankings')
        fig.tight_layout();fig.savefig(plots/f'{dataset}_grid.png',dpi=160);plt.close(fig)
    best=[]
    for dataset,g in summary.groupby('dataset'):
        for objective,column,ascending in [('prediction','mae_mean',True),('discovery','recall_auc_mean',False)]:
            b=g.sort_values(column,ascending=ascending).iloc[0].to_dict();b['objective']=objective;best.append(b)
    pd.DataFrame(best).to_csv(OUT/'descriptive_best.csv',index=False)
    print(json.dumps(audit,indent=2));print(pd.DataFrame(best).to_string(index=False))

if __name__=='__main__':main()
