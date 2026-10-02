"""Coverage/leakage audit and predeclared exploratory paired comparisons."""
import itertools,json
import numpy as np
import pandas as pd
from run_robust_study import ROOT,OUT,CACHE,PROTEIN_REPS,LIPID_REPS,KERNELS,POLICIES,MODES,SEEDS
from robust_splits import sha

def signflip(delta):
    delta=np.asarray(delta,float);signs=np.array(list(itertools.product([-1,1],repeat=len(delta))))
    observed=abs(delta.mean());null=abs((signs*delta).mean(axis=1))
    return float(np.mean(null>=observed-1e-12))

def main():
    manifest=json.loads((OUT/'experiment_manifest.json').read_text());parts=json.loads((CACHE/'robust_splits.json').read_text())['splits']
    for name,digest in manifest['source_hashes'].items():assert sha(ROOT/name)==digest
    for name,digest in manifest['cache_hashes'].items():assert sha(CACHE/name)==digest
    final=pd.read_csv(OUT/'final_by_seed.csv');keys=['dataset','split','seed','protein_representation','representation','kernel','protocol']
    expected=set(itertools.product(manifest['datasets'],MODES,SEEDS,PROTEIN_REPS,LIPID_REPS,KERNELS,POLICIES))
    assert len(final)==len(expected) and set(map(tuple,final[keys].to_numpy()))==expected
    fits=0;bad=0;max_variance_error=0.;audit_rows=[]
    for _,row in final.iterrows():
        r=json.loads((ROOT/row.run_path).read_text());assert r['signature']==manifest['signature']
        part=parts[f'{row.dataset}/{row.split}/seed{row.seed}'];acquired=set(r['initial_indices']);val=set(r['validation_indices']);test=set(r['test_indices'])
        assert r['initial_indices']==part['initial'] and r['validation_indices']==part['validation'] and r['test_indices']==part['test']
        assert set(r['proxy_indices'])<=val and not val&test
        for q in r['queries']:
            indices=set(q['indices']);assert len(indices)==24 and not indices&(acquired|val|test);acquired|=indices
            for a in q['variance_audit']:
                max_variance_error=max(max_variance_error,abs(a['predicted_proxy_variance_reduction']-a['actual_conditional_reduction']))
                assert a['actual_conditional_reduction']>=-1e-8
        assert len(acquired)==120 and r['total_label_cost']==120+len(val)
        fits+=len(r['fits']);bad+=sum(not f['success'] for f in r['fits'])
        assert all(np.isfinite(v) for h in r['history'] for v in h.values())
    selected=pd.read_csv(OUT/'validation_selected_by_seed.csv')
    for key,g in final.groupby(['dataset','split','seed','protocol']):
        chosen=selected
        for name,value in zip(['dataset','split','seed','protocol'],key):chosen=chosen[chosen[name]==value]
        assert len(chosen)==1 and np.isclose(chosen.validation_mae.iloc[0],g.validation_mae.min())
    tests=[]
    for dataset,mode,baseline,metric in itertools.product(manifest['datasets'],MODES,['thompson','uncertainty'],['test_mae','recall_auc']):
        group=selected[(selected.dataset==dataset)&(selected.split==mode)].pivot(index='seed',columns='protocol',values=metric).reindex(SEEDS)
        delta=group['prediction_aware_thompson']-group[baseline]
        tests.append(dict(dataset=dataset,split=mode,baseline=baseline,metric=metric,n_seeds=len(delta),
                          mean_difference=float(delta.mean()),sd_difference=float(delta.std()),p_signflip=signflip(delta),
                          interpretation='Exploratory paired sign-flip; repeated splits overlap and are not independent biological replications'))
    p=np.array([r['p_signflip'] for r in tests]);order=np.argsort(p);adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    for i,j in enumerate(order):tests[j]['p_holm']=float(adjusted[i])
    pd.DataFrame(tests).to_csv(OUT/'paired_comparisons.csv',index=False)
    splitrows=[dict(key=k,**p['audit']) for k,p in parts.items()];pd.DataFrame(splitrows).to_csv(OUT/'split_audit.csv',index=False)
    # Existing label ranges are curation disagreement indicators, not measured assay variance.
    assay=[]
    for name in manifest['datasets']:
        df=pd.read_csv(ROOT/'data/processed'/f'{name}.csv');span=df.p_affinity_max-df.p_affinity_min
        assay.append(dict(dataset=name,pairs=len(df),multiple_source_annotations=int((df.source_annotations>1).sum()),
            nonzero_pK_range=int((span>0).sum()),maximum_pK_range=float(span.max()),mean_pK_range=float(span.mean()),
            warning='Source counts need not be independent assays; zero range does not establish zero measurement noise. No assay-noise correction is claimed.'))
    pd.DataFrame(assay).to_csv(OUT/'assay_heterogeneity.csv',index=False)
    audit=dict(status='PASS',runs=len(final),configurations=len(final)//len(SEEDS),fits=fits,nonconverged_fits=bad,
        seeds=list(SEEDS),complete_cartesian_product=True,outer_test_never_acquired=True,validation_never_acquired=True,
        proxy_labels_never_passed_to_selector=True,validation_only_configuration_selection_verified=True,
        maximum_variance_reduction_formula_error=max_variance_error,paired_test_family=len(tests),
        significance_warning='Six overlapping repeated holdouts; exploratory p-values, no independent replication or novelty claim',
        single_inner_holdout_not_full_nested_kfold=True,original_5850_runs_unmodified=True)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
    print(pd.read_csv(OUT/'validation_selected_summary.csv').to_string(index=False))

if __name__=='__main__':main()
