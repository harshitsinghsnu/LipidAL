"""Verify adaptation training boundaries and complete matched coverage."""
import itertools,json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from run_adaptation_study import ROOT,CACHE,OUT,ROBUST,SEEDS,MODES,KERNELS,POLICIES
from robust_splits import sha

def main():
    manifest=json.loads((OUT/'experiment_manifest.json').read_text());assert sha(ROOT/'run_adaptation_study.py')==manifest['source_sha256']
    for n,h in manifest['source_hashes'].items():assert sha(ROOT/n)==h
    assert sha(ROBUST/'experiment_manifest.json')==manifest['robust_manifest_sha256']
    assert sha(CACHE/'lipid_ssl/manifest.json')==manifest['ssl_manifest_sha256']
    assert sha(CACHE/'lipid_ssl/representations.npz')==manifest['ssl_features_sha256']
    parts=json.loads((CACHE/'robust_splits.json').read_text())['splits'];f=pd.read_csv(OUT/'final_by_seed.csv');variants=manifest['variants']
    keys=['dataset','split','seed','variant','kernel','protocol'];expected=set(itertools.product(['biodolphin_Kd','biodolphin_Ki'],MODES,SEEDS,variants,KERNELS,POLICIES))
    assert len(f)==len(expected) and set(map(tuple,f[keys].to_numpy()))==expected
    for dataset,mode,seed in itertools.product(['biodolphin_Kd','biodolphin_Ki'],MODES,SEEDS):
        part=parts[f'{dataset}/{mode}/seed{seed}'];ad=OUT/'adapters'/dataset/mode/f'seed{seed}';meta=json.loads((ad/'metadata.json').read_text())
        assert meta['training_indices']==part['initial'] and meta['affinity_labels_used']==24
        assert not set(meta['training_indices'])&(set(part['test'])|set(part['validation'])|set(part['pool']))
        assert sha(ad/'features.npz')==meta['feature_sha256']
    for _,row in f.iterrows():
        r=json.loads((ROOT/row.run_path).read_text());part=parts[f'{row.dataset}/{row.split}/seed{row.seed}'];used=set(r['initial_indices'])
        for q in r['queries']:
            assert len(q['indices'])==24 and not used&set(q['indices']);used.update(q['indices'])
        assert len(used)==120 and not used&(set(part['test'])|set(part['validation']))
        assert r['total_label_cost']==120+len(part['validation'])
        if row.variant!='frozen_mlm':assert r['signature']==manifest['signature']
    def scaff(s):return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),includeChirality=False) or 'ACYCLIC'
    corpus=pd.read_csv(CACHE/'lipid_ssl/training_smiles.csv');benchmark=np.load(CACHE/'representations.npz')['smiles']
    assert not set(corpus.lipid_smiles)&set(benchmark)
    assert not {scaff(s) for s in corpus.lipid_smiles}&{scaff(s) for s in benchmark}
    selected=pd.read_csv(OUT/'validation_selected_by_seed.csv')
    for key,g in f.groupby(['dataset','split','seed','variant','protocol']):
        s=selected
        for name,value in zip(['dataset','split','seed','variant','protocol'],key):s=s[s[name]==value]
        assert len(s)==1 and np.isclose(s.validation_mae.iloc[0],g.validation_mae.min())
    report=dict(status='PASS',comparisons=len(f),reused_frozen_baselines=int((f.variant=='frozen_mlm').sum()),
        new_campaigns=int((f.variant!='frozen_mlm').sum()),adapter_fits=36,adapter_labels_per_fit=24,
        adapter_initial_only_verified=True,validation_only_kernel_selection_verified=True,ssl_training_molecules=len(corpus),
        ssl_benchmark_scaffold_overlap=0,all_features_frozen_before_acquisition=True,
        nonconverged_fit_count_including_reused=int(f.nonconverged_fits.sum()))
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
