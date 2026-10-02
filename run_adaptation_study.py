"""Matched lipid MLM and initial-label-only residual-adapter ablations.

ESM-2 35M + ChemBERTa-MLM, three kernels, six policies, same 36 partitions.
Original 648 frozen-baseline runs are reused, not refit or selected by test score.
"""
from pathlib import Path
import json,itertools,time,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from grid_gp import Geometry
from run_robust_study import ROOT,CACHE,OUT as ROBUST,run_one,MODES,SEEDS,KERNELS,POLICIES
from robust_splits import sha
OUT=ROOT/'results/adaptation_study'

def adapt(p,l,y_initial,initial,seed,dest):
    import torch
    from torch import nn
    torch.manual_seed(seed);torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
    pm=p[initial].mean(0);ps=np.maximum(p[initial].std(0),1e-4);lm=l[initial].mean(0);ls=np.maximum(l[initial].std(0),1e-4)
    assert len(y_initial)==len(initial)
    pn=(p-pm)/ps;ln=(l-lm)/ls;yc=float(y_initial.mean());ys=max(float(y_initial.std()),1e-6)
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__();self.pa=nn.Sequential(nn.Linear(p.shape[1],32),nn.GELU(),nn.Linear(32,p.shape[1]))
            self.la=nn.Sequential(nn.Linear(l.shape[1],32),nn.GELU(),nn.Linear(32,l.shape[1]))
            for branch in (self.pa,self.la):nn.init.zeros_(branch[-1].weight);nn.init.zeros_(branch[-1].bias)
            self.pp=nn.Linear(p.shape[1],32);self.lp=nn.Linear(l.shape[1],32);self.head=nn.Linear(96,1)
        def forward(self,p,l):
            ap=p+.1*self.pa(p);al=l+.1*self.la(l);hp=torch.tanh(self.pp(ap));hl=torch.tanh(self.lp(al))
            return ap,al,self.head(torch.cat([hp,hl,hp*hl],dim=1)).squeeze(1)
    model=Adapter().to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    tp=torch.tensor(pn[initial],dtype=torch.float32,device=device);tl=torch.tensor(ln[initial],dtype=torch.float32,device=device)
    target=torch.tensor((y_initial-yc)/ys,dtype=torch.float32,device=device);losses=[]
    model.train()
    for epoch in range(100):
        opt.zero_grad();ap,al,pred=model(tp,tl)
        loss=((pred-target)**2).mean()+.01*((ap-tp).square().mean()+(al-tl).square().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        ap,al,_=model(torch.tensor(pn,dtype=torch.float32,device=device),torch.tensor(ln,dtype=torch.float32,device=device))
        # Return to original feature units: a zero adapter exactly recovers baseline geometry.
        adapted_p=ap.cpu().numpy()*ps+pm;adapted_l=al.cpu().numpy()*ls+lm
    dest.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),dest/'adapter_weights.pt')
    np.savez_compressed(dest/'features.npz',protein=adapted_p,lipid=adapted_l)
    metadata=dict(training_indices=initial.tolist(),affinity_labels_used=len(initial),epochs=100,rank=32,residual_scale=.1,
        optimizer='AdamW lr=0.001 weight_decay=0.01',objective='standardized affinity MSE + 0.01 residual penalty',seed=seed,
        train_loss_initial=losses[0],train_loss_final=losses[-1],early_stopping=False,validation_or_test_labels_used=False,
        frozen_across_all_acquisition_cycles=True,method='Residual feature adapters on frozen encoders; not attention LoRA',
        feature_sha256=sha(dest/'features.npz'))
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2));return adapted_p,adapted_l

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parts=json.loads((CACHE/'robust_splits.json').read_text())['splits'];pro=np.load(CACHE/'protein_representations.npz')
    lip=np.load(CACHE/'extended_representations.npz');ssl=np.load(CACHE/'lipid_ssl/representations.npz')
    config=dict(version=1,source_sha256=sha(__file__),robust_manifest_sha256=sha(ROBUST/'experiment_manifest.json'),
        ssl_manifest_sha256=sha(CACHE/'lipid_ssl/manifest.json'),ssl_features_sha256=sha(CACHE/'lipid_ssl/representations.npz'),
        variants=['frozen_mlm','lipid_ssl','initial_label_adapter'],protein='esm2_35m',lipid='chemberta_mlm',
        expected_runs=2*len(MODES)*len(SEEDS)*len(KERNELS)*len(POLICIES)*3,
        source_hashes={n:sha(ROOT/n) for n in ['grid_gp.py','run_robust_study.py','prediction_aware_acquisition.py']})
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();manifest=OUT/'experiment_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:raise ValueError('Adaptation experiment changed')
    manifest.write_text(json.dumps(dict(signature=signature,**config),indent=2));records=[];started=time.time()
    pl={s:i for i,s in enumerate(pro['sequences'])};ll={s:i for i,s in enumerate(lip['smiles'])};sl={s:i for i,s in enumerate(ssl['smiles'])}
    with threadpool_limits(limits=1):
        for dataset in ('biodolphin_Kd','biodolphin_Ki'):
            df=pd.read_csv(ROOT/'data/processed'/f'{dataset}.csv');p=pro['esm2_35m'][[pl[s] for s in df.protein_sequence]].astype(float)
            l=lip['chemberta_mlm'][[ll[s] for s in df.lipid_smiles]].astype(float);lx=ssl['chemberta_lipid_ssl'][[sl[s] for s in df.lipid_smiles]].astype(float)
            for mode,seed in itertools.product(MODES,SEEDS):
                part=parts[f'{dataset}/{mode}/seed{seed}'];initial=np.array(part['initial']);ad=OUT/'adapters'/dataset/mode/f'seed{seed}'
                if (ad/'metadata.json').exists():
                    meta=json.loads((ad/'metadata.json').read_text());assert meta['training_indices']==initial.tolist() and sha(ad/'features.npz')==meta['feature_sha256']
                    cached=np.load(ad/'features.npz');ap,al=cached['protein'],cached['lipid']
                else:ap,al=adapt(p,l,df.affinity.to_numpy()[initial],initial,seed,ad)
                for variant,(pf,lf) in dict(frozen_mlm=(p,l),lipid_ssl=(p,lx),initial_label_adapter=(ap,al)).items():
                    geom=Geometry(lf,pf.astype(float),initial)
                    for kernel,policy in itertools.product(KERNELS,POLICIES):
                        conf=dict(dataset=dataset,split=mode,seed=seed,variant=variant,kernel=kernel,protocol=policy)
                        if variant=='frozen_mlm':
                            path=ROBUST/'runs'/dataset/mode/'esm2_35m/chemberta_mlm'/kernel/policy/f'seed{seed}.json'
                            r=json.loads(path.read_text())
                        else:
                            path=OUT/'runs'/dataset/mode/variant/kernel/policy/f'seed{seed}.json'
                            if path.exists():r=json.loads(path.read_text());assert r['signature']==signature
                            else:
                                r=run_one(df,geom,kernel,policy,part,seed);r.update(config=conf,signature=signature)
                                path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r));tmp.replace(path)
                        h=r['history'];records.append(dict(**conf,**h[-1],recall_auc=float(np.trapz([v['recall_top10'] for v in h],[v['labelled'] for v in h])/96),
                            run_path=path.relative_to(ROOT).as_posix(),total_label_cost=r['total_label_cost'],nonconverged_fits=sum(not f['success'] for f in r['fits'])))
                progress=dict(completed=len(records),expected=config['expected_runs'],elapsed_seconds=time.time()-started)
                (OUT/'progress.json').write_text(json.dumps(progress));print(progress,flush=True)
    final=pd.DataFrame(records);assert len(final)==config['expected_runs'];final.to_csv(OUT/'final_by_seed.csv',index=False)
    selected=final.sort_values(['validation_mae','kernel'],kind='stable').groupby(['dataset','split','seed','protocol','variant'],sort=False).head(1)
    selected.to_csv(OUT/'validation_selected_by_seed.csv',index=False)
    selected.groupby(['dataset','split','protocol','variant']).agg(n_seeds=('seed','nunique'),mae_mean=('test_mae','mean'),mae_sd=('test_mae','std'),r2_mean=('test_r2','mean'),
        recall_mean=('recall_top10','mean'),recall_auc_mean=('recall_auc','mean')).reset_index().to_csv(OUT/'validation_selected_summary.csv',index=False)
    print('COMPLETE',len(final),'adaptation comparisons (648 reused baselines)',flush=True)

if __name__=='__main__':main()
