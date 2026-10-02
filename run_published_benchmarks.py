"""Reproducible offline active learning on downloaded measurements only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from protein_lipid_active_learning import run_active_learning

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'processed'
OUT=ROOT/'results'/'published'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    datasets={n:DATA/(n+'.csv') for n in ['biodolphin_Kd','biodolphin_Ki','traak_a_Kd1','traak_b_Kd1']}
    targets=pd.read_csv(DATA/'per_target_manifest.csv')
    for _,row in targets[targets.n_pairs>=10].iterrows():
        datasets[row.dataset]=DATA/row.file
    all_curves=[]; final=[]; protocols={}
    methods=[('random','random',0),('greedy','greedy',0),('uncertainty','uncertainty',0),
             ('ei','ei',0),('thompson','thompson',0),('thompson_diverse','thompson',0.2)]
    with threadpool_limits(limits=1):
        for name,path in datasets.items():
            df=pd.read_csv(path)
            initial,batch,rounds=(24,8,5) if len(df)>=100 else (4,1,4)
            protocols[name]=dict(n_pairs=len(df),initial=initial,batch=batch,rounds=rounds,
                                 split='seeded random unique pair; 20% fixed held-out',seeds=[7,19,42])
            for seed in [7,19,42]:
                for label,method,diversity in methods:
                    h,s,e=run_active_learning(df,initial,batch,rounds,method,seed=seed,diversity=diversity)
                    run=OUT/name/f'{label}_seed{seed}';run.mkdir(parents=True,exist_ok=True)
                    h.to_csv(run/'learning_curve.csv',index=False)
                    s.to_csv(run/'acquired_measurements.csv',index=False)
                    e.to_csv(run/'prediction_sensitivity.csv')
                    # Save exact frozen test and initial row identities for audit.
                    idx=np.random.default_rng(seed).permutation(len(df));nt=max(1,round(.2*len(df)))
                    df.iloc[idx[:nt]].to_csv(run/'held_out.csv',index=False)
                    df.iloc[idx[nt:nt+initial]].to_csv(run/'initial_labels.csv',index=False)
                    h=h.assign(dataset=name,method=label,seed=seed)
                    all_curves.append(h);final.append(h.iloc[-1].to_dict())
                print(name,'seed',seed,'complete',flush=True)
    curves=pd.concat(all_curves,ignore_index=True);curves.to_csv(OUT/'all_learning_curves.csv',index=False)
    finals=pd.DataFrame(final);finals.to_csv(OUT/'final_by_seed.csv',index=False)
    summary=finals.groupby(['dataset','method']).agg(
        mae_mean=('test_mae','mean'),mae_sd=('test_mae','std'),
        recall_mean=('top10pct_recall','mean'),recall_sd=('top10pct_recall','std'),
        r2_mean=('test_r2','mean'),labelled=('labelled','first')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps(protocols,indent=2),encoding='utf8')
    fig,axes=plt.subplots(len(datasets),2,figsize=(11,3*len(datasets)),squeeze=False)
    for row,name in enumerate(datasets):
        for col,(metric,ylabel) in enumerate([('test_mae','Held-out MAE (pK)'),('top10pct_recall','Top-10% pool recall')]):
            for label,_,_ in methods:
                sub=curves[(curves.dataset==name)&(curves.method==label)]
                g=sub.groupby('labelled')[metric].agg(['mean','std'])
                ax=axes[row,col];ax.plot(g.index,g['mean'],label=label)
                ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],alpha=.1)
            ax.set(title=name,xlabel='Measured pairs',ylabel=ylabel)
            if row==0 and col==1:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
