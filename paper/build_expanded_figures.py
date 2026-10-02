"""Expanded paired-encoder comparisons, real portal screenshots and source tables."""
import sys,json,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE/'python_tools'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=HERE/'figures';TABLE=HERE/'results_tables';TABLE.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
reps=['ecfp','maccs','chemberta','chemberta_mlm','molformer']
names=['ECFP','MACCS','ChemBERTa\nMTR','ChemBERTa\nMLM','MoLFormer']
frame=pd.read_csv(ROOT/'results/expanded_grid/final_by_seed.csv')
fixed=frame[(frame.kernel=='matern32') & (frame.protocol=='thompson')]
fig,axes=plt.subplots(2,2,figsize=(10,7))
for j,dataset in enumerate(['biodolphin_Kd','biodolphin_Ki']):
    for row,(metric,title) in enumerate([('test_mae','MAE (pK; lower is better)'),('recall_auc','Recall AUC (higher is better)')]):
        ax=axes[row,j];tab=fixed[fixed.dataset==dataset].groupby('representation')[metric].agg(['mean','std']).reindex(reps)
        ax.bar(np.arange(5),tab['mean'],yerr=tab['std'],capsize=3,color=['#7896a5','#b6a18a','#087e83','#c58d3b','#7964a5'])
        ax.set_xticks(np.arange(5),names,fontsize=8);ax.set_ylabel(title)
        ax.set_title(f'({chr(65+row*2+j)}) BioDolphin '+('$K_d$' if j==0 else '$K_i$'),loc='left')
        if row:ax.set_ylim(0,1)
fig.tight_layout();fig.savefig(FIG/'fig6_representations.pdf',bbox_inches='tight');fig.savefig(FIG/'fig6_representations.png',dpi=220,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,9))
for ax,name,title in zip(axes.flat,['a_selection','b_curves','c_comparison','d_campaign'],
                       ['(A) Experiment selection','(B) Four-panel learning curves','(C) Matched representation comparison','(D) Local upload campaign controls']):
    ax.imshow(plt.imread(FIG/'dashboard'/f'{name}.png'));ax.axis('off');ax.set_title(title,loc='left',fontsize=11)
fig.tight_layout();fig.savefig(FIG/'fig7_dashboard.pdf',bbox_inches='tight');fig.savefig(FIG/'fig7_dashboard.png',dpi=200,bbox_inches='tight');plt.close(fig)
for name in ['summary.csv','final_by_seed.csv','representation_thompson_matern32.csv','audit.json','experiment_manifest.json']:
    shutil.copy2(ROOT/'results/expanded_grid'/name,TABLE/name)
for name in ['representation_manifest.json','extended_representation_manifest.json']:
    shutil.copy2(ROOT/'data/features'/name,TABLE/name)
summary=pd.read_csv(TABLE/'summary.csv');rows=[]
for dataset,g in summary.groupby('dataset'):
    for role,column,ascending in [('prediction','mae_mean',True),('discovery','recall_auc_mean',False)]:
        optimum=g[column].min() if ascending else g[column].max()
        for _,r in g[np.isclose(g[column],optimum,rtol=0,atol=1e-12)].iterrows():rows.append(dict(role=role,**r.to_dict()))
pd.DataFrame(rows).to_csv(TABLE/'descriptive_winners_including_ties.csv',index=False)
sources=[ROOT/'results/expanded_grid/summary.csv',ROOT/'results/expanded_grid/final_by_seed.csv',FIG/'dashboard/capture.json']
(HERE/'expanded_figure_audit.json').write_text(json.dumps(dict(runs=len(frame),configurations=len(summary),
    main_composites=7,panels_per_composite=4,screenshot_source='Local running portal; not a claimed public deployment',
    inputs={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
print('Built transformer comparison, four-screenshot dashboard panel and complete result tables.')
