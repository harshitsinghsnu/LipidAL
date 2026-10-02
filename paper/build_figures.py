"""Four-panel manuscript figures and complete 4--5-panel source-figure atlas."""
from pathlib import Path
import sys, json, hashlib, textwrap, shutil
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'python_tools'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
ROOT=HERE.parent
RES=ROOT/'results'
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
S=pd.read_csv(RES/'expanded_grid/summary.csv')
C=pd.read_csv(RES/'expanded_grid/all_learning_curves.csv')
protocols=list(S.protocol.unique())
colors=dict(zip(protocols,plt.get_cmap('tab20').colors))

def panel(ax,i,title):
    ax.set_title(f'({chr(65+i)}) {title}',loc='left',fontweight='bold')
def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10,7))
for j,dataset in enumerate(['biodolphin_Kd','biodolphin_Ki']):
    frame=S[S.dataset==dataset]
    for protocol,g in frame.groupby('protocol'):
        axes[0,j].scatter(g.mae_mean,g.recall_mean,label=protocol.replace('ucb_',''),s=18,alpha=.7,color=colors[protocol])
    panel(axes[0,j],j,f'{dataset}: prediction vs discovery')
    axes[0,j].set(xlabel='Held-out MAE (pK)',ylabel='Final top-10% recall')
    # Average within each seed across the 25 model configurations, then report
    # seed mean and SD: configurations are not independent replications.
    curves=C[(C.dataset==dataset)].groupby(['protocol','seed','labelled']).recall_top10.mean().reset_index()
    for protocol,g in curves.groupby('protocol'):
        agg=g.groupby('labelled').recall_top10.agg(['mean','std'])
        axes[1,j].plot(agg.index,agg['mean'],color=colors[protocol],label=protocol.replace('ucb_',''))
        axes[1,j].fill_between(agg.index,agg['mean']-agg['std'],agg['mean']+agg['std'],color=colors[protocol],alpha=.09)
    panel(axes[1,j],j+2,f'{dataset}: protocol recall trajectories')
    axes[1,j].set(xlabel='Acquired labels',ylabel='Top-10% recall',ylim=(0,1.03))
axes[0,1].legend(fontsize=6,ncol=2)
save(fig,'fig1_performance')

fig,axes=plt.subplots(2,2,figsize=(11,9))
for i,(dataset,metric) in enumerate([(d,m) for d in ['biodolphin_Kd','biodolphin_Ki'] for m in ['mae_mean','recall_auc_mean']]):
    ax=axes.flat[i]
    tab=S[S.dataset==dataset].pivot(index=['representation','kernel'],columns='protocol',values=metric)
    im=ax.imshow(tab.to_numpy(),aspect='auto',cmap='viridis_r' if metric=='mae_mean' else 'viridis')
    ax.set_yticks(range(len(tab)),[f'{a}/{b}' for a,b in tab.index],fontsize=6)
    ax.set_xticks(range(len(tab.columns)),[s.replace('ucb_','') for s in tab.columns],rotation=55,ha='right',fontsize=6)
    panel(ax,i,f'{dataset}: '+('MAE (lower)' if metric=='mae_mean' else 'Recall AUC (higher)'))
    fig.colorbar(im,ax=ax,fraction=.045)
save(fig,'fig2_grid')

fig,axes=plt.subplots(2,2,figsize=(10,7))
for i,(dataset,protocol) in enumerate([(d,p) for d in ['biodolphin_Kd','biodolphin_Ki'] for p in ['ucb_explore_heavy','ucb_exploit_heavy']]):
    ax=axes.flat[i]
    tables=[pd.read_csv(RES/'paper_explainability'/dataset/protocol/f'seed{s}'/'feature_evolution.csv',index_col=0) for s in [7,19,42]]
    means=np.mean([t.to_numpy() for t in tables],axis=0)
    chosen=np.argsort(-means[-1])[:5]
    for bit in chosen:
        a=np.array([t.iloc[:,bit].to_numpy() for t in tables])
        line=ax.plot(range(11),a.mean(0),label=str(bit))[0]
        ax.fill_between(range(11),np.maximum(0,a.mean(0)-a.std(0,ddof=1)),a.mean(0)+a.std(0,ddof=1),color=line.get_color(),alpha=.1)
    panel(ax,i,f'{dataset}: '+protocol.replace('ucb_',''))
    ax.set(xlabel='AL cycle',ylabel='Mean |bit SHAP| (pK)')
    ax.legend(title='ECFP bit',fontsize=6,ncol=3,title_fontsize=7)
save(fig,'fig3_temporal')

fig,axes=plt.subplots(2,2,figsize=(10,7))
E=pd.read_csv(RES/'explainability/summary.csv')
for j,d in enumerate(['biodolphin_Kd','biodolphin_Ki']):
    g=E[E.dataset==d].groupby('protocol')[['lipid_mean_abs_shap_pK','protein_mean_abs_shap_pK']].mean()
    ax=axes[0,j];positions=np.arange(len(g))
    ax.bar(positions-.18,g.iloc[:,0],.36,label='Lipid');ax.bar(positions+.18,g.iloc[:,1],.36,label='Protein')
    ax.set_xticks(positions,[s.replace('ucb_','') for s in g.index],rotation=15,fontsize=7)
    ax.set_ylabel('Mean |two-group SHAP| (pK)');ax.legend(fontsize=7)
    panel(ax,j,f'{d}: selected-model attributions')
P=pd.read_csv(RES/'paper_explainability/cross_protocol_stability.csv')
for d,g in P[P.dataset.isin(['biodolphin_Kd','biodolphin_Ki'])].groupby('dataset'):
    tab=g.groupby('cycle').top10_jaccard.agg(['mean','std']);ax=axes[1,0]
    line=ax.plot(tab.index,tab['mean'],label=d)[0]
    ax.fill_between(tab.index,np.maximum(0,tab['mean']-tab['std']),np.minimum(1,tab['mean']+tab['std']),alpha=.1,color=line.get_color())
axes[1,0].set(xlabel='Cycle',ylabel='Cross-protocol top-10 Jaccard',ylim=(0,1.02));axes[1,0].legend(fontsize=7)
panel(axes[1,0],2,'ECFP-linear protocol stability')
best=S.loc[S.groupby('dataset').mae_mean.idxmin()]
small=best[~best.dataset.isin(['biodolphin_Kd','biodolphin_Ki'])]
axes[1,1].bar(range(len(small)),small.r2_mean,color='#b34d58')
axes[1,1].set_xticks(range(len(small)),['TTR-A' if '889' in s else 'TTR-B' if 'e1ec' in s else 'TRAAK-a' if 'traak_a' in s else 'TRAAK-b' for s in small.dataset],fontsize=7)
axes[1,1].axhline(0,color='black',lw=.8);axes[1,1].set_ylabel('Mean held-out R²')
panel(axes[1,1],3,'Small panels: lowest-MAE configurations')
save(fig,'fig4_explanations')

from rdkit import Chem
from rdkit.Chem import Draw
molecules=[];legends=[];atomlists=[];bondlists=[]
for dataset in ['biodolphin_Kd','biodolphin_Ki']:
    mapping=json.loads((RES/'paper_explainability'/dataset/'ucb_explore_heavy/seed7/fragment_mapping.json').read_text())
    data=pd.read_csv(ROOT/'data/processed'/f'{dataset}.csv')
    for fragment in mapping[:2]:
        env=sorted(fragment['environments'],key=lambda e:(e['radius']==0,e['radius'],e['center']))[0]
        molecules.append(Chem.MolFromSmiles(data.iloc[fragment['representative_row']].lipid_smiles))
        legends.append(f'({chr(65+len(legends))}) {dataset} | bit {fragment["bit"]}\nMeasured pK {fragment["representative_pK"]:.2f}; retrospective example')
        atomlists.append(env['atoms']);bondlists.append(env['bonds'])
svg=Draw.MolsToGridImage(molecules,molsPerRow=2,subImgSize=(550,370),legends=legends,
                        highlightAtomLists=atomlists,highlightBondLists=bondlists,useSVG=True)
(FIG/'fig5_fragments.svg').write_text(svg,encoding='utf-8')
renderPDF.drawToFile(svg2rlg(str(FIG/'fig5_fragments.svg')),str(FIG/'fig5_fragments.pdf'))

# Complete source atlas, including all original figures without editing them.
sources=sorted(p for p in RES.rglob('*') if p.suffix.lower() in ('.png','.svg'))
sizes=[];remaining=len(sources)
while remaining:
    size=4 if remaining in (4,8,12) else 5
    if remaining<size:raise ValueError('Cannot partition into groups of 4--5')
    sizes.append(size);remaining-=size
W,H=A3
c=canvas.Canvas(str(HERE/'all_figures_atlas.pdf'),pagesize=A3)
c.setTitle('Complete protein-lipid figure atlas: all source figures')
index=[];cursor=0
for page,size in enumerate(sizes,1):
    c.setFont('Helvetica-Bold',13);c.drawString(25,H-26,f'Figure S{page}: source-figure panels ({size})')
    c.setFont('Helvetica',8);c.drawString(25,H-42,'Supplementary atlas. Zoom for details; panel paths identify dataset, method, seed and analysis.')
    rows=2 if size==4 else 3;cellw=(W-60)/2;cellh=(H-85)/rows
    for slot,p in enumerate(sources[cursor:cursor+size]):
        col=slot%2;row=slot//2;left=25+col*(cellw+10);bottom=H-65-(row+1)*cellh
        rel=p.relative_to(ROOT).as_posix()
        c.setFont('Helvetica',7)
        label=f'({chr(65+slot)}) {rel}'
        for line,text in enumerate(textwrap.wrap(label,93)):
            c.drawString(left,bottom+cellh-10-line*9,text)
        boxh=cellh-46
        if p.suffix=='.svg':
            drawing=svg2rlg(str(p))
            scale=min(cellw/drawing.width,boxh/drawing.height)
            c.saveState();c.translate(left+(cellw-drawing.width*scale)/2,bottom+5);c.scale(scale,scale)
            renderPDF.draw(drawing,c,0,0);c.restoreState()
        else:
            c.drawImage(str(p),left,bottom+5,width=cellw,height=boxh,preserveAspectRatio=True,anchor='c')
        index.append(dict(atlas_page=page,figure=f'S{page}',panel=chr(65+slot),source=rel,
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    cursor+=size;c.showPage()
c.save()
pd.DataFrame(index).to_csv(HERE/'figure_manifest.csv',index=False)
assert len(index)==len(sources) and all(n in (4,5) for n in sizes)
(HERE/'figure_audit.json').write_text(json.dumps(dict(source_figures=len(sources),atlas_pages=len(sizes),
    panels_per_atlas_page=sizes,main_figures=7,panels_per_main_figure=4,all_sources_included=True),indent=2))
shutil.copy2(HERE/'template/ACML_camera_ready/jmlr.cls',HERE/'jmlr.cls')
print('Built five four-panel manuscript figures and atlas:',len(sources),'source figures;',len(sizes),'pages')
