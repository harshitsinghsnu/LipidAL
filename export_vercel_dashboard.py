"""Explicit allowlisted, read-only public-site export. Never exports user_runs."""
from pathlib import Path
import json,hashlib,shutil
from dashboard import ROOT,catalog
from extended_features import REPRESENTATIONS
from extended_methods import KERNELS,PROTOCOLS,STATIC_PHASES

def main():
    out=ROOT/'vercel_dashboard';out.mkdir(exist_ok=True)
    rows=catalog();assets=set()
    public_reports=['results/expanded_grid/'+n for n in ['summary.csv','final_by_seed.csv','audit.json','representation_thompson_matern32.csv']]
    for row in rows:
        if row['kind']=='Full benchmark':
            row['downloads']=[dict(name=Path(p).name,path=p) for p in public_reports]
        for asset in row['plots']+row['downloads']:
            path=Path(asset['path'])
            if not path.parts or path.parts[0]!='results' or path.parts[1] not in ('explainability','paper_explainability','expanded_grid'):
                raise ValueError('Non-allowlisted asset: '+str(path))
            source=(ROOT/path).resolve()
            if not source.is_relative_to(ROOT/'results') or not source.is_file():raise ValueError(str(source))
            assets.add(path.as_posix())
    # Explicit derived-result allowlist: never copy checkpoints, raw data or uploads.
    extra={
        'robust_study_v2':['index.html','policy_panels.png','policy_panels.pdf','protein_panels.png','protein_panels.pdf','adaptation_panels.png','adaptation_panels.pdf','audit.json','experiment_manifest.json','final_by_seed.csv','summary.csv','validation_selected_summary.csv','validation_selected_by_seed.csv','protein_validation_selected_by_seed.csv','paired_comparisons.csv','split_audit.csv','assay_heterogeneity.csv'],
        'adaptation_study':['audit.json','final_by_seed.csv','validation_selected_by_seed.csv','validation_selected_summary.csv'],
        'explanation_robustness':['audit.json','fingerprint_collisions.csv','collision_affinity_strata.csv','background_stability.csv'],
        'structure_contacts':['contact_audit.json','contact_summary.csv']}
    for folder,names in extra.items():
        for name in names: assets.add('results/'+folder+'/'+name)
    assets.update(['ROBUST_STUDY.md','paper/main_updated.pdf','paper/all_figures_atlas.pdf','paper/updated_composite_supplement.pdf'])
    # Portable local browsing after cloning, without distributing GP checkpoints.
    (ROOT/'web/catalog.json').write_text(json.dumps(rows,separators=(',',':'),allow_nan=False),encoding='utf-8')
    methods=dict(representations=REPRESENTATIONS,kernels=KERNELS,protocols=PROTOCOLS,static_protocols=list(STATIC_PHASES))
    html=(ROOT/'web/dashboard.html').read_text(encoding='utf-8')
    html=html.replace('__CATALOG__',json.dumps(rows,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
    html=html.replace('__TOKEN__','null').replace('__METHODS__',json.dumps(methods))
    html=html.replace("const url=p=>'../../'+p;","const url=p=>p;")
    html=html.replace('Browsing works directly from this file.','This public-ready export is read-only. Uploaded data are not sent to this site.')
    html=html.replace('Uploads cannot run from a static HTML file.','Online AL is not deployed. Run the local application for measured-data campaigns.')
    (out/'index.html').write_text(html,encoding='utf-8')
    # Preserve links from the new report back to the original explorer.
    for folder in ['paper_explainability','explainability']:
        rel='results/'+folder+'/index.html'
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../../index.html"><a href="../../index.html">Open dashboard</a>',encoding='utf-8')
    manifest=[]
    for rel in sorted(assets):
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
        manifest.append(dict(path=rel,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size))
    for folder in ['paper_explainability','explainability']:
        rel='results/'+folder+'/index.html';target=out/rel
        manifest.append(dict(path=rel,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size))
    (out/'export_manifest.json').write_text(json.dumps(dict(mode='read-only',public_deployment_verified=False,
        benchmark_runs=sum(r['kind']=='Full benchmark' for r in rows),explanation_reports=sum(r['kind']!='Full benchmark' for r in rows),
        robust_campaigns=9720,adaptation_comparisons=1944,adaptation_new_campaigns=1296,
        excluded=['user_runs','raw source datasets','model weights','Python runtime','full GP checkpoints'],assets=manifest),indent=2))
    # No build, Python runtime or upload endpoint is shipped.
    (out/'vercel.json').write_text(json.dumps(dict(version=2,framework=None,
        headers=[dict(source='/(.*)',headers=[dict(key='X-Content-Type-Options',value='nosniff'),dict(key='X-Frame-Options',value='DENY')])]),indent=2))
    allowed={Path(r['path']) for r in manifest}|{Path(n) for n in ['index.html','vercel.json','export_manifest.json']}
    unexpected=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.relative_to(out) not in allowed and '.vercel' not in p.relative_to(out).parts]
    if unexpected:raise ValueError('Review unexpected export files before publishing: '+str(unexpected))
    print('Exported',len(rows),'views;',len(assets),'assets;',round(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())/1e6,1),'MB')

if __name__=='__main__':main()
