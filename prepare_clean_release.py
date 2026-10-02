"""Assemble an allowlisted public tree in the separately cloned repository.

No deletion, force-push, raw data, model weights, user uploads or credentials.
"""
from pathlib import Path
import json, shutil, hashlib, re
ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/'LipidAL-release'

def main():
    assert (DEST/'.git').is_dir(),'Clone the intended repository before staging'
    manifest=json.loads((ROOT/'vercel_dashboard/export_manifest.json').read_text())
    assert manifest['robust_campaigns']==9720
    files=set()
    files.update(p.relative_to(ROOT) for p in ROOT.glob('*.py'))
    files.update(p.relative_to(ROOT) for p in ROOT.glob('*.md'))
    files.update(Path(n) for n in ['requirements.txt','requirements-dev.txt','.gitignore','.gitattributes','.vercelignore','vercel.json'])
    files.update(p.relative_to(ROOT) for p in (ROOT/'web').glob('*') if p.is_file())
    files.update(p.relative_to(ROOT) for p in (ROOT/'vercel_dashboard').rglob('*') if p.is_file() and '.vercel' not in p.parts)
    # Mirror the reviewed derived report assets for local app/file browsing.
    files.update(Path(row['path']) for row in manifest['assets'])
    files.update(Path('results')/folder/'summary.csv' for folder in ['explainability','paper_explainability'])
    for folder in ['results/robust_study_v2','results/adaptation_study','results/explanation_robustness','results/structure_contacts','results/expanded_grid','results/full_grid','results/published']:
        files.update(p.relative_to(ROOT) for p in (ROOT/folder).glob('*') if p.suffix in ('.csv','.json','.md'))
    files.update(p.relative_to(ROOT) for p in (ROOT/'paper').glob('*') if p.suffix in ('.tex','.cls','.md','.py','.json','.csv'))
    files.update(Path('paper')/n for n in ['main_updated.pdf','main_expanded.pdf','updated_composite_supplement.pdf','all_figures_atlas.pdf'])
    for folder in ['paper/figures','paper/results_tables','paper/updated_results_tables']:
        files.update(p.relative_to(ROOT) for p in (ROOT/folder).rglob('*') if p.suffix in ('.png','.svg','.pdf','.csv','.json'))
    # Manifests contain provenance but not sequence arrays or pretrained weights.
    files.update(p.relative_to(ROOT) for p in (ROOT/'data/features').glob('*manifest.json'))
    files.add(Path('data/features/lipid_ssl/manifest.json'))
    for name in ['deployment_verification.json','vercel_export_validation.json']:
        if (ROOT/name).exists():files.add(Path(name))
    blocked={'user_runs','__pycache__','node_modules','tex_env','python_tools','ui_test_tools','study_tools','.vercel','raw','processed','runs','adapters'}
    rows=[]
    secret=re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9]{40,}')
    for rel in sorted(files):
        assert not set(rel.parts)&blocked,rel
        assert not rel.name.startswith('.env'),rel
        source=ROOT/rel
        assert source.is_file(),source
        assert source.stat().st_size<95_000_000,('Large file',rel)
        if source.suffix in ('.py','.html','.md','.json','.txt','.tex','.csv'):
            assert not secret.search(source.read_text(encoding='utf-8',errors='replace')),('Credential-like content',rel)
        target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        rows.append(dict(path=rel.as_posix(),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    allowed={r['path'] for r in rows}|{'release_manifest.json'}
    extras=[p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file() and not p.name.startswith('.env') and not set(p.relative_to(DEST).parts)&{'.git','.vercel','__pycache__'} and p.relative_to(DEST).as_posix() not in allowed]
    assert not extras,('Unreviewed files remain in release tree',extras)
    (DEST/'release_manifest.json').write_text(json.dumps(dict(mode='allowlisted-public-release',files=rows,excluded=sorted(blocked),credential_pattern_scan='PASS',raw_source_data_included=False,checkpoints_included=False),indent=2))
    print('Release files:',len(rows),'total MB:',round(sum(r['bytes'] for r in rows)/1e6,1))

if __name__=='__main__':main()
