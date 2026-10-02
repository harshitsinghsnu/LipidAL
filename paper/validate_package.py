"""Check full figure coverage, PDF structure and package the manuscript sources."""
from pathlib import Path
import sys,json,zipfile,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'python_tools'))
import pymupdf
import pandas as pd
root=HERE.parent
manifest=pd.read_csv(HERE/'figure_manifest.csv')
sources={p.relative_to(root).as_posix() for p in (root/'results').rglob('*') if p.suffix in ('.png','.svg')}
assert set(manifest.source)==sources and len(manifest)==len(sources)
assert manifest.source.is_unique
assert manifest.groupby('atlas_page').size().isin([4,5]).all()
for _,r in manifest.iterrows():
    assert hashlib.sha256((root/r.source).read_bytes()).hexdigest()==r.sha256
atlas=pymupdf.open(HERE/'all_figures_atlas.pdf')
assert len(atlas)==manifest.atlas_page.max()
preview=HERE/'previews';preview.mkdir(exist_ok=True)
for idx in [0,22,44]:
    atlas[idx].get_pixmap(matrix=pymupdf.Matrix(1,1)).save(preview/f'atlas_{idx+1}.png')
audit=dict(source_figures=len(sources),atlas_pages=len(atlas),all_source_hashes_match=True,
           panels_per_main_figure=4,atlas_panels_all_4_or_5=True)
audit['template_url']='https://www.acml-conf.org/2026/downloads/ACML_camera_ready.zip'
audit['class_sha256']=hashlib.sha256((HERE/'jmlr.cls').read_bytes()).hexdigest()
assert (HERE/'jmlr.cls').read_bytes()==(HERE/'template/ACML_camera_ready/jmlr.cls').read_bytes()
pdf=HERE/'main_expanded.pdf'
if pdf.exists():
    main=pymupdf.open(pdf);audit['main_pages']=len(main);audit['main_pdf']=pdf.name
    audit['within_16_page_limit']=len(main)<=16
    assert len(main)<=16
    alltext=''.join(page.get_text() for page in main)
    assert '??' not in alltext
    assert '5,850' in alltext and 'Interactive Dashboard' in alltext
    assert 'Frozen transformers' in alltext
    for idx in range(len(main)):
        main[idx].get_pixmap(matrix=pymupdf.Matrix(1,1)).save(preview/f'main_{idx+1}.png')
else:
    audit['main_pdf_compiled']=False
(HERE/'package_validation.json').write_text(json.dumps(audit,indent=2))
files=[p for p in HERE.iterdir() if p.suffix in ('.tex','.cls','.pdf','.csv','.json','.md','.py') and p.name!='main.pdf']
files+=list((HERE/'figures').glob('*.pdf'))
files+=list((HERE/'results_tables').glob('*'))
files+=list((HERE/'figures/dashboard').glob('*'))
with zipfile.ZipFile(HERE/'acml_expanded_source_package.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in files:
        archive.write(p,p.relative_to(HERE).as_posix())
print(json.dumps(audit,indent=2))
