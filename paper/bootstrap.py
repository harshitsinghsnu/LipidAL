"""Download official format and portable compiler; record provenance."""
from pathlib import Path
import requests, zipfile, io, json, hashlib
ROOT=Path(__file__).resolve().parent
items={
 'template': 'https://www.acml-conf.org/2026/downloads/ACML_camera_ready.zip',
 'compiler': 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip'}
manifest=[]
for name,url in items.items():
    folder=ROOT/name;folder.mkdir(exist_ok=True)
    if name=='compiler':
        url='https://api.github.com/repos/tectonic-typesetting/tectonic/releases/assets/490851968?download=1'
        r=requests.get(url,headers={'Accept':'application/octet-stream'},timeout=60)
    else:
        r=requests.get(url,timeout=60)
    r.raise_for_status()
    archive=zipfile.ZipFile(io.BytesIO(r.content))
    for entry in archive.infolist():
        target=(folder/entry.filename).resolve()
        if not target.is_relative_to(folder.resolve()):raise ValueError('Unsafe archive member')
    archive.extractall(folder)
    manifest.append(dict(name=name,url=url,sha256=hashlib.sha256(r.content).hexdigest(),files=archive.namelist()))
(ROOT/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
