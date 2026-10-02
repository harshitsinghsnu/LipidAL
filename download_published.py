"""Download official published source files; no synthetic fallback."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
BASE = 'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41589-020-00659-5/MediaObjects/'
SOURCES = {'BioDolphin_v1.1.zip': 'https://biodolphin.chemistry.gatech.edu/BioDolphin_vr1.1.zip'}
SOURCES.update({f'traak_source_{i}.xlsx': BASE + f'41589_2020_659_MOESM{i}_ESM.xlsx' for i in range(3,7)})
SOURCES['traak_supplement.pdf'] = BASE + '41589_2020_659_MOESM1_ESM.pdf'

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, url in SOURCES.items():
        path = RAW / name
        if not path.exists():
            part = path.with_suffix(path.suffix + '.part')
            subprocess.run(['curl.exe', '-L', '--fail', '--retry', '2', '--max-time', '600',
                            '-C', '-', '-o', str(part), url], check=True)
            part.replace(path)
        if path.suffix in ['.zip','.xlsx']:
            with zipfile.ZipFile(path) as archive:
                bad=archive.testzip()
                if bad: raise ValueError(f'Corrupt archive member: {bad}')
        manifest.append(dict(file=name, url=url, bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (RAW / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')

if __name__ == '__main__':
    main()
