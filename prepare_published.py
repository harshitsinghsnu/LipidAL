"""Curate BioDolphin affinity subsets without inventing missing labels."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'

def joined(x):
    return ';'.join(sorted(set(x.dropna().astype(str))))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW / 'BioDolphin_v1.1.zip') as z:
        member = next(n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX'))
        with z.open(member) as f:
            df = pd.read_csv(f, low_memory=False)
    audit = {'source_rows': len(df), 'datasets': {}}
    manifests = []
    for endpoint in ['Kd', 'Ki']:
        value = pd.to_numeric(df[f'complex_avgAffinity_{endpoint}(nM)'], errors='coerce')
        good = value.notna() & np.isfinite(value) & (value > 0)
        d = df.loc[good].copy()
        d['affinity'] = 9 - np.log10(value[good])
        d['protein_sequence'] = d.protein_Sequence.str.upper().str.replace(r'\s+', '', regex=True)
        d['lipid_smiles'] = d.lipid_Isomeric_smiles.fillna(d.lipid_Canonical_smiles)
        d['protein_id'] = d.protein_UniProt_ID
        d['target_name'] = d.protein_Name
        d['lipid_id'] = d.lipid_Ligand_ID_CCD
        # Preserve the source's reported averages and provenance; they are not new assays.
        d.to_csv(OUT / f'biodolphin_{endpoint}_source_rows.csv', index=False)
        cache = {}
        for s in d.lipid_smiles.dropna().unique():
            mol = Chem.MolFromSmiles(s)
            cache[s] = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None
        d['lipid_smiles'] = d.lipid_smiles.map(cache)
        d = d.dropna(subset=['protein_sequence', 'lipid_smiles'])
        d = d[d.protein_sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+')]
        # Conservatively exclude annotations with censored values in any source field.
        rawcols = [c for c in d if c.startswith('complex_Binding_Affinity_')]
        censored = d[rawcols].fillna('').astype(str).apply(lambda c: c.str.contains(r'[<>~≈≤≥]')).any(axis=1)
        excluded = int(censored.sum())
        d = d[~censored]
        group = d.groupby(['protein_sequence', 'lipid_smiles'], as_index=False)
        clean = group.agg(affinity=('affinity', 'median'), protein_id=('protein_id', joined),
                          target_name=('target_name', joined), lipid_id=('lipid_id', joined),
                          source_ids=('BioDolphinID', joined), pdb_ids=('complex_PDB_ID', joined),
                          pubmed_ids=('complex_PubMed_ID', joined), source_annotations=('affinity', 'size'),
                          p_affinity_min=('affinity', 'min'), p_affinity_max=('affinity', 'max'))
        clean['endpoint'] = endpoint
        clean['source_doi'] = '10.1038/s42004-024-01384-z'
        clean.to_csv(OUT / f'biodolphin_{endpoint}.csv', index=False)
        audit['datasets'][endpoint] = dict(numeric_source_rows=int(good.sum()), censored_rows_excluded=excluded,
                                          unique_pairs=len(clean))
        per = OUT / 'per_target'; per.mkdir(exist_ok=True)
        for seq, g in clean.groupby('protein_sequence'):
            key = hashlib.sha256(seq.encode()).hexdigest()[:12]
            name = f'biodolphin_{endpoint}_{key}'
            g.to_csv(per / (name + '.csv'), index=False)
            manifests.append(dict(dataset=name, endpoint=endpoint, protein_id=g.protein_id.iloc[0],
                                  target_name=g.target_name.iloc[0], n_pairs=len(g),
                                  file=f'per_target/{name}.csv', runnable=len(g)>=12))
    pd.DataFrame(manifests).sort_values('n_pairs',ascending=False).to_csv(OUT/'per_target_manifest.csv',index=False)
    (OUT/'curation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit,indent=2))

if __name__ == '__main__': main()
