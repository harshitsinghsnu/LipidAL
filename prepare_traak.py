"""Prepare two experimentally measured TRAAK Kd1 panels (Schrecke et al.)."""
from pathlib import Path
from urllib.parse import quote
import json
import requests
import numpy as np
import pandas as pd
from rdkit import Chem

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'
OUT=ROOT/'data'/'processed'
# Names are transcribed from Supplementary Table 1 in the downloaded PDF.
# Ordering matches the titration blocks in Source Data Figs. 2 and 3.
NAMES=[
 '1-palmitoyl-2-oleoyl-sn-glycero-3-phosphate',
 '1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoethanolamine',
 '1-palmitoyl-2-oleoyl-sn-glycero-3-phosphocholine',
 '1-palmitoyl-2-oleoyl-sn-glycero-3-phospho-L-serine',
 "1-palmitoyl-2-oleoyl-sn-glycero-3-phospho-(1\u0027-rac-glycerol)",
 '1-stearoyl-2-arachidonoyl-sn-glycero-3-phosphoethanolamine',
 '1-stearoyl-2-docosahexaenoyl-sn-glycero-3-phosphoethanolamine',
 '1-arachidonoyl-2-hydroxy-sn-glycero-3-phosphate',
 '1-(1Z-octadecenyl)-2-arachidonoyl-sn-glycero-3-phosphoethanolamine',
 '1,2-dioleoyl-sn-glycero-3-phosphoethanolamine',
 '1-oleoyl-2-hydroxy-sn-glycero-3-phosphate',
 '1-palmitoyl-2-hydroxy-sn-glycero-3-phosphate',
 '1-palmitoyl-2-hydroxy-sn-glycero-3-phosphoethanolamine',
 '1-stearoyl-2-arachidonoyl-sn-glycero-3-phosphate',
 '1,2-dioleoyl-sn-glycero-3-phosphoethanol',
 '1-(1Z-octadecenyl)-2-oleoyl-sn-glycero-3-phosphoethanolamine',
]

def fetch_json(url,path):
    if path.exists(): return json.loads(path.read_text())
    r=requests.get(url,timeout=30); r.raise_for_status()
    data=r.json(); path.write_text(json.dumps(data,indent=2),encoding='utf8'); return data

def main():
    cache=RAW/'traak_identifiers'; cache.mkdir(exist_ok=True)
    structures={}; failures=[]
    for i,name in enumerate(NAMES):
        if i==10: name='1-oleoyl-sn-glycero-3-phosphate'
        if i==11: name='1-hexadecanoyl-sn-glycero-3-phosphate'
        url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'+quote(name,safe='')+'/property/IsomericSMILES/JSON'
        try:
            props=fetch_json(url,cache/f'lipid_{i}.json')['PropertyTable']['Properties']
            # Manual disambiguation from Table 1: sn-glycerol stereochemistry and
            # cis oleoyl / explicitly 1Z plasmalogen. Other hits lack/wrong stereo.
            selected_cid={2:5497103,15:42607457}.get(i)
            if selected_cid: props=[p for p in props if p['CID']==selected_cid]
            if len(props)!=1: raise ValueError('Ambiguous structure identity')
            p=props[0]; s=p.get('SMILES',p.get('IsomericSMILES'))
            mol=Chem.MolFromSmiles(s)
            if mol is None:raise ValueError('Invalid structure')
            structures[i]=(Chem.MolToSmiles(mol),p['CID'],url)
            print(i,p['CID'],flush=True)
        except Exception as e:
            failures.append({'index':i,'name':name,'reason':str(e),'url':url})
            print('Excluded unresolved',i,str(e),flush=True)
    # The paper's longer construct is Q9NYG8-2 1:290, N104Q/N108Q;
    # the shorter construct removes its first 26 residues. Fusion tags omitted.
    u='https://rest.uniprot.org/uniprotkb/Q9NYG8-2.fasta'
    fp=cache/'Q9NYG8-2.fasta'
    if not fp.exists():
        r=requests.get(u,timeout=30);r.raise_for_status();fp.write_text(r.text,encoding='utf8')
    seq=''.join(fp.read_text().splitlines()[1:])[:290]
    assert len(seq)==290 and seq[103]=='N' and seq[107]=='N', 'Verify UniProt isoform numbering'
    seq=seq[:103]+'Q'+seq[104:107]+'Q'+seq[108:]
    for file,iso,protein in [(3,'a',seq[26:]),(4,'b',seq)]:
        d=pd.read_excel(RAW/f'traak_source_{file}.xlsx',header=None)
        starts=d.index[d[0].eq('Kd')].tolist(); assert len(starts)==len(NAMES)
        rows=[]
        for i,start in enumerate(starts):
            if i not in structures:continue
            assert d.iloc[start+2,0]=='Kd1'
            replicate=d.iloc[start+2,1:4].astype(float).to_numpy()
            kd=float(np.mean(replicate)); assert np.isclose(kd,float(d.iloc[start+2,4]))
            smiles,cid,url=structures[i]
            rows.append(dict(protein_sequence=protein,protein_id=f'TRAAK_{iso}',target_name=f'TRAAK K2P4.1{iso}',
                             lipid_smiles=smiles,lipid_id=f'CID{cid}',lipid_name=NAMES[i],
                             affinity=6-np.log10(kd),kd_uM=kd,kd_sd_uM=float(np.std(replicate,ddof=1)),
                             endpoint='Kd1',source_doi='10.1038/s41589-020-00659-5',
                             source_file=f'traak_source_{file}.xlsx',source_excel_row=start+3,
                             source_block=str(d.iloc[start,1]),structure_url=url))
        pd.DataFrame(rows).to_csv(OUT/f'traak_{iso}_Kd1.csv',index=False)
        print('TRAAK',iso,len(rows),flush=True)
    (OUT/'traak_unresolved_structures.json').write_text(json.dumps(failures,indent=2),encoding='utf8')

if __name__=='__main__':main()
