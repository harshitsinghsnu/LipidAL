"""Illustrative experimental-structure contact checks for pre-existing SHAP examples.

Not docking, not new affinity labels, not a prospective experiment. A contact is
an explicitly defined <=4 A heavy-atom proximity to the recorded protein chain.
"""
from pathlib import Path
import json,zipfile,hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/structure_contacts';CACHE=ROOT/'data/raw/contact_pdbs'

def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    columns=['BioDolphinID','complex_PDB_ID','complex_Receptor_Chain','complex_Ligand_Chain','complex_Residue_number_of_the_ligand','lipid_Ligand_ID_CCD']
    with zipfile.ZipFile(ROOT/'data/raw/BioDolphin_v1.1.zip') as z:
        name=next(n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX'));source=pd.read_csv(z.open(name),usecols=columns,dtype=str).set_index('BioDolphinID')
    records=[]
    for dataset in ('biodolphin_Kd','biodolphin_Ki'):
        df=pd.read_csv(ROOT/'data/processed'/f'{dataset}.csv')
        mapping=json.loads((ROOT/'results/paper_explainability'/dataset/'ucb_explore_heavy/seed7/fragment_mapping.json').read_text())
        for fragment in mapping[:2]:
            row=df.iloc[fragment['representative_row']];record=dict(dataset=dataset,row_index=int(fragment['representative_row']),bit=int(fragment['bit']),status='not_mapped')
            try:
                ids=str(row.source_ids).split(';');sid=next(s for s in ids if s in source.index);s=source.loc[sid]
                if isinstance(s,pd.DataFrame):s=s.iloc[0]
                pdb=s.complex_PDB_ID.lower();protein_chain=s.complex_Receptor_Chain;ligand_chain=s.complex_Ligand_Chain
                if len(protein_chain)!=1 or len(ligand_chain)!=1:raise ValueError('Multi-character chain requires mmCIF mapping; not silently guessed')
                residue=str(int(float(s.complex_Residue_number_of_the_ligand)));ccd=s.lipid_Ligand_ID_CCD
                path=CACHE/(pdb+'.pdb');url='https://files.rcsb.org/download/'+pdb.upper()+'.pdb'
                if not path.exists():
                    response=requests.get(url,timeout=60);response.raise_for_status();path.write_bytes(response.content)
                text=path.read_text();atoms=[];protein=[];seen=set()
                # First crystallographic model only; no biological assembly expansion.
                for line in text.splitlines():
                    if line.startswith('ENDMDL'):break
                    if not line.startswith(('ATOM  ','HETATM')) or line[16] not in (' ','A'):continue
                    element=line[76:78].strip();coord=np.array([float(line[a:b]) for a,b in ((30,38),(38,46),(46,54))])
                    if element in ('H','D'):continue
                    if line.startswith('ATOM  ') and line[21]==protein_chain:protein.append((coord,line[17:20].strip()+':'+line[22:27].strip()))
                    if line.startswith('HETATM') and line[21]==ligand_chain and line[22:26].strip()==residue and line[17:20].strip()==ccd:
                        key=line[12:16].strip()
                        if key not in seen:atoms.append(line);seen.add(key)
                if not atoms or not protein:raise ValueError('Recorded chain/residue not present in PDB coordinates')
                ligand=Chem.MolFromPDBBlock('\n'.join(atoms)+'\nEND\n',sanitize=False,removeHs=True)
                template=Chem.MolFromSmiles(row.lipid_smiles)
                if ligand is None or ligand.GetNumAtoms()!=template.GetNumAtoms():raise ValueError('Incomplete or different ligand graph; not forcing atom correspondence')
                ligand=AllChem.AssignBondOrdersFromTemplate(template,ligand)
                matches=ligand.GetSubstructMatches(template,uniquify=False,useChirality=False,maxMatches=1000)
                if not matches:raise ValueError('No complete template/PDB graph match')
                xyz=ligand.GetConformer().GetPositions();px=np.vstack([a for a,_ in protein]);dist=np.linalg.norm(xyz[:,None,:]-px[None,:,:],axis=-1)
                env=sorted(fragment['environments'],key=lambda e:(e['radius']==0,e['radius'],e['center']))[0]
                fractions=[];residues=set()
                for match in matches:
                    selected=np.array([match[i] for i in env['atoms']],dtype=int);near=dist[selected]<=4.
                    fractions.append(float(np.any(near,axis=1).mean()));residues.update(protein[i][1] for i in np.flatnonzero(np.any(near,axis=0)))
                record.update(status='mapped',source_id=sid,pdb_id=pdb,protein_chain=protein_chain,ligand_chain=ligand_chain,ligand_residue=residue,ccd=ccd,
                    template_atoms=template.GetNumAtoms(),fragment_atoms=len(env['atoms']),graph_mappings=len(matches),mapping_search_capped=len(matches)==1000,
                    fragment_contact_fraction_min=min(fractions),fragment_contact_fraction_max=max(fractions),contact_residues=sorted(residues),
                    all_ligand_contact_fraction=float((dist.min(axis=1)<=4).mean()),
                    distance_cutoff_angstrom=4.,structure_url=url,structure_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except (ValueError,KeyError,StopIteration,requests.RequestException) as error:record['reason']=str(error)
            records.append(record);print(record,flush=True)
    (OUT/'contact_audit.json').write_text(json.dumps(dict(records=records,mapped=sum(r['status']=='mapped' for r in records),
        scope='Four pre-existing affinity-prioritized fragment examples, possibly repeated molecules. Descriptive proximity, not causal attribution or predictive validation.',
        atom_mapping='Full heavy-atom bond-order graph match, ignoring chirality; symmetry mappings retained, not assumed unique',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    pd.DataFrame(records).to_csv(OUT/'contact_summary.csv',index=False)

if __name__=='__main__':main()
