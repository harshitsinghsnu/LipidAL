"""Cache the three molecular representations in the reference author's featurizer.

ECFP radius=4, 4096 bits, MACCS 167-bit RDKit output (bit 0 unused), and
ChemBERTa-77M-MTR CLS embeddings. No invented embeddings or network fallback.
"""
import os
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'data'/'features'
MODEL='DeepChem/ChemBERTa-77M-MTR'
REVISION='66b895cab8adebea0cb59a8effa66b2020f204ca'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--offline',action='store_true');args=ap.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    frames=[pd.read_csv(ROOT/'data'/'processed'/f'{n}.csv') for n in
            ['biodolphin_Kd','biodolphin_Ki','traak_a_Kd1','traak_b_Kd1']]
    smiles=sorted(set(pd.concat(frames).lipid_smiles))
    digest=hashlib.sha256('\n'.join(smiles).encode()).hexdigest()
    target=CACHE/'representations.npz';meta=CACHE/'representation_manifest.json'
    if target.exists() and meta.exists() and json.loads(meta.read_text())['smiles_sha256']==digest:
        print('Verified matching feature cache:',target);return
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=4,fpSize=4096,includeChirality=False)
    ecfp=[];maccs=[]
    for s in smiles:
        mol=Chem.MolFromSmiles(s)
        if mol is None:raise ValueError('Invalid measured lipid: '+s)
        a=np.zeros(4096,np.float32);DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),a);ecfp.append(a)
        b=np.zeros(167,np.float32);DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol),b);maccs.append(b)
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    local=snapshot_download(MODEL,revision=REVISION,local_dir=str(CACHE/'chemberta_model'),
                            allow_patterns=['*.json','merges.txt','vocab.json','pytorch_model.bin'],
                            local_files_only=args.offline,max_workers=2)
    tok=AutoTokenizer.from_pretrained(local,local_files_only=True)
    config=AutoConfig.from_pretrained(local,local_files_only=True)
    # Use restricted tensor-only loading; no trust_remote_code and no pickle objects.
    state=torch.load(Path(local)/'pytorch_model.bin',map_location='cpu',weights_only=True)
    model=AutoModel.from_config(config,add_pooling_layer=False)
    # Published checkpoint is a Roberta multitask head; retain the pretrained base.
    if any(k.startswith('roberta.') for k in state):
        state={k.removeprefix('roberta.'):v for k,v in state.items() if k.startswith('roberta.')}
    # Older transformers persisted this deterministic buffer; current versions regenerate it.
    state.pop('embeddings.position_ids',None)
    model.load_state_dict(state,strict=True)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device).eval()
    embs=[];truncated=[]
    for i,s in enumerate(smiles):
        if len(tok(s,truncation=False)['input_ids'])>512:truncated.append(i)
    with torch.inference_mode():
        for start in range(0,len(smiles),16):
            inp=tok(smiles[start:start+16],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
            embs.append(model(**inp).last_hidden_state[:,0,:].cpu().numpy())
            if start%160==0:print(f'ChemBERTa {start}/{len(smiles)}',flush=True)
    embeddings=np.vstack(embs)
    assert np.isfinite(embeddings).all()
    np.savez_compressed(target,smiles=np.array(smiles),ecfp=np.vstack(ecfp),maccs=np.vstack(maccs),chemberta=embeddings)
    meta.write_text(json.dumps(dict(smiles_sha256=digest,n_lipids=len(smiles),model=MODEL,revision=REVISION,
        ecfp=dict(radius=4,bits=4096,chirality=False),maccs_bits=167,pooling='CLS',
        max_tokens=512,truncated_indices=truncated,embedding_dim=embeddings.shape[1],device=device),indent=2))
    print('Saved',target,flush=True)

if __name__=='__main__':main()
