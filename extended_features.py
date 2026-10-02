"""Pinned, frozen molecular encoders; no synthetic embedding fallback.

CLI downloads only with --download. Upload inference is always local/offline.
MoLFormer uses full sequences (rotary positions support this); sequences beyond
its 202-token pretraining regime are explicitly counted, never silently cut.
"""
import argparse, hashlib, json, os
from pathlib import Path
from functools import lru_cache
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'data/features'
REPRESENTATIONS=('ecfp','maccs','chemberta','chemberta_mlm','molformer')
MODELS={
 'chemberta':('DeepChem/ChemBERTa-77M-MTR','66b895cab8adebea0cb59a8effa66b2020f204ca','chemberta_model'),
 'chemberta_mlm':('DeepChem/ChemBERTa-77M-MLM','ed8a5374f2024ec8da53760af91a33fb8f6a15ff','chemberta_mlm_model'),
 'molformer':('ibm-research/MoLFormer-XL-both-10pct','361063d0ad524ef77cf39b08469f6be770dc550f','molformer_model')}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@lru_cache(maxsize=3)
def encoder(rep):
    os.environ.setdefault('HF_MODULES_CACHE',str(CACHE/'hf_modules'))
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    folder=CACHE/MODELS[rep][2]
    if not (folder/'config.json').exists():
        raise ValueError(f'{rep} weights missing. Run python extended_features.py --download first.')
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    if rep=='molformer':
        # Official, pinned and reviewed architecture only. Verify code before import.
        for name,digest in MOLFORMER_CODE.items():
            if sha(folder/name)!=digest: raise ValueError('MoLFormer implementation checksum mismatch: '+name)
        config=AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=True,deterministic_eval=True)
        model=AutoModel.from_config(config,trust_remote_code=True)
        from safetensors.torch import load_file
        state=load_file(str(folder/'model.safetensors'))
        if any(k.startswith('molformer.') for k in state):
            state={k.removeprefix('molformer.'):v for k,v in state.items() if k.startswith('molformer.')}
    else:
        config=AutoConfig.from_pretrained(folder,local_files_only=True)
        model=AutoModel.from_config(config,add_pooling_layer=False)
        state=torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True)
        if any(k.startswith('roberta.') for k in state):
            state={k.removeprefix('roberta.'):v for k,v in state.items() if k.startswith('roberta.')}
        state.pop('embeddings.position_ids',None)
    model.load_state_dict(state,strict=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return tokenizer,model,device

# Filled with SHA256 of the reviewed pinned upstream files; never trust arbitrary code.
MOLFORMER_CODE={
 'configuration_molformer.py':'b88ea8d4b7b5e54f4f186cc7a230eff308928020a030f153a038fd12c05e3bed',
 'modeling_molformer.py':'6f1ef72022de2c69e95661899422a7bb39a40a2cc5a6cb6216f14e9b7d84559c'}

def encode(smiles,rep,cancel=None):
    if rep not in REPRESENTATIONS: raise ValueError('Unsupported representation')
    smiles=list(smiles)
    if rep in ('ecfp','maccs'):
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=4,fpSize=4096,includeChirality=False)
        arrays=[]
        for s in smiles:
            if cancel is not None and cancel.is_set(): raise InterruptedError('Cancelled during featurization.')
            mol=Chem.MolFromSmiles(s)
            if mol is None: raise ValueError('Invalid SMILES')
            fp=gen.GetFingerprint(mol) if rep=='ecfp' else MACCSkeys.GenMACCSKeys(mol)
            a=np.zeros(fp.GetNumBits(),np.float32);DataStructs.ConvertToNumpyArray(fp,a);arrays.append(a)
        return np.vstack(arrays),dict(representation=rep,dimension=len(arrays[0]),truncated=0)
    import torch
    tok,model,device=encoder(rep)
    # MoLFormer pretraining removed stereochemistry; record this lossy transform.
    texts=[Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=False) for s in smiles] if rep=='molformer' else smiles
    lengths=[len(tok(s,truncation=False)['input_ids']) for s in texts]
    limit=1024 if rep=='molformer' else 512
    if max(lengths)>limit: raise ValueError(f'{rep}: input exceeds {limit} tokens. No truncation or substitute embedding is allowed.')
    arrays=[]
    with torch.inference_mode():
        for start in range(0,len(texts),8):
            if cancel is not None and cancel.is_set(): raise InterruptedError('Cancelled during transformer encoding.')
            inputs=tok(texts[start:start+8],padding=True,truncation=False,return_tensors='pt').to(device)
            output=model(**inputs)
            vector=output.pooler_output if rep=='molformer' else output.last_hidden_state[:,0,:]
            arrays.append(vector.cpu().numpy())
    x=np.vstack(arrays)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite pretrained embeddings')
    return x,dict(representation=rep,model=MODELS[rep][0],revision=MODELS[rep][1],dimension=x.shape[1],
                  pooling='masked mean' if rep=='molformer' else 'CLS',stereochemistry_removed=rep=='molformer',
                  max_tokens=max(lengths),beyond_pretraining_length=sum(n>202 for n in lengths) if rep=='molformer' else 0,
                  truncated=0,device=device)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');args=ap.parse_args()
    if args.download:
        os.environ.setdefault('HF_HUB_DISABLE_XET','1')
        from huggingface_hub import snapshot_download
        for rep,(repo,revision,folder) in MODELS.items():
            snapshot_download(repo,revision=revision,local_dir=str(CACHE/folder),max_workers=2,
                allow_patterns=['*.json','merges.txt','model.safetensors','pytorch_model.bin','configuration_molformer.py','modeling_molformer.py'])
    base=np.load(CACHE/'representations.npz',allow_pickle=False)
    output={'smiles':base['smiles']};metadata={}
    for rep in ('chemberta_mlm','molformer'):
        print('Encoding',rep,flush=True)
        output[rep],metadata[rep]=encode(base['smiles'],rep)
        print(metadata[rep],flush=True)
    np.savez_compressed(CACHE/'extended_representations.npz',**output)
    (CACHE/'extended_representation_manifest.json').write_text(json.dumps(dict(models=metadata,
        base_features_sha256=sha(CACHE/'representations.npz'),features_sha256=sha(CACHE/'extended_representations.npz'),
        encoder_sha256=sha(__file__),model_code_sha256=MOLFORMER_CODE),indent=2))

if __name__=='__main__': main()
