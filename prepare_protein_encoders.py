"""Frozen ESM-2 protein features, preserving every residue without truncation."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from protein_lipid_active_learning import protein_features
ROOT=Path(__file__).resolve().parent;CACHE=ROOT/'data/features'
MODELS={
 'esm2_8m':('esm2_t6_8M_UR50D','c731040fcd8d73dceaa04b0a8e6329b345b0f5df'),
 'esm2_35m':('esm2_t12_35M_UR50D','6fbf070e65b0b7291e7bbcd451118c216cff79d8')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import torch
    from transformers import AutoTokenizer,AutoConfig,AutoModel
    from safetensors.torch import load_file
    frames=[pd.read_csv(ROOT/'data/processed'/f'biodolphin_{endpoint}.csv') for endpoint in ('Kd','Ki')]
    sequences=sorted(set(pd.concat(frames).protein_sequence));result=dict(sequences=np.array(sequences),composition=np.vstack([protein_features(s) for s in sequences]))
    meta=dict(sequences=len(sequences),sequence_sha256=hashlib.sha256('\n'.join(sequences).encode()).hexdigest(),
              source_sha256=sha(__file__),models={},no_affinity_training=True)
    for name,(repo,revision) in MODELS.items():
        folder=CACHE/repo
        tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        config=AutoConfig.from_pretrained(folder,local_files_only=True)
        model=AutoModel.from_config(config,add_pooling_layer=False)
        state=load_file(str(folder/'model.safetensors'))
        if any(k.startswith('esm.') for k in state):state={k.removeprefix('esm.'):v for k,v in state.items() if k.startswith('esm.')}
        state.pop('embeddings.position_ids',None)
        # Older exports persisted an unused absolute table even for rotary ESM-2.
        # The inspected current forward uses this tensor only in absolute mode.
        if config.position_embedding_type=='rotary':state.pop('embeddings.position_embeddings.weight',None)
        model.load_state_dict(state,strict=True)
        device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device).eval()
        chunks=[(i,s[start:start+1000]) for i,s in enumerate(sequences) for start in range(0,len(s),1000)]
        chunks.sort(key=lambda item:len(item[1]));emb=np.zeros((len(sequences),config.hidden_size),np.float64);counts=np.zeros(len(sequences))
        with torch.inference_mode():
            for start in range(0,len(chunks),4):
                batch=chunks[start:start+4];inputs=tokenizer([s for _,s in batch],padding=True,return_tensors='pt',return_special_tokens_mask=True)
                mask=(inputs['attention_mask']*(1-inputs.pop('special_tokens_mask'))).to(device)
                out=model(**inputs.to(device)).last_hidden_state
                sums=(out*mask.unsqueeze(-1)).sum(1).cpu().numpy();lengths=mask.sum(1).cpu().numpy()
                for (idx,seq),vec,n in zip(batch,sums,lengths):
                    assert n==len(seq),'Residue/token mismatch';emb[idx]+=vec;counts[idx]+=n
                if start%200==0:print(name,start,'/',len(chunks),flush=True)
        assert np.array_equal(counts,np.array([len(s) for s in sequences]))
        result[name]=(emb/counts[:,None]).astype(np.float32);assert np.isfinite(result[name]).all()
        meta['models'][name]=dict(repository='facebook/'+repo,revision=revision,weights_sha256=sha(folder/'model.safetensors'),
            dimension=config.hidden_size,pooling='Residue mean, BOS/EOS/padding excluded; disjoint <=1000-residue chunks weighted by residue count',
            proteins_chunked=sum(len(s)>1000 for s in sequences),device=device)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    np.savez_compressed(CACHE/'protein_representations.npz',**result)
    meta['features_sha256']=sha(CACHE/'protein_representations.npz')
    (CACHE/'protein_representation_manifest.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)

if __name__=='__main__':main()
