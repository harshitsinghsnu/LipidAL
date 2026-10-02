"""Label-free lipid-domain MLM adaptation on BioDolphin-only SMILES.

Conservatively excludes every benchmark scaffold (including all acyclic lipids)
before training, so all outer splits share one frozen external-domain encoder.
This is a small available corpus, not an LMSD-scale pretraining claim.
"""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT=Path(__file__).resolve().parent;CACHE=ROOT/'data/features';OUT=CACHE/'lipid_ssl';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scaffold(s):return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),includeChirality=False) or 'ACYCLIC'

def main():
    import torch
    from transformers import AutoTokenizer,AutoConfig,AutoModelForMaskedLM
    torch.manual_seed(7);np.random.seed(7)
    base=np.load(CACHE/'representations.npz');benchmark=list(base['smiles']);blocked={scaffold(s) for s in benchmark}
    archive=ROOT/'data/raw/BioDolphin_v1.1.zip'
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX'))
        frame=pd.read_csv(z.open(name),usecols=['lipid_Isomeric_smiles','lipid_Canonical_smiles'])
    raw=frame.lipid_Isomeric_smiles.fillna(frame.lipid_Canonical_smiles).dropna().astype(str).unique()
    corpus=set();invalid=0
    for s in raw:
        mol=Chem.MolFromSmiles(s)
        if mol is None:invalid+=1;continue
        canonical=Chem.MolToSmiles(mol,canonical=True)
        if scaffold(canonical) not in blocked:corpus.add(canonical)
    folder=CACHE/'chemberta_mlm_model';tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    corpus=sorted(s for s in corpus if len(tok(s,truncation=False)['input_ids'])<=512)
    if len(corpus)<100:raise ValueError('Insufficient scaffold-disjoint corpus; do not fabricate pretraining data')
    assert not set(corpus)&set(benchmark) and not {scaffold(s) for s in corpus}&blocked
    pd.DataFrame({'lipid_smiles':corpus}).to_csv(OUT/'training_smiles.csv',index=False)
    config=AutoConfig.from_pretrained(folder,local_files_only=True);model=AutoModelForMaskedLM.from_config(config)
    state=torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True)
    state.pop('roberta.embeddings.position_ids',None);model.load_state_dict(state,strict=True)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01);logs=[]
    for epoch in range(3):
        model.train();order=np.random.default_rng(7+epoch).permutation(len(corpus));losses=[]
        for start in range(0,len(order),16):
            inputs=tok([corpus[i] for i in order[start:start+16]],padding=True,return_tensors='pt',return_special_tokens_mask=True)
            special=inputs.pop('special_tokens_mask').to(device).bool();inputs=inputs.to(device)
            original=inputs['input_ids'].clone();eligible=inputs['attention_mask'].bool()&~special
            selected=(torch.rand(original.shape,device=device)<.15)&eligible
            if not selected.any():selected[tuple(torch.nonzero(eligible)[0])]=True
            labels=original.clone();labels[~selected]=-100;draw=torch.rand(original.shape,device=device)
            inputs['input_ids'][selected&(draw<.8)]=tok.mask_token_id
            randommask=selected&(draw>=.8)&(draw<.9);randomtokens=torch.randint(len(tok),original.shape,device=device)
            inputs['input_ids'][randommask]=randomtokens[randommask]
            opt.zero_grad();loss=model(**inputs,labels=labels).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite MLM objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        logs.append(dict(epoch=epoch+1,training_loss=float(np.mean(losses)),steps=len(losses)))
        print('Lipid MLM',logs[-1],flush=True)
    model.eval();model.save_pretrained(OUT/'model',safe_serialization=True);tok.save_pretrained(OUT/'model')
    arrays=[]
    with torch.inference_mode():
        for start in range(0,len(benchmark),16):
            inp=tok(benchmark[start:start+16],padding=True,truncation=False,return_tensors='pt').to(device)
            arrays.append(model.roberta(**inp).last_hidden_state[:,0,:].cpu().numpy())
    features=np.vstack(arrays);assert np.isfinite(features).all()
    np.savez_compressed(OUT/'representations.npz',smiles=base['smiles'],chemberta_lipid_ssl=features)
    meta=dict(corpus_source='BioDolphin v1.1 unlabeled SMILES only',raw_unique_smiles=len(raw),invalid_smiles=invalid,training_smiles=len(corpus),
        benchmark_scaffolds_excluded=len(blocked),scaffold_overlap_with_benchmark=0,affinity_labels_used=0,
        base='DeepChem/ChemBERTa-77M-MLM',base_revision='ed8a5374f2024ec8da53760af91a33fb8f6a15ff',seed=7,
        fixed_epochs=3,lr=1e-5,batch_size=16,mask_probability=.15,logs=logs,pooling='CLS',
        source_sha256=sha(__file__),archive_sha256=sha(archive),corpus_sha256=sha(OUT/'training_smiles.csv'),features_sha256=sha(OUT/'representations.npz'),
        scope='Continued MLM on a small conservative scaffold-disjoint lipid corpus; not LMSD-scale or a new foundation model')
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)

if __name__=='__main__':main()
