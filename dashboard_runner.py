"""Validated local CSV campaigns using the unchanged full-grid GP engine."""
import io, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from threadpoolctl import threadpool_limits
from grid_gp import Geometry, ExactPairGP
from extended_methods import KERNELS, PROTOCOLS, SCHEDULES, STATIC_PHASES, select
from extended_features import REPRESENTATIONS, encode
from run_full_grid import evaluate, split
from protein_lipid_active_learning import protein_features, AA


def validate_upload(raw, config):
    if len(raw)>5*1024*1024: raise ValueError('CSV exceeds 5 MB.')
    df=pd.read_csv(io.BytesIO(raw))
    required=['protein_sequence','lipid_smiles','affinity']
    if not set(required)<=set(df): raise ValueError('Required columns: '+', '.join(required))
    if not 10<=len(df)<=1000: raise ValueError('Use 10–1000 measured pairs per local campaign.')
    if df[required].isna().any().any(): raise ValueError('Missing sequence, SMILES or affinity. No labels will be imputed.')
    if 'endpoint' in df and df.endpoint.dropna().nunique()>1: raise ValueError('Use a single endpoint per campaign; split mixed Kd/Ki data first.')
    df=df.copy()
    df['protein_sequence']=df.protein_sequence.astype(str).str.replace(r'\s+','',regex=True).str.upper()
    for i,s in enumerate(df.protein_sequence):
        if not s or len(s)>20000 or set(s)-set(AA): raise ValueError(f'Row {i+2}: sequence must contain standard amino-acid letters only (max 20,000).')
    smiles=[]
    for i,s in enumerate(df.lipid_smiles.astype(str)):
        if len(s)>10000: raise ValueError(f'Row {i+2}: SMILES is too long.')
        mol=Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms()==0: raise ValueError(f'Row {i+2}: invalid lipid SMILES.')
        smiles.append(Chem.MolToSmiles(mol,canonical=True))
    df['lipid_smiles']=smiles
    if df.duplicated(['protein_sequence','lipid_smiles']).any(): raise ValueError('Duplicate sequence/canonical-SMILES pairs. Curate duplicates before uploading.')
    y=pd.to_numeric(df.affinity,errors='coerce').to_numpy(float)
    if not np.isfinite(y).all(): raise ValueError('Affinity must contain finite measured numbers, not censored strings.')
    mode=config.get('units','pK')
    if mode!='pK':
        if mode not in ('Kd_nM','Kd_uM','Ki_nM','Ki_uM'): raise ValueError('Unsupported affinity units.')
        if np.any(y<=0): raise ValueError('Raw Kd/Ki values must be positive.')
        y=-np.log10(y*(1e-9 if mode.endswith('nM') else 1e-6))
    df['affinity']=y
    if np.std(y)<1e-10: raise ValueError('All affinities are identical; this cannot benchmark affinity learning.')
    cfg={}
    for k,v in [('initial',10),('batch',5),('cycles',5),('seed',7)]:
        value=config.get(k,v)
        if isinstance(value,bool) or float(value)!=int(value): raise ValueError(f'{k} must be an integer.')
        cfg[k]=int(value)
    if cfg['initial']<2 or cfg['batch']<1 or not 1<=cfg['cycles']<=20 or not 0<=cfg['seed']<=2**32-1:
        raise ValueError('Initial >= 2, batch >= 1, cycles 1–20 and seed 0–4294967295 are required.')
    cfg.update(representation=config.get('representation','ecfp'),kernel=config.get('kernel','tanimoto'),protocol=config.get('protocol','thompson'),units=mode)
    if cfg['representation'] not in REPRESENTATIONS or cfg['kernel'] not in KERNELS or cfg['protocol'] not in PROTOCOLS:
        raise ValueError('Unsupported representation, kernel or acquisition policy.')
    if cfg['protocol'] not in STATIC_PHASES and cfg['cycles']!=10:
        raise ValueError('Paper phase schedules require exactly 10 cycles.')
    available=len(df)-max(2,round(.2*len(df)))
    final=cfg['initial']+cfg['batch']*cfg['cycles']
    if final>available or final>500: raise ValueError(f'Budget is {final}; maximum is min(acquisition pool {available}, 500 labels). Reduce initial/batch/cycles.')
    return df,cfg


def run_campaign(df,cfg,dest,update,cancel):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    df.to_csv(dest/'input_normalized.csv',index=False)
    p=np.vstack([protein_features(s) for s in df.protein_sequence])
    if cancel.is_set(): raise InterruptedError('Cancelled before fitting.')
    update(status='running',message='Encoding measured lipid structures with '+cfg['representation'])
    x,representation_metadata=encode(df.lipid_smiles,cfg['representation'],cancel)
    test,labelled,pool=split(len(df),cfg['seed'],cfg['initial']);initial=labelled.copy();campaign=np.r_[labelled,pool]
    rng=np.random.default_rng(cfg['seed']+104729);y=df.affinity.to_numpy(float)
    phases=STATIC_PHASES
    schedule=phases[cfg['protocol']]*cfg['cycles'] if cfg['protocol'] in phases else SCHEDULES[cfg['protocol']]
    history=[];acquired=[];fits=[]
    with threadpool_limits(limits=1):
        model=ExactPairGP(Geometry(np.array(x),p,initial),cfg['kernel'])
        for cycle in range(cfg['cycles']+1):
            if cancel.is_set(): raise InterruptedError('Cancelled between cycles; partial outputs retained.')
            model.fit(labelled,y[labelled]);fits.append(dict(round=cycle,**model.diagnostics))
            metrics,pred,sd=evaluate(model,test,labelled,campaign,y)
            metrics.update(round=cycle,phase='initial' if cycle==0 else schedule[cycle-1]);history.append(metrics)
            pd.DataFrame(history).to_csv(dest/'learning_curves.csv',index=False)
            update(history=list(history),cycle=cycle,status='running',message=f'Fitted cycle {cycle}/{cfg["cycles"]}')
            if cycle<cfg['cycles']:
                q=select(model,pool,schedule[cycle],cfg['batch'],rng,incumbent=float(y[labelled].max()))
                acquired.append(dict(round=cycle+1,phase=schedule[cycle],indices=q.tolist()))
                labelled=np.r_[labelled,q];pool=pool[~np.isin(pool,q)]
    assert not set(test)&set(labelled)
    result=dict(config=cfg,representation_metadata=representation_metadata,history=history,acquired=acquired,fits=fits,initial_indices=initial.tolist(),test_indices=test.tolist(),
                extension_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['extended_features.py','extended_methods.py','dashboard_runner.py']},
                test_predictions=pred.tolist(),test_sd=sd.tolist(),input_sha256=hashlib.sha256((dest/'input_normalized.csv').read_bytes()).hexdigest(),
                engine_sha256=hashlib.sha256((Path(__file__).parent/'grid_gp.py').read_bytes()).hexdigest(),
                interpretation='Retrospective simulation on measured labels; 20% random-pair test holdout; larger pK is better.')
    (dest/'result.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame({'row_index':test,'measured_pK':y[test],'predicted_pK':pred,'observation_sd_pK':sd}).to_csv(dest/'test_predictions.csv',index=False)
    pd.DataFrame([{'round':q['round'],'row_index':i} for q in acquired for i in q['indices']]).to_csv(dest/'acquired_pairs.csv',index=False)
    return result
