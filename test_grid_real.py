"""Numerical checks on downloaded real observations; requires prepared features."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime
from threadpoolctl import threadpool_limits
from grid_gp import Geometry,ExactPairGP,KERNELS,SCHEDULES
from protein_lipid_active_learning import protein_features

def main():
    root=Path(__file__).resolve().parent
    features=np.load(root/'data/features/representations.npz')
    df=pd.read_csv(root/'data/processed/biodolphin_Kd.csv')
    lookup={s:i for i,s in enumerate(features['smiles'])};rows=[lookup[s] for s in df.lipid_smiles]
    p=np.vstack(df.protein_sequence.map(protein_features));train=np.arange(20);y=df.affinity.to_numpy()
    with threadpool_limits(limits=1):
        for rep in ['ecfp','maccs','chemberta']:
            g=Geometry(features[rep][rows],p,train)
            for kernel in KERNELS:
                m=ExactPairGP(g,kernel);m.train=train;m.y=(y[train]-y[train].mean())/y[train].std()
                _,grad=m.objective(m.theta)
                numerical=approx_fprime(m.theta,lambda t:m.objective(t)[0],1e-6)
                assert np.allclose(grad,numerical,atol=2e-4,rtol=2e-3),(rep,kernel)
                assert np.linalg.eigvalsh(m.cov(train,train)).min()>-1e-8
                m.fit(train,y[train]);assert m.diagnostics['nll']<=m.diagnostics['initial_nll']+1e-6
                mu,sd,obs=m.predict(np.arange(30));mu2,cov=m.predict(np.arange(30),joint=True)
                assert np.allclose(mu,mu2) and np.allclose(sd**2,np.diag(cov),atol=1e-6)
                assert (obs>=sd).all();m.thompson(np.arange(30),np.random.default_rng(7))
    assert SCHEDULES['ucb_sandwich']=='EEXXXXXXEE'
    assert SCHEDULES['ucb_gradual']=='EEEBBBBXXX'
    assert SCHEDULES['ucb_alternate']=='EXEXEXEXEX'
    print('All 15 real-data kernel/feature numerical checks passed; schedules verified.')

if __name__=='__main__':main()
