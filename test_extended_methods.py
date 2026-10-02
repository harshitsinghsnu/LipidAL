"""Regression checks on measured TRAAK data, no generated affinity data."""
import unittest, tempfile, threading
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from extended_features import CACHE, REPRESENTATIONS, encode
from extended_methods import KERNELS, PROTOCOLS, SCHEDULES, select, improvement_scores
from grid_gp import Geometry,ExactPairGP, select as old_select, PROTOCOLS as OLD
from run_full_grid import split
from protein_lipid_active_learning import protein_features
from dashboard_runner import validate_upload,run_campaign

class ExtendedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(CACHE.parent/'processed/traak_a_Kd1.csv').read_bytes()
        cls.df=pd.read_csv(CACHE.parent/'processed/traak_a_Kd1.csv')
    def test_all_combinations_real_pairs(self):
        df=self.df;y=df.affinity.to_numpy(float)
        base=np.load(CACHE/'representations.npz');extra=np.load(CACHE/'extended_representations.npz')
        lookup={s:i for i,s in enumerate(base['smiles'])};rows=[lookup[s] for s in df.lipid_smiles]
        p=np.vstack([protein_features(s) for s in df.protein_sequence]);test,initial,pool=split(len(df),7,2)
        with threadpool_limits(limits=1):
            for rep in REPRESENTATIONS:
                x=(base if rep in base.files else extra)[rep][rows];geom=Geometry(x,p,initial)
                for kernel in KERNELS:
                    model=ExactPairGP(geom,kernel).fit(initial,y[initial])
                    for protocol in PROTOCOLS:
                        with self.subTest(rep=rep,kernel=kernel,protocol=protocol):
                            phase=SCHEDULES[protocol][0]
                            q=select(model,pool,phase,2,np.random.default_rng(7),incumbent=y[initial].max())
                            self.assertEqual(len(set(q)),2);self.assertTrue(set(q)<=set(pool));self.assertFalse(set(q)&set(test))
                            if protocol in OLD:
                                np.testing.assert_array_equal(q,old_select(model,pool,phase,2,np.random.default_rng(7)))
                            np.testing.assert_array_equal(q,select(model,pool,phase,2,np.random.default_rng(7),incumbent=y[initial].max()))
    def test_transformers_deterministic_and_padding(self):
        smiles=self.df.lipid_smiles.tolist()[:8]
        for rep in ('chemberta','chemberta_mlm','molformer'):
            with self.subTest(rep=rep):
                a,meta=encode(smiles,rep);b,_=encode(smiles,rep);one,_=encode(smiles[:1],rep)
                self.assertTrue(np.isfinite(a).all());self.assertEqual(meta['truncated'],0)
                np.testing.assert_array_equal(a,b)
                np.testing.assert_allclose(a[0],one[0],atol=2e-5,rtol=2e-5)
    def test_transformer_uploads(self):
        for rep,protocol in [('chemberta','ei'),('chemberta_mlm','pi'),('molformer','thompson_diverse')]:
            with self.subTest(rep=rep),tempfile.TemporaryDirectory(prefix='lipid_transformer_') as temp:
                cfg=dict(initial=2,batch=1,cycles=2,seed=7,representation=rep,kernel='matern32',protocol=protocol,units='pK')
                df,cfg=validate_upload(self.raw,cfg)
                result=run_campaign(df,cfg,Path(temp)/'run',lambda **kw:None,threading.Event())
                self.assertEqual(len(result['history']),3);self.assertEqual(result['representation_metadata']['representation'],rep)
    def test_ei_pi_limits(self):
        y=np.sort(self.df.affinity.to_numpy(float));incumbent=float(np.median(y))
        ei,pi=improvement_scores(y,np.zeros_like(y),incumbent)
        np.testing.assert_allclose(ei,np.maximum(y-incumbent-.01,0))
        np.testing.assert_array_equal(pi,(y-incumbent-.01>0).astype(float))

if __name__=='__main__':unittest.main()
