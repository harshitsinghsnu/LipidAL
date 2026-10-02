"""Numerical and leakage-boundary checks on published pairs only."""
import unittest,json,inspect
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_limits
from grid_gp import Geometry,ExactPairGP
from prediction_aware_acquisition import select,variance_gain,POLICIES
from run_robust_study import ROOT,CACHE,run_one
from robust_splits import groups

class RobustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=pd.read_csv(ROOT/'data/processed/biodolphin_Kd.csv')
        cls.splits=json.loads((CACHE/'robust_splits.json').read_text())
        f=np.load(CACHE/'representations.npz');p=np.load(CACHE/'protein_representations.npz')
        lr={s:i for i,s in enumerate(f['smiles'])};pr={s:i for i,s in enumerate(p['sequences'])}
        cls.part=cls.splits['splits']['biodolphin_Kd/cold_sequence40/seed7'];idx=np.array(cls.part['initial'])
        cls.geom=Geometry(f['ecfp'][[lr[s] for s in cls.df.lipid_smiles]],p['esm2_8m'][[pr[s] for s in cls.df.protein_sequence]].astype(float),idx)
        with threadpool_limits(limits=1):cls.model=ExactPairGP(cls.geom,'linear').fit(idx,cls.df.affinity.to_numpy()[idx])
    def test_partitions(self):
        for key,p in self.splits['splits'].items():
            a=set(p['initial'])|set(p['pool']);v=set(p['validation']);t=set(p['test'])
            self.assertFalse(a&v or a&t or v&t);self.assertEqual(len(p['initial']),24)
            if '/cold_sequence40/' in key:
                self.assertLessEqual(p['audit']['max_train_test_global_identity'],.4)
                self.assertEqual(p['audit']['accession_overlap'],0)
            if '/scaffold/' in key:
                df=pd.read_csv(ROOT/'data/processed'/(key.split('/')[0]+'.csv'))
                g=groups(df,'scaffold',None,None)
                self.assertFalse(set(g[list(a)])&set(g[list(t)]))
    def test_acquisition_all_policies(self):
        pool=np.array(self.part['pool'][:30]);proxy=np.array(self.part['validation'][:8])
        with threadpool_limits(limits=1):
            for method in POLICIES:
                q,audit=select(self.model,pool,proxy,method,4,np.random.default_rng(7))
                self.assertEqual(len(set(q)),4);self.assertTrue(set(q)<=set(pool))
                for item in audit:
                    self.assertAlmostEqual(item['predicted_proxy_variance_reduction'],item['actual_conditional_reduction'],places=8)
                    self.assertGreaterEqual(item['actual_conditional_reduction'],-1e-10)
                np.testing.assert_array_equal(q,select(self.model,pool,proxy,method,4,np.random.default_rng(7))[0])
    def test_variance_against_direct_conditioning(self):
        pool=np.array(self.part['pool'][:20]);proxy=np.array(self.part['validation'][:8]);idx=np.r_[pool,proxy]
        with threadpool_limits(limits=1):
            _,cov=self.model.predict(idx,joint=True);noise=self.model.noise*self.model.scale**2
            gain=variance_gain(cov,len(pool),noise)
            # Independently condition a candidate block using a solve, no refit.
            q=3;conditional=cov[len(pool):,len(pool):]-cov[len(pool):,[q]]@np.linalg.solve(cov[np.ix_([q],[q])]+noise*np.eye(1),cov[[q],len(pool):])
            self.assertAlmostEqual(gain[q],float(np.diag(cov)[len(pool):].mean()-np.diag(conditional).mean()),places=10)
    def test_no_final_test_access_in_selector(self):
        self.assertEqual(list(inspect.signature(select).parameters),['model','pool','proxy','policy','batch','rng'])
        proxy=np.array(self.part['validation'][:8]);pool=np.array(self.part['pool'][:30])
        self.assertFalse(set(proxy)&set(self.part['test']))
        with self.assertRaises(ValueError):select(self.model,pool,pool[:2],'prediction_aware_thompson',2,np.random.default_rng(7))

if __name__=='__main__':unittest.main()
