"""Integration tests use real published TRAAK pairs; no dummy affinities."""
import io,json,tempfile,time,threading,unittest
from pathlib import Path
import pandas as pd
from dashboard import create_app,catalog,ROOT
from dashboard_runner import validate_upload,run_campaign

class DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(ROOT/'data/processed/traak_a_Kd1.csv').read_bytes()
        cls.cfg=dict(initial=2,batch=1,cycles=2,seed=7,representation='maccs',kernel='linear',protocol='thompson',units='pK')
    def test_validation(self):
        df,cfg=validate_upload(self.raw,self.cfg)
        self.assertEqual(len(df),16)
        for config in [dict(self.cfg,cycles=20),dict(self.cfg,kernel='bad'),dict(self.cfg,protocol='ucb_gradual')]:
            with self.assertRaises(ValueError):validate_upload(self.raw,config)
        duplicated=pd.concat([df,df.iloc[:1]]).to_csv(index=False).encode()
        with self.assertRaises(ValueError):validate_upload(duplicated,self.cfg)
        raw_units=pd.read_csv(io.BytesIO(self.raw));raw_units['affinity']=raw_units.kd_uM
        converted,_=validate_upload(raw_units.to_csv(index=False).encode(),dict(self.cfg,units='Kd_uM'))
        self.assertLess(abs(converted.affinity-df.affinity).max(),1e-10)
    def test_catalog(self):
        rows=catalog()
        self.assertEqual(sum(r['kind']!='Full benchmark' for r in rows),90)
        self.assertGreaterEqual(sum(r['kind']=='Full benchmark' for r in rows),2160)
        keys=[tuple(r[k] for k in ('kind','dataset','representation','kernel','protocol','seed')) for r in rows]
        self.assertEqual(len(keys),len(set(keys)))
        for r in rows:
            for p in r['plots']+r['downloads']:self.assertTrue((ROOT/p['path']).is_file(),p)
    def test_http_campaign(self):
        with tempfile.TemporaryDirectory(prefix='lipidlab_test_') as temp:
            app=create_app(temp);client=app.test_client();token=app.extensions['dashboard_token']
            self.assertEqual(client.get('/results/paper_explainability/index.html').status_code,200)
            self.assertEqual(client.get('/',headers={'Host':'attacker.example'}).status_code,403)
            self.assertEqual(client.get('/results/paper_explainability/../../dashboard.py').status_code,404)
            self.assertEqual(client.post('/api/jobs').status_code,403)
            self.assertEqual(client.post('/api/jobs',headers={'X-Dashboard-Token':token,'Origin':'https://attacker.example'}).status_code,403)
            response=client.post('/api/jobs',headers={'X-Dashboard-Token':token},data={'file':(io.BytesIO(self.raw),'measured.csv'),'config':json.dumps(self.cfg)})
            self.assertEqual(response.status_code,202,response.get_json());ident=response.json['id']
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                status=client.get('/api/jobs/'+ident).json
                if status['status'] not in ('queued','running'):break
                time.sleep(.1)
            self.assertEqual(status['status'],'complete',status)
            self.assertEqual(len(status['history']),3)
            self.assertEqual(client.get('/api/jobs/'+ident+'/download').status_code,200)
            result=json.loads((Path(temp)/ident/'result.json').read_text())
            acquired=set(result['initial_indices'])|{i for r in result['acquired'] for i in r['indices']}
            self.assertFalse(acquired&set(result['test_indices']))
            self.assertEqual(len(acquired),4)
            app.extensions['dashboard_executor'].shutdown()
    def test_cancel(self):
        df,cfg=validate_upload(self.raw,self.cfg);cancel=threading.Event();cancel.set()
        with tempfile.TemporaryDirectory(prefix='lipidlab_cancel_') as temp:
            with self.assertRaises(InterruptedError):run_campaign(df,cfg,Path(temp)/'run',lambda **kw:None,cancel)

if __name__=='__main__':unittest.main()
