"""Build offline dashboards and serve the optional localhost AL runner."""
from pathlib import Path
import json, secrets, threading, uuid, io, zipfile, argparse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from flask import Flask, request, jsonify, send_file, redirect, abort
from dashboard_runner import validate_upload, run_campaign
from extended_features import REPRESENTATIONS
from extended_methods import KERNELS, PROTOCOLS, STATIC_PHASES

ROOT=Path(__file__).resolve().parent

def catalog():
    # Clean public clones retain a derived browsing catalog, not GP checkpoints.
    saved=ROOT/'web/catalog.json'
    if saved.exists() and not any((ROOT/'results/full_grid/runs').glob('*/*/*/*/seed*.json')) and not any((ROOT/'results/expanded_grid/runs').glob('*/*/*/*/seed*.json')):
        return json.loads(saved.read_text(encoding='utf-8'))
    rows=[]
    # Benchmarks and explanations are distinct views: lack of SHAP must not hide a run.
    for folder in ('full_grid','expanded_grid'):
        source=ROOT/'results'/folder
        for path in sorted((source/'runs').glob('*/*/*/*/seed*.json')):
            r=json.loads(path.read_text());history=r['history']
            area=sum((b['labelled']-a['labelled'])*(a['recall_top10']+b['recall_top10'])/2 for a,b in zip(history,history[1:]))
            area/=history[-1]['labelled']-history[0]['labelled']
            downloads=[dict(name='Run audit JSON',path=path.relative_to(ROOT).as_posix())]
            summary=ROOT/'results/expanded_grid/summary.csv'
            if summary.exists(): downloads.append(dict(name='Full grid summary CSV',path=summary.relative_to(ROOT).as_posix()))
            rows.append(dict(kind='Full benchmark',**r['config'],plots=[],history=history,
                metrics={**history[-1],'recall_auc':area,'nonconverged_fits':sum(not f['success'] for f in r['fits'])},
                downloads=downloads))
    for kind,folder in [('Temporal ECFP','paper_explainability'),('Grouped SHAP','explainability')]:
        source=ROOT/'results'/folder/'summary.csv'
        if not source.exists(): continue
        for r in pd.read_csv(source).to_dict('records'):
            rel=Path(r['output']) if kind=='Grouped SHAP' else Path('results')/folder/r['output']
            # CSV outputs use Windows paths; normalize for browser URLs.
            prefix=str(rel).replace('\\','/')
            files=['feature_evolution.png','affinity_distributions.png','highlighted_fragments.svg'] if kind=='Temporal ECFP' else ['global_importance.png','local_explanation.png']
            downloads=['fragment_ranking.csv','temporal_stability.csv','cycle_audit.csv','sampling.json'] if kind=='Temporal ECFP' else ['global_importance.csv','local_shap.csv','explained_pairs.csv','acquisition_candidates.csv','metadata.json']
            rows.append(dict(kind=kind,dataset=r['dataset'],representation=r['representation'],kernel=r['kernel'],protocol=r['protocol'],seed=int(r['seed']),
                             plots=[dict(name=f.replace('_',' ').rsplit('.',1)[0].title(),path=prefix+'/'+f) for f in files],
                             downloads=[dict(name=f,path=prefix+'/'+f) for f in downloads],metrics={k:v for k,v in r.items() if isinstance(v,(int,float))}))
    return rows

def page(token=None):
    data=json.dumps(catalog(),allow_nan=False).replace('<','\\u003c')
    methods=dict(representations=REPRESENTATIONS,kernels=KERNELS,protocols=PROTOCOLS,static_protocols=list(STATIC_PHASES))
    return (ROOT/'web/dashboard.html').read_text(encoding='utf-8').replace('__CATALOG__',data).replace('__TOKEN__',json.dumps(token)).replace('__METHODS__',json.dumps(methods))

def build():
    html=page()
    for folder in ('paper_explainability','explainability'):
        dest=ROOT/'results'/folder/'index.html'
        if dest.exists() and not dest.with_name('index_legacy.html').exists():
            dest.with_name('index_legacy.html').write_bytes(dest.read_bytes())
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(html,encoding='utf-8')

def create_app(job_root=None):
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=6*1024*1024
    token=secrets.token_urlsafe(32);jobs={};lock=threading.Lock();executor=ThreadPoolExecutor(max_workers=1)
    job_root=Path(job_root or ROOT/'results/user_runs')
    app.extensions['dashboard_jobs']=jobs;app.extensions['dashboard_token']=token;app.extensions['dashboard_executor']=executor
    @app.before_request
    def local_only():
        if request.host.split(':')[0] not in ('127.0.0.1','localhost'): abort(403)
        if request.method=='POST':
            if request.headers.get('Origin') not in (None,request.host_url.rstrip('/')): abort(403)
            if not secrets.compare_digest(request.headers.get('X-Dashboard-Token',''),token): abort(403)
    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff';response.headers['X-Frame-Options']='DENY'
        response.headers['Cache-Control']='no-store';return response
    @app.errorhandler(413)
    def too_large(e): return jsonify(error='Upload too large. Maximum CSV size is 5 MB.'),413
    @app.get('/')
    def home(): return redirect('/results/paper_explainability/index.html')
    @app.get('/results/<path:name>')
    def assets(name):
        if name in ('paper_explainability/index.html','explainability/index.html'): return page(token)
        target=(ROOT/'results'/name).resolve()
        allowed=[(ROOT/'results'/p).resolve() for p in ('paper_explainability','explainability','full_grid','expanded_grid','published','robust_study_v2','adaptation_study','explanation_robustness','structure_contacts')]
        if not any(target.is_relative_to(p) for p in allowed) or target.suffix not in ('.png','.svg','.csv','.json','.html','.pdf'): abort(404)
        if not target.is_file(): abort(404)
        return send_file(target)
    @app.get('/paper/main_updated.pdf')
    def manuscript(): return send_file(ROOT/'paper/main_updated.pdf')
    @app.get('/ROBUST_STUDY.md')
    def study_methods(): return send_file(ROOT/'ROBUST_STUDY.md',mimetype='text/plain')
    @app.post('/api/jobs')
    def start():
        upload=request.files.get('file')
        if not upload or not upload.filename.lower().endswith('.csv'): return jsonify(error='Choose a CSV file.'),400
        try:
            cfg=json.loads(request.form.get('config','{}'))
            if not isinstance(cfg,dict): raise ValueError('Invalid configuration.')
            df,cfg=validate_upload(upload.read(5*1024*1024+1),cfg)
        except Exception as e: return jsonify(error=str(e)),400
        with lock:
            if any(j['status'] in ('queued','running') for j in jobs.values()): return jsonify(error='A campaign is already running. Finish or cancel it first.'),409
            ident=uuid.uuid4().hex;cancel=threading.Event()
            jobs[ident]=dict(status='queued',cycle=0,history=[],message='Validated; starting campaign.',config=cfg,cancel=cancel)
        def update(**kw):
            with lock: jobs[ident].update(kw)
        def work():
            try:
                result=run_campaign(df,cfg,job_root/ident,update,cancel)
                update(status='complete',message='Campaign complete. Downloads are ready.',nonconverged_fits=sum(not f['success'] for f in result['fits']))
            except InterruptedError as e: update(status='cancelled',message=str(e))
            except Exception as e: update(status='error',message='Campaign failed: '+str(e))
        executor.submit(work)
        return jsonify(id=ident),202
    @app.get('/api/jobs/<ident>')
    def status(ident):
        with lock:
            if ident not in jobs: abort(404)
            return jsonify({k:v for k,v in jobs[ident].items() if k!='cancel'})
    @app.post('/api/jobs/<ident>/cancel')
    def cancel(ident):
        with lock:
            if ident not in jobs: abort(404)
            jobs[ident]['cancel'].set()
        return jsonify(message='Cancellation requested; current GP operation must finish first.')
    @app.get('/api/jobs/<ident>/download')
    def download(ident):
        with lock:
            if ident not in jobs or jobs[ident]['status']!='complete': abort(404)
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('result.json','learning_curves.csv','test_predictions.csv','acquired_pairs.csv'):
                z.write(job_root/ident/name,name)
        out.seek(0);return send_file(out,mimetype='application/zip',as_attachment=True,download_name=f'al_{ident[:8]}.zip')
    return app

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--build-only',action='store_true');ap.add_argument('--port',type=int,default=8765);args=ap.parse_args()
    build()
    if not args.build_only: create_app().run(host='127.0.0.1',port=args.port,debug=False,use_reloader=False)
