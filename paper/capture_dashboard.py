"""Capture four actual local-portal views, without uploading user data."""
import sys,json,threading,hashlib,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ui_test_tools'))
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server
from dashboard import create_app

def main():
    out=HERE/'figures/dashboard';out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='paper_dashboard_') as temp:
        app=create_app(temp);server=make_server('127.0.0.1',8768,app,threaded=True)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with sync_playwright() as pw:
                browser=pw.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
                page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1.5)
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto('http://127.0.0.1:8768/results/paper_explainability/index.html')
                for key,value in dict(kind='Full benchmark',dataset='biodolphin_Kd',representation='chemberta_mlm',kernel='matern32',protocol='thompson',seed='7').items():
                    page.locator('#filter_'+key).select_option(value)
                assert '5850' in page.locator('#selectionCount').inner_text()
                page.locator('#filters').locator('..').screenshot(path=str(out/'a_selection.png'))
                page.locator('#benchmarkPanel .chart-grid').screenshot(path=str(out/'b_curves.png'))
                page.locator('#representationTable').screenshot(path=str(out/'c_comparison.png'))
                page.locator('#runNav').click()
                page.locator('#representation').select_option('molformer');page.locator('#protocol').select_option('thompson_diverse')
                page.locator('#runForm').screenshot(path=str(out/'d_campaign.png'))
                assert not errors,errors;browser.close()
            (out/'capture.json').write_text(json.dumps(dict(source='Actual local Flask portal; not a public Vercel deployment',
                selected=dict(dataset='biodolphin_Kd',representation='chemberta_mlm',kernel='matern32',protocol='thompson',seed=7),
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.png')},javascript_errors=errors),indent=2))
        finally:
            server.shutdown();thread.join();app.extensions['dashboard_executor'].shutdown()

if __name__=='__main__':main()
