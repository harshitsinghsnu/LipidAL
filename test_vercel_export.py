"""Validate the read-only export, links and disabled upload behavior locally."""
import sys,json,threading,hashlib
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'vercel_dashboard'
sys.path.insert(0,str(ROOT/'ui_test_tools'))
from playwright.sync_api import sync_playwright

def main():
    manifest=json.loads((OUT/'export_manifest.json').read_text())
    assert manifest['benchmark_runs']==5850 and manifest['explanation_reports']==90
    for row in manifest['assets']:
        assert 'user_runs' not in row['path'] and '/runs/' not in row['path']
        assert hashlib.sha256((OUT/row['path']).read_bytes()).hexdigest()==row['sha256']
    server=ThreadingHTTPServer(('127.0.0.1',8769),partial(SimpleHTTPRequestHandler,directory=str(OUT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1440,height=1000));errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto('http://127.0.0.1:8769/')
            assert '5850' in page.locator('#selectionCount').inner_text()
            assert page.locator('#benchmarkPanel svg').count()==4
            assert page.locator('#filter_representation option').count()==5
            assert page.locator('#filter_protocol option').count()==13
            link=page.locator('#downloads a').first.get_attribute('href')
            assert page.request.get('http://127.0.0.1:8769/'+link).status==200
            page.locator('#filter_kind').select_option('Grouped SHAP')
            page.wait_for_function("document.getElementById('plotImage').complete && document.getElementById('plotImage').naturalWidth>0")
            page.locator('#runNav').click()
            assert page.locator('#start').is_disabled()
            assert 'read-only' in page.locator('#offline').inner_text()
            page.locator('#browseNav').click()
            page.locator('#fragmentShortcut').click()
            page.wait_for_function("document.getElementById('plotImage').complete && document.getElementById('plotImage').naturalWidth>0")
            assert 'highlighted_fragments.svg' in page.locator('#plotImage').get_attribute('src')
            paper=page.locator('#paperLink').get_attribute('href')
            response=page.request.get('http://127.0.0.1:8769/'+paper)
            assert response.status==200 and response.body().startswith(b'%PDF')
            page.locator('#robustLink').click()
            page.wait_for_url('**/robust_study_v2/index.html')
            page.wait_for_function("document.querySelectorAll('#new_protein_representation option').length===3")
            assert page.locator('#new_representation option').count()==5
            assert page.locator('#new_split option').count()==3
            assert page.locator('#new_protocol option').count()==6
            page.locator('#new_protocol').select_option('prediction_aware_thompson')
            page.locator('#new_study').select_option('Adaptation: 1,944 comparisons (648 reused)')
            assert page.locator('#new_variant option').count()==3
            for a in page.locator('a[href]').all():
                href=a.get_attribute('href')
                if not href.startswith(('http:','https:','#')):
                    from urllib.parse import urljoin
                    assert page.request.get(urljoin(page.url,href)).status==200,href
            page.set_viewport_size(dict(width=390,height=844))
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+2')
            assert not errors,errors;browser.close()
        report=dict(status='PASS',benchmark_runs=5850,explanation_reports=90,download_links=True,
            actual_explanation_image=True,uploads_disabled=True,asset_hashes_verified=True,javascript_errors=errors,
            robust_campaigns=9720,adaptation_comparisons=1944,new_dropdowns=True,highlighted_fragments=True,manuscript_download=True,
            mobile_overflow=False,public_deployment_verified=False)
        (ROOT/'vercel_export_validation.json').write_text(json.dumps(report,indent=2));print(report)
    finally:server.shutdown();thread.join()

if __name__=='__main__':main()
