"""Check the public site without authentication, including all asset paths."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,sys,time
from urllib.parse import urljoin,quote
import requests
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'ui_test_tools'))
from playwright.sync_api import sync_playwright

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--url',required=True);args=ap.parse_args()
    base=args.url.rstrip('/')+'/'
    assert base.startswith('https://')
    manifest=json.loads((ROOT/'vercel_dashboard/export_manifest.json').read_text())
    failures=[]
    def check(row):
        url=urljoin(base,quote(row['path']))
        for attempt in range(3):
            try:
                r=requests.head(url,timeout=30,allow_redirects=True)
                if r.status_code==200:return None
                reason=str(r.status_code)
            except requests.RequestException as e:reason=type(e).__name__
            time.sleep(.3)
        return dict(path=row['path'],error=reason)
    with ThreadPoolExecutor(max_workers=8) as executor:
        failures=[x for x in executor.map(check,manifest['assets']) if x]
    assert not failures,failures[:20]
    # Verify the actual PDF bytes, not just a successful HTML error page.
    pdf=requests.get(urljoin(base,'paper/main_updated.pdf'),timeout=60)
    assert pdf.status_code==200 and pdf.content.startswith(b'%PDF')
    assert hashlib.sha256(pdf.content).hexdigest()==hashlib.sha256((ROOT/'paper/main_updated.pdf').read_bytes()).hexdigest()
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
        page=browser.new_page(viewport=dict(width=1440,height=1000))
        page.on('pageerror',lambda e:errors.append(str(e)))
        response=page.goto(base,wait_until='networkidle',timeout=120000)
        assert response.status==200
        assert page.locator('#benchmarkPanel svg').count()==4
        page.locator('#fragmentShortcut').click()
        page.wait_for_function("document.getElementById('plotImage').complete && document.getElementById('plotImage').naturalWidth>0")
        assert 'highlighted_fragments.svg' in page.locator('#plotImage').get_attribute('src')
        page.locator('#runNav').click();assert page.locator('#start').is_disabled()
        page.locator('#browseNav').click();page.locator('#robustLink').click()
        page.wait_for_url('**/robust_study_v2/**')
        page.wait_for_function("document.querySelectorAll('#new_protein_representation option').length===3")
        page.locator('#new_split').select_option('cold_sequence40')
        page.locator('#new_protocol').select_option('prediction_aware_thompson')
        assert page.locator('#new_representation option').count()==5
        page.locator('#new_study').select_option('Adaptation: 1,944 comparisons (648 reused)')
        assert page.locator('#new_variant option').count()==3
        page.set_viewport_size(dict(width=390,height=844))
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+2')
        assert not errors,errors
        browser.close()
    report=dict(status='PASS',url=base,authenticated_requests=False,asset_paths_checked=len(manifest['assets']),
        manuscript_sha256=hashlib.sha256(pdf.content).hexdigest(),public_browser_checks=True,highlighted_structures=True,
        new_study_filters=True,uploads_disabled=True,mobile_overflow=False,javascript_errors=errors,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (ROOT/'deployment_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
