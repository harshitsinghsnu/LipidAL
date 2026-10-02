"""Headless browser smoke test; uploads real TRAAK data to a temporary server."""
import sys,tempfile,threading,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'ui_test_tools'))
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server
from dashboard import create_app,build

def main():
    build();shots=ROOT/'web/screenshots';shots.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lipidlab_browser_') as temp:
        app=create_app(temp);server=make_server('127.0.0.1',8766,app,threaded=True)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
                page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto('http://127.0.0.1:8766/results/paper_explainability/index.html')
                assert page.locator('#filter_kind').input_value()=='Full benchmark'
                assert page.locator('#benchmarkPanel svg').count()==4
                assert page.locator('#representationTable tbody').count()==0
                assert page.locator('#representationTable tr').count()==6
                page.screenshot(path=str(shots/'desktop_full_benchmark.png'),full_page=True)
                page.locator('#filter_kind').select_option('Temporal ECFP')
                page.locator('#filter_seed').select_option('19')
                page.wait_for_function("document.querySelector('#plotImage').complete && document.querySelector('#plotImage').naturalWidth > 0")
                assert 'seed19' in page.locator('#plotImage').get_attribute('src')
                page.locator('#plotTabs button').nth(2).click()
                page.wait_for_function("document.querySelector('#plotImage').complete && document.querySelector('#plotImage').naturalWidth > 0")
                page.screenshot(path=str(shots/'desktop_explorer.png'),full_page=True)
                page.locator('#runNav').click()
                page.locator('#csvFile').set_input_files(str(ROOT/'data/processed/traak_a_Kd1.csv'))
                page.locator('#start').click();page.locator('#runError').wait_for(state='visible')
                assert 'Budget' in page.locator('#runError').inner_text()
                page.locator('#initial').fill('2');page.locator('#batch').fill('1');page.locator('#cycles').fill('2')
                assert page.locator('#representation option').count()==5
                assert page.locator('#kernel option').count()==5
                assert page.locator('#protocol option').count()==13
                page.locator('#representation').select_option('molformer')
                page.locator('#protocol').select_option('ei')
                assert not page.locator('#cycles').evaluate('(e)=>e.readOnly')
                page.locator('#start').click();page.locator('#downloadJob').wait_for(state='visible',timeout=60000)
                assert 'complete' in page.locator('#jobStatus').inner_text().lower()
                with page.expect_download() as download:page.locator('#downloadJob').click()
                assert download.value.suggested_filename.endswith('.zip')
                page.screenshot(path=str(shots/'desktop_campaign.png'),full_page=True)
                page.set_viewport_size({'width':390,'height':844});page.locator('#browseNav').click()
                page.locator('#filter_kind').select_option('Full benchmark')
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                page.screenshot(path=str(shots/'mobile_explorer.png'),full_page=True)
                page.goto((ROOT/'results/paper_explainability/index.html').as_uri())
                page.locator('#runNav').click();assert page.locator('#offline').is_visible()
                assert page.locator('#start').is_disabled()
                assert not errors,errors
                browser.close()
            (shots/'validation.json').write_text(json.dumps(dict(dropdowns=True,images=True,real_csv_upload=True,
                invalid_budget_rejected=True,completed_campaign=True,zip_download=True,mobile_no_overflow=True,
                offline_guidance=True,javascript_errors=errors),indent=2))
        finally:
            server.shutdown();thread.join();app.extensions['dashboard_executor'].shutdown()
    print('PASS: browser dropdowns, plots, upload validation, real AL run, ZIP, mobile and offline behavior.')

if __name__=='__main__':main()
