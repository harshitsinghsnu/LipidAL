"""Capture real read-only dashboard views and assemble updated paper assets."""
from pathlib import Path
import sys, json, hashlib, shutil, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'paper';FIG=HERE/'figures';SHOTS=FIG/'updated_dashboard'
sys.path.insert(0,str(ROOT/'ui_test_tools'))
from playwright.sync_api import sync_playwright
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    SHOTS.mkdir(exist_ok=True)
    for name in ['policy_panels','protein_panels','adaptation_panels']:
        for ext in ['png','pdf']:shutil.copy2(ROOT/'results/robust_study_v2'/f'{name}.{ext}',FIG/f'{name}.{ext}')
    server=ThreadingHTTPServer(('127.0.0.1',8771),partial(SimpleHTTPRequestHandler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    errors=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
            page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto('http://127.0.0.1:8771/results/paper_explainability/index.html')
            assert page.locator('#benchmarkPanel svg').count()==4
            page.screenshot(path=str(SHOTS/'a_navigation.png'))
            page.locator('#fragmentShortcut').click()
            page.wait_for_function("document.getElementById('plotImage').complete && document.getElementById('plotImage').naturalWidth>0")
            assert 'highlighted_fragments.svg' in page.locator('#plotImage').get_attribute('src')
            page.locator('#plotImage').evaluate("e=>e.scrollIntoView({block:'start'})")
            page.screenshot(path=str(SHOTS/'b_fragments.png'))
            page.locator('#robustLink').click()
            page.wait_for_url('**/robust_study_v2/index.html')
            page.wait_for_function("document.querySelectorAll('#new_protein_representation option').length===3")
            assert page.locator('#new_protein_representation option').count()==3
            assert page.locator('#new_representation option').count()==5
            assert page.locator('#new_protocol option').count()==6
            page.locator('#new_split').select_option('cold_sequence40')
            page.locator('#new_protocol').select_option('prediction_aware_thompson')
            page.locator('#newExplorer').screenshot(path=str(SHOTS/'c_robust.png'))
            page.locator('#new_study').select_option('Adaptation: 1,944 comparisons (648 reused)')
            assert page.locator('#new_variant option').count()==3
            page.locator('#new_variant').select_option('initial_label_adapter')
            page.locator('#newExplorer').screenshot(path=str(SHOTS/'d_adaptation.png'))
            page.set_viewport_size(dict(width=390,height=844))
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2'), 'Mobile horizontal overflow'
            assert not errors,errors
            browser.close()
    finally:server.shutdown();thread.join()
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    names=['a_navigation','b_fragments','c_robust','d_adaptation']
    for ax,name,title in zip(axes.flat,names,['(a) Dashboard navigation','(b) Highlighted structures','(c) Leakage-controlled explorer','(d) Adaptation explorer']):
        ax.imshow(plt.imread(SHOTS/f'{name}.png'));ax.axis('off');ax.set_title(title,loc='left')
    for ext in ['png','pdf']:fig.savefig(FIG/f'updated_dashboard.{ext}',dpi=180)
    plt.close(fig)
    # Four additional panels summarize diagnostics without implying causal validity.
    audit=json.loads((ROOT/'results/explanation_robustness/audit.json').read_text())
    stability=pd.read_csv(ROOT/'results/explanation_robustness/background_stability.csv')
    contacts=pd.read_csv(ROOT/'results/structure_contacts/contact_summary.csv')
    split=pd.read_csv(ROOT/'results/robust_study_v2/split_audit.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    c=audit['collisions'];axes[0,0].bar(['Multiple IDs','One ID'],[c['bits_with_multiple_unfolded_ids'],c['observed_bits']-c['bits_with_multiple_unfolded_ids']],color=['#ba6c43','#16868a']);axes[0,0].set_title('(a) Observed ECFP folding collisions');axes[0,0].set_ylabel('Observed fingerprint positions')
    stability.groupby(['kernel','comparison']).top10_jaccard.mean().unstack().plot.bar(ax=axes[0,1],rot=0);axes[0,1].set_title('(b) Small-query nonlinear stability');axes[0,1].set_ylim(0,1.05);axes[0,1].set_ylabel('Mean top-ten Jaccard')
    cold=split[split.key.str.contains('cold_sequence40')];axes[1,0].bar(range(len(cold)),cold.max_train_test_global_identity);axes[1,0].axhline(.4,color='red',linestyle='--');axes[1,0].set_title('(c) Cold train/test global identity');axes[1,0].set_xlabel('Endpoint/seed partition');axes[1,0].set_ylabel('Maximum global sequence identity')
    axes[1,1].bar(range(4),contacts.fragment_contact_fraction_min,label='Selected fragment');axes[1,1].plot(range(4),contacts.all_ligand_contact_fraction,'ko',label='Whole ligand');axes[1,1].set_xticks(range(4),[f'{r.pdb_id}\nbit {r.bit}' for r in contacts.itertuples()]);axes[1,1].set_title('(d) Four examples; only two complexes');axes[1,1].set_ylabel('Atom fraction within 4 Å of protein');axes[1,1].legend()
    for ext in ['png','pdf']:fig.savefig(FIG/f'explanation_checks.{ext}',dpi=180)
    plt.close(fig)
    (SHOTS/'capture.json').write_text(json.dumps(dict(source='Generated read-only dashboard served over local HTTP; not a claim of public hosting',javascript_errors=errors,mobile_overflow=False,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SHOTS.glob('*.png')}),indent=2))
    print('Updated four-panel assets and browser checks complete')

if __name__=='__main__':main()
