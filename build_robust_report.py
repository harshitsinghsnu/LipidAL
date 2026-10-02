"""Render completed, validation-selected studies without rerunning experiments."""
from pathlib import Path
import html
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/robust_study_v2'
SPLITS = ['random_pair', 'cold_sequence40', 'scaffold']
LABELS = ['Random pair', 'Cold sequence', 'Scaffold']

def panels(frame, group, metrics, filename, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, endpoint in enumerate(['Kd', 'Ki']):
        data = frame[frame.dataset == 'biodolphin_'+endpoint]
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for name, subset in data.groupby(group):
                values = subset.groupby('split')[metric].mean().reindex(SPLITS)
                ax.plot(range(3), values, marker='o', label=name.replace('_', ' '))
            ax.set_xticks(range(3), LABELS)
            ax.set_title(f'({chr(97+2*row+col)}) {endpoint}: {label}')
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle(title+'\nSix-seed descriptive means; configurations chosen by inner validation')
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT/f'{filename}.{suffix}', dpi=170)
    plt.close(fig)

def main():
    base = pd.read_csv(OUT/'validation_selected_summary.csv')
    protein = pd.read_csv(OUT/'protein_validation_selected_by_seed.csv')
    protein = protein[protein.protocol == 'prediction_aware_thompson']
    adaptation = pd.read_csv(ROOT/'results/adaptation_study/validation_selected_summary.csv')
    adaptation = adaptation[adaptation.protocol == 'prediction_aware_thompson']
    # Public browser records contain derived metrics, never sequences or labels.
    main_runs = pd.read_csv(OUT/'final_by_seed.csv')
    main_runs['study'] = 'Main: 9,720 campaigns'
    main_runs['variant'] = 'frozen'
    adapt_runs = pd.read_csv(ROOT/'results/adaptation_study/final_by_seed.csv')
    adapt_runs['study'] = 'Adaptation: 1,944 comparisons (648 reused)'
    adapt_runs['protein_representation'] = 'esm2_35m'
    adapt_runs['representation'] = 'chemberta_mlm'
    columns = ['study','dataset','split','protein_representation','representation','variant','kernel','protocol','seed',
               'test_mae','test_r2','recall_top10','recall_auc','validation_mae','labelled','total_label_cost','nonconverged_fits']
    records = pd.concat([main_runs[columns], adapt_runs[columns]], ignore_index=True)
    explorer = (ROOT/'web/robust_explorer.html').read_text(encoding='utf-8').replace(
        '__ROBUST_RECORDS__', records.to_json(orient='records', double_precision=10).replace('<','\\u003c'))
    panels(base, 'protocol', [('mae_mean', 'Test MAE (pK; lower better)'),
           ('recall_mean', 'Pool top-decile recall (higher better)')], 'policy_panels', 'Prediction versus discovery')
    panels(protein, 'protein_representation', [('test_mae', 'Test MAE (pK)'),
           ('recall_auc', 'Pool recall AUC')], 'protein_panels', 'Protein representations: prediction-aware Thompson')
    panels(adaptation, 'variant', [('mae_mean', 'Test MAE (pK)'),
           ('recall_mean', 'Pool top-decile recall')], 'adaptation_panels', 'Lipid adaptation: ESM-2 35M + ChemBERTa MLM')
    table = base[base.protocol.isin(['thompson', 'prediction_aware_thompson'])].copy()
    sections = []
    for name, title in [('policy_panels', 'Prediction–discovery trade-off'),
                        ('protein_panels', 'Frozen protein representations'),
                        ('adaptation_panels', 'Lipid-domain adaptation')]:
        sections.append(f'<section><h2>{title}</h2><img src="{name}.png" alt="Four panels comparing {html.escape(title)}"><p><a href="{name}.pdf">Vector PDF</a></p></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Leakage-controlled protein–lipid study</title><style>body{font:16px/1.6 system-ui;background:#eef3f8;color:#18283b;max-width:1200px;margin:auto;padding:24px}section,header{background:white;padding:24px;margin:20px 0;border-radius:14px}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}a{color:#075ba8}.note{border-left:4px solid #e2a12b;padding:14px;background:#fff7df}</style>
<header><h1>Leakage-controlled protein–lipid active learning</h1><p>Completed retrospective study · 9,720 main campaigns + 1,296 new adaptation campaigns.</p>
<p class="note">No demonstrated overall winner or statistical superiority. The new selector trades discovery for prediction on some tasks. These results do not constitute prospective validation.</p>
<p>Three splits, six seeds, three protein representations, five lipid representations, three kernels and six acquisition policies. Each campaign acquires 120 labels; additional inner-validation labels are explicitly charged. Test labels never select configurations.</p>
<p><a href="../../ROBUST_STUDY.md">Full methods and limitations</a> · <a href="audit.json">Main audit</a> · <a href="../adaptation_study/audit.json">Adaptation audit</a> · <a href="validation_selected_summary.csv">All selected results</a></p></header>'''
    page += explorer + ''.join(sections)
    page += '<section><h2>Thompson comparison</h2><p>Recall is measured in the acquisition pool, not the outer test set. Lines connect split regimes for readability, not time. See raw per-seed files for variability; plots show descriptive means only.</p><div class="scroll">'+table.to_html(index=False, float_format=lambda x:f'{x:.3f}')+'</div></section>'
    page += '''<section><h2>Explanation checks</h2><p>3,539 of 3,980 observed ECFP positions have multiple unfolded Morgan identifiers. Bit importance is not a unique chemical motif. Nonlinear individual-bit estimates cover only four models, two fixed queries each, and three backgrounds.</p><p>Four selected fragment examples map to heavy-atom contacts within 4 Å in two experimental complexes (4OAS and 6CDJ). These are illustrations, not independent mechanistic validation.</p><p><a href="../explanation_robustness/fingerprint_collisions.csv">Collision audit</a> · <a href="../explanation_robustness/background_stability.csv">Background sensitivity</a> · <a href="../structure_contacts/contact_summary.csv">Experimental structure contacts</a></p></section>
<section><h2>Interpretation limits</h2><p>Repeated nested holdouts use one inner validation split, not full nested K-fold. Six overlapping seeds and 24 corrected tests are underpowered: the minimum attainable Holm-adjusted exact two-sided p-value is 0.75. No equivalence claim is justified.</p><p>Lipid SSL used only 790 scaffold-disjoint molecules. Supervised residual feature adapters used only the 24 initial labels and were frozen before acquisition; these are not attention LoRA. Two ESM-2 sizes are not two different protein-model families. Assay noise, prospective evaluation and broad contact-map validation remain open.</p><p><a href="../../paper/main_updated.pdf">Updated full manuscript</a> · <a href="../paper_explainability/index.html">Original benchmark and highlighted fragments</a></p></section></html>'''
    (OUT/'index.html').write_text(page, encoding='utf-8')
    print('Built', OUT/'index.html')

if __name__ == '__main__':
    main()
