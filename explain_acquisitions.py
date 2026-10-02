"""Replay acquisition decisions without retraining, and build an HTML report."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from grid_gp import Geometry, ExactPairGP
from protein_lipid_active_learning import protein_features
from run_full_grid import ROOT, OUT as GRID, datasets, split
from explain_grid import sha


REASONS = {
    'R': 'Uniform random sampling; prediction and uncertainty did not determine selection.',
    'E': 'Explore: rank by predictive observation standard deviation.',
    'X': 'Exploit: rank by predicted mean affinity (higher pK).',
    'B': 'Balanced UCB: rank by 0.5 * predicted mean + 0.5 * observation SD.',
    'T': 'Thompson: rank by one correlated latent-posterior function draw; not by mean alone.',
}


def replay(record, df, x):
    initial = np.asarray(record['initial_indices'])
    protein = np.vstack([protein_features(s) for s in df.protein_sequence])
    geom = Geometry(x, protein, initial)
    labelled = initial.copy()
    seed = record['config']['seed']
    _, _, pool = split(len(df), seed, len(initial))
    rng = np.random.default_rng(seed+104729)
    y = df.affinity.to_numpy(float)
    rows = []
    decisions = []
    checked = 0
    for batch in record['acquired']:
        step, phase = batch['round'], batch['phase']
        selected = np.asarray(batch['indices'], int)
        if not len(selected):
            decisions.append(dict(round=step, phase=phase, selected_count=0, reason='Zero-budget phase: no acquisition.', replay_matches=True))
            continue
        fit = max((f for f in record['fits'] if f['round'] < step), key=lambda f: f['round'])
        model = ExactPairGP(geom, record['config']['kernel'])
        model.theta = np.array([fit['params'][n] for n in model.names])
        model.train = labelled
        values = y[labelled]
        model.center, model.scale = float(values.mean()), max(float(values.std()), 1e-6)
        model.y = (values-model.center)/model.scale
        model.mean = fit['params']['mean']
        model.noise = np.exp(fit['params']['log_noise'])
        k = model.cov(labelled, labelled)+(model.noise+1e-8)*np.eye(len(labelled))
        model.cf = cho_factor(k, lower=True, check_finite=False)
        model.alpha = cho_solve(model.cf, model.y-model.mean, check_finite=False)
        mean, latent_sd, obs_sd = model.predict(pool)
        if phase == 'R':
            score = np.full(len(pool), np.nan)
            predicted_selection = rng.choice(pool, len(selected), replace=False)
            ranks = np.full(len(pool), np.nan)
        else:
            score = model.thompson(pool, rng) if phase == 'T' else (obs_sd if phase == 'E' else (mean if phase == 'X' else .5*(mean+obs_sd)))
            order = np.argsort(-score, kind='stable')
            predicted_selection = pool[order[:len(selected)]]
            ranks = np.empty(len(pool), int)
            ranks[order] = np.arange(1, len(pool)+1)
        if not np.array_equal(predicted_selection, selected):
            raise AssertionError(f'Acquisition replay mismatch at round {step}: {record["config"]}')
        checked += 1
        flags = np.isin(pool, selected)
        for j, row in enumerate(pool):
            # All eligible candidates, not just selected rows, for fair comparison.
            rows.append(dict(round=step, phase=phase, row_index=int(row), selected=bool(flags[j]),
                             predicted_mean_pK=mean[j], latent_sd_pK=latent_sd[j], observation_sd_pK=obs_sd[j],
                             acquisition_score=score[j], acquisition_rank=ranks[j]))
        decisions.append(dict(round=step, phase=phase, selected_count=len(selected), candidate_count=len(pool),
                              reason=REASONS[phase], replay_matches=True,
                              preceding_fit_converged=fit['success']))
        labelled = np.r_[labelled, selected]
        pool = pool[~flags]
    return pd.DataFrame(rows), decisions, checked


def build_report(root, summary, replay_audit):
    settings = json.loads((root/'manifest.json').read_text())
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>Protein-lipid explainability</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}td,th{padding:7px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}img{max-width:48%}details{margin:20px 0}code{background:#eee}a{color:#17657c}</style></head><body>',
             '<h1>Protein-lipid model explainability</h1>',
             f'<p>{len(summary)} saved final models explained. {replay_audit["verified_nonempty_rounds"]} nonempty acquisition rounds replayed with exact agreement.</p>',
             '<p>Default selection: lowest mean test-MAE model, highest top-10% recall-AUC model, and lowest-MAE Thompson model per dataset; three seeds. These are descriptive selections on the same test results, not independently validated winners.</p>',
             '<h2>How to read these results</h2><ul>',
             '<li>Positive SHAP contributions increase predicted pKd/pKi (stronger predicted affinity). Baseline is the mean model prediction on the recorded acquired-training background.</li>',
             '<li>Protein-versus-lipid values are exact for a two-group replacement game. Detailed values use grouped permutation SHAP. These are different grouping games: do not add their attributions together.</li>',
             '<li>Fingerprint analysis isolates 20 high-variance training bits and groups all remaining bits. ChemBERTa uses 12 dimension blocks, not chemical functional-group labels. Protein descriptors stay jointly grouped, not residue-wise.</li>',
             f'<li>Up to {settings["samples"]} held-out pairs and {settings["background"]} training backgrounds per model; two runs of {settings["permutations"]} antithetic permutations. Global importance is only over this explained subset. Repeat differences measure sampling sensitivity, not confidence intervals.</li>',
             '<li>Feature replacement may produce off-manifold fingerprint/embedding combinations. These computational perturbations are not additional experimental data. SHAP is not causal, structural, binding-site or residue-contact evidence.</li>',
             '<li>Single-target protein attributions are zero because the protein is constant, not because proteins are biologically irrelevant. Small panels and random-pair splits retain the benchmark limitations.</li>',
             '<li>Final-prediction SHAP does not explain a Thompson random draw. Acquisition CSVs separately show the mean, uncertainty, actual replayed draw/score, rank and selection for every eligible candidate.</li></ul>',
             '<p><a href="summary.csv">All model summaries</a> | <a href="validation.json">SHAP validation</a> | <a href="acquisition_validation.json">Decision replay validation</a></p>']
    for _, row in summary.iterrows():
        dest = ROOT/row.output
        rel = dest.relative_to(root).as_posix()
        title = f'{row.dataset} | {row.representation} / {row.kernel} / {row.protocol} | seed {row.seed}'
        parts.append(f'<details><summary>{html.escape(title)}</summary>')
        parts.append(f'<p>Roles: {html.escape(row.selection_roles)}. Mean absolute two-group contributions: lipid {row.lipid_mean_abs_shap_pK:.3f} pK; protein {row.protein_mean_abs_shap_pK:.3f} pK. SHAP repeat mean absolute difference: {row.repeat_mean_abs_difference_pK:.4f} pK.</p>')
        for file in ('global_importance.png', 'local_explanation.png'):
            parts.append(f'<a href="{rel}/{file}"><img loading="lazy" src="{rel}/{file}" alt="{file}"></a>')
        for file in ('global_importance.csv', 'local_shap.csv', 'explained_pairs.csv', 'fingerprint_annotations.json', 'metadata.json', 'acquisition_candidates.csv', 'acquisition_decisions.json'):
            parts.append(f'<p><a href="{rel}/{file}">{file}</a></p>')
        parts.append('</details>')
    parts.append('</body></html>')
    (root/'index.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--explanations', type=Path, default=ROOT/'results'/'explainability')
    args = ap.parse_args()
    root = args.explanations.resolve()
    summary = pd.read_csv(root/'summary.csv')
    manifest = json.loads((GRID/'experiment_manifest.json').read_text())
    assert sha(ROOT/'grid_gp.py') == manifest['engine_sha256']
    assert sha(ROOT/'run_full_grid.py') == manifest['runner_sha256']
    feature_file = ROOT/'data/features/representations.npz'
    assert sha(feature_file) == manifest['feature_sha256']
    features = np.load(feature_file, allow_pickle=False)
    lookup = {s: i for i, s in enumerate(features['smiles'])}
    frames = {}
    checked = 0
    with threadpool_limits(limits=1):
        for i, row in summary.iterrows():
            path = GRID/'runs'/row.dataset/row.representation/row.kernel/row.protocol/f'seed{row.seed}.json'
            record = json.loads(path.read_text())
            dest = ROOT/row.output
            meta = json.loads((dest/'metadata.json').read_text())
            assert record['signature'] == manifest['signature'] and sha(path) == meta['checkpoint_sha256']
            if row.dataset not in frames:
                data = datasets()[row.dataset]
                assert sha(data) == manifest['data_sha256'][row.dataset]
                frames[row.dataset] = pd.read_csv(data)
            df = frames[row.dataset]
            x = features[row.representation][[lookup[s] for s in df.lipid_smiles]]
            candidates, decisions, count = replay(record, df, x)
            candidates.to_csv(dest/'acquisition_candidates.csv', index=False)
            (dest/'acquisition_decisions.json').write_text(json.dumps(decisions, indent=2))
            checked += count
            print(f'{i+1}/{len(summary)} decision replay verified', flush=True)
    audit = dict(models=len(summary), verified_nonempty_rounds=checked, all_selected_indices_match=True,
                 method='Rebuild each saved posterior; replay original RNG and acquisition without refitting.',
                 script_sha256=sha(Path(__file__)), experiment_signature=manifest['signature'])
    (root/'acquisition_validation.json').write_text(json.dumps(audit, indent=2))
    build_report(root, summary, audit)
    print('Report:', root/'index.html')


if __name__ == '__main__':
    main()
