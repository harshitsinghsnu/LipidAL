"""Temporal, ungrouped ECFP SHAP following paper section 2.6.

Uses saved linear-kernel campaigns: exact linear Shapley solution at fixed query
protein, checked with KernelExplainer. Does not retrain or use pool labels in SHAP.
"""
import json
import html
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Draw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grid_gp import Geometry, ExactPairGP, SCHEDULES
from protein_lipid_active_learning import protein_features
from run_full_grid import ROOT, OUT as GRID, datasets, sha, split

OUT = ROOT/'results'/'paper_explainability'
PROTOCOLS = ('ucb_explore_heavy', 'ucb_exploit_heavy')


def restore(record, df, geometry, cycle):
    train = list(record['initial_indices'])
    for query in record['acquired']:
        if query['round'] <= cycle:
            train.extend(query['indices'])
    fit = max((f for f in record['fits'] if f['round'] <= cycle), key=lambda f: f['round'])
    gp = ExactPairGP(geometry, 'linear')
    gp.train = np.array(train)
    y = df.affinity.to_numpy(float)[train]
    gp.center, gp.scale = float(y.mean()), max(float(y.std()), 1e-6)
    gp.mean = fit['params']['mean']
    gp.theta = np.array([fit['params'][n] for n in gp.names])
    gp.noise = np.exp(fit['params']['log_noise'])
    gp.cf = cho_factor(gp.cov(gp.train, gp.train)+(gp.noise+1e-8)*np.eye(len(train)), lower=True)
    gp.alpha = cho_solve(gp.cf, (y-gp.center)/gp.scale-gp.mean)
    test = np.array(record['test_indices'])
    pred, _, sd = gp.predict(test)
    mae = np.mean(abs(pred-df.affinity.to_numpy(float)[test]))
    assert abs(mae-record['history'][cycle]['test_mae']) < 1e-7
    if cycle == 10:
        np.testing.assert_allclose(pred, record['test_predictions'], atol=1e-7, rtol=0)
        np.testing.assert_allclose(sd, record['test_sd'], atol=1e-7, rtol=0)
    return gp


def weights(gp, x, initial, rows):
    par = dict(zip(gp.names, gp.theta))
    kp = np.ones((len(rows), len(gp.train)))
    if gp.g.multi_target:
        r = np.sqrt(3*gp.g.protein_d2[np.ix_(rows, gp.train)])/np.exp(par['log_protein_length'])
        kp = (1+r)*np.exp(-r)
    scale = max(float(np.mean((x[initial]**2).sum(1))), 1e-12)
    return (kp*gp.alpha)@x[gp.train]*(gp.scale*np.exp(par['log_signal'])/scale)


def kernel_audit(x, background, w, intercept, expected, seed):
    # For affine f, replacing empirical background by its mean preserves v(S)
    # for EVERY coalition exactly. No feature grouping or dimension truncation.
    baseline = background.mean(0)
    active = np.flatnonzero(abs(x-baseline) > 1e-12)
    if not len(active):
        return 0.
    offset = intercept+baseline@w-baseline[active]@w[active]
    def predict(z):
        return offset+np.asarray(z)@w[active]
    np.random.seed(seed)
    explainer = shap.KernelExplainer(predict, baseline[None, active])
    result = np.asarray(explainer.shap_values(x[None, active], nsamples=2*len(active)+128,
                                            l1_reg=0, silent=True))[0]
    error = float(np.max(abs(result-expected[active])))
    assert error < 1e-6, error
    return error


def fragments(df, x, importance, bits, dest):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=4096, includeChirality=False)
    results = []
    distributions = []
    mols, legends, highlights, bondlights = [], [], [], []
    for bit in bits:
        members = np.flatnonzero(x[:, bit] > 0)
        if not len(members):
            continue
        # Retrospective interpretation only: labels never enter model/SHAP.
        ordered = sorted(members, key=lambda i: (-float(df.iloc[i].affinity), int(i)))
        row = int(ordered[0])
        mol = Chem.MolFromSmiles(df.iloc[row].lipid_smiles)
        extra = rdFingerprintGenerator.AdditionalOutput()
        extra.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=extra)
        envs = []
        for center, radius in extra.GetBitInfoMap()[int(bit)]:
            bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
            atoms = {center}
            for b in bonds:
                bond = mol.GetBondWithIdx(b)
                atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=bonds, canonical=True)
            envs.append(dict(center=int(center), radius=int(radius), atoms=sorted(atoms), bonds=bonds, fragment_smiles=smi))
        # Preserve all collision environments; deterministic nonzero-radius representative.
        rep = sorted(envs, key=lambda e: (e['radius'] == 0, e['radius'], e['center']))[0]
        prevalence = len(members)/len(df)
        results.append(dict(bit=int(bit), mean_abs_shap=float(importance[bit]), frequency_pairs=len(members),
                            frequency_unique_lipids=int(df.iloc[members].lipid_smiles.nunique()),
                            prevalence=prevalence, combined_score=float(prevalence*importance[bit]),
                            representative_row=row, representative_pK=float(df.iloc[row].affinity),
                            representative_fragment=rep['fragment_smiles'], environments=envs,
                            affinity_source='all measured rows, retrospective only; may include held-out rows'))
        for idx in range(len(df)):
            distributions.append(dict(bit=int(bit), row_index=idx, present=bool(x[idx, bit]),
                                      affinity=float(df.iloc[idx].affinity), protein_id=str(df.iloc[idx].get('protein_id', ''))))
        mols.append(mol)
        legends.append(f'Bit {bit} | pK {df.iloc[row].affinity:.2f}\n{len(members)}/{len(df)} pairs')
        highlights.append(rep['atoms'])
        bondlights.append(rep['bonds'])
    results.sort(key=lambda r: -r['combined_score'])
    (dest/'fragment_mapping.json').write_text(json.dumps(results, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != 'environments'} for r in results]).to_csv(dest/'fragment_ranking.csv', index=False)
    pd.DataFrame(distributions).to_csv(dest/'affinity_distributions.csv', index=False)
    if mols:
        svg = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(500, 330), legends=legends,
                                  highlightAtomLists=highlights, highlightBondLists=bondlights, useSVG=True)
        (dest/'highlighted_fragments.svg').write_text(svg, encoding='utf-8')
    fig, ax = plt.subplots(figsize=(11, 5))
    top = bits[:5]
    data, labels = [], []
    for bit in top:
        for present in (True, False):
            vals = df.affinity.to_numpy()[x[:, bit].astype(bool) == present]
            data.append(vals if len(vals) else np.array([np.nan]))
            labels.append(f'{bit}\n'+('present' if present else 'absent'))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel('Measured pK (retrospective; not a validation test)')
    ax.set_title('Top bits: affinity distributions with and without the bit')
    fig.tight_layout()
    fig.savefig(dest/'affinity_distributions.png', dpi=150)
    plt.close(fig)


def campaign(name, protocol, seed, df, x, manifest):
    path = GRID/'runs'/name/'ecfp'/'linear'/protocol/f'seed{seed}.json'
    record = json.loads(path.read_text())
    assert record['signature'] == manifest['signature']
    dest = OUT/name/protocol/f'seed{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    initial = np.array(record['initial_indices'])
    protein = np.vstack([protein_features(s) for s in df.protein_sequence])
    geom = Geometry(x, protein, initial)
    all_importance, audits, tops, samples = [], [], [], []
    max_error, ke_error = 0., 0.
    for cycle in range(11):
        gp = restore(record, df, geom, cycle)
        pool = np.setdiff1d(np.arange(len(df)), np.r_[gp.train, record['test_indices']])
        rng = np.random.default_rng(seed*1000+cycle)
        rows = rng.choice(pool, min(100, len(pool)), replace=False)
        background = rng.choice(gp.train, min(50, len(gp.train)), replace=False)
        assert not set(rows) & set(gp.train) and not set(rows) & set(record['test_indices'])
        w = weights(gp, x, initial, rows)
        intercept = gp.center+gp.scale*gp.mean
        phi = (x[rows]-x[background].mean(0))*w
        base = intercept+w@x[background].mean(0)
        pred = gp.predict(rows)[0]
        error = float(np.max(abs(base+phi.sum(1)-pred)))
        assert error < 1e-6
        max_error = max(error, max_error)
        if cycle == 10:
            ke_error = kernel_audit(x[rows[0]], x[background], w[0], intercept, phi[0], seed)
        importance = abs(phi).mean(0)
        ranked = np.argsort(-importance, kind='stable')[:10]
        all_importance.append(importance)
        tops.append(set(ranked.tolist()))
        audits.append(dict(cycle=cycle, labelled=len(gp.train), unqueried=len(pool), explained=len(rows),
                           background=len(background), phase='initial' if cycle == 0 else SCHEDULES[protocol][cycle-1],
                           additivity_error=error, model_fit_round=max(f['round'] for f in record['fits'] if f['round']<=cycle),
                           top10=ranked.tolist()))
        samples.append(dict(cycle=cycle, query_indices=rows.tolist(), background_indices=background.tolist()))
        np.savez_compressed(dest/f'cycle{cycle:02d}.npz', shap=phi, expected_values=base, predictions=pred,
                            query_indices=rows, background_indices=background, mean_abs_shap=importance)
    importance = np.array(all_importance)
    pd.DataFrame(importance, columns=[f'bit_{i}' for i in range(4096)]).rename_axis('cycle').to_csv(dest/'feature_evolution.csv')
    pd.DataFrame(audits).to_csv(dest/'cycle_audit.csv', index=False)
    (dest/'sampling.json').write_text(json.dumps(samples, indent=2))
    final_bits = np.argsort(-importance[-1], kind='stable')[:10]
    fig, ax = plt.subplots(figsize=(11, 6))
    for bit in final_bits:
        ax.plot(range(11), importance[:, bit], marker='.', label=f'bit {bit}')
    ax.set_xticks(range(11), [f'{i}\n'+('initial' if i==0 else SCHEDULES[protocol][i-1]) for i in range(11)])
    ax.set_xlabel('AL cycle / actual acquisition phase (E=explore, X=exploit)')
    ax.set_ylabel('Mean absolute per-bit SHAP (pK)')
    ax.set_title(f'{name}: ECFP / linear / {protocol} / seed {seed}')
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(dest/'feature_evolution.png', dpi=150)
    plt.close(fig)
    fragments(df, x, importance[-1], final_bits, dest)
    stability = []
    for cycle in range(1, 11):
        union = tops[cycle] | tops[cycle-1]
        active = (importance[cycle] > 1e-12) | (importance[cycle-1] > 1e-12)
        rho = float(spearmanr(importance[cycle, active], importance[cycle-1, active]).statistic) if active.sum()>1 else None
        stability.append(dict(cycle=cycle, top10_jaccard=len(tops[cycle]&tops[cycle-1])/len(union),
                              active_bit_spearman=rho))
    pd.DataFrame(stability).to_csv(dest/'temporal_stability.csv', index=False)
    result = dict(dataset=name, protocol=protocol, seed=seed, kernel='linear', representation='ecfp',
                  cycles=11, explanations=sum(a['explained'] for a in audits), max_additivity_error=max_error,
                  final_kernel_explainer_max_error=ke_error, top10_jaccard_initial_final=len(tops[0]&tops[-1])/len(tops[0]|tops[-1]),
                  final_max_bit_importance=float(importance[-1].max()), checkpoint_sha256=sha(path),
                  output=str(dest.relative_to(OUT)))
    (dest/'completed.json').write_text(json.dumps(result, indent=2))
    return result


def report(summary):
    comparisons = []
    for name, group in summary.groupby('dataset'):
        for seed in (7, 19, 42):
            rows = group[group.seed == seed]
            arrays = [pd.read_csv(OUT/r.output/'feature_evolution.csv', index_col=0).to_numpy() for _, r in rows.iterrows()]
            for cycle in range(11):
                a, b = arrays[0][cycle], arrays[1][cycle]
                ta, tb = set(np.argsort(-a)[:10]), set(np.argsort(-b)[:10])
                active = (a>1e-12) | (b>1e-12)
                comparisons.append(dict(dataset=name, seed=seed, cycle=cycle, top10_jaccard=len(ta&tb)/len(ta|tb),
                                         active_bit_spearman=float(spearmanr(a[active], b[active]).statistic) if active.sum()>1 else np.nan))
    pd.DataFrame(comparisons).to_csv(OUT/'cross_protocol_stability.csv', index=False)
    body = ['<!doctype html><meta charset="utf-8"><title>Paper-style lipid explainability</title>',
            '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}details{margin:20px}a{color:#17657c}</style>',
            '<h1>Temporal ECFP-bit explainability</h1>',
            '<p>36 matched ECFP/linear campaigns, explore-heavy versus exploit-heavy, three seeds, six panels. All 4096 bits, no grouping. Each cycle samples up to 100 unqueried pairs and 50 acquired-background lipids.</p>',
            '<p>Exact per-bit Shapley solution for the affine lipid predictor at a fixed query protein. KernelExplainer independently checks one final query per campaign. This is an adaptation, not a rerun of the paper on its targets or every kernel.</p>',
            '<p>Top ten bits are selected after SHAP, not prefiltered. Evolution plots track final top-ten bits; every cycle has its own top-ten list in cycle_audit.csv. Pool/background changes can change importance without a change in the model.</p>',
            '<p>Highlighted fragments come from the highest measured-affinity bit-containing pair. Affinity distributions and fragment choices use retrospective labels, including held-out rows, for interpretation only. Pooled panels mix proteins; these distributions cannot establish target-specific SAR or binding mechanisms. No 3-D interaction or residue claims are made.</p>',
            '<p>Fragment score = pair prevalence × mean absolute bit SHAP (explicit implementation choice; the paper does not specify the algebra). All representative-molecule collision environments are retained. Tiny panels exhaust almost all candidates: their curves are exploratory.</p>',
            '<p><a href="summary.csv">Summary</a> | <a href="cross_protocol_stability.csv">Protocol stability</a> | <a href="validation.json">Validation</a></p>']
    for _, r in summary.iterrows():
        rel = Path(r.output).as_posix()
        title = html.escape(f'{r.dataset} / {r.protocol} / seed {r.seed}')
        body.append(f'<details><summary>{title}</summary>')
        for file in ('feature_evolution.png','affinity_distributions.png','highlighted_fragments.svg'):
            body.append(f'<img loading="lazy" src="{rel}/{file}" alt="{file}">')
        for file in ('fragment_ranking.csv','fragment_mapping.json','feature_evolution.csv','temporal_stability.csv','cycle_audit.csv','sampling.json'):
            body.append(f'<p><a href="{rel}/{file}">{file}</a></p>')
        body.append('</details>')
    (OUT/'index.html').write_text('\n'.join(body), encoding='utf-8')


def main():
    manifest = json.loads((GRID/'experiment_manifest.json').read_text())
    assert sha(ROOT/'grid_gp.py') == manifest['engine_sha256']
    assert sha(ROOT/'run_full_grid.py') == manifest['runner_sha256']
    file = ROOT/'data/features/representations.npz'
    assert sha(file) == manifest['feature_sha256']
    cache = np.load(file, allow_pickle=False)
    lookup = {s:i for i,s in enumerate(cache['smiles'])}
    OUT.mkdir(parents=True, exist_ok=True)
    config = dict(script_sha256=sha(Path(__file__)), grid_signature=manifest['signature'],
                  kernels=['linear'], protocols=PROTOCOLS, seeds=[7,19,42], cycles=list(range(11)),
                  n_query=100, n_background=50, bits=4096, shap_version=shap.__version__,
                  estimator='Exact linear per-bit Shapley; KernelExplainer audit final first query, l1_reg=0',
                  protein_adaptation='hold query protein fixed; backgrounds supply lipids only',
                  paper='https://doi.org/10.1039/D5DD00436E', authors_code_commit='c77e3a6a77e0207f995815c22f3146ae9b046777')
    # JSON normalizes tuples so compare serialized structures.
    config = json.loads(json.dumps(config))
    marker = OUT/'manifest.json'
    if marker.exists() and json.loads(marker.read_text()) != config:
        raise RuntimeError('Changed code/config: preserve existing results and use a new output directory')
    marker.write_text(json.dumps(config, indent=2))
    results=[]
    with threadpool_limits(limits=1):
        for name, path in datasets().items():
            assert sha(path) == manifest['data_sha256'][name]
            df = pd.read_csv(path)
            x = cache['ecfp'][[lookup[s] for s in df.lipid_smiles]].astype(float)
            for protocol in PROTOCOLS:
                for seed in (7,19,42):
                    done = OUT/name/protocol/f'seed{seed}'/'completed.json'
                    if done.exists():
                        result = json.loads(done.read_text())
                        source = GRID/'runs'/name/'ecfp'/'linear'/protocol/f'seed{seed}.json'
                        assert result['checkpoint_sha256'] == sha(source)
                    else:
                        result = campaign(name, protocol, seed, df, x, manifest)
                    results.append(result)
                    pd.DataFrame(results).to_csv(OUT/'summary.csv', index=False)
                    print(f'{len(results)}/36 {name} {protocol} seed={seed}', flush=True)
    summary = pd.DataFrame(results)
    validation = dict(campaigns=len(results), cycles=int(summary.cycles.sum()), explanations=int(summary.explanations.sum()),
                      max_additivity_error=float(summary.max_additivity_error.max()),
                      max_kernel_explainer_disagreement=float(summary.final_kernel_explainer_max_error.max()),
                      kernel_explainer_checks=len(results), complete=len(results)==36)
    (OUT/'validation.json').write_text(json.dumps(validation, indent=2))
    report(summary)
    print(json.dumps(validation, indent=2))


if __name__=='__main__':
    main()
