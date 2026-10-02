"""Post-hoc grouped SHAP for saved exact GPs; never retrains or acquires labels.

Default: descriptive prediction/discovery winners and best-MAE Thompson for each
dataset, all three seeds. --all explains every completed grid run instead.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import shap
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid_gp import Geometry, ExactPairGP
from protein_lipid_active_learning import protein_features
from run_full_grid import ROOT, OUT as GRID, datasets, sha


def distance(a, b):
    return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T, 0)


def median_distance(x):
    d = distance(x, x)
    # Match Geometry's exactly-zero diagonal, even with floating point rounding.
    np.fill_diagonal(d, 0)
    positive = d[d > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


class SavedPredictor:
    """Restore final posterior from saved parameters, without optimizing again."""
    def __init__(self, record, df, lipid):
        self.record = record
        self.x = np.asarray(lipid, float)
        self.p = np.vstack([protein_features(s) for s in df.protein_sequence])
        initial = np.asarray(record['initial_indices'], int)
        train = initial.tolist()
        for batch in record['acquired']:
            train.extend(batch['indices'])
        self.train = np.asarray(train, int)
        self.test = np.asarray(record['test_indices'], int)
        assert not set(train) & set(self.test)
        self.scaler = StandardScaler().fit(self.p[initial])
        self.ps = self.scaler.transform(self.p)
        self.lipid_scale = median_distance(self.x[initial])
        self.protein_scale = median_distance(self.ps[initial])
        self.linear_scale = max(float(np.mean((self.x[initial]**2).sum(1))), 1e-12)
        self.par = record['fits'][-1]['params']
        self.kernel = record['config']['kernel']
        y = df.affinity.to_numpy(float)[self.train]
        self.center, self.scale = float(y.mean()), max(float(y.std()), 1e-6)
        self.tx = self.x[self.train]
        self.tp = self.ps[self.train]
        self.tnorm = (self.tx**2).sum(1)
        k = self.cross(self.tx, self.p[self.train])
        cf = cho_factor(k + (np.exp(self.par['log_noise'])+1e-8)*np.eye(len(train)),
                        lower=True, check_finite=False)
        self.alpha = cho_solve(cf, (y-self.center)/self.scale-self.par['mean'], check_finite=False)
        self.cf = cf
        pred, sd = self.predict(self.x[self.test], self.p[self.test], uncertainty=True)
        self.prediction_error = float(np.max(abs(pred-record['test_predictions'])))
        self.sd_error = float(np.max(abs(sd-record['test_sd'])))
        if self.prediction_error > 1e-6 or self.sd_error > 1e-6:
            raise AssertionError(f'Restored predictions differ: {self.prediction_error}, {self.sd_error}')

    def kernel_stats(self, dot, norm, pd):
        par = self.par
        if self.kernel == 'linear':
            kl = dot/self.linear_scale
        elif self.kernel == 'tanimoto':
            total = norm[..., None]+self.tnorm
            kl = dot/np.maximum(total-dot, 1e-12)
            kl = np.where(total < 1e-12, 1., kl)
        else:
            d = np.maximum(norm[..., None]+self.tnorm-2*dot, 0)
            d = d/self.lipid_scale/np.exp(2*par['log_lipid_length'])
            if self.kernel == 'rbf':
                kl = np.exp(-.5*d)
            elif self.kernel == 'rq':
                a = np.exp(par['log_rq_alpha'])
                kl = np.exp(-a*np.log1p(d/(2*a)))
            else:
                r = np.sqrt(3*d)
                kl = (1+r)*np.exp(-r)
        if 'log_protein_length' in par:
            r = np.sqrt(3*np.maximum(pd, 0)/self.protein_scale)/np.exp(par['log_protein_length'])
            kl = kl*(1+r)*np.exp(-r)
        return np.exp(par['log_signal'])*kl

    def cross(self, x, p):
        return self.kernel_stats(x@self.tx.T, (x*x).sum(1), distance(self.scaler.transform(p), self.tp))

    def predict(self, x, p, uncertainty=False):
        cross = self.cross(np.asarray(x), np.asarray(p))
        mean = self.center+self.scale*(self.par['mean']+cross@self.alpha)
        if not uncertainty:
            return mean
        diag = np.exp(self.par['log_signal'])*((x*x).sum(1)/self.linear_scale if self.kernel == 'linear' else np.ones(len(x)))
        solved = cho_solve(self.cf, cross.T, check_finite=False)
        latent = np.maximum(diag-np.einsum('ij,ji->i', cross, solved), 1e-12)*self.scale**2
        return mean, np.sqrt(latent+np.exp(self.par['log_noise'])*self.scale**2)


class CoalitionGame:
    """v(S)=mean of predictions over acquired-label backgrounds.

All protein descriptors form one coherent group. Lipid groups partition every
dimension. Sufficient statistics make this exactly equivalent to predicting
masked feature vectors, without materializing thousands of 4096-bit vectors.
"""
    def __init__(self, model, row, background, groups):
        self.model = model
        self.groups = groups
        bx, x = model.x[background], model.x[row]
        self.base_dot = bx@model.tx.T
        self.base_norm = (bx*bx).sum(1)
        self.delta_dot = np.array([(x[g]-bx[:, g])@model.tx[:, g].T for g in groups])
        self.delta_norm = np.array([(x[g]**2-bx[:, g]**2).sum(1) for g in groups])
        self.base_pd = distance(model.ps[background], model.tp)
        self.target_pd = distance(model.ps[[row]], model.tp)
        self.cache = {}

    def __call__(self, masks):
        masks = np.asarray(masks, float)
        keys = [tuple(m) for m in masks]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        for start in range(0, len(missing), 128):
            chunk = missing[start:start+128]
            m = np.asarray(chunk)
            dot = self.base_dot[None]+np.einsum('mg,gbt->mbt', m[:, :-1], self.delta_dot, optimize=True)
            norm = self.base_norm[None]+m[:, :-1]@self.delta_norm
            pd = self.base_pd[None]+m[:, -1, None, None]*(self.target_pd-self.base_pd)[None]
            k = self.model.kernel_stats(dot, norm, pd)
            values = self.model.center+self.model.scale*(self.model.par['mean']+np.mean(k@self.model.alpha, axis=1))
            self.cache.update(zip(chunk, values.tolist()))
        return np.array([self.cache[k] for k in keys])


def feature_groups(model, representation, top_bits):
    d = model.x.shape[1]
    if representation == 'chemberta':
        groups = [g for g in np.array_split(np.arange(d), 12) if len(g)]
        names = [f'ChemBERTa dims {g[0]}-{g[-1]}' for g in groups]
    else:
        # Feature selection is training-input-only, not based on held-out labels.
        variance = model.tx.var(0)
        ranked = np.argsort(-variance, kind='stable')
        selected = [int(i) for i in ranked if variance[i] > 0][:top_bits]
        groups = [np.array([i]) for i in selected]
        names = [f'{representation.upper()} bit {i}' for i in selected]
        rest = np.setdiff1d(np.arange(d), selected)
        if len(rest):
            groups.append(rest)
            names.append('Remaining lipid bits (joint)')
    assert sorted(np.concatenate(groups).tolist()) == list(range(d))
    return groups, names+['Protein descriptors (joint)']


def choose_runs(all_runs=False):
    summary = pd.read_csv(GRID/'summary.csv')
    choices = {}
    if all_runs:
        selected = [(row, 'all') for _, row in summary.iterrows()]
    else:
        selected = []
        for _, frame in summary.groupby('dataset'):
            selected.extend([(frame.sort_values('mae_mean').iloc[0], 'prediction'),
                             (frame.sort_values('recall_auc_mean', ascending=False).iloc[0], 'discovery'),
                             (frame[frame.protocol == 'thompson'].sort_values('mae_mean').iloc[0], 'thompson')])
    for row, role in selected:
        for seed in (7, 19, 42):
            p = GRID/'runs'/row.dataset/row.representation/row.kernel/row.protocol/f'seed{seed}.json'
            choices.setdefault(p, []).append(role)
    return choices


def fingerprint_annotations(df, rep, groups, background, rows):
    """Map actual bit occurrences, not assumed unique functional groups."""
    annotations = []
    if rep == 'chemberta':
        return annotations
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=4096, includeChirality=False)
    for row in sorted(set(background.tolist()+rows.tolist())):
        mol = Chem.MolFromSmiles(df.iloc[row].lipid_smiles)
        info = None
        if rep == 'ecfp':
            extra = rdFingerprintGenerator.AdditionalOutput()
            extra.AllocateBitInfoMap()
            generator.GetFingerprint(mol, additionalOutput=extra)
            info = extra.GetBitInfoMap()
        for g in groups:
            if len(g) != 1:
                continue
            bit = int(g[0])
            entry = dict(row_index=int(row), bit=bit, representation=rep)
            if rep == 'maccs':
                smarts, threshold = MACCSkeys.smartsPatts.get(bit, ('?', 0))
                pattern = Chem.MolFromSmarts(smarts) if smarts != '?' else None
                matches = mol.GetSubstructMatches(pattern) if pattern is not None else ()
                entry.update(smarts=smarts, count_threshold=threshold, atom_matches=matches,
                             special_case=(pattern is None))
            else:
                occurrences = []
                for center, radius in info.get(bit, ()):
                    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
                    atoms = {center}
                    for bond in bonds:
                        b = mol.GetBondWithIdx(bond)
                        atoms.update([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
                    fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=bonds)
                    occurrences.append(dict(center=int(center), radius=int(radius), atoms=sorted(atoms), fragment_smiles=fragment))
                entry['occurrences'] = occurrences
            annotations.append(entry)
    return annotations


def plots(dest, values, names, predictions, baselines):
    importance = np.mean(abs(values), axis=0)
    order = np.argsort(importance)[-15:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(names)[order], importance[order], color='#247d91')
    ax.set_xlabel('Mean |group SHAP| (pK units; explained held-out subset)')
    ax.set_title('Grouped feature importance; not causal effects')
    fig.tight_layout()
    fig.savefig(dest/'global_importance.png', dpi=150)
    plt.close(fig)
    order = np.argsort(abs(values[0]))[-12:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(names)[order], values[0, order], color=np.where(values[0, order] >= 0, '#b94050', '#247d91'))
    ax.axvline(0, color='black', linewidth=.6)
    ax.set_xlabel('Signed contribution to predicted pK (positive = stronger predicted affinity)')
    ax.set_title(f'First explained pair: baseline {baselines[0]:.3f} -> prediction {predictions[0]:.3f}\nLargest 12 groups shown; full additive values in local_shap.csv')
    fig.tight_layout()
    fig.savefig(dest/'local_explanation.png', dpi=150)
    plt.close(fig)


def explain_one(path, roles, df, lipid, dest, args, manifest):
    record = json.loads(path.read_text())
    if record['signature'] != manifest['signature']:
        raise AssertionError('Run signature does not match experiment')
    model = SavedPredictor(record, df, lipid)
    rng = np.random.default_rng(args.seed)
    background = rng.choice(model.train, min(args.background, len(model.train)), replace=False)
    rows = rng.choice(model.test, min(args.samples, len(model.test)), replace=False)
    assert set(background).issubset(model.train) and not set(background) & set(model.test)
    groups, names = feature_groups(model, record['config']['representation'], args.top_bits)
    values, blocks, differences, baselines, predictions = [], [], [], [], []
    for row in rows:
        game = CoalitionGame(model, row, background, groups)
        count = len(names)
        def calculate(seed):
            exp = shap.PermutationExplainer(game, np.zeros((1, count)), feature_names=names, seed=seed)
            return exp(np.ones((1, count)), max_evals=(2*count+1)*args.permutations, silent=True)
        first = calculate(args.seed+int(row))
        second = calculate(args.seed+100000+int(row))
        v = .5*(first.values[0]+second.values[0])
        base = float(first.base_values[0])
        prediction = float(model.predict(model.x[[row]], model.p[[row]])[0])
        assert abs(base+v.sum()-prediction) < 1e-6
        coarse = CoalitionGame(model, row, background, [np.arange(model.x.shape[1])])
        empty, lip, prot, full = coarse([[0, 0], [1, 0], [0, 1], [1, 1]])
        block = [.5*((lip-empty)+(full-prot)), .5*((prot-empty)+(full-lip))]
        assert abs(empty+sum(block)-prediction) < 1e-6
        assert abs(empty-base) < 1e-6
        values.append(v)
        differences.append(abs(first.values[0]-second.values[0]))
        blocks.append(block)
        baselines.append(base)
        predictions.append(prediction)
    values, blocks, differences = np.array(values), np.array(blocks), np.array(differences)
    dest.mkdir(parents=True, exist_ok=True)
    local = []
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            local.append(dict(row_index=int(row), feature_group=name, shap_pK=values[i, j],
                              repeat_abs_difference_pK=differences[i, j]))
    pd.DataFrame(local).to_csv(dest/'local_shap.csv', index=False)
    pd.DataFrame(dict(feature_group=names, mean_abs_shap_pK=abs(values).mean(0),
                      mean_shap_pK=values.mean(0), repeat_mean_abs_difference_pK=differences.mean(0))).sort_values(
                          'mean_abs_shap_pK', ascending=False).to_csv(dest/'global_importance.csv', index=False)
    pair_rows = df.iloc[rows].copy()
    pair_rows.insert(0, 'row_index', rows)
    pair_rows['baseline_pK'] = baselines
    pair_rows['prediction_pK'] = predictions
    pair_rows['lipid_block_shap_pK'] = blocks[:, 0]
    pair_rows['protein_block_shap_pK'] = blocks[:, 1]
    pair_rows['reconstructed_pK'] = np.array(baselines)+values.sum(1)
    pair_rows.to_csv(dest/'explained_pairs.csv', index=False)
    meta = dict(config=record['config'], selection_roles=roles, background_indices=background.tolist(),
                explained_indices=rows.tolist(), groups=[dict(name=n, lipid_dimensions=g.tolist()) for n, g in zip(names, groups)],
                protein_group='all 26 descriptors jointly; not residue attribution',
                prediction_restore_max_error=model.prediction_error, observation_sd_restore_max_error=model.sd_error,
                max_additivity_error=float(np.max(abs(np.array(baselines)+values.sum(1)-predictions))),
                permutations_per_repeat=args.permutations, repeats=2,
                repeat_mean_abs_difference_pK=float(differences.mean()),
                source_checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha(path),
                experiment_signature=manifest['signature'])
    (dest/'metadata.json').write_text(json.dumps(meta, indent=2))
    annotations = fingerprint_annotations(df, record['config']['representation'], groups, background, rows)
    (dest/'fingerprint_annotations.json').write_text(json.dumps(annotations, indent=2))
    plots(dest, values, names, predictions, baselines)
    return dict(**record['config'], selection_roles=';'.join(roles), explained_pairs=len(rows), background_pairs=len(background),
                lipid_mean_abs_shap_pK=float(abs(blocks[:, 0]).mean()), protein_mean_abs_shap_pK=float(abs(blocks[:, 1]).mean()),
                repeat_mean_abs_difference_pK=float(differences.mean()),
                prediction_restore_max_error=model.prediction_error, max_additivity_error=meta['max_additivity_error'],
                output=str(dest.relative_to(ROOT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all', action='store_true', help='Explain all 2160 checkpoints, not just representative configurations')
    ap.add_argument('--samples', type=int, default=8, help='Random held-out pairs per checkpoint')
    ap.add_argument('--background', type=int, default=8, help='Acquired training pairs only')
    ap.add_argument('--permutations', type=int, default=8, help='Antithetic permutations in each of two independent repeats')
    ap.add_argument('--top-bits', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--limit', type=int, default=0, help='Diagnostic checkpoint limit')
    ap.add_argument('--output', type=Path, default=ROOT/'results'/'explainability')
    args = ap.parse_args()
    if min(args.samples, args.background, args.permutations, args.top_bits) < 1 or args.limit < 0:
        ap.error('Sample counts must be positive; limit must be nonnegative')
    manifest = json.loads((GRID/'experiment_manifest.json').read_text())
    feature_file = ROOT/'data'/'features'/'representations.npz'
    if sha(feature_file) != manifest['feature_sha256']:
        raise AssertionError('Feature cache changed since grid run')
    for file, field in [('grid_gp.py', 'engine_sha256'), ('run_full_grid.py', 'runner_sha256')]:
        if sha(ROOT/file) != manifest[field]:
            raise AssertionError('Grid source changed since saved experiment')
    files = datasets()
    for name, file in files.items():
        if sha(file) != manifest['data_sha256'][name]:
            raise AssertionError('Dataset changed: '+name)
    features = np.load(feature_file, allow_pickle=False)
    lookup = {s: i for i, s in enumerate(features['smiles'])}
    choices = list(choose_runs(args.all).items())
    expected = len(choices)
    if args.limit:
        choices = choices[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(experiment_signature=manifest['signature'], script_sha256=sha(Path(__file__)),
                  summary_sha256=sha(GRID/'summary.csv'), all=args.all, samples=args.samples,
                  background=args.background, permutations=args.permutations, top_bits=args.top_bits,
                  seed=args.seed, expected_selected_runs=expected, limit=args.limit,
                  shap_version=shap.__version__, numpy_version=np.__version__, python=platform.python_version())
    config_path = args.output/'manifest.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Explanation configuration changed; use a new --output directory')
    config_path.write_text(json.dumps(config, indent=2))
    results = []
    frames = {}
    start = time.time()
    with threadpool_limits(limits=1):
        for i, (path, roles) in enumerate(choices):
            conf = json.loads(path.read_text())['config']
            dataset, rep = conf['dataset'], conf['representation']
            if dataset not in frames:
                frames[dataset] = pd.read_csv(files[dataset])
            df = frames[dataset]
            x = features[rep][[lookup[s] for s in df.lipid_smiles]]
            dest = args.output/dataset/rep/conf['kernel']/conf['protocol']/f"seed{conf['seed']}"
            checkpoint = dest/'completed_summary.json'
            if checkpoint.exists():
                meta = json.loads((dest/'metadata.json').read_text())
                assert meta['checkpoint_sha256'] == sha(path)
                result = json.loads(checkpoint.read_text())
            else:
                result = explain_one(path, roles, df, x, dest, args, manifest)
                checkpoint.write_text(json.dumps(result, indent=2))
            results.append(result)
            pd.DataFrame(results).to_csv(args.output/'summary.csv', index=False)
            print(f'{i+1}/{len(choices)} {dataset} {rep}/{conf["kernel"]}/{conf["protocol"]} seed={conf["seed"]} elapsed={time.time()-start:.0f}s', flush=True)
    summary = pd.DataFrame(results)
    audit = dict(expected_selected_runs=expected, completed=len(results), complete=len(results) == expected,
                 maximum_prediction_restore_error=float(summary.prediction_restore_max_error.max()),
                 maximum_additivity_error=float(summary.max_additivity_error.max()),
                 background='acquired training rows only; no held-out rows',
                 explanation='final posterior mean in pK; NOT Thompson random draws or acquisition scores')
    (args.output/'validation.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
