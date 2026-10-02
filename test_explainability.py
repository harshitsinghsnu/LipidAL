"""Verify explanations against saved experiments and real measured pairs."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from explain_grid import SavedPredictor, CoalitionGame, feature_groups, choose_runs
from run_full_grid import ROOT, OUT, datasets
from grid_gp import KERNELS


def main():
    features = np.load(ROOT/'data/features/representations.npz', allow_pickle=False)
    lookup = {s: i for i, s in enumerate(features['smiles'])}
    count = 0
    with threadpool_limits(limits=1):
        for dataset in ('biodolphin_Kd', 'traak_a_Kd1'):
            df = pd.read_csv(datasets()[dataset])
            for rep in ('ecfp', 'maccs', 'chemberta'):
                x = features[rep][[lookup[s] for s in df.lipid_smiles]]
                for kernel in KERNELS:
                    p = OUT/'runs'/dataset/rep/kernel/'random'/'seed7.json'
                    record = json.loads(p.read_text())
                    model = SavedPredictor(record, df, x)
                    background = model.train[:3]
                    row = model.test[0]
                    groups, names = feature_groups(model, rep, 5)
                    game = CoalitionGame(model, row, background, groups)
                    rng = np.random.default_rng(17)
                    masks = np.r_[np.zeros((1, len(names))), np.ones((1, len(names))),
                                  rng.integers(0, 2, size=(5, len(names)))]
                    for mask, value in zip(masks, game(masks)):
                        lx = model.x[background].copy()
                        pr = model.p[background].copy()
                        for keep, group in zip(mask[:-1], groups):
                            if keep:
                                lx[:, group] = model.x[row, group]
                        if mask[-1]:
                            pr[:] = model.p[row]
                        np.testing.assert_allclose(value, model.predict(lx, pr).mean(), atol=1e-7, rtol=1e-7)
                    block = CoalitionGame(model, row, background, [np.arange(x.shape[1])])
                    empty, lip, prot, full = block([[0, 0], [1, 0], [0, 1], [1, 1]])
                    protein_phi = .5*((prot-empty)+(full-lip))
                    if dataset.startswith('traak'):
                        assert abs(protein_phi) < 1e-9, 'Constant protein should have zero attribution'
                    count += 1
    selections = choose_runs()
    assert len(selections) > 0 and len(choose_runs(True)) == 2160
    print(f'PASS: {count} real-data posteriors (all 15 representation/kernel combinations, pooled and single-target).')
    print('Saved prediction/uncertainty parity, optimized masking vs direct predictions, constant protein null effect, full-grid selection passed.')


if __name__ == '__main__':
    main()
