"""Audit temporal explanations and exact linear shortcut on real observations."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from paper_explainability import OUT, ROOT, GRID, datasets, weights, restore, Geometry
from explain_grid import SavedPredictor
from protein_lipid_active_learning import protein_features


def main():
    summary = pd.read_csv(OUT/'summary.csv')
    assert len(summary) == 36
    features = np.load(ROOT/'data/features/representations.npz', allow_pickle=False)
    lookup = {s:i for i,s in enumerate(features['smiles'])}
    frames = {name:pd.read_csv(path) for name,path in datasets().items()}
    total = 0
    with threadpool_limits(limits=1):
        for _, row in summary.iterrows():
            df = frames[row.dataset]
            x = features['ecfp'][[lookup[s] for s in df.lipid_smiles]].astype(float)
            record = json.loads((GRID/'runs'/row.dataset/'ecfp'/'linear'/row.protocol/f'seed{row.seed}.json').read_text())
            train = set(record['initial_indices'])
            for cycle in range(11):
                if cycle:
                    train.update(record['acquired'][cycle-1]['indices'])
                result = np.load(OUT/row.output/f'cycle{cycle:02d}.npz', allow_pickle=False)
                queries, background = result['query_indices'], result['background_indices']
                assert len(set(queries)) == len(queries) and len(set(background)) == len(background)
                assert set(background).issubset(train)
                assert not set(queries)&(train|set(record['test_indices']))
                assert len(background) == min(50, len(train))
                assert len(queries) == min(100, len(df)-len(train)-len(record['test_indices']))
                assert result['shap'].shape == (len(queries), 4096)
                assert np.isfinite(result['shap']).all()
                np.testing.assert_allclose(result['expected_values']+result['shap'].sum(1), result['predictions'], atol=1e-7)
                np.testing.assert_allclose(abs(result['shap']).mean(0), result['mean_abs_shap'], atol=1e-12)
                total += len(queries)
            # Independently compare coefficient formula to arbitrary masked input
            # predictions through the already-validated saved GP implementation.
            saved = SavedPredictor(record, df, x)
            gp = restore(record, df, Geometry(x, saved.p, np.array(record['initial_indices'])), 10)
            query = np.array(record['test_indices'][:1])
            w = weights(gp, x, np.array(record['initial_indices']), query)[0]
            hybrid = x[query].copy()
            hybrid[:, ::7] = x[saved.train[0], ::7]
            got = saved.predict(hybrid, saved.p[query])[0]
            expected = gp.center+gp.scale*gp.mean+hybrid[0]@w
            np.testing.assert_allclose(got, expected, atol=1e-7)
            # Representative must actually be the highest-affinity bit carrier.
            for fragment in json.loads((OUT/row.output/'fragment_mapping.json').read_text()):
                members = np.flatnonzero(x[:, fragment['bit']])
                assert fragment['representative_row'] in members
                assert fragment['representative_pK'] == float(df.iloc[members].affinity.max())
    assert total == int(summary.explanations.sum())
    print(f'PASS: 36 campaigns, 396 cycles, {total} explanations; masks/background/split checks, full-bit additivity, GP coefficient parity and fragment affinity priority.')


if __name__=='__main__':
    main()
