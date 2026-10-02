# Full representation x kernel x acquisition comparison

The new grid is separate from the initial fixed-kernel baseline in
`results/published/`. Its output directory is `results/full_grid/`.

## Coverage

- Representations: ECFP radius 4 / 4096 bits, MACCS, ChemBERTa-77M-MTR CLS embeddings.
- Lipid kernels: Tanimoto, linear, RBF, rational quadratic, Matern-3/2.
- Protocols: all seven paper schedules plus Thompson sampling.
- Datasets: BioDolphin Kd, BioDolphin Ki, two TRAAK isoforms, two transthyretin constructs.
- Seeds: 7, 19, 42, matched across all configurations.

Total: **6 x 3 x 5 x 8 x 3 = 2,160 runs**, including 1,890 paper-protocol runs
and 270 Thompson runs. There are 720 distinct configurations averaged over seeds.
The representation encodings follow the authors' published featurizer; their
radius-4 ECFP implementation corresponds to ECFP8 terminology, despite ECFP4
terminology elsewhere in the paper. RDKit MACCS has 167 bits, with bit 0 unused.
ChemBERTa is pinned to revision `66b895cab8adebea0cb59a8effa66b2020f204ca`;
its 384-dimensional CLS embeddings are computed on the GPU. No lipid SMILES
required truncation in this corpus (see representation_manifest.json).

## Acquisition schedules

E = uncertainty, X = predicted affinity, B = 0.5 mean + 0.5 predictive SD,
R = uniform random. These strings describe ten sequential acquisition phases.

| Protocol | Schedule |
|---|---|
| Random | RRRRRRRRRR |
| UCB-balanced | BBBBBBBBBB |
| UCB-alternate | EXEXEXEXEX |
| UCB-sandwich | EEXXXXXXEE |
| UCB-explore-heavy | EEEEEEEXXX |
| UCB-exploit-heavy | XXXXXXXEEE |
| UCB-gradual | EEEBBBBXXX |
| Thompson | Ten joint latent-posterior draws, one per batch |

UCB uses predictive observation uncertainty to follow the canonical authors'
implementation. Thompson uses latent function covariance (not observation noise).
This grid does not include EI, PI, or diversity reranking; those are additional
methods, not part of the seven-protocol table in the paper.

## Fit and evaluation

The exact Gaussian process learns signal and observation-noise variances,
constant mean, applicable lipid length scale, RQ shape, and protein length scale
by maximizing marginal likelihood. Optimization uses analytic-gradient
L-BFGS-B, at most 60 iterations per fit, warm-starting from the previous round.
Every fit records termination status, iterations, parameter values and initial/
final negative log likelihood. Input scaling is fitted to initial labelled
inputs and frozen. Targets are standardized using the acquired labels at each
round. Numeric verification compares analytic and finite-difference gradients,
covariance positive semidefiniteness, posterior marginal/joint agreement, and
likelihood improvement, on real BioDolphin observations.

For pooled protein-lipid data the lipid kernel is multiplied by a Matern-3/2
protein-composition kernel with a learned length scale. On a single protein
sequence this factor equals one. This is a domain adaptation, not an exact
reproduction of the original single-target ligand-only model. The optimizer
also differs from the authors' Adam training routine. SHAP is performed separately
on saved posteriors, without changing these experiments; see
[EXPLAINABILITY.md](EXPLAINABILITY.md).

BioDolphin uses the paper's budget of 60 initial labels and 10 batches of 30
(360 total) with a disjoint 20% test set. This differs from the earlier
64-label baseline; comparisons across those two outputs are not budget-matched.

TRAAK and transthyretin contain only 16 and 11 observations per panel. Their
budget is explicitly adapted to two initial labels and six additional labels,
distributed across ten phase slots as [0,1,0,1,1,0,1,0,1,1]. Zero-budget slots
perform no acquisition or refit. Thus the schedule names retain their phase
positions but these tiny-panel runs do not replicate the paper's experimental
budget, and some phases have no queries. Interpret them only as exploratory.

Recall covers the top 2%, 5% and 10% of the acquisition pool and includes
initial labels. Ties at the threshold are included; the target count uses a
ceiling. Recall AUC integrates over the number of labels and divides by the
acquired-label range. Test MAE/RMSE/R2 and 90% interval coverage use only the
frozen held-out rows. Test labels never guide acquisition or GP fitting.

The split is random by deduplicated pair, not cold-protein/family/scaffold.
Homology, chemical similarity and heterogeneous assay provenance remain
limitations. Tiny panels have only 2-3 test rows. Reported best configurations
are descriptive comparisons over the same held-out data, not independently
validated winners; selecting a deployment model requires validation/nested
evaluation. More seeds are needed to establish robust differences.

## Run, resume, inspect

```powershell
python prepare_grid_features.py
python test_grid_real.py
python run_full_grid.py
python summarize_full_grid.py
```

Use `prepare_grid_features.py --offline` after downloading the model. Re-running
the grid resumes only matching, completed checkpoints. Dataset, feature, engine
and runner hashes are recorded; changed configurations are refused in the same
output directory. No synthetic fallback is used.

Outputs in `results/full_grid/`:

- `progress.json`: live count, expected total, most recent configuration.
- `summary.csv`: three-seed means/SD for all 720 configurations.
- `final_by_seed.csv`, `all_learning_curves.csv`: all numerical results.
- `paired_vs_random_summary.csv`: within-seed, same-kernel/representation gains.
- `top5_by_test_mae.csv`, `top5_by_recall_auc.csv`: descriptive rankings.
- `plots/*_grid.png`: full grid heatmaps for every dataset.
- `validation.json`: completeness, split separation, schedules and budget checks.
- `optimizer_nonconvergence.json`: any fits reaching the iteration limit or
  otherwise terminating without the optimizer convergence criterion.
- `runs/.../seed*.json`: test/initial/acquired row indices, fitted hyperparameters,
  predictions, uncertainty, learning curves and execution time.

## Completed results

All 2,160 runs completed; the audit found no missing or extra combinations.
Split separation, matched initialization, schedules and budgets passed.
Of 18,000 fitted models, 17,994 met the optimizer convergence criterion;
six reached the 60-iteration limit and are explicitly logged. Their final
likelihood objectives improved, but they should not be called converged.

Descriptive lowest held-out MAE configurations (means over three seeds):

| Dataset | Representation / kernel / protocol | MAE (pK) | R2 |
|---|---|---:|---:|
| BioDolphin Kd | ECFP / linear / random | 0.643 | 0.667 |
| BioDolphin Ki | MACCS / Matern-3/2 / explore-heavy | 0.455 | 0.847 |

For Thompson sampling, the lowest-MAE configurations were ChemBERTa /
Matern-3/2 on Kd (MAE 0.705, R2 0.624, final top-10% recall 96.7%) and
MACCS / RBF on Ki (MAE 0.490, R2 0.836, recall 100%). The latter includes
one iteration-limited intermediate fit. Discovery and prediction rankings
differ: highest recall-AUC configurations are recorded separately in
`results/full_grid/descriptive_best.csv`. These comparisons use 360 labels
on BioDolphin, not the earlier 64-label budget. The tiny per-target panels
still have negative mean R2 even for their lowest-MAE configurations.

## Source links

- [Paper, Methods and Table 1](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00436e).
- [Authors' featurizer](https://github.com/meyresearch/explainable_AL/blob/main/explainable_al/featuriser.py).
- [Authors' canonical GP/acquisition code](https://github.com/meyresearch/explainable_AL/blob/main/explainable_al/active_learning.py).
