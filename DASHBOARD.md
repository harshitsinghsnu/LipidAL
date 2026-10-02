# LipidLab local dashboard

Start from `D:\design\proteinlipid`:

```powershell
python dashboard.py
```

Open **http://127.0.0.1:8765** in your browser. Keep the terminal running.
Use `--port 8767` if the default port is occupied. Stop with Ctrl+C.
This is a localhost research application, not a production multi-user service.
It does not submit data to a remote service or start a public server.

## Explore saved results

Both `results/paper_explainability/index.html` and
`results/explainability/index.html` now show the responsive dashboard.
Analysis, dataset, representation, kernel, policy and seed dropdowns are linked:
only available combinations are offered. Plot tabs show temporal importance,
affinity distributions, molecular highlights, or global/local grouped SHAP.
The default **Full benchmark** view reads every completed run in `full_grid`
and `expanded_grid`, not just selected explanations. Its four-panel plot shows
MAE, discovery recall, RMSE and interval coverage. A matched-representation
table compares mean and sample SD across seeds for the selected kernel/policy.
The separate explanation views retain 36 temporal and 54 final-model reports;
new benchmark combinations are explicitly marked when SHAP is not computed.
Numerical audit JSON files remain downloadable. Original pages are preserved
as `index_legacy.html`. Rebuild static pages after finishing additional runs;
the local server reads current checkpoints on page refresh.

Opening an HTML file directly supports browsing. **Running AL requires the
Python server**, because static HTML cannot execute the GP model. File mode
disables Run and provides the server-start instructions; it does not pretend
to run an experiment. No external JavaScript/CDN dependencies are needed.

## Upload and run

The CSV must contain `protein_sequence`, `lipid_smiles`, and `affinity`.
No fabricated sample data are provided. A real small test dataset is already
available at `data/processed/traak_a_Kd1.csv`; use pK units, initial=2,
batch=1 and cycles=2 to verify the workflow. Its tiny test set is not a reliable
measure of predictive quality.

- Every pair needs a numeric measured affinity. This is retrospective
  pool-based AL, **not an oracle for unmeasured candidates**.
- Select pKd/pKi directly, or raw Kd/Ki in nM or micromolar units. Raw values
  must be positive and are converted to pK; larger pK is better.
- Use one endpoint and standard amino-acid letters only. Whitespace is removed
  from sequences and case is normalized. SMILES are validated/canonicalized;
  duplicate sequence/canonical-SMILES pairs are rejected, not aggregated.
- Limits: 5 MB CSV, 10–1000 rows, 1–20 cycles, at most 500 final acquired labels.
  A 20% random-pair test set is frozen. The requested acquisition budget must
  fit in the remaining pool; no silent clipping or dummy fallback occurs.
- Uploads support all **five representations**, **five lipid kernels**, and
  **13 acquisition policies** listed below. Scheduled policies require ten
  cycles; static policies accept 1–20. Frozen transformer inference runs locally
  from downloaded weights. Missing weights cause an explicit error, not a fallback.
- New campaigns use the unchanged `grid_gp.py` optimized exact GP and the
  same product-protein kernel, initial-input scaling, acquired-only target
  normalization and acquisition definitions as the full grid.

Progress displays cycle-wise MAE, R² and top-decile pool recall. Cancel takes
effect between GP operations, not mid-factorization. One campaign runs at a
time. Closing the browser does not stop it. A session can resume polling after
a reload while the same server process remains alive. Server restart loses
the in-memory job status, but completed files remain on disk.

New outputs use unique IDs under `results/user_runs/<id>/`. The normalized input
CSV is retained locally (including optional metadata); do not upload sensitive
data without considering local storage. Download ZIPs contain learning curves,
acquired row indices, test predictions and the full result/fit diagnostics, but
not the original input CSV. Existing results and manuscript figures are not
overwritten. Upload runs do **not** automatically generate SHAP analysis.

## Rebuild and test

```powershell
python dashboard.py --build-only
python test_dashboard.py
python test_extended_methods.py
python test_dashboard_browser.py
```

The browser test additionally needs Playwright (installed locally in
`ui_test_tools`) and the Chrome executable configured in that test. It uses
real TRAAK measurements and a temporary output folder. It checks linked
dropdowns, plot loading, invalid-budget feedback, a completed AL campaign,
ZIP downloading, mobile overflow and static-file guidance. Screenshots are
in `web/screenshots/`.

If you regenerate the original explanation reports with their original scripts,
run `python dashboard.py --build-only` afterward to restore the dashboard pages.
Those experiment scripts remain unchanged to preserve their provenance hashes.

The server enforces localhost hostnames, same-origin/token-protected POSTs,
upload limits and restricted result-file paths. These safeguards do not make
it an authenticated public hosting service. Do not expose it through a public
tunnel or change the host binding for multi-user deployment.

## Supported experiment suite

“All” means the Cartesian product of this explicit supported suite, not every
acquisition function or kernel in the literature. The original GP engine and
original benchmark artifacts are unchanged.

| Lipid representation | Features | Frozen encoder / pooling |
|---|---:|---|
| ECFP | 4096 | Morgan radius 4, no chirality |
| MACCS | 167 | RDKit structural keys |
| ChemBERTa-MTR | 384 | DeepChem/ChemBERTa-77M-MTR, CLS |
| ChemBERTa-MLM | 384 | DeepChem/ChemBERTa-77M-MLM, CLS |
| MoLFormer-XL | 768 | ibm-research/MoLFormer-XL-both-10pct, masked mean |

Each is crossed with Tanimoto (dot-product form for dense embeddings), linear,
RBF, rational quadratic and Matérn 3/2. Protein features remain 26 sequence
composition descriptors with a Matérn 3/2 product factor: these are molecular
transformer experiments, **not protein-transformer or structure-aware models**.

The 13 policies are random, six paper UCB schedules (balanced, alternate,
sandwich, explore-heavy, exploit-heavy, gradual), joint-posterior Thompson,
greedy predicted mean, predictive observation uncertainty, expected improvement
(EI), probability of improvement (PI), and Thompson + diversity.

EI and PI maximize improvement over the best **acquired** pK with fixed
threshold ξ=0.01 pK and latent posterior SD. Batches take the top marginal
scores; these are not joint qEI/qPI or noisy EI. Thompson + diversity takes one
joint posterior draw, standardizes its candidate scores and greedily adds
0.25 times `1-exp(-nearest_selected_lipid_distance²/2)` within a batch. This is
a transparent heuristic, not a claimed new or state-of-the-art algorithm.
Constants were fixed before the expanded benchmark, not tuned on test scores.

MoLFormer preprocessing removes stereochemistry to match its pretraining.
ChemBERTa CLS encodings retain the provided SMILES tokenization. No affinity
labels are used in encoder training or feature extraction. Model revisions and
reviewed custom-code checksums are pinned in `extended_features.py`. All model
state keys load strictly; missing pretrained tensors are an error. Published
lipids require at most 156 ChemBERTa tokens and 154 MoLFormer tokens, without
truncation. Uploaded MoLFormer strings above its 202-token training regime are
encoded in full up to a 1024-token safety limit and counted in result metadata;
performance there is out-of-distribution. ChemBERTa rejects >512 tokens.

Model sources: [ChemBERTa-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM),
[ChemBERTa-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR),
[MoLFormer](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct).
These are established pretrained encoders, not a claim to cover all 2026 models.

## Reproduce the expanded benchmark

```powershell
python extended_features.py --download
python run_expanded_grid.py
python audit_expanded_grid.py
python dashboard.py --build-only
```

After the initial download, `python extended_features.py` works offline.
Do not regenerate features while a benchmark is running. Hash-checked resumable
checkpoints prevent mixing different feature caches or code versions.
The matched design contains **5,850 runs**: 5 representations × 5 kernels ×
13 policies × 6 panels × 3 seeds. It reuses the 2,160 original verified runs and
adds 3,690 runs under `results/expanded_grid/runs`. `progress.json` is the live
completion counter; `run_index.json` is the completed cross-directory audit
index. `summary.csv`, `final_by_seed.csv`, and `all_learning_curves.csv` in that
directory combine original and extension runs without duplicating them.

The six panels include overlapping pooled/per-target BioDolphin subsets and
two TRAAK series; they are not six independent publications. Budgets, splits,
seeds, GP fitting and protein features match the original experiment. Rankings
are descriptive multiple comparisons, not nested-validation estimates.
The updated manuscript (`paper/main_expanded.pdf`) includes the expanded
experiment and actual dashboard screenshots. SHAP artifacts remain the original
selected cohort. See `VERCEL.md` for the tested read-only hosting export;
no public deployment is claimed.

### Completed run and validation

All 5,850 runs (1,950 three-seed configurations) completed. The audit in
`results/expanded_grid/audit.json` verifies the exact Cartesian product,
hashes, splits, budgets, finite outputs and absence of test/acquisition overlap.
There were 26 optimizer non-convergences among 48,750 fits; these are retained
and flagged, not silently dropped. Eight integration/unit tests, the original
15 numerical kernel checks, and the browser real-data upload checks passed.

For a fixed Matérn-3/2 + Thompson comparison, BioDolphin Kd MAE is 0.7055 for
ChemBERTa-MTR, 0.7107 for ChemBERTa-MLM, and 0.7532 for MoLFormer. On Ki the
corresponding values are 0.5421, 0.5404 and 1.3281. These are three-seed means,
not significance claims; pretrained transformers do not uniformly improve
protein–lipid prediction. The complete matched table is
`results/expanded_grid/representation_thompson_matern32.csv`.
