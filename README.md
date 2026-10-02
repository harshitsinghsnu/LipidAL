# Protein-lipid affinity active learning on published measurements

## Current release

**Live dashboard: [lipidal.vercel.app](https://lipidal.vercel.app)**.
Open the [new-study explorer](https://lipidal.vercel.app/results/robust_study_v2/index.html)
or [download the current paper](https://lipidal.vercel.app/paper/main_updated.pdf).

Repository: [harshitsinghsnu/LipidAL](https://github.com/harshitsinghsnu/LipidAL).
The current full manuscript is [main_updated.pdf](paper/main_updated.pdf),
with [LaTeX source](paper/main_updated.tex), an
[updated four-panel supplement](paper/updated_composite_supplement.pdf), and
the [complete historical figure atlas](paper/all_figures_atlas.pdf).
The older `main_expanded.pdf` and `main.tex` describe the earlier study only.

The public dashboard is a **read-only** export. It includes the original
5,850-run benchmark, 90 explanation reports, all 9,720 new main campaigns,
and 1,944 adaptation comparisons (648 reused baselines). It does not receive
uploads or run cloud AL jobs. See [deployment instructions](VERCEL.md).
For local measured-data campaigns, install `requirements.txt` and run
`python dashboard.py`. The clean repository provides a derived browsing
catalog but excludes raw datasets, pretrained weights and GP checkpoints.
Research reruns require downloading/preparing these assets; no synthetic
fallback data are supplied. Source data retain their respective licenses.

## New leakage-controlled study

The [new study report](results/robust_study_v2/index.html) adds cold-sequence and
scaffold evaluation, frozen ESM-2 representations, prediction-aware Thompson,
lipid-domain SSL and initial-label adapters, plus nonlinear/collision/contact
explanation checks. **11,016 new unique AL campaigns are complete.** Read
[methods, results and limitations](ROBUST_STUDY.md) before comparing these
120-label experiments with the older benchmark below. Improvements are mixed;
statistical superiority and prospective validity are not established. The
current manuscript and dashboard export include this study; older manuscript
snapshots are retained only for provenance.

## Interactive dashboard

Run `python dashboard.py` from this folder and open **http://127.0.0.1:8765**.
Browse saved experiments with dropdowns, or upload measured CSV pairs to run
new local AL cycles. See [DASHBOARD.md](DASHBOARD.md) for input requirements,
limits and validation. Directly opened HTML files support browsing only.

The portal's **Full benchmark** view exposes every completed configuration,
separately from the selected SHAP reports. The expanded supported suite is
**ECFP, MACCS, ChemBERTa-MTR, ChemBERTa-MLM and MoLFormer-XL**, crossed with
all five paper lipid kernels and 13 acquisition policies. All are also
available for local CSV campaigns. See [DASHBOARD.md](DASHBOARD.md#supported-experiment-suite)
for exact definitions and limitations. Expanded results/checkpoints are in
`results/expanded_grid/`; all **5,850 matched runs are complete**, including
2,160 reused original runs. The [audit](results/expanded_grid/audit.json)
verifies complete coverage and no test/acquisition overlap. The updated
[15-page manuscript](paper/main_expanded.pdf) covers the expanded grid and
includes dashboard screenshots. SHAP analyses remain the original selected cohort.
The [Vercel-ready read-only export](VERCEL.md) is prepared but not publicly deployed.

An anonymous ACML-format manuscript draft is in [paper/main.tex](paper/main.tex),
with four-panel manuscript figures and a [complete figure atlas](paper/all_figures_atlas.pdf).
See [paper/README.md](paper/README.md) for build and submission limitations.

The complete paper representation/kernel/protocol grid, plus Thompson sampling,
is documented in [FULL_GRID.md](FULL_GRID.md). Its outputs are in
`results/full_grid/`; the results below describe the earlier fixed-kernel baseline.

SHAP and acquisition-decision explanations are now available in
[the interactive report](results/explainability/index.html). See
[EXPLAINABILITY.md](EXPLAINABILITY.md) for methods, limitations and rerun commands.
The completed explanation run covers 54 selected final models across all six
panels and all three seeds, not every one of the 2,160 grid checkpoints.

The newer [paper-style temporal report](results/paper_explainability/index.html)
tracks individual ECFP bits across learning cycles for 36 matched linear-kernel,
explore-heavy/exploit-heavy campaigns. It adds affinity distributions,
highlighted fragments and cross-protocol stability; adaptations are documented
in [EXPLAINABILITY.md](EXPLAINABILITY.md).

Real data have been downloaded, curated, and evaluated. No synthetic or imputed
affinities are used. There are **two independent publications**, yielding these
endpoint/target subsets (not four independent source studies):

| Dataset | Unique pairs | Endpoint | File under data/processed/ |
|---|---:|---|---|
| BioDolphin v1.1 | 864 | pKd | biodolphin_Kd.csv |
| BioDolphin v1.1 | 549 | pKi | biodolphin_Ki.csv |
| TRAAK K2P4.1a | 16 | pKd1 | traak_a_Kd1.csv |
| TRAAK K2P4.1b | 16 | pKd1 | traak_b_Kd1.csv |

## Curation

BioDolphin's full official release has 127,359 structural interaction entries.
1,292 have a positive numeric Kd annotation and 958 have a positive numeric Ki
annotation. These are not independent measured pairs. Structure/sequence
validation, conservative exclusion of censored source annotations, and
deduplication produce the counts above. Kd and Ki stay separate and may overlap.
The published BioDolphin release imports some annotations from BindingDB,
PDBbind and other sources; it is not our previous BindingDB/LIPID MAPS proxy.
Respect upstream terms when redistributing downloaded annotations.

TRAAK Source Data Figures 2 and 3 contain triplicates for 16 lipids each.
We use the mean **Kd1**, the first sequential binding event, and convert
micromolar to pKd. Replicates and higher binding events are not independent
training examples. The supplementary PDF contains additional measurements that
have not been parsed into the benchmark. Source Data Figures 4 and Extended
Data 7 contain functional/gel measurements and are not affinity training labels.

PubChem structures are resolved from the supplement's chemical names. Two
ambiguous searches were resolved by the specified sn-glycerol/double-bond
stereochemistry, with explicit CIDs in prepare_traak.py. Protein sequences use
the published constructs: Q9NYG8-2 residues 1-290 with N104Q/N108Q; isoform a
removes its first 26 residues. Fusion tags are omitted from sequence features.
The isoforms are evaluated separately.

All exact-sequence BioDolphin target datasets are in data/processed/per_target/.
per_target_manifest.csv gives counts and paths. Two transthyretin constructs
have 11 pairs each and are also benchmarked. Most targets have insufficient
distinct lipids for meaningful target-specific learning curves.

## Reproduce

Run from this folder:

```powershell
python -m pip install -r requirements.txt
python download_published.py
python prepare_published.py
python prepare_traak.py
python run_published_benchmarks.py
```

Downloads have URLs, byte counts and SHA-256 checksums in
data/raw/download_manifest.json. PubChem responses and the UniProt sequence
are cached. BioDolphin source rows retain source IDs, PDB IDs, PubMed IDs and
original affinity annotations. curation_audit.json records filtering counts.

## Experiments and results

Open results/published/summary.csv and learning_curves.png.
Every individual run contains held-out rows, initial labels, acquired
measurements, learning curves and grouped prediction sensitivities.
protocol.json records budgets and splits.

The GP multiplies a protein Matern-5/2 kernel by a lipid Tanimoto kernel on
chiral ECFP4 fingerprints. Protein inputs are amino-acid composition and simple
physicochemical descriptors. Hyperparameters are fixed baseline settings,
not optimized/calibrated. Comparisons include random, greedy, uncertainty, EI,
joint-posterior Thompson sampling, and Thompson sampling with diversity.
UCB, PI and other kernels are available in the module but not in this benchmark.
BoTorch acquisitions, learned kernels, ESM embeddings and a complete reproduction
of the reference paper are **not implemented** in this baseline.

Three seeds (7,19,42) each hold out 20% of unique pairs, with identical initial
labels across methods. BioDolphin uses 24 initial labels plus five batches of
eight (64 total). Small panels use four initial labels plus four single queries
(eight total). Recall measures the top 10% of the acquisition pool, including
initial labels; MAE/R2 use the frozen test set. Test labels never guide queries.

Thompson sampling improves pooled BioDolphin hit recovery over random but has
higher held-out MAE. Greedy/EI recover more hits while sacrificing global
prediction accuracy. See the CSV for means and sample SD across seeds.

These are exploratory random-pair benchmarks, not tests of generalization to
unseen protein families/scaffolds. Homologous proteins and related lipids can
cross the split. Small target panels have only 2-3 held-out pairs per seed, so
R2 is especially unstable. BioDolphin's reported averages may mix assay
conditions; the consensus median is not a new experimental measurement.
BioDolphin uses a broad lipid definition (including polyketides), not exclusively
membrane phospholipids. Grouped permutation outputs are prediction sensitivities,
not causal explanations or SHAP values. The separate `results/explainability/`
outputs contain actual grouped SHAP for the optimized full-grid models.

## Sources

- [BioDolphin paper](https://doi.org/10.1038/s42004-024-01384-z) and
  [official release](https://biodolphin.chemistry.gatech.edu/BioDolphin_vr1.1.zip).
- [TRAAK study, Schrecke et al.](https://doi.org/10.1038/s41589-020-00659-5).
- [Reference active-learning paper](https://doi.org/10.1039/D5DD00436E).
