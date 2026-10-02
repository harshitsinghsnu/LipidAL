# SHAP and acquisition explainability

## Paper-style temporal analysis (new)

Open [the temporal ECFP report](results/paper_explainability/index.html).
This follows the analysis structure in sections 2.6 and 3.5 of the reference
paper, rather than the final-model grouped SHAP analysis described below.

The analysis uses 36 saved **ECFP / linear** campaigns: explore-heavy and
exploit-heavy, six panels, three seeds, with the initial model and all ten AL
cycles. The kernel is fixed to make a matched protocol comparison, not chosen
separately from each protocol's test results. This is not a temporal analysis
of every kernel or a complete reproduction of the paper's original experiments.

| Aspect | Implementation |
|---|---|
| Explanation population | Random sample of up to 100 unqueried pool pairs at each cycle; frozen test rows excluded |
| Background | Up to 50 acquired-training lipids at that cycle |
| Features | All 4096 ECFP bits individually; no top-variance prefilter or remainder group |
| Importance | Mean absolute SHAP; top ten retained separately at every cycle |
| Evolution | Trace final top-ten bits through all cycles; per-cycle lists also saved |
| Fragment mapping | Highest measured-affinity bit-containing pair; RDKit atom environments, canonical fragment SMILES and highlighted structures |
| Association | Present-versus-absent measured-affinity distributions, exported row-wise |
| Stability | Top-ten Jaccard and active-bit Spearman across consecutive cycles and matched protocols |

### Exact computation and deviations

At a fixed protein, a GP with a linear lipid kernel has an affine posterior
mean in the lipid fingerprint, even with the multiplicative protein kernel:
`prediction = intercept + sum_j weight_j(protein) * lipid_bit_j`.
Thus its exact marginal-background per-bit Shapley values are
`weight_j(protein) * (lipid_bit_j - mean_background_bit_j)`.
We compute this exact solution for all explained rows, instead of sampling
Kernel SHAP independently for every prediction. `shap.KernelExplainer` checks
one final-cycle query independently in each of the 36 campaigns, with no L1
feature selection (`l1_reg=0`, `nsamples=2*M+128` for M varying dimensions).
For that check the empirical background is compressed to its mean: for an
affine predictor this preserves the expected prediction for **every coalition**,
and therefore preserves the complete Shapley game exactly. It does not group
features or approximate the original 50-background game.

The paper specifies KernelExplainer but does not specify its coalition budget
or regularization in section 2.6. Our estimator settings and exact linear
shortcut are explicit adaptations, not a claim of byte-for-byte code equivalence.
The shortcut is valid here only for the linear lipid kernel. It must not be
applied to Tanimoto, RBF, rational-quadratic or Matern lipid models.

On pooled panels each query's protein is held fixed while lipids are replaced.
This isolates lipid contributions conditional on that protein; it is **not**
a decomposition into protein and lipid contributions. Each protein can have a
different baseline and bit weights. On single-target panels it becomes the
ordinary ligand-only game. As in the earlier grid, our protein factor and GP
optimizer differ from the original single-target ligand study.

Small panels have fewer than 100 unqueried/50 background observations and use
all available. Their original two-initial/eight-final-label budgets remain
unchanged. Zero-query phases reuse the same fitted posterior; sampled background
or pool composition can still affect the explanation. Consequently, temporal
changes cannot automatically be attributed to learning alone.

The fragment ranking uses `pair prevalence * mean absolute bit SHAP`; the paper
describes a frequency/importance combination without specifying its algebra.
All environments in the representative molecule are retained to show collisions;
one deterministic nonzero-radius environment is used for its highlight when
available. Unique-lipid counts are also saved. Absent bits can have nonzero
SHAP, and a single depicted fragment does not exhaust a hashed bit's meaning.

Affinity-prioritized representatives and affinity distributions use all measured
rows **retrospectively**, including held-out rows. Those labels never guide
training, acquisition, SHAP, or top-bit selection. This is descriptive chemical
interpretation, not independent validation. On pooled BioDolphin panels,
cross-protein confounding prevents interpreting affinity distributions as
target-specific SAR. No 3-D binding pose, protein-contact panel, or residue
interaction is produced: those would require validated structural evidence.

Run from this project folder:

```powershell
python paper_explainability.py
python test_paper_explainability.py
```

Outputs are isolated in `results/paper_explainability/`. Each cycle's compressed
NPZ stores all per-pair/per-bit SHAP values, baselines, predictions and sampled
indices. CSVs, SVG molecular highlights and the HTML report provide the paper-
style views. Earlier explanations and benchmark results are preserved.

Completed and validated: **36 campaigns, 396 cycle snapshots, 14,754 pair
explanations**. Maximum additive reconstruction error was 9.86e-14 pK;
maximum disagreement with the 36 independent KernelExplainer checks was
9.23e-11 pK. Real-data tests also passed for the full-bit arrays, acquired-only
backgrounds, unqueried-only explanation rows, saved-posterior reconstruction,
masked-input coefficient formula and highest-affinity fragment representatives.
Final top-ten overlap between the two protocols averaged Jaccard 0.545 on
both pooled BioDolphin endpoints across three seeds; this is moderate overlap,
not evidence that fragment rankings are protocol-independent.

References: [paper, sections 2.6 and 3.5](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00436e),
[authors' fragment viewer, pinned revision](https://github.com/meyresearch/explainable_AL/blob/c77e3a6a77e0207f995815c22f3146ae9b046777/apps/shap_app/shapey.py).

## Earlier final-model grouped analysis

Open [the HTML report](results/explainability/index.html) or
[the model summary](results/explainability/summary.csv).

## Completed scope

The default run explains **54 final fitted models**: three representative
configurations per dataset, six dataset panels, and seeds 7, 19 and 42.
Configurations are the lowest mean test-MAE model, the highest mean top-10%
recall-AUC model, and the lowest mean test-MAE Thompson model. They are selected
descriptively from the completed grid, not independently validated winners.
This selection does not change acquisition, training or the existing results.

The default does not explain every grid checkpoint. `--all` enables all 2,160.
At most eight randomly selected held-out pairs are explained per model, with
eight randomly selected acquired-training pairs as the empirical background
(all available if fewer). No test rows are used as background. Selections are
seeded, and row indices, feature groups and checkpoint hashes are recorded.

## What is explained

1. **Final predicted affinity**, in pKd/pKi units. Positive contributions mean
   higher predicted pK, hence stronger predicted affinity. They are not changes
   in raw Kd/Ki. Baseline is the mean posterior prediction on background pairs.
2. **Protein versus lipid**, using exact two-player Shapley values. For each
   held-out pair, evaluate background/background, target-lipid/background-protein,
   background-lipid/target-protein, and target/target. Average predictions over
   the background, then average each group's marginal effect in both orders.
3. **Detailed grouped features**, using `shap.PermutationExplainer`. ECFP/MACCS
   use the 20 highest-variance bits in the final acquired training inputs as
   individual players; all remaining bits form another player. Thus no omitted
   bits silently disappear. ChemBERTa uses 12 contiguous blocks of 32 embedding
   dimensions. All 26 protein descriptors form one joint player, preserving
   their within-protein compositional dependencies during replacement.
4. **Acquisition decisions**, reconstructed independently for every round of
   these 54 campaigns. Candidate CSVs contain predicted mean, latent and
   observation SD, acquisition score/rank and selected status. Random acquisition
   has no model-based score. Thompson scores are replayed correlated posterior
   draws, not the posterior mean or a fictitious deterministic UCB score.

The two-player and detailed games use different groupings: their contributions
must not be added together or assumed interchangeable. Detailed lipid-group
attributions need not sum to the separately calculated two-player lipid value.

Each detailed explanation averages two independent SHAP runs, each with eight
forward/reverse permutation pairs. `repeat_abs_difference_pK` reports their
absolute difference as a Monte Carlo sensitivity diagnostic, not a confidence
interval. Increasing permutations reduces Monte Carlo error but does not fix
background choice or model uncertainty. Additivity alone does not establish
that individual approximate contributions have converged.

## Chemistry and biological interpretation

`fingerprint_annotations.json` maps the individually explained ECFP bits to
actual atom-centered environments (center, radius, atoms and fragment SMILES)
on explained and background molecules. Hash collisions and multiple occurrences
are retained; a bit is not a unique functional group. MACCS entries include
RDKit SMARTS, count thresholds and atom matches; special-case keys are flagged.
These are annotations of bit occurrences, **not atom-level SHAP values**.

ChemBERTa block importance is importance in embedding space, not an explanation
of a named chemical group or SMILES token. Protein contributions reflect the
model's sequence-composition representation, not residues, binding sites,
membrane orientation, contacts, or a structural binding mechanism. Protein
SHAP is zero on a constant-protein panel by construction.

Replacement may generate chemically invalid/off-manifold fingerprint or
embedding combinations and unfamiliar protein-lipid combinations. These are
computational SHAP perturbations, not dummy training datasets or invented
affinity measurements. The GP is never trained on them. This is a marginal
background-replacement game, not conditional SHAP constrained to valid lipids.
Correlated features, background sampling, grouping and the limited explained
subset affect attribution. The `Remaining lipid bits` group may dominate and
must not be mistaken for one identifiable chemical motif.

SHAP explains the model, not biological causality or predictive correctness.
The existing tiny-panel, heterogeneous-assay, random-pair split and multiple-
comparison limitations remain. A persuasive binding mechanism would need
additional structural and experimental evidence.

## Validation

The final posterior is rebuilt from the original acquired labels, initial-only
input scaling, saved hyperparameters and exact GP equations, **without any
optimization or refitting**. All held-out means and observation SDs must match
the original checkpoint within 1e-6. Source, feature and dataset hashes are
checked before execution; original grid files are untouched.

For the completed 54-model run, maximum restored prediction error was below
3e-13 pK and maximum SHAP additivity error below 2e-13 pK. Tests on real pooled
and single-target observations cover all 15 representation/kernel combinations,
check optimized coalition predictions against explicit masked-feature
predictions, and verify zero protein effects on constant-protein panels.
Every nonempty acquisition round must reproduce its recorded selected indices
in the original order; a mismatch stops the report build.

## Reproduce and extend

Run from `D:\design\proteinlipid`:

```powershell
python test_explainability.py
python explain_grid.py
python explain_acquisitions.py
```

The standard run resumes matching completed explanation checkpoints. Changing
settings or code requires a new output directory to preserve previous analyses.
To explain every grid model, using the same default approximation settings:

```powershell
python explain_grid.py --all --output results/explainability_all
python explain_acquisitions.py --explanations results/explainability_all
```

For more extensive explanations of the representative models:

```powershell
python explain_grid.py --samples 32 --background 32 --permutations 32 --top-bits 50 --output results/explainability_extended
python explain_acquisitions.py --explanations results/explainability_extended
```

Each model directory contains local signed SHAP values, global mean absolute
importance over its explained subset, two plots, annotated held-out pairs,
fingerprint mappings, provenance, all-candidate acquisition traces and decision
reasons. The report links these artifacts. Original experimental summary CSVs
remain separate and unchanged.

Method reference: [official SHAP PermutationExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.PermutationExplainer.html).
