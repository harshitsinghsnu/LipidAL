# Leakage-controlled and prediction-aware study

Completed study: [four-panel report](results/robust_study_v2/index.html). The previous 5,850-run benchmark is unchanged. This study adds **11,016 unique AL campaigns**: 9,720 main campaigns plus 1,296 adaptation campaigns. The adaptation comparison table contains 1,944 entries because it also reuses 648 main-study baselines.

## Evaluation and representations

The main grid crosses two pooled BioDolphin endpoints (864 Kd and 549 Ki pairs), three split regimes, six seeds, three protein representations, five lipid representations, three kernels, and six policies. Tiny target panels are not used to support these comparisons.

- Splits: random pair; cold sequence/accession groups; Bemis–Murcko scaffold groups. All acyclic molecules form one scaffold group.
- Cold groups are connected components joining proteins with global alignment identity >40% or identical recorded accessions. Global Needleman–Wunsch alignment uses BLOSUM62, gap-open 10, gap-extension 1, and matches/alignment-columns identity. This is not curated family separation or a guarantee against shared local domains. Observed maximum train/test identity is below 40% in all cold splits.
- Seeds: 7, 19, 42, 73, 101, 137. Approximately 20% outer test and 20% of the remainder inner validation, subject to whole-group allocation.
- Acquisition: 24 initial labels and four batches of 24, totaling 120. Inner-validation labels are an additional charged cost, never acquired. The exact cost is saved per run.
- Protein features: composition baseline, frozen ESM-2 8M (320 dimensions), frozen ESM-2 35M (480). Residue mean pooling excludes special tokens. Five sequences longer than 1,000 residues use residue-weighted nonoverlapping chunk pooling, losing cross-chunk context. These are two scales of one model family, not ProtT5 or learned contact maps.
- Lipid features: ECFP, MACCS, ChemBERTa MTR, ChemBERTa MLM, MoLFormer. Kernels: linear, RBF, Matérn-3/2. Policies: random, uncertainty, balanced UCB, Thompson, prediction-aware Thompson, integrated variance reduction.
- Per policy, representation/kernel selection uses only final inner-validation MAE, with deterministic ties. This is repeated nested holdout with one inner split, not full nested K-fold. Protein-specific comparisons hold the protein representation fixed during selection.

Frozen encoder provenance and exact checkpoint revisions are in `data/features/protein_representation_manifest.json`. ESM methods/checkpoints follow the [official ESM repository](https://github.com/facebookresearch/esm). The new study uses float64 GP geometry for numerical stability.

## Prediction-aware Thompson

For each batch, draw a correlated latent GP function. Score each candidate by an equal-weight combination of its sampled-affinity rank and its predictive variance-reduction rank. The latter is the mean of `C(proxy,x)^2 / (C(x,x) + noise_variance)` over a fixed proxy set of at most 64 inner-validation inputs. Update posterior covariance after each selected candidate. The proxy's labels and all outer-test inputs/labels are unavailable to the acquisition selector.

The weight is fixed at 0.5, not optimized on test results. Pure variance reduction is included as an ablation. This combines established ideas: variance reduction is not expected MAE reduction and is not a new theoretical acquisition principle; see [Cohn et al.](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/cohn96a-html/statmodels.html) and prior [multiobjective batch selection](https://proceedings.mlr.press/v84/gupta18a.html). Novelty and superiority are not established by this implementation.

## Main results

Six-seed means after validation-only configuration selection; MAE in pK units. Recall is recovery of top-decile acquisition-pool compounds, not an outer-test classification metric.

| Endpoint / split | Thompson MAE | Prediction-aware MAE | Thompson recall | Prediction-aware recall |
|---|---:|---:|---:|---:|
| Kd random | 1.017 | 0.933 | 51.3% | 31.1% |
| Kd cold sequence | 1.541 | 1.406 | 59.6% | 23.3% |
| Kd scaffold | 1.305 | 1.277 | 51.0% | 26.4% |
| Ki random | 0.895 | 0.702 | 64.3% | 33.3% |
| Ki cold sequence | 1.419 | 1.428 | 66.4% | 31.9% |
| Ki scaffold | 1.469 | 1.336 | 87.9% | 34.0% |

The selector changes the prediction–discovery trade-off; it does not jointly improve both metrics. Random acquisition remains competitive. Cold and scaffold prediction are substantially harder. Representation improvements are mixed; for example, within prediction-aware Thompson on cold Ki, composition / ESM-2 8M / ESM-2 35M give MAE 1.446 / 1.411 / 1.368 when lipid representation and kernel are selected within each protein group.

Paired sign-flip tests compare prediction-aware Thompson against Thompson and uncertainty, on MAE and recall AUC over six endpoint/split combinations (24 tests). None survives Holm correction. Importantly, six seeds imply a minimum two-sided exact p-value of 0.03125, and therefore a minimum Holm-adjusted value of 0.75 in this family. This analysis is underpowered by construction, not evidence of equivalence. Repeated holdouts overlap and are not independent biological replicates. Future confirmatory experiments need independently justified sample size and predeclared primary comparisons.

These 120-label results cannot be compared directly with the earlier 360-label study. The dataset was already used in earlier development; new splits are not a new external or prospective dataset.

## Lipid adaptation

Self-supervised continued ChemBERTa MLM training used **790** valid BioDolphin SMILES after excluding every benchmark scaffold and overlength molecule. Three fixed epochs were followed by frozen embeddings. This is a small domain-adaptation pilot, not a large LMSD/SwissLipids pretraining experiment. Training loss alone is not evidence of generalization.

Supervised residual bottleneck feature adapters were fitted separately for each partition using only its 24 initial labels, for 100 fixed epochs, then frozen across all kernels/policies/cycles. There are 36 such fits. No pooled full-label supervision, held-out labels, or future acquired labels enter adaptation. These adapters operate on frozen features; they are not attention LoRA or transformer weight fine-tuning.

Matched comparisons fix ESM-2 35M and ChemBERTa MLM, vary three kernels and six policies, and select kernels using inner validation. Prediction-aware cold Kd MAE is 1.431 frozen / 1.416 adapter / 1.520 SSL; cold Ki is 1.567 / 1.692 / 1.562. Scaffold Kd is 1.295 / 1.252 / 1.219. Improvements are inconsistent; these are descriptive comparisons, without a separate confirmatory significance claim.

## Explainability and assay checks

Across 549 lipids, 3,539 of 3,980 observed ECFP positions contain multiple unfolded Morgan identifiers. Distinct 32-bit identifiers are themselves imperfect proxies for distinct chemical environments. High/low-affinity collision comparisons use all labels retrospectively and are not independent structure–activity validation.

Nonlinear individual-bit permutation Shapley estimates cover four prespecified historical ECFP RBF/Matérn Thompson models, two fixed test queries each, three acquired-training backgrounds, and two sampling repeats (48 explanations). Protein features are held fixed. Mean cross-background top-ten Jaccard is 0.698 for RBF and 0.721 for Matérn; same-background repeats are 0.970 and 1.000. Additivity checks verify reconstruction, not sampling convergence or mechanistic validity.

Four selected fragment explanations map to experimental structures [4OAS](https://files.rcsb.org/download/4OAS.pdb) and [6CDJ](https://files.rcsb.org/download/6CDJ.pdb), only two distinct complexes. Full heavy-atom graph matches include symmetry alternatives; graph mapping ignores chirality. Every selected fragment atom lies within 4 Å of the recorded protein chain under the enumerated mappings. This is a geometric contact check, not a causal attribution or binding-energy calculation. First-model, recorded-chain coordinates are used without biological assembly expansion.

Assay disagreement is summarized in `assay_heterogeneity.csv`, not eliminated or modeled as a calibrated noise process. Consensus ranges and source annotations are not complete independent assay replicates. Zero observed range does not establish noiseless measurements.

## Artifacts and reproducibility

- [Main audit](results/robust_study_v2/audit.json): 9,720 campaigns, 48,600 fits, 23 nonconverged fits, split/selection checks passed.
- [Adaptation audit](results/adaptation_study/audit.json): 1,944 comparisons, 36 initial-only adapters, no SSL benchmark-scaffold overlap; six nonconverged fits including reused baselines.
- [Per-seed selected results](results/robust_study_v2/validation_selected_by_seed.csv), [paired tests](results/robust_study_v2/paired_comparisons.csv), [split audit](results/robust_study_v2/split_audit.csv).
- [Nonlinear explanation audit](results/explanation_robustness/audit.json), [contacts](results/structure_contacts/contact_summary.csv).

Implementation entry points are `prepare_protein_encoders.py`, `robust_splits.py`, `prediction_aware_acquisition.py`, `run_robust_study.py`, `parallel_robust_study.py`, `adapt_lipid_encoder.py`, `run_adaptation_study.py`, `explanation_robustness.py`, and `structure_contact_audit.py`. Pinned model files and manifests are under `data/features/`. Sequence alignment uses isolated `parasail==1.3.4` in `study_tools`; do not prepend that directory to Python's path and shadow the environment's NumPy.

To recheck existing results and rebuild the report in the configured project environment:

```powershell
python -m unittest test_robust_study.py
python audit_robust_study.py
python audit_adaptation_study.py
python build_robust_report.py
```

The earlier `results/robust_study/` attempt is explicitly invalidated following a float32 GP geometry failure; use only `results/robust_study_v2/`. Source/data hashes bind the completed runs to their implementations. Original study outputs remain unchanged.

Still outstanding: prospective/time-stamped external validation, larger independent lipid corpora, distinct protein-model families such as ProtT5, calibrated assay-aware noise modeling, broad contact-map analysis, and adequately powered confirmatory testing. These results are now included in [the updated manuscript](paper/main_updated.pdf) and the static dashboard export. Older manuscript snapshots are retained for provenance only.
