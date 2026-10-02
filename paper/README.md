# ACML-format manuscript draft

## Current full study

Use **[main_updated.pdf](main_updated.pdf)** and `main_updated.tex`: the current
15-page manuscript includes the original 5,850 campaigns, 9,720 leakage-controlled
campaigns and 1,296 additional adaptation campaigns. Eight four-panel figures
appear in the paper. `updated_composite_supplement.pdf` contains all seven older
main figures and five new composites; `all_figures_atlas.pdf` retains all 223
historical source plots. `updated_results_tables/` holds the full new tables and
audits. `acml_updated_source_package.zip` is the current source package.

Build with `tectonic main_updated.tex --keep-logs`, then run
`python package_updated.py`. `build_updated_assets.py` generates actual local
dashboard captures and new diagnostics. The package audit checks the adopted
16-page limit, references, text overflow, figure hashes and unchanged template.
The linked public repository identifies its owner: anonymize/remove links and
metadata before any double-blind submission. Hosting status is kept in
`deployment_status.tex`; do not invent a deployment URL.

The material below documents the **superseded original expanded-grid paper**,
not the current full-study manuscript.

Title: **Discovery, Prediction, and Explanation in Active Learning for
Protein--Lipid Binding Affinity**.

- `main.tex`: anonymous conference-track manuscript using the official 2026
  ACML `jmlr` class. Authors and affiliations are intentionally blank.
- `main_expanded.pdf`: current compiled **15-page** manuscript covering all
  5,850 runs (within the adopted 16-page limit). `main_expanded.tex` wraps
  `main.tex`. The older `main.pdf` was locked by an open viewer and remains
  the superseded 12-page snapshot; do not use it for the expanded results.
- `figures/fig1_performance.pdf` through `fig7_dashboard.pdf`: seven
  four-panel composites. New figures compare all five encoders and show four
  actual dashboard screenshots. Charts are vector; UI screenshots are raster.
- `all_figures_atlas.pdf`: all 223 pre-existing result figures, arranged into
  45 supplementary composites with four or five source figures per page.
- `figure_manifest.csv`: exact source paths, page/panel mapping and SHA-256
  hashes. SVG fragment illustrations are embedded as vectors in the atlas.
- `acml_expanded_source_package.zip`: current LaTeX sources, class, PDF figures,
  supplementary atlas, screenshot sources, complete result tables and provenance
  files; no local Python environment. The old source ZIP is a legacy snapshot.
- `results_tables/`: all 1,950 configuration summaries and 5,850 per-seed
  outcomes, tied-winner table, matched-encoder table and experiment/feature audits.

The format comes from the [official ACML 2026 conference-track instructions](https://www.acml-conf.org/2026/calls/papers/)
and [official template](https://www.acml-conf.org/2026/downloads/ACML_camera_ready.zip).
The main-paper limit is 16 pages including references and appendices; the full
figure atlas is a separate supplement. The official class is preserved and
its license notice retained. The manuscript follows the template's empty
author convention and suppressed page numbering.

## Build

Upload the source ZIP to Overleaf and select `main.tex`, or use a normal LaTeX
installation with the `jmlr` dependencies. Run PDFLaTeX twice. With the isolated
compiler environment installed locally, run from this folder:

```powershell
conda run --prefix .\tex_env --no-capture-output tectonic main_expanded.tex --keep-logs
```

To regenerate the figures from the existing project results:

```powershell
python capture_dashboard.py
python build_figures.py
python build_expanded_figures.py
conda run --prefix .\tex_env --no-capture-output tectonic main_expanded.tex --keep-logs
python validate_package.py
```

Figure assembly additionally uses matplotlib, pandas, NumPy, ReportLab and
svglib; PDF validation uses PyMuPDF. The latter rendering dependencies were
installed into this folder's `python_tools`, not into the experiment environment.
Source-data calculations and original experiment files are not changed.
Dashboard capture additionally uses the project's local Playwright installation
and Chrome, with a temporary localhost server. Screenshots are genuine local
application views, not mockups or evidence of public hosting.
The isolated `tex_env` and `python_tools` folders are local build tools and
are deliberately excluded from the source ZIP. The standalone GitHub compiler
download failed; the successful compiler installation used conda-forge.

## Scientific and submission status

This is a research draft, not a claim of acceptance or submission readiness.
It reports post-hoc descriptive selections and negative small-panel findings.
No novel acquisition algorithm, prospective validation, causal mechanism,
binding-site identification or superiority over untested external baselines
is claimed. Three seeds and random-pair splits do not establish robust
generalization. Larger target-specific data, grouped splits, independent model
selection, stronger baselines and additional explanation controls remain
important before submission. Biological interpretation and references need
author review. Add the real author details only for the appropriate version.

The atlas includes legacy 64-label baseline plots for completeness, clearly
identified by `results/published/` paths. They are not budget-matched evidence
against the current 360-label grid. The full atlas is intentionally extensive;
reviewers need not read supplementary material, so the main text retains the
principal methods, outcomes and limitations.

The ACML 2026 main submission deadline shown by the official site was July 5,
2026, already past at drafting time. This package adopts that year's format;
it does not imply eligibility to submit now. Recheck the target edition's
instructions before submission. No external submission or upload was made.
