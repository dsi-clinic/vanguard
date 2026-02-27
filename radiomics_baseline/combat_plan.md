# ComBat Plan (ISPY2-First) + Visualization Plan

## Goal
- Reduce site/scanner-related feature shift while preserving pCR signal.
- Evaluate harmonization in a leakage-safe way.
- Prioritize CV mean AUC and CV std (ignore held-out test until split/CV exclusivity is fixed).

## Key Rule
- Never fit ComBat on full data before CV.
- For each CV fold:
  - Fit ComBat parameters on train fold only.
  - Apply learned transform to validation fold.

## Data/Eval Setup
- Cohort: ISPY2 only (first phase).
- Features: current imaging-only radiomics feature table(s).
- CV protocol:
  - Primary: Grouped CV by site (or scanner center proxy if available).
  - Secondary: standard stratified CV for sensitivity check.
- Ranking metric:
  - Primary: mean CV AUC.
  - Secondary: CV AUC std (lower is better).

## Harmonization Experiments
- Baselines:
  1. No harmonization.
  2. Site-wise z-score (train-fold fit only).
- ComBat variants:
  3. ComBat parametric.
  4. ComBat non-parametric.
  5. ComBat with biological covariates preserved (pCR not included; optional age/tumor burden if appropriate and available).
- Optional extension:
  6. ComBat-GAM if strong nonlinear continuous confounders are present.

## Minimal Run Matrix
- Keep model fixed initially (same classifier + selector) and only swap harmonization.
- Start with the current strongest CV-oriented settings:
  - `bin100_f2d1_kinsubonly_kbest40_corr080`
  - `bin100_f2d1_all_mrmr20_corr080`
  - `bin8_f2d1_kinsubonly_kbest50_corr095`
- Run each of the three settings across the four harmonization modes:
  - none, z-score, ComBat-param, ComBat-nonparam
- Total initial matrix: 12 runs.

## Visualization Plan (Before vs After ComBat)

## 1) PCA scatter (paired panels)
- Panel A: before ComBat.
- Panel B: after ComBat.
- Color: site. Marker: pCR.
- Use identical axis limits and same PCA basis handling for fair comparison.
- Purpose: visually assess site cluster separation reduction.

## 2) Feature distributions by site (selected site-sensitive features)
- Pick 6-10 features with strongest site association pre-ComBat.
- Plot violin/ridge distributions per site:
  - left column = before
  - right column = after
- Purpose: verify site mean/scale offsets are reduced.

## 3) Site-mean heatmap
- Rows: site.
- Columns: selected features.
- Values: standardized site-wise mean feature values.
- Show before and after side-by-side.
- Purpose: global view of batch effect attenuation.

## 4) Site centroid distance heatmap
- Compute pairwise distances between site centroids in standardized feature space.
- Show before and after.
- Purpose: quantify reduction in inter-site divergence.

## 5) “Signal retained” summary bars
- Metric A: site-prediction CV AUC (should decrease after harmonization).
- Metric B: pCR-prediction CV AUC (should be stable or improve).
- Plot with error bars (std over folds).
- Purpose: confirm harmonization removes nuisance signal more than biological signal.

## Suggested Figure Order for Student Readability
1. PCA before/after (intuitive big picture).
2. Distribution plots for exemplar features.
3. Heatmaps (site means + site distances).
4. Signal-retained summary chart (decision figure).

## Interpretation Heuristics
- Good outcome:
  - Site separability drops.
  - pCR CV mean is stable or up.
  - pCR CV std does not increase materially.
- Bad outcome:
  - Site separability drops but pCR CV mean also drops notably.
  - Indicates over-correction or biological signal removal.

## Deliverables to Save
- `harmonization_summary.csv`:
  - experiment_id, harmonization_mode, cv_auc_mean, cv_auc_std, site_auc_mean, site_auc_std
- Figure bundle:
  - `combat_pca_before_after.png`
  - `combat_feature_distributions_before_after.png`
  - `combat_site_mean_heatmap_before_after.png`
  - `combat_site_distance_heatmap_before_after.png`
  - `combat_signal_retained_summary.png`

## Next Step After ISPY2
- Once CV-stable harmonization is identified in ISPY2:
  - Freeze the harmonization protocol.
  - Re-run in multi-site setting to test external generalization behavior.

## Implemented Wiring (Current Repo)
- Training supports:
  - `--harmonization-mode {none,zscore_site,combat_param,combat_nonparam}`
  - `--harmonization-batch-col <labels column>`
  - `--harmonization-covariate <labels column>` (repeatable)
  - `--cv-only` (skips test-set evaluation; writes CV-first metrics)
- `run_experiment.py` passes these keys from YAML `train.*`.
- `run_ablations.py` now includes `auc_train_cv_std` in summary CSV and respects explicit `paths.extract_outdir` in base configs.

## Ready-to-Run Matrix Configs
- Base configs:
  - `configs/exp_combat_matrix_all_mrmr20.yaml`
  - `configs/exp_combat_matrix_all_kbest40.yaml`
  - `configs/exp_combat_matrix_kinsub_kbest50.yaml`
- Sweep configs:
  - `configs/sweep_combat_matrix_all_mrmr20.yaml`
  - `configs/sweep_combat_matrix_all_kbest40.yaml`
  - `configs/sweep_combat_matrix_kinsub_kbest50.yaml`
- Convenience runner:
  - `scripts/run_combat_matrix.sh`
