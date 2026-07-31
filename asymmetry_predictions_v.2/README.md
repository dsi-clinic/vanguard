# asymmetry_predictions_v.2

`asymmetry_predictions_v.2` is the selected DUKE vessel/asymmetry pCR
model with lower maximum marginal feature_confounder correlation.

`asymmetry_predictions_v.1` refers to the original all-feature vessel
asymmetry model previously reported as `logreg_all`.

## Selection Rule

Remove vessel features with max absolute Spearman correlation >= `0.25`
to any audited covariate. The technical confounders are `xy_spacing_mm`
and `z_spacing_mm`. `tumor_size_tumor_voxels` is treated separately as a
biological/clinical covariate used to test whether vessel signal is
independent of tumor extent.

This supports the narrower claim of lower maximum marginal
feature_confounder correlation for retained features. It does not prove
that the multivariate model has lower technical-confounder sensitivity or
will generalize outside Duke.

## Result

- Cases: `273`
- Features kept: `104`
- Features removed: `11`
- Mean AUC: `0.597 +/- 0.044`
- Mean AP: `0.311 +/- 0.016`
- Max marginal feature/covariate |r| after filtering:
  `0.228`
- Max marginal feature/technical-confounder |r| after filtering:
  `0.228`
- Max marginal feature/tumor-size |r| after filtering:
  `0.205`

## OOF Model-Level Covariate Audit

Out-of-fold predicted pCR probabilities were correlated with the audited
covariates as a small model-level check.

- `y_prob` vs `xy_spacing_mm`: Spearman r `0.014`
  (`n=273`)
- `y_prob` vs `z_spacing_mm`: Spearman r `-0.102`
  (`n=273`)
- `y_prob` vs `tumor_size_tumor_voxels`
  (biological/clinical covariate): Spearman r `0.020`
  (`n=273`)

See `oof_prediction_covariate_audit.csv` for p-values and absolute
correlations.

## Derived Inputs

### Tumor Size Plus Vessel Asymmetry Feature Table

- Contains: 273 shared DUKE cases with `case_id`, `dataset`, `pcr`, 11
  tumor-size features, and 115 vessel/asymmetry features before v2
  filtering.
- Produced by: the shared 273-case DUKE tumor/vessel comparison workflow
  documented in
  `vessel_tumor_comparisons/shared_273_case_comparison/README.md`, using
  `tabular/duke_final_vessel_asymmetry_features.csv` plus
  `vessel_tumor_comparisons/duke_tumor_size_only_ablation/runs/tumor_size_only/features_engineered_labeled.csv`.
- Local source path: `/ess/home/home1/aakrithiram/vanguard/vessel_tumor_comparisons/shared_273_case_comparison/inputs/tumor_size_plus_vessel_asymmetry.csv`
- Durable shared copy: `/gpfs/data/karczmar-lab/workspaces/aakrithiram/asymmetry_predictions_v2_inputs/tumor_size_plus_vessel_asymmetry.csv`
- SHA256: `3d8dfff4399d967af21050a11290c47cd54bfaf1ab5fd3b6974156a81496f2f6`

### Spacing Table

- Contains: per-case scanner spacing covariates, including
  `xy_spacing_mm` and `z_spacing_mm` after `patient_id` is renamed to
  `case_id`.
- Produced by: Sarit's data-visualization spacing export workflow.
- Shared path: `/gpfs/data/karczmar-lab/workspaces/saritbose/outputs/data_viz/spacing_by_hospital.csv`
- SHA256: `2003a83518c558ed8d3954c4ba9d68a4537b505682c4a96183bb6817a2f4fe49`

## Exact Command

```bash
micromamba run -n vanguard python asymmetry_predictions_v.2/run_asymmetry_predictions_v2.py --overwrite
```

## Files

- `run_asymmetry_predictions_v2.py`: build/evaluate script.
- `asymmetry_predictions_v2_features.csv`: filtered model input table.
- `removed_nuisance_features.csv`: removed features and audited
  covariate scores.
- `all_feature_nuisance_scores.csv`: full feature/covariate audit.
- `filtered_feature_nuisance_scores.csv`: feature/covariate audit after
  filtering.
- `cv_metrics.csv`: fold-level and mean/std pCR metrics.
- `oof_predictions.csv`: out-of-fold predictions.
- `oof_prediction_covariate_audit.csv`: model-level OOF y_prob
  correlation audit against spacing and tumor size.
- `auc_asymmetry_predictions_v2.png`: AUC plot.
- `nuisance_before_after_heatmap.png`: audited covariate before/after
  heatmap.
- `asymmetry_predictions_v1_nuisance_heatmap.png`: covariate heatmap for
  `asymmetry_predictions_v.1`; separately generated exploratory artifact.
- `asymmetry_predictions_v2_nuisance_heatmap.png`: matching covariate heatmap
  for `asymmetry_predictions_v.2`; separately generated exploratory artifact.
- `run_metadata.json`: exact paths and feature counts.
