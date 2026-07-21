# asymmetry_predictions_v.2

`asymmetry_predictions_v.2` is the selected nuisance-filtered DUKE
vessel/asymmetry pCR model.

`asymmetry_predictions_v.1` refers to the original all-feature vessel
asymmetry model previously reported as `logreg_all`.

## Selection Rule

Remove vessel features with max absolute Spearman correlation >= `0.25`
to any of: `xy_spacing_mm`, `z_spacing_mm`, `tumor_size_tumor_voxels`.

## Result

- Cases: `273`
- Features kept: `104`
- Features removed: `11`
- Mean AUC: `0.597 +/- 0.044`
- Mean AP: `0.311 +/- 0.016`
- Max nuisance |r| after filtering: `0.228`

## Files

- `run_asymmetry_predictions_v2.py`: build/evaluate script.
- `asymmetry_predictions_v2_features.csv`: filtered model input table.
- `removed_nuisance_features.csv`: removed features and nuisance scores.
- `all_feature_nuisance_scores.csv`: full nuisance audit.
- `filtered_feature_nuisance_scores.csv`: nuisance audit after filtering.
- `cv_metrics.csv`: fold-level and mean/std pCR metrics.
- `oof_predictions.csv`: out-of-fold predictions.
- `auc_asymmetry_predictions_v2.png`: AUC plot.
- `nuisance_before_after_heatmap.png`: nuisance before/after heatmap.
- `asymmetry_predictions_v1_nuisance_heatmap.png`: nuisance heatmap for
  `asymmetry_predictions_v.1`.
- `asymmetry_predictions_v2_nuisance_heatmap.png`: matching nuisance heatmap
  for `asymmetry_predictions_v.2`.
- `run_metadata.json`: exact paths and feature counts.
