# Model-Family Ablation Orientation (Issue #116)

## What currently exists

- **Tabular training entrypoint**
  - `train_tabular.py`
  - Builds features via `tabular_cohort.py`, then runs evaluator CV and saves metrics/predictions/plots.
- **Model families in code**
  - logistic regression (`model: lr`)
  - random forest (`model: rf`)
  - XGBoost (`model: xgb`) in `tabular_models.py`
- **Feature-arm ablations**
  - `run_ablation_matrix.py` (single entrypoint for feature arms, optional **model-family** and **split-mode** grids)
  - driven by YAML such as `configs/independent_signal.yaml` or `configs/model_family_matrix.yaml`
- **Shared evaluation**
  - `evaluation/` handles folds and metrics (including **AUC** and **AP** in fold aggregates when both classes appear)

## Existing tracked result summaries

- `run_ablation_matrix.py` writes:
  - `ablation_summary.csv`, `ablation_fold_auc.csv`
- Independent-signal artifacts referenced elsewhere:
  - `results/independent_signal_q3_summary.csv`

## Issue #116 setup (same runner as feature ablation)

- **Config:** `configs/model_family_matrix.yaml`
  - `ablation_arms`: fixed feature arms
  - `model_families`: `lr`, `rf`, `xgb`
  - `split_mode_matrix`: one mode (`cv`) so run IDs look like `clinical_plus_tumor_size__lr__cv`
  - `model_family_overrides`: per-family tuning (e.g. nested tuning on for `lr`)
  - `baseline_run_name`: `clinical_plus_tumor_size__lr__cv` for delta vs baseline column
- **Slurm:** `slurm/submit_model_family_matrix.slurm` (submit from repo root; logs under `logs/`)
- **Outputs:** under `experiments/model_family_matrix_ispy2/` (or your `OUTDIR`):
  - `ablation_summary.csv` (**includes `auc_mean` and `auc_std` over folds**)
  - `runs/<run_name>/metrics.json`, nested tuning under each run folder (no overwrites)
- **PR snapshot:** `results/model_family_matrix_ispy2_summary.csv`

## Observed AUC means (prior run; std in CSV)

- `clinical_plus_tumor_size`: lr > rf > xgb
- `clinical_plus_tumor_size_plus_kinematic`: lr > rf > xgb
- `clinical_plus_tumor_size_plus_vessel_all`: xgb > rf > lr

## Conclusion (Issue #116) — families to carry forward

**Recommend carrying forward logistic regression (`lr`) and XGBoost (`xgb`).** On the matrix run in `results/model_family_matrix_ispy2_summary.csv`, **`lr` had the best mean AUC on the two simpler arms** (clinical + tumor_size, and + kinematic). **On the richest arm** (clinical + tumor_size + morph + graph + kinematic), **`xgb` had the best mean AUC** (with the lowest AUC std among the three families on that arm). **Random forest** was never the top family on any arm in this matrix, so it is optional at most—not a default carry-forward.

Follow-up robustness (#117): `docs/model_family_robustness_117.md`.
