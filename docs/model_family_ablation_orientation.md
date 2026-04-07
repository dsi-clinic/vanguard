# Model-Family Ablation Orientation (Issue #116)

## What currently exists

- **Tabular training entrypoint**
  - `train_tabular.py`
  - Builds features via `tabular_cohort.py`, then runs evaluator CV and saves metrics/predictions/plots.
- **Current model families in code (before this issue)**
  - logistic regression (`model: lr`)
  - random forest (`model: rf`)
  - implemented in `tabular_models.py`
- **Feature-arm ablations already present**
  - `run_ablation_matrix.py`
  - driven by `configs/ablation.yaml` and `configs/independent_signal.yaml`
  - compares canonical blocks (`clinical`, `tumor_size`, `morph`, `graph`, `kinematic`)
- **Shared evaluation**
  - `evaluation/` package handles consistent folds and metrics
  - outputs include aggregated fold metrics and per-fold predictions

## Existing tracked result summaries

- `run_ablation_matrix.py` writes:
  - `ablation_summary.csv`
  - `ablation_fold_auc.csv`
- Independent-signal tracked artifacts referenced in top-level README:
  - `results/independent_signal_q3_summary.csv`
  - `results/independent_signal_q3_auc_summary.png`

## What was added for model-family matrix

- **New model family support**
  - XGBoost added in `tabular_models.py` (`model: xgb`)
- **New matrix entrypoint**
  - `run_model_family_matrix.py`
  - runs fixed feature arms across configurable families
  - saves one merged table:
    - `model_family_matrix_summary.csv`
  - includes requested columns:
    - `arm_name`, `model_family`, `feature_selection_mode`, `nested_tuning_on`,
      `auc_mean/std`, `ap_mean/std`
- **Reproducible config**
  - `configs/model_family_matrix.yaml`
  - focused starting arms:
    - `clinical + tumor_size`
    - `clinical + tumor_size + kinematic`
    - `clinical + tumor_size + morph + graph + kinematic`
  - families:
    - `lr`, `rf`, `xgb`
- **Slurm launcher**
  - `slurm/submit_model_family_matrix.slurm`

## Run outputs and conclusions (completed run)

Run used:

- Config: `configs/model_family_matrix.yaml`
- Slurm script: `slurm/submit_model_family_matrix.slurm`
- Output root: `experiments/model_family_matrix_ispy2`
- Merged table: `experiments/model_family_matrix_ispy2/model_family_matrix_summary.csv`
- PR-safe snapshot: `results/model_family_matrix_ispy2_summary.csv`

Observed AUC means from the completed matrix:

- `clinical_plus_tumor_size`
  - `lr`: **0.5710**
  - `rf`: 0.5448
  - `xgb`: 0.5266
- `clinical_plus_tumor_size_plus_kinematic`
  - `lr`: **0.5867**
  - `rf`: 0.5764
  - `xgb`: 0.5620
- `clinical_plus_tumor_size_plus_vessel_all`
  - `xgb`: **0.6064**
  - `rf`: 0.6034
  - `lr`: 0.5906

Answers to issue guidance:

- Does a stronger nonlinear model beat logistic regression?
  - **Yes, on the richest vessel arm** (`clinical_plus_tumor_size_plus_vessel_all`), where `xgb` is best.
- On all arms, or only on richer vessel arms?
  - **Only on richer vessel arms** in this focused matrix. `lr` remains best on the simpler two arms.
- Is the extra complexity worth it?
  - **Conditionally yes**: nonlinear models are most useful when the full vessel feature set is present.

Recommended families to carry into the next project step:

- **`lr`**: strongest and simplest baseline across the two simpler arms.
- **`xgb`**: best peak performance on the richest arm.

`rf` is close to `xgb` on the richest arm and can be kept as a tie-breaker, but it is not the top pick overall for follow-up experiments.

