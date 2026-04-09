# Model-family robustness (Issue #117)

## What this checks

For top families from Issue #116 (`lr`, `xgb`), we compare:

- **Standard CV** (`split_mode_matrix` entry with `use_group_split: false`)
- **Site-exclusive group CV** (`use_group_split: true`, `group_col: site`)

`stratum_col: tumor_subtype` stays on so subtype metrics appear in each run’s `metrics.json`.

Frozen feature arm: **clinical + tumor_size + morph + graph + kinematic**.

## Reproducible artifacts

- Config: `configs/model_family_robustness.yaml`
- Same runner as feature ablation: `run_ablation_matrix.py`
- Slurm: `slurm/submit_model_family_robustness.slurm`
- Outputs (example): `experiments/model_family_robustness_ispy2/`
  - `ablation_summary.csv` — **AUC mean and `auc_std` over folds**
  - `ablation_subtype_summary.csv` — subtype AUC rows (`export_subtype_summary: true`)
- PR snapshots: `results/model_family_robustness_ispy2_summary.csv`, `results/model_family_robustness_ispy2_subtype_summary.csv`

## Overall results (AUC mean ± std, prior run)

| Model | Standard CV | Site-exclusive CV |
|-------|-------------|---------------------|
| `lr`  | 0.595 ± 0.032 | 0.523 ± 0.060 |
| `xgb` | 0.606 ± 0.028 | 0.608 ± 0.054 |

## Plain-English conclusion

- **LR** loses a lot of mean AUC under site-exclusive folds (possible site-related signal in standard CV).
- **XGB** keeps similar **mean** AUC; **std** rises under site splits.
- **Most robust on mean AUC under site splits:** **XGB**; LR remains the simpler baseline but needs cautious interpretation under site-exclusive evaluation.

Refs #117
