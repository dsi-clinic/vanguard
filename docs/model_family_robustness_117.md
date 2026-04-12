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

## Conclusion (Issue #117) — robustness claim supported by site-exclusive CV

**Claim we think the numbers support:** On the full vessel arm, **mean AUC for XGBoost stayed essentially the same** when switching from standard CV to **site-exclusive** folds (≈0.606 vs ≈0.608 in `results/model_family_robustness_ispy2_summary.csv`), while **mean AUC for logistic regression dropped substantially** (≈0.595 vs ≈0.523). So **XGBoost looks more stable under a “no shared site between train and validation” evaluation** on this setup; **LR’s standard-CV number is harder to trust** as a guide to out-of-site behavior here. We are **not** claiming XGB is universally better—only that **under this stress test, LR’s performance collapses more than XGB’s**, and XGB’s fold-to-fold spread widens under site splits (higher `auc_std`) even though the mean stays flat.

Refs #117
