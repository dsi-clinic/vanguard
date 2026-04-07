# Model-family robustness (Issue #117)

## What this checks

For the top families from Issue #116 (`lr`, `xgb`), we compare:

- **Standard stratified CV** (`use_group_split: false`)
- **Site-exclusive group-aware CV** (`use_group_split: true`, `group_col: site`)

We keep `stratum_col: tumor_subtype` so subtype-specific AUC appears in evaluator outputs.

Frozen feature arm for this run: **clinical + tumor_size + morph + graph + kinematic** (`clinical_plus_tumor_size_plus_vessel_all`).

## Reproducible artifacts

- Config: `configs/model_family_robustness.yaml`
- Runner: `run_model_family_robustness.py`
- Slurm: `slurm/submit_model_family_robustness.slurm`
- Full run outputs (on cluster): `experiments/model_family_robustness_ispy2/`
- PR snapshots: `results/model_family_robustness_ispy2_summary.csv`, `results/model_family_robustness_ispy2_subtype_summary.csv`

## Overall results (AUC mean ± std)

| Model | Standard CV | Site-exclusive CV |
|-------|-------------|---------------------|
| `lr`  | 0.595 ± 0.032 | 0.523 ± 0.060 |
| `xgb` | 0.606 ± 0.028 | 0.608 ± 0.054 |

## Plain-English conclusion

- **Logistic regression** loses a lot of mean AUC when evaluated with site-exclusive folds. That suggests **site leakage or site-driven signal** may be inflating standard-CV performance for LR on this arm.
- **XGBoost** keeps a similar **mean** AUC under site-exclusive folds, but **fold variability increases** (higher std). So it looks **more stable in average level** across split strategies, with **more variance** between folds when sites are held out.
- Subtype AUCs are not identical across strata (see subtype CSV); **luminal_b** and **triple_negative** are relatively weaker under site CV for both models, which is worth mentioning in write-ups.

**Most robust for “does not collapse under site splits” on mean AUC:** **`xgb`**.  
**Simplest baseline to interpret** remains **`lr`**, but it should be interpreted cautiously under site-exclusive evaluation for this arm.
