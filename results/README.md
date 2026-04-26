# Project 2 — Tabular model selection & baseline-arm comparison

Summary of outputs for **Issues #116, #117, and #118**. All three are driven by the unified `run_ablation_matrix.py` pipeline so the runs are directly comparable. Per-issue background and run instructions live in `docs/`.

## TL;DR

- **Top tabular setup:** `clinical + tumor_size + vessel_all` (morph + graph + kinematic), evaluated with **5-fold CV** on **n = 808 ISPY2 unilateral cases**.
  - **XGBoost:** AUC **0.606 ± 0.028**, AP **0.415 ± 0.020**
  - **Logistic regression (nested-tuned):** AUC **0.595 ± 0.033**, AP **0.396 ± 0.034**
- **Vs. tumor-size-only baseline** (LR AUC 0.571 ± 0.046): full vessel stack adds **+0.024 (LR)** / **+0.035 (XGB)** AUC; only the full stack moves both families consistently above baseline.
- **Vs. radiomics baseline** (Bella, last quarter — AUC 0.594 on a 447-case held-out test split): tabular `+ vessel_all` matches it on a larger, consistent CV cohort.
- **Robustness (site-exclusive CV):** XGB stays flat (0.606 → 0.608 AUC). LR collapses (0.595 → 0.523).

## Issues at a glance

| Issue | What it asks | Top-level deliverable | Key output(s) |
|---|---|---|---|
| **#116** | Model-family ablation across feature arms | Pick families to carry forward | `results/model_family_matrix_ispy2_summary.csv` |
| **#117** | Robustness for top families (standard CV vs. site-exclusive group CV) | Robustness claim | `results/model_family_robustness_ispy2_summary.csv`, `..._subtype_summary.csv` |
| **#118** | Frozen-setup vessel-arm comparison vs. radiomics baseline | Best vessel-feature recipe + radiomics comparison | `experiments/issue118_baseline_arms/ablation_summary.csv` |

---

## Issue #116 — Model-family ablation matrix

**Question:** Across `lr`, `rf`, `xgb`, which family wins on each feature arm?

**Setup**
- Runner: `run_ablation_matrix.py`
- Config: `configs/model_family_matrix.yaml`
- Slurm: `slurm/submit_model_family_matrix.slurm`
- Per-family overrides: nested tuning **on** for `lr`, off for `rf` / `xgb`
- Doc: `docs/model_family_ablation_orientation.md`

**Key metrics (CV AUC mean ± std)**

| Arm | `lr` | `rf` | `xgb` | Winner |
|---|---|---|---|---|
| `clinical_plus_tumor_size` | **0.571 ± 0.046** | 0.545 ± 0.060 | 0.527 ± 0.045 | lr |
| `clinical_plus_tumor_size_plus_kinematic` | **0.587 ± 0.041** | 0.576 ± 0.038 | 0.562 ± 0.059 | lr |
| `clinical_plus_tumor_size_plus_vessel_all` | 0.591 ± 0.034 | 0.603 ± 0.038 | **0.606 ± 0.028** | xgb |

**Conclusion (carry-forward):** Keep **`lr` and `xgb`**. LR wins on the simpler arms, XGBoost takes over once all three vessel blocks are stacked (with the lowest std on that arm). RF is never the top family on any arm.

---

## Issue #117 — Robustness for top families

**Question:** On the frozen `+ vessel_all` arm, do `lr` and `xgb` hold up under a stricter "no shared site between train and validation" evaluation?

**Setup**
- Runner: `run_ablation_matrix.py` (same one as #116)
- Config: `configs/model_family_robustness.yaml`
- Slurm: `slurm/submit_model_family_robustness.slurm`
- Frozen arm: `clinical + tumor_size + morph + graph + kinematic`
- Split modes: `standard_cv` vs. `site_group_cv` (group_col = `site`)
- Subtype output: `ablation_subtype_summary.csv` (stratum = `tumor_subtype`)
- Doc: `docs/model_family_robustness_117.md`

**Key metrics (AUC mean ± std)**

| Model | Standard CV | Site-exclusive CV | Δ (mean) |
|---|---|---|---|
| `lr` | 0.595 ± 0.032 | 0.523 ± 0.060 | **−0.072** |
| `xgb` | 0.606 ± 0.028 | 0.608 ± 0.054 | **+0.002** |

**Conclusion (robustness claim):** Under site-exclusive folds, **XGBoost mean AUC is essentially unchanged**, while **LR drops substantially**. We are not claiming XGB is universally better — only that under this stress test LR's standard-CV number is harder to trust as a guide to out-of-site behavior, and XGB's mean stays flat (with a wider fold-to-fold spread).

---

## Issue #118 — Tabular baseline arms vs. radiomics

**Question:** With one frozen setup, how do the five vessel arms compare to each other and to the radiomics baseline from last quarter?

**Setup**
- Runner: `run_ablation_matrix.py`
- Config: `configs/issue118_baseline_arms.yaml`
- Slurm: `slurm/submit_issue118_baseline_arms.slurm`
- Families: `lr` (nested-tuned), `xgb`
- Split: 5-fold CV (`use_group_split: false`)
- Cohort: **n = 808** ISPY2 unilateral (`dataset_include: [ISPY2]`, `bilateral_filter: False`) — see "Cohort note" below
- Radiomics reference: `radiomics/outputs/peri5_multiphase_logreg/training/metrics.json` → AUC **0.594** (447-case test split; not paired, used as a reference line only)
- Doc: `docs/issue118_baseline_comparison.md`

**Key metrics (CV AUC mean ± std, AP mean ± std, ΔAUC vs. `clinical_plus_tumor_size__lr__cv`)**

| Arm | Family | AUC | AP | ΔAUC vs. baseline |
|---|---|---|---|---|
| `clinical_plus_tumor_size` | lr | 0.571 ± 0.046 | 0.357 ± 0.033 | — (baseline) |
| `clinical_plus_tumor_size` | xgb | 0.527 ± 0.045 | 0.348 ± 0.032 | −0.044 |
| `+ morph` | lr | 0.587 ± 0.032 | 0.376 ± 0.023 | +0.016 |
| `+ morph` | xgb | 0.580 ± 0.031 | 0.390 ± 0.038 | +0.009 |
| `+ graph` | lr | 0.586 ± 0.057 | 0.380 ± 0.056 | +0.015 |
| `+ graph` | xgb | 0.527 ± 0.016 | 0.349 ± 0.008 | −0.044 |
| `+ kinematic` | lr | 0.587 ± 0.041 | 0.381 ± 0.037 | +0.016 |
| `+ kinematic` | xgb | 0.562 ± 0.059 | 0.387 ± 0.046 | −0.009 |
| `+ vessel_all` (morph + graph + kinematic) | **lr** | **0.595 ± 0.033** | **0.396 ± 0.034** | **+0.024** |
| `+ vessel_all` | **xgb** | **0.606 ± 0.028** | **0.415 ± 0.020** | **+0.035** |

**Cohort note (n = 808):** This is not a sample-loss bug. The pipeline parses 1506 centerline studies, applies the ISPY2 + unilateral prefilters, then keeps only cases with all of (centerline `.npy`, tumor mask, tumor-graph JSON) — every one of the 808 ISPY2 unilateral cases passes all three. End-to-end trace is in `docs/issue118_baseline_comparison.md` ("Cohort-size investigation").

**Bottom line:**
- Best overall tabular setup is `+ vessel_all` with **XGBoost** on AUC and AP.
- Single vessel blocks alone give only marginal gains; the full vessel stack is the only configuration where both families clear the tumor-size-only baseline.
- Tabular `+ vessel_all` matches the radiomics reference AUC on a larger CV cohort.

---

## File index

| Path | Issue | Notes |
|---|---|---|
| `results/model_family_matrix_ispy2_summary.csv` | #116 | Per-arm × per-family AUC mean / std |
| `results/model_family_robustness_ispy2_summary.csv` | #117 | Standard CV vs. site-exclusive CV |
| `results/model_family_robustness_ispy2_subtype_summary.csv` | #117 | Per-tumor-subtype AUC rows |
| `experiments/issue118_baseline_arms/ablation_summary.csv` | #118 | Five arms × two families, AUC / AP / Δ |
| `experiments/issue118_baseline_arms/ablation_fold_auc.csv` | #118 | Per-fold AUC for paired comparisons |
| `docs/model_family_ablation_orientation.md` | #116 | Background + reproduction |
| `docs/model_family_robustness_117.md` | #117 | Background + reproduction |
| `docs/issue118_baseline_comparison.md` | #118 | Background + reproduction + cohort trace |
