# Issue #118 — Tabular baseline arms vs radiomics (oriented)

## What this delivers

A reproducible rerun of the **five vessel baseline arms** from the issue description, using the **frozen families** from #116 / #117:

- **`lr`**: nested tuning on (`issue118_baseline_arms_lr.yaml`)
- **`xgb`**: nested tuning off (`issue118_baseline_arms_xgb.yaml`)

Arms (both configs):

1. `clinical + tumor_size` (baseline; deltas vs this arm are in `ablation_summary.csv`)
2. `clinical + tumor_size + morph`
3. `clinical + tumor_size + graph`
4. `clinical + tumor_size + kinematic`
5. `clinical + tumor_size + morph + graph + kinematic` (`clinical_plus_tumor_size_plus_vessel_all`)

## Run on the cluster

From the repo root:

```bash
cd slurm
sbatch submit_issue118_baseline_arms.slurm
```

Optional overrides (fresh output roots, same configs):

```bash
LR_OUT=../experiments/issue118_baseline_arms_lr_run2 \
XGB_OUT=../experiments/issue118_baseline_arms_xgb_run2 \
MERGED_DIR=../experiments/issue118_baseline_arms_merged_run2 \
sbatch submit_issue118_baseline_arms.slurm
```

## Outputs

Per model family:

- `experiments/issue118_baseline_arms_lr/` — `ablation_summary.csv`, `runs/<arm>/metrics.json`, …
- `experiments/issue118_baseline_arms_xgb/` — same layout

Merged:

- `experiments/issue118_baseline_arms_merged/issue118_baseline_arms_combined.csv` — long table with `model_family` column

`ablation_summary.csv` already includes **`auc_mean_delta_vs_clinical_plus_tumor_size`** when the baseline arm is present.

## Radiomics baseline row (you fill from `metrics.json`)

Strongest / closest matched radiomics run on your side is typically under the radiomics workflow, e.g.:

`radiomics/outputs/peri5_multiphase_logreg/training/metrics.json`

**Do not assume metrics are comparable 1:1.** In the short write-up for the PR / issue, spell out at least:

| Topic | Tabular (#118) | Radiomics (example peri-5 multiphase) |
|--------|------------------|----------------------------------------|
| Cohort filter | ISPY2, `bilateral_filter: false` (YAML) | MAMA-MIA paths in radiomics config |
| Labels | `pcr_labels.csv` / `pcr` column | radiomics `labels` / `label_column` in that config |
| Splits | stratified k-fold CV in evaluator | train/test split in radiomics pipeline |
| Features | vessel + clinical tabular blocks | PyRadiomics / peri-tumoral extraction |

Pull **`auc_test`**, **`auc_train_cv`**, or whatever your radiomics `metrics.json` exposes, and add **one row** (or small table) next to the best tabular arm(s) from `issue118_baseline_arms_combined.csv`.

## Short answers for the issue checklist (after runs)

Use the merged CSV and subtype/strata plots under each `runs/<arm>/` as needed:

- **Which single vessel block helped most?** — Compare `auc_mean_delta_vs_clinical_plus_tumor_size` for morph-only, graph-only, kinematic-only rows (per `model_family`).
- **Does the full combo beat any single block?** — Compare `clinical_plus_tumor_size_plus_vessel_all` to the three single-block arms.
- **Is the gain over clinical + tumor_size meaningful?** — Judge delta magnitude and `auc_std`; tie to #117 if you report site-exclusive behavior elsewhere.
- **vs radiomics** — Same cohort label, different pipeline; keep the mismatch table above visible.

Refs #118
