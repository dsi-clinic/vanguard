# Issue #118 — Tabular baseline arms vs radiomics (oriented)

## What this delivers

A reproducible rerun of the **five vessel baseline arms** from the issue description, using **one** `run_ablation_matrix.py` job with:

- `model_families: [lr, xgb]`
- `split_mode_matrix: [{ name: cv, use_group_split: false }]`
- per-family overrides (nested tuning on for `lr`, off for `xgb`)

Arms:

1. `clinical + tumor_size` (baseline for deltas)
2. `clinical + tumor_size + morph`
3. `clinical + tumor_size + graph`
4. `clinical + tumor_size + kinematic`
5. `clinical + tumor_size + morph + graph + kinematic`

Run IDs look like `clinical_plus_tumor_size__lr__cv` and `clinical_plus_tumor_size__xgb__cv`.

## Run on the cluster

From the **repository root**:

```bash
cd ~/vanguard
sbatch slurm/submit_issue118_baseline_arms.slurm
```

Optional overrides:

```bash
CONFIG=configs/issue118_baseline_arms.yaml \
OUTDIR=experiments/issue118_baseline_arms_run2 \
sbatch slurm/submit_issue118_baseline_arms.slurm
```

Logs: `logs/issue118-baseline-<jobid>.out`

## Outputs

- `OUTDIR/ablation_summary.csv` — includes **`auc_std`** and **`ap_mean` / `ap_std`** when AP is defined on each fold
- `OUTDIR/runs/<run_name>/` — per-run `metrics.json`, predictions, plots
- `baseline_run_name: clinical_plus_tumor_size__lr__cv` fills delta columns vs that row

## Radiomics baseline row

Use your strongest matched radiomics `metrics.json` (e.g. under `radiomics/outputs/.../training/`). Document cohort, splits, labels, and preprocessing differences vs this tabular CV setup.

Refs #118
