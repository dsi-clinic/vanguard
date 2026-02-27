# Radiomics Baseline (Current Runbook)

This directory contains the radiomics baseline for pCR prediction.

Current recommended workflow is:
1. extract features once (cached, resumable),
2. run CV-focused training sweeps,
3. use centralized evaluator outputs (`metrics.json`, `cv/metrics.json`, `predictions.csv`, plots).

## Quick Start (Current)

## Environment
```bash
micromamba activate vanguard
cd /home/annawoodard/gt/vanguard/crew/amy/radiomics_baseline
```

## 1) Feature extraction (cached + resumable)
```bash
python scripts/radiomics_extract.py \
  --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --labels labels.csv \
  --splits splits_train_test_ready.csv \
  --output /net/projects2/vanguard/annawoodard/radiomics_baseline/outputs/shared_extraction/my_extract \
  --params pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern "{pid}.nii.gz" \
  --peri-radius-mm 5 \
  --peri-mode 3d \
  --n-jobs 8
```

## 2) CV-only training (recommended)
Use this when test split is not independent from CV.

```bash
python scripts/radiomics_train.py \
  --train-features /net/projects2/vanguard/annawoodard/radiomics_baseline/outputs/shared_extraction/my_extract/features_train_final.csv \
  --test-features /net/projects2/vanguard/annawoodard/radiomics_baseline/outputs/shared_extraction/my_extract/features_test_final.csv \
  --labels labels.csv \
  --output /net/projects2/vanguard/annawoodard/radiomics_baseline/outputs/my_model/training \
  --classifier logistic \
  --logreg-penalty elasticnet \
  --logreg-l1-ratio 0.5 \
  --corr-threshold 0.9 \
  --k-best 50 \
  --feature-selection kbest \
  --grid-search \
  --cv-folds 5 \
  --cv-only
```

## 3) Single YAML experiment
```bash
python scripts/run_experiment.py configs/exp_peri5_multiphase_logreg.yaml
```

## 4) Sweep from YAML grid
```bash
python scripts/run_ablations.py configs/sweep_feature_selection.yaml \
  --generated-dir configs/generated
```

## 5) Run ComBat matrix (12 runs)
```bash
bash scripts/run_combat_matrix.sh
```

This runs:
- `configs/sweep_combat_matrix_all_mrmr20.yaml`
- `configs/sweep_combat_matrix_all_kbest40.yaml`
- `configs/sweep_combat_matrix_kinsub_kbest50.yaml`

Each sweep evaluates harmonization modes:
- `none`
- `zscore_site`
- `combat_param`
- `combat_nonparam`

## Central Evaluator Outputs
`radiomics_train.py` writes centralized evaluator artifacts directly.

Typical output:
```text
outputs/<experiment>/training/
  metrics.json
  predictions.csv
  plots/roc_curve.png
  cv/metrics.json
  cv/metrics_per_fold.json
  cv/predictions.csv
  cv/plots/roc_curve.png
```

Use CV metrics for ranking:
- `auc_train_cv`
- `auc_train_cv_std`

## Harmonization / ComBat Flags
Available in `radiomics_train.py` and YAML `train.*`:
- `--harmonization-mode {none,zscore_site,combat_param,combat_nonparam}`
- `--harmonization-batch-col <labels column>` (default: `site`)
- `--harmonization-covariate <labels column>` (repeatable)
- `--cv-only`

Important:
- Harmonization is fold-safe in CV (fit on fold-train, apply to fold-val).
- Do not include `pcr` as harmonization covariate.

## Updated YAML Keys (train block)
```yaml
train:
  cv_folds: 5
  cv_only: true
  harmonization_mode: combat_param
  harmonization_batch_col: site
  harmonization_covariates:
    - tumor_subtype
```

## Caching + Resume Notes
Extraction uses per-patient checkpoint cache under `output/_checkpoint`.

Cache is invalidated by changes to extraction fingerprint inputs, including:
- image patterns,
- mask pattern,
- peri radius/mode,
- force-2d settings,
- non-scalar handling settings,
- params YAML,
- label override.

`run_ablations.py` now respects explicit `paths.extract_outdir` in base configs, so you can force reuse of an existing extraction cache.

## What Changed Since Older Workflow
Compared with older scripts/configs used before this branch:
- Central evaluator integration is native in `radiomics_train.py` (no adapter script workflow needed).
- CV-only mode was added (`--cv-only`) to avoid using non-independent test metrics.
- Fold-safe harmonization added (`zscore_site`, `combat_param`, `combat_nonparam`).
- `auc_train_cv_std` is now included in ablation summary outputs.
- Extraction checkpointing/resume is robust and intended for routine reuse.
- ComBat matrix configs and runner script were added for immediate harmonization experiments.

## Practical Guidance
- Prefer CV-first model ranking (`auc_train_cv`, then `auc_train_cv_std`).
- Keep extraction fixed while sweeping model/harmonization knobs.
- Run heavy compute through Slurm, not on head/login node.

## Troubleshooting
- `mrmr` missing: install `mrmr-selection` in env.
- `xgboost` missing: install `xgboost` if needed.
- Label mismatch errors: verify `patient_id` consistency across labels/features/splits.
- Empty class in fold: reduce `cv_folds` or adjust filtering.
