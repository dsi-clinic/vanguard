# ML Pipeline - pCR Prediction

## Overview

This directory contains the config-driven pCR prediction pipeline.

Entrypoint:

```bash
python ml_pipeline/pcr_prediction.py --config ml_pipeline/config_pcr.yaml
```

## What the pipeline loads

- Vascular features from per-case morphometry JSON files in `data_paths.feature_dir`
- Clinical features from `data_paths.clinical_excel` when `use_clinical: true`
- Radiomics features from `data_paths.radiomics_csv` when `use_radiomics: true`

Each feature source is controlled by `feature_toggles` in `ml_pipeline/config_pcr.yaml`.

## Outputs

The pipeline writes outputs under `experiment_setup.base_outdir`, or `--outdir` if
provided:

- `features_complete.csv`
- `model_rf.pkl` or `model_lr.pkl`

## Notes

- Run from the repo root so imports resolve correctly.
- The config file is the runtime source of truth for paths and feature toggles.
