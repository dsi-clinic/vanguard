# Radiomics Baseline (pCR)

## Run
python radiomics_baseline/baseline_pcr_radiomics.py \
  --images /path/to/images \
  --masks  /path/to/masks \
  --labels radiomics_baseline/labels.csv \
  --split  radiomics_baseline/splits_train_test_ready_0000.csv \
  --output radiomics_baseline/outdir_rf_0000 \
  --params radiomics_baseline/pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0000.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --classifier rf \
  --rf-n-estimators 500 \
  --n-proc 8

## Parameters
- YAML sets: binWidth, normalize, resampledPixelSpacing=[1,1,1], enabled image types, feature classes.
- Choose DCE phase via `--image-pattern` (e.g., `_0001.nii.gz`).

## Outputs
- features_train.csv, features_test.csv
- predictions.csv, metrics.json
- roc_test.png, pr_curve.png, calibration_curve.png
- model.pkl

## Notes
- Train only on TRAIN split; evaluate on TEST.
- AUC_test > 0.5 is considered better-than-chance; otherwise metrics.json includes commentary.
- Parallel feature extraction via `--n-proc`.
