# Radiomics Baseline (pCR)

## Radiomics Baseline

This folder implements a two-stage radiomics baseline to predict pathologic
complete response (pCR) from breast DCE-MRI.

1. **Extract** PyRadiomics features from MRI volumes and tumor masks to CSVs.
2. **Train** machine-learning models (logistic regression, random forest,
   extreme gradient boosting) on those feature tables, optionally including
   subtype, to predict pCR.

We separate extraction and training because extraction is slow but training is
fast — you can extract once and reuse the same features across many models.

---


## Files

- `radiomics_extract.py`

Stage 1: run PyRadiomics on all patients in the split and write feature tables.

  - **Inputs - CLI arguments:**
      - Supports:
        - multiple DCE phases
        - optional peritumoral shell
        - tumor mask from NIfTI

    - `--images`  
      Root directory containing MRI volumes.
    - `--masks`  
      Root directory containing NIfTI tumor masks.
    - `--labels`  
      CSV with at least columns `patient_id,pcr[,subtype]`.
    - `--split`  
      CSV with at least a `patient_id` column and a train/test indicator.
    - `--output`  
      Output directory where feature tables and split label CSVs are written.
    - `--params`  
      PyRadiomics YAML configuration (bin width, resampling, feature classes).
    - `--image-pattern`  
      Comma-separated template(s) for image paths relative to `--images`, e.g.  
      `"{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz"`.
    - `--mask-pattern`  
      Template for mask paths relative to `--masks`, e.g. `"{pid}.nii.gz"`.
    - `--peri-radius-mm`  
      Optional peritumoral shell width in millimeters (0 = tumor only).
    - `--n-proc`  
      Number of worker processes.
  
  - **Outputs:** outputs to the chosen folder:
    - `features_train.csv` — one row per training case, columns = radiomic features + `patient_id`
    - `features_test.csv` — one row per test case, columns = radiomic features + `patient_id`
    - `train_labels_split.csv` - labels for training data
    - `test_labels_split.csv` - labels for testing data


- `radiomics_train.py`
  - **Inputs:**
    - `--train-features`: path to the CSV produced by the extractor for the train split  
    - `--test-features`: path to the CSV produced by the extractor for the test split  
    - `--labels`: CSV that maps `patient_id` to the label you want to predict
  - Reads the CSVs above + the master `labels.csv`, sanitizes to numeric, (optionally) appends subtype, and trains a model
  - **Outputs:**
    - `metrics.json`
    - `predictions.csv`
    - `roc_test.png`, `pr_curve.png`, `calibration_curve.png`
    - `model.pkl`

- `pyradiomics_params.yaml`  
  - PyRadiomics settings (bin width, resampling, feature classes).

- `labels.csv`  
  -  **Columns:** patient_id,pcr,subtype


---------------------------


## Example Code:
Below are example commands you can copy-paste and adjust to your paths.

# 1. Run feature extraction
**Example: 5 mm Peri, Multi-Phase (0001, 0002)**

python radiomics_baseline/radiomics_extract.py \
  --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --labels radiomics_baseline/labels.csv \
  --split  radiomics_baseline/splits_train_test_ready.csv \
  --output radiomics_baseline/experiments/extract_peri5_multiphase \
  --params radiomics_baseline/pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --peri-radius-mm 5 \
  --n-proc 8



# 2. Train the baseline model on the extracted features
**Example: Logistic, Subtype Included**
Additionally, with elastic net penalty, correlation pruning, SelectKBest,
and cross-validated grid search.

  python radiomics_baseline/radiomics_train.py \
      --train-features experiments/extract_peri5_multiphase/features_train.csv \
      --test-features  experiments/extract_peri5_multiphase/features_test.csv \
      --labels         labels.csv \
      --output         outputs/elasticnet_corr0.9_k50_cv5 \
      --classifier     logistic \
      --logreg-penalty elasticnet \
      --logreg-l1-ratio 0.5 \
      --corr-threshold 0.9 \
      --k-best         50 \
      --grid-search \
      --cv-folds       5 \
      --include-subtype

**Example: Random Forest**
  python radiomics_baseline/radiomics_train.py \
  --train-features radiomics_baseline/experiments/extract_peri5_multiphase/features_train.csv \
  --test-features  radiomics_baseline/experiments/extract_peri5_multiphase/features_test.csv \
  --labels         radiomics_baseline/labels.csv \
  --output         radiomics_baseline/experiments/train_rf_peri5_multiphase \
  --classifier     rf \
  --rf-n-estimators 500 \
  --rf-max-depth 8

**Example: XGBoost**
  python radiomics_baseline/radiomics_train.py \
    --train-features radiomics_baseline/experiments/extract_peri5_multiphase/features_train.csv \
    --test-features  radiomics_baseline/experiments/extract_peri5_multiphase/features_test.csv \
    --labels         radiomics_baseline/labels.csv \
    --output         radiomics_baseline/experiments/train_xgb_peri5_multiphase \
    --classifier     xgb \
    --corr-threshold 0.9 \
    --k-best         50 \
    --grid-search
