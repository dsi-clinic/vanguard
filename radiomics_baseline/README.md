# Radiomics Baseline (pCR)

## Radiomics Baseline

This folder runs a 2-step pipeline:

1. **Extract** PyRadiomics features from the MRI volumes + masks to CSVs.
2. **Train** a ML models (logistic / random forest) on those features CSVs, optionally adding subtype, to predict pCR.

We split it like this because extraction is slow, but training is fast — so you can extract once and try many models.

---------------------------


## Files

- `radiomics_extract.py`  
  - **Inputs:** a directory of images, a directory of masks, a split file (or list of case IDs), and a radiomics parameter YAML (if you want non-default PyRadiomics settings).
  - Runs PyRadiomics on all patients in the split. Supports:
    - multiple DCE phases (comma-separated `--image-pattern`)
    - optional peritumoral shell (`--peri-radius-mm`)
    - tumor mask from NIfTI
  - **Outputs:** outputs to the chosen folder:
    - `features_train.csv` — one row per training case, columns = radiomic features + `case_id`
    - `features_test.csv` — one row per test case, columns = radiomic features + `case_id`
    - `train_labels_split.csv`
    - `test_labels_split.csv`


- `radiomics_train.py`
  - **Inputs:**
    - `--train-features`: path to the CSV produced by the extractor for the train split  
    - `--test-features`: path to the CSV produced by the extractor for the test split  
    - `--labels`: CSV that maps `case_id` to the label you want to predict
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
Below are example commands you can copy-paste and adjust to your paths. Run them from the project root (the folder that contains `radiomics_baseline/`).

# 1 Run feature extraction
**Base Code:**
python radiomics_baseline/radiomics_extract.py \
  --images-dir /path/to/images \
  --masks-dir /path/to/masks \
  --splits radiomics_baseline/splits.csv \
  --params radiomics_baseline/radiomics_params.yaml \
  --output-dir radiomics_baseline/experiments/extract_peri5_multiphase


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



# 2 Train the baseline model on the extracted features
**Example: Logistic, Subtype Included, 5 mm Peri, Multi-Phase**

  python radiomics_baseline/radiomics_train.py \
  --train-features radiomics_baseline/experiments/extract_peri5_multiphase/features_train.csv \
  --test-features  radiomics_baseline/experiments/extract_peri5_multiphase/features_test.csv \
  --labels         radiomics_baseline/labels.csv \
  --output         radiomics_baseline/experiments/train_logreg_subtype_peri5_multiphase \
  --classifier     logistic \
  --include-subtype

**Example: Random Forest, 5 mm Peri, Multi-Phase**
  python radiomics_baseline/radiomics_train.py \
  --train-features radiomics_baseline/experiments/extract_peri5_multiphase/features_train.csv \
  --test-features  radiomics_baseline/experiments/extract_peri5_multiphase/features_test.csv \
  --labels         radiomics_baseline/labels.csv \
  --output         radiomics_baseline/experiments/train_rf_peri5_multiphase \
  --classifier     rf \
  --rf-n-estimators 500 \
  --rf-max-depth 8

