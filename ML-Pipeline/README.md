
# ML Pipeline - PCR Prediction

## Overview
This project implements a machine learning pipeline for pCR (pathologic complete response) prediction using `pcr_prediction.py`.

## What It Does
The `pcr_prediction.py` script trains and evaluates machine learning models to predict pCR outcomes based on input parameters and experimental conditions.

## Goals
- Predict pCR outcomes and identify key feature importance factors
- Optimize parameters based on predictive models
- Reduce experimental iterations through data-driven modeling approaches

## Requirements
- Python 3.8+
- All required Python packages are installed via the project’s `environment.yml`

## Running the Code

```bash
python pcr_prediction.py [options]
```


### Command Line Arguments
- `--feature-dir DIR` - Directory of per-case JSON feature files (one <case_id>.json per case)
- `--labels FILE/DIR` - CSV file or directory of per-case variant JSONs (labels)
- `--label-column NAME` - Binary label column to learn (e.g., `pcr`)
- `--id-column NAME` - ID column name in labels table (default: `case_id`)
- `--outdir DIR` - Output directory for features, engineered csv, model and metrics
- `--model {rf,lr}` - Model type: `rf` (RandomForest) or `lr` (LogisticRegression) (default: `rf`)
- `--test-size FLOAT` - Fraction of data to hold out as test set (default: `0.2`)
- `--val-size FLOAT` - Fraction of remaining data used for validation (default: `0.1`)
- `--random-state INT` - Random seed for reproducibility (default: `42`)
- `--random-baseline` - Run a random baseline on the same splits for comparison
- `--bootstrap-n INT` - If >0, bootstrap test metrics with N resamples to get 95% CIs
- `--plots` - Save ROC/PR curves and confusion matrix PNGs
- `--save-intermediate-checks` - Emit feature sanity ranges to CSV for pipeline verification
- `--delong` - Run DeLong test to compare model AUC vs random baseline
- `--ensemble-runs INT` - If >0, repeat train/val/test with different seeds to get AUC/AP distribution
- `--ensemble-hist` - Save histogram PNGs of ensemble AUC/AP if `--ensemble-runs>0`

### Example
```bash
python3 ML-Pipeline/pcr_prediction.py     --cow-feature-dir vessel_segmentations/processed_3D     --labels pcr_labels.csv     --label-column pcr     --id-column patient_id     --outdir out_pcr_rebecca     --model rf     --plots     --test-size 0.2     --val-size 0.2
```

## Output
- Confusion Matrix
- Top 20 Feature importance bar chart
- Feature importance CSV
- Model evaluation metrics (accuracy, precision, recall, F1)
- Visualizations
- Features CSV
- Features Engineered CSV


## Important Notes
- Ensure input feature files are valid JSON with consistent schema across cases
- Ensure labels CSV contains the specified `--id-column` and `--label-column`
- Models are cached in `--outdir` for reuse and inspection
- Training requires sufficient memory for large datasets; consider `--test-size` and `--val-size` for memory-constrained environments
- Use `--random-baseline` and `--delong` for statistical validation of model performance
- Use `--ensemble-runs` to assess model stability across different random seeds

