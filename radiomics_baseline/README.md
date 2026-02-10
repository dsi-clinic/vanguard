# Radiomics Baseline for pCR Prediction

This folder implements a radiomics-based baseline to predict pathologic complete response (pCR) from breast DCE-MRI.

## Overview

The radiomics pipeline has three stages:

1. **Extract**: Extract PyRadiomics features from MRI volumes and tumor masks
2. **Train**: Train machine learning models on extracted features
3. **Evaluate** *(optional)*: Use centralized evaluation framework for standardized metrics

We separate extraction and training because extraction is slow (~1 hour per experiment) but training is fast (~1 minute) — extract once and reuse features across many models.

---

## Directory Structure

```
radiomics_baseline/
├── scripts/
│   ├── radiomics_extract.py      # Stage 1: Feature extraction
│   ├── radiomics_train.py         # Stage 2: Model training
│   ├── run_ablations.py           # Run multiple experiments from config
│   ├── run_experiment.py          # Run single experiment
│   ├── eval_adapter.py            # Adapter for evaluation framework
│   └── test_eval_integration.py   # Test evaluation integration
├── configs/
│   ├── ablation_summary.csv       # Results from ablation studies
│   └── generated/                 # Auto-generated experiment configs
├── outputs/                       # Model outputs (gitignored)
│   ├── shared_extraction/         # Extracted features (reused)
│   └── peri5_multiphase_.../      # Individual model outputs
├── logs/                          # SLURM/experiment logs
├── figures/                       # Saved plots
├── labels.csv                     # Patient labels (pcr, tumor_subtype)
├── splits_train_test_ready.csv    # Train/test split assignments
├── pyradiomics_params.yaml        # PyRadiomics configuration
└── README.md                      # This file
```

---

## Stage 1: Feature Extraction

**Script:** `scripts/radiomics_extract.py`

Extracts PyRadiomics features from MRI volumes and tumor masks.

### Key Features
- Multiple DCE phases (e.g., 0001, 0002)
- Peritumoral shell extraction (2D ring or 3D isotropic)
- Parallel processing with `--n-jobs`
- Outputs reusable feature CSVs

### Usage

```bash
python scripts/radiomics_extract.py \
  --images /path/to/images \
  --masks /path/to/masks \
  --labels labels.csv \
  --splits splits_train_test_ready.csv \
  --output outputs/shared_extraction/my_features \
  --params pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern "{pid}.nii.gz" \
  --peri-radius-mm 5 \
  --peri-mode 2d \
  --n-jobs 8
```

### Key Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--images` | Root directory with MRI volumes | `/net/projects2/vanguard/MAMA-MIA-syn60868042/images` |
| `--masks` | Root directory with tumor masks | `/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert` |
| `--labels` | CSV with patient_id, pcr, tumor_subtype | `labels.csv` |
| `--splits` | CSV with patient_id, split (train/test) | `splits_train_test_ready.csv` |
| `--output` | Output directory for features | `outputs/shared_extraction/peri5_2d` |
| `--image-pattern` | Template for image paths | `"{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz"` |
| `--peri-radius-mm` | Peritumoral shell width (mm) | `5` (or `0` for tumor only) |
| `--peri-mode` | Shell type: `2d` (ring) or `3d` (isotropic) | `2d` |
| `--force-2d` | Force 2D feature extraction | *(flag)* |
| `--n-jobs` | Number of parallel workers | `8` |

### Outputs

```
outputs/shared_extraction/my_features/
├── features_train_final.csv    # Training features (patient_id as index)
├── features_test_final.csv     # Test features (patient_id as index)
├── train_labels_split.csv      # Training labels
└── test_labels_split.csv       # Test labels
```

---

## Stage 2: Model Training

**Script:** `scripts/radiomics_train.py`

Trains machine learning models on extracted features.

### Key Features
- Classifiers: Logistic Regression, Random Forest, XGBoost
- Feature selection: correlation pruning, SelectKBest
- Grid search with cross-validation
- Optional subtype inclusion as feature
- Generates metrics, plots, and saved model

### Usage

```bash
python scripts/radiomics_train.py \
  --train-features outputs/shared_extraction/my_features/features_train_final.csv \
  --test-features outputs/shared_extraction/my_features/features_test_final.csv \
  --labels labels.csv \
  --output outputs/my_model/training \
  --classifier rf \
  --corr-threshold 0.9 \
  --k-best 50 \
  --grid-search \
  --cv-folds 5 \
  --include-subtype
```

### Key Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--classifier` | Model type: logistic, rf, xgb | `rf` |
| `--corr-threshold` | Drop features with correlation > threshold | `0.9` |
| `--k-best` | Select top K features | `50` |
| `--grid-search` | Use GridSearchCV for hyperparameters | *(flag)* |
| `--cv-folds` | Number of CV folds for grid search | `5` |
| `--include-subtype` | Add tumor subtype as feature | *(flag)* |

#### Classifier-Specific Arguments

**Logistic Regression:**
- `--logreg-penalty`: `l1`, `l2`, `elasticnet`
- `--logreg-l1-ratio`: L1 ratio for elasticnet (0-1)

**Random Forest:**
- `--rf-n-estimators`: Number of trees (default: 100)
- `--rf-max-depth`: Max tree depth

**XGBoost:**
- Grid search automatically tunes hyperparameters

### Outputs

```
outputs/my_model/training/
├── metrics.json              # AUC, accuracy, sensitivity, specificity
├── predictions.csv           # Per-patient predictions
├── model.pkl                 # Trained model
├── roc_test.png             # ROC curve
├── pr_curve.png             # Precision-recall curve
├── calibration_curve.png    # Calibration plot
└── predictions_cv_train.csv # Cross-validation predictions
```

---

## Running Ablation Studies

**Script:** `scripts/run_ablations.py`

Run multiple experiments from a YAML config file.

### Example Config

```yaml
# configs/my_ablation.yaml
base_config:
  images: /path/to/images
  masks: /path/to/masks
  labels: labels.csv
  splits: splits_train_test_ready.csv

ablations:
  peri_mode: [2d, 3d]
  peri_radius_mm: [2.5, 5]
  classifier: [logistic, rf, xgb]
```

### Usage

```bash
python scripts/run_ablations.py \
  configs/my_ablation.yaml \
  --generated-dir configs/generated
```

This generates individual configs and runs all combinations.

### SLURM Batch Jobs

For parallel execution on a cluster:

```bash
# Submit 2D and 3D sweeps in parallel
sbatch scripts/slurm_sweep_peri2d.sh
sbatch scripts/slurm_sweep_peri3d.sh

# Check logs
tail -f logs/sweep_peri2d_*.out
```

Results are saved to `configs/ablation_summary.csv`.

---

## Stage 3: Evaluation Framework Integration (Optional)

Use the centralized evaluation framework for standardized metrics and plots.

### Quick Start

Add these imports to your workflow:

```python
from scripts.eval_adapter import (
    create_evaluator_from_radiomics_data,
    create_train_test_results,
    save_evaluation_results,
)
```

### Integration with radiomics_train.py

Add at the end of your `radiomics_train.py` script:

```python
# After training and predictions
evaluator = create_evaluator_from_radiomics_data(
    X_train=Xtr,
    y_train=ytr,
    patient_ids_train=Xtr_raw.index,
    model_name=f"radiomics_{args.classifier}",
)

results = create_train_test_results(
    y_true=yte,
    y_pred=pred_test,
    y_prob=proba_test,
    patient_ids=Xte_raw.index,
    model_name=f"radiomics_{args.classifier}",
    stratum=yte_df.get('tumor_subtype', None),  # Optional subgroup analysis
)

# Save with standardized format
eval_out_dir = out_dir.parent / f"{out_dir.name}_standardized"
save_evaluation_results(evaluator, results, eval_out_dir)
```

### What You Get

Using the evaluation framework provides:

- ✅ **Standardized metrics** - Consistent with other model systems
- ✅ **Styled visualizations** - Publication-ready plots
- ✅ **Subgroup analysis** - Automatic per-subtype metrics
- ✅ **Consistent format** - Easy comparison across models

### Output Structure

```
output_dir/
└── radiomics_rf/          # model_name
    └── test/              # run_name
        ├── metrics.json          # Standardized format
        ├── predictions.csv       # With patient_id, y_true, y_pred, y_prob, stratum
        ├── roc_curve.png        # Styled ROC plot
        ├── pr_curve.png         # Precision-recall curve
        └── calibration_curve.png # Calibration plot
```

### Testing

Test the integration:

```bash
cd scripts
python test_eval_integration.py
```

### Using Evaluation Framework Directly

You can also use `/evaluation/` framework functions directly:

```python
from evaluation import Evaluator
from evaluation.metrics import compute_binary_metrics

# Compute metrics
metrics = compute_binary_metrics(y_true, y_pred, y_prob)

# Create evaluator for k-fold CV
evaluator = Evaluator(X, y, patient_ids, model_name="radiomics_rf")
splits = evaluator.create_kfold_splits(n_splits=5, stratify=True)
# ... train on each fold and aggregate results
```

### Key Point

- ❌ No modifications to `/evaluation/` folder
- ✅ Use evaluation framework as a library (import and use)
- ✅ `scripts/eval_adapter.py` provides simple helper functions

---

## Complete Example Workflow

### 1. Extract Features (Once)

```bash
python scripts/radiomics_extract.py \
  --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --labels labels.csv \
  --splits splits_train_test_ready.csv \
  --output outputs/shared_extraction/peri5_2d_multiphase \
  --params pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern "{pid}.nii.gz" \
  --peri-radius-mm 5 \
  --peri-mode 2d \
  --n-jobs 8
```

### 2. Train Multiple Models (Reusing Features)

```bash
# Logistic Regression
python scripts/radiomics_train.py \
  --train-features outputs/shared_extraction/peri5_2d_multiphase/features_train_final.csv \
  --test-features outputs/shared_extraction/peri5_2d_multiphase/features_test_final.csv \
  --labels labels.csv \
  --output outputs/peri5_2d_logreg/training \
  --classifier logistic \
  --logreg-penalty elasticnet \
  --logreg-l1-ratio 0.5 \
  --corr-threshold 0.9 \
  --k-best 50 \
  --grid-search \
  --cv-folds 5 \
  --include-subtype

# Random Forest
python scripts/radiomics_train.py \
  --train-features outputs/shared_extraction/peri5_2d_multiphase/features_train_final.csv \
  --test-features outputs/shared_extraction/peri5_2d_multiphase/features_test_final.csv \
  --labels labels.csv \
  --output outputs/peri5_2d_rf/training \
  --classifier rf \
  --corr-threshold 0.9 \
  --k-best 50 \
  --grid-search \
  --cv-folds 5 \
  --include-subtype

# XGBoost
python scripts/radiomics_train.py \
  --train-features outputs/shared_extraction/peri5_2d_multiphase/features_train_final.csv \
  --test-features outputs/shared_extraction/peri5_2d_multiphase/features_test_final.csv \
  --labels labels.csv \
  --output outputs/peri5_2d_xgb/training \
  --classifier xgb \
  --corr-threshold 0.9 \
  --k-best 50 \
  --grid-search \
  --cv-folds 5 \
  --include-subtype
```

### 3. Compare Results

```bash
# View metrics from all models
for dir in outputs/peri5_2d_*/training; do
  echo "=== $(basename $(dirname $dir)) ==="
  python -c "import json; print('AUC:', json.load(open('$dir/metrics.json'))['auc_test'])"
done
```

---

## Data Files

### labels.csv

```csv
patient_id,pcr,tumor_subtype
ISPY2_123,1,luminal_a
DUKE_456,0,triple_negative
...
```

- **patient_id**: Unique identifier
- **pcr**: Binary outcome (0 = no pCR, 1 = pCR)
- **tumor_subtype**: Optional, for subgroup analysis

### splits_train_test_ready.csv

```csv
patient_id,split
ISPY2_123,train
DUKE_456,test
...
```

- **patient_id**: Must match labels.csv
- **split**: `train` or `test`

### pyradiomics_params.yaml

PyRadiomics configuration:
- Bin width, resampling settings
- Enabled feature classes (shape, firstorder, glcm, etc.)

---

## Recent Results

### Peritumor Sweep (2D vs 3D)

Best models from recent ablation study:

| Rank | Configuration | AUC |
|------|--------------|-----|
| 1 | 3D, force_2d=False, 2.5mm, RF | 0.6329 |
| 2 | 2D, force_2d=False, 2.5mm, RF | 0.6259 |
| 3 | 3D, force_2d=False, 2.5mm, Logistic | 0.6230 |
| 4 | 2D, force_2d=False, 5mm, RF | 0.6187 |
| 5 | 3D, force_2d=True, 2.5mm, RF | 0.6141 |

**Key findings:**
- 3D peritumor mode slightly outperforms 2D (avg AUC: 0.593 vs 0.587)
- 2.5mm peritumor radius better than 5mm
- Random Forest consistently achieves highest test AUC
- XGBoost shows overfitting (high train AUC, lower test AUC)

See `configs/ablation_summary.csv` for full results.

---

## Tips & Best Practices

### Feature Extraction
- **Reuse features**: Extract once, train many models
- **Parallel processing**: Use `--n-jobs 8` or more
- **Shared extraction**: Save to `outputs/shared_extraction/` for reuse

### Model Training
- **Feature selection**: Use `--corr-threshold 0.9 --k-best 50` to reduce overfitting
- **Grid search**: Add `--grid-search --cv-folds 5` for hyperparameter tuning
- **Subtype**: Include with `--include-subtype` if you have tumor subtype data

### Ablation Studies
- Use YAML configs for reproducibility
- Run parallel sweeps with SLURM on cluster
- Check `configs/ablation_summary.csv` for results

### Evaluation
- Use evaluation framework for standardized outputs
- Enables comparison with non-imaging baseline, deep learning models
- Automatic subgroup analysis by tumor subtype

---

## Troubleshooting

### Common Issues

**Q: Feature extraction is slow**
- Use `--n-jobs 8` or more for parallel processing
- Consider submitting SLURM jobs for large experiments

**Q: Model overfitting**
- Increase `--corr-threshold` (e.g., 0.95)
- Reduce `--k-best` (e.g., 30)
- Use `--grid-search` with proper CV

**Q: Evaluation framework import errors**
- Install seaborn: `micromamba install -n vanguard -c conda-forge seaborn`
- Ensure you're in the vanguard environment

**Q: Patient ID mismatch**
- Ensure patient_id in labels.csv matches features CSV index
- Check splits_train_test_ready.csv has all patients

---

## Dependencies

Installed in the `vanguard` conda environment:

- Python 3.11+
- numpy, pandas, scikit-learn
- pyradiomics
- xgboost
- matplotlib, seaborn
- joblib

---

## Related Documentation

- `/evaluation/` - Centralized evaluation framework
- `/examples/baseline_model_example.py` - Example using evaluation framework
- `scripts/eval_adapter.py` - Helper functions for radiomics integration
