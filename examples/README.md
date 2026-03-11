# Examples

Example scripts and sample data for running the evaluation pipeline.

## Purpose

This directory provides a runnable entry point to the centralized evaluation system: load data (synthetic, CSV, or Excel-driven), define a baseline model (random or logistic), and run k-fold or train/test evaluation with optional cohort selection.

## Contents

| File | Description |
|------|-------------|
| `baseline_model_example.py` | Main example script: runs random or logistic baseline with k-fold or train/test evaluation; supports CSV features/labels, Excel metadata, and selection criteria (datasets, sites, tumor types, laterality). |
| `example_features.csv` | Sample feature matrix (patient_id + numeric columns) for testing CSV-based runs. |
| `example_labels.csv` | Sample labels (patient_id, label, optional stratum) for testing CSV-based runs. |
| [README_CSV_EXAMPLE.md](README_CSV_EXAMPLE.md) | CSV format specification for `--features` and `--labels`. |

## How to run

**Random baseline with default synthetic data** (recommended for a quick check):

```bash
python examples/baseline_model_example.py --model random --output results/baseline_example
```

**With your own features and labels (CSV):**

```bash
python examples/baseline_model_example.py --model random \
  --features path/to/features.csv \
  --labels path/to/labels.csv \
  --output results/baseline_example
```

**With Excel metadata and dataset/site selection** (see main README §6.4):

```bash
python examples/baseline_model_example.py --model random \
  --excel-metadata path/to/clinical_and_imaging_info.xlsx \
  --datasets iSpy2 --output results/ispy2
```

**Logistic regression baseline** (with synthetic or provided data):

```bash
python examples/baseline_model_example.py --model logistic --output results/logistic_example
```

For more options (YAML config, unilateral-only, stratified export), see the main [README](../README.md) §6.4 (Evaluation Framework) and the Pipeline Workflow section. CSV column requirements are documented in [README_CSV_EXAMPLE.md](README_CSV_EXAMPLE.md).
