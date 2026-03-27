# Examples

This directory contains small example scripts and sample data for the shared evaluation framework.

These examples are not the main project pipeline. They are here so a new student can see the expected input format and output structure without needing the full breast MRI dataset.

## Contents

| File | Purpose |
|------|---------|
| `baseline_model_example.py` | Small end-to-end example that runs a random or logistic baseline through the evaluation framework |
| `example_features.csv` | Sample case-level feature table |
| `example_labels.csv` | Sample labels table |
| `README_CSV_EXAMPLE.md` | Required CSV columns for the sample example |

## Quick Check

```bash
micromamba activate vanguard
python examples/baseline_model_example.py --model random --output results/baseline_example
```

## Using Your Own CSV Files

```bash
python examples/baseline_model_example.py --model random \
  --features path/to/features.csv \
  --labels path/to/labels.csv \
  --output results/baseline_example
```

## Using Excel Metadata

```bash
python examples/baseline_model_example.py --model random \
  --excel-metadata path/to/clinical_and_imaging_info.xlsx \
  --datasets iSpy2 \
  --output results/ispy2
```

For more detail on the evaluation layer, see the main [README](../README.md) and [evaluation/README.md](../evaluation/README.md).
