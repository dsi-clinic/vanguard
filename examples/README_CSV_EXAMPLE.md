# Example CSV format for baseline_model_example.py

When running the baseline with `--features` and `--labels`, use CSVs in the following format.

## features.csv

- **patient_id** (optional): Unique identifier per row. If present, used to align with labels.
- **Remaining columns**: Numeric features only (one column per feature).

Example (`example_features.csv`):

```csv
patient_id,feature_1,feature_2,feature_3
patient_001,0.12,-0.45,1.23
patient_002,-0.89,0.67,-0.34
...
```

## labels.csv

- **patient_id**: Must match `patient_id` in features (used for inner join).
- **label**: Binary outcome (0 or 1), e.g. non-pCR / pCR.
- **stratum** or **subtype** (optional): Subgroup for validation reporting (e.g. tumor subtype). If present, the evaluator prints and saves overall AUC and per-stratum AUC.

Example (`example_labels.csv`):

```csv
patient_id,label,stratum
patient_001,0,HR+ HER2-
patient_002,1,HR+ HER2-
patient_003,0,HR- HER2+
...
```

## Run with example files

```bash
python examples/baseline_model_example.py --model random \
  --features examples/example_features.csv \
  --labels examples/example_labels.csv \
  --output results/baseline_example
```

With stratum in the labels, the script will print a validation summary (overall AUC and AUC per stratum) and save it in `metrics.json` under `validation_summary`.
