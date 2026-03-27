# Evaluation

This package is the shared evaluation layer for the repository. Its job is to make different models comparable by using the same data splits, the same metrics, and the same output format.

## What This Package Does

- creates train/validation folds
- supports group-aware splits, for example keeping all cases from one site together when needed
- filters cohorts by dataset, site, subtype, and laterality
- computes metrics such as ROC AUC
- saves predictions, summary metrics, and standard plots in a consistent layout

## What This Package Does Not Do

- feature engineering
- model training logic
- PyTorch datasets or dataloaders
- graph construction

## Common Terms

- `fold`
  - one train/validation split in cross-validation
- `group-aware split`
  - a split that keeps related samples together, for example all studies from the same site
- `prediction table`
  - a table with one row per patient and columns for the true label and model output

## When To Use `train_tabular.py`

Use `train_tabular.py` when:

- your inputs are patient-level feature tables
- you want the existing tabular pipeline
- you want to train the current elastic-net or tree-based models from config

## When To Use `evaluation/` Directly

Use `evaluation/` directly when:

- you are adding a new model family such as a GNN
- your model has its own training loop
- you still want to reuse the same folds, metrics, and saved outputs

In short:

- `train_tabular.py` is one concrete model pipeline
- `evaluation/` is the shared comparison framework behind it

## Minimal Output Contract

To use the evaluator, your model only needs to produce a patient-level prediction table with these columns:

- `patient_id`
- `y_true`
- `y_pred`
- `y_prob`

For multi-fold runs, each fold is stored as a `FoldResults` object. The framework then combines those fold results into one run-level summary and writes metrics, predictions, and plots to disk.

## Main Files

| File | Purpose |
|------|---------|
| `evaluator.py` | Main class for split creation, result aggregation, and output saving |
| `build_splits.py` | Shared helpers for creating configured train/validation folds |
| `kfold.py` | Lower-level fold generation utilities |
| `selection.py` | Cohort filtering by dataset, site, subtype, and laterality |
| `metrics.py` | Binary classification metrics |
| `visualizations.py` | ROC and precision-recall plotting helpers |
| `random_baseline.py` | Random-baseline comparison utilities |
| `utils.py` | Small helpers such as prediction-table preparation |
| `types.py` | Lightweight result containers used during aggregation |

## Reusing This For A New Model

A future GNN should keep its own training code in `train_gnn.py`, then call into `evaluation/` for the parts that should stay consistent across models.

Typical pattern:

1. Load patient IDs and labels.
2. Create an evaluator and ask it for splits.
3. For each split, train the model using your own code.
4. Convert validation predictions into the standard prediction table.
5. Aggregate the fold results and save them.

## Example Skeleton

```python
from evaluation import Evaluator, FoldResults
from evaluation.utils import prepare_predictions_df

patient_ids = patient_manifest_df["patient_id"]
labels = patient_manifest_df["label"]

evaluator = Evaluator(
    X=patient_manifest_df,
    y=labels,
    patient_ids=patient_ids,
    model_name="gnn_model",
    random_state=42,
)

splits = evaluator.create_kfold_splits(
    n_splits=5,
    groups=patient_manifest_df["site"].to_numpy(),
    stratify_labels=patient_manifest_df["tumor_subtype"].to_numpy(),
)

fold_results = []
for split in splits:
    y_true, y_pred, y_prob, fold_patient_ids = run_your_model_for_one_fold(split)
    pred_df = prepare_predictions_df(
        patient_ids=fold_patient_ids,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fold=split.fold_idx,
    )
    fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))

kfold_results = evaluator.aggregate_kfold_results(fold_results)
evaluator.save_results(kfold_results, output_dir)
```

## Related Files

- [`../train_tabular.py`](../train_tabular.py)
- [`../train_gnn.py`](../train_gnn.py)
- [`../configs/ispy2.yaml`](../configs/ispy2.yaml)
- [`../examples/baseline_model_example.py`](../examples/baseline_model_example.py)
