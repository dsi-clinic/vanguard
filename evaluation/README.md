# Evaluation

Centralized evaluation system for model evaluation with k-fold cross-validation, cohort selection, random baseline comparison, and metrics/visualizations.

## Purpose

This package provides:

- **K-fold cross-validation** with optional group-stratified splits (e.g. site-exclusive folds) and stratification by labels or strata
- **Cohort selection** to restrict evaluation to specific datasets, sites, tumor types, or laterality (unilateral/bilateral)
- **Random baseline** distribution (AUC under random predictions) and empirical p-values
- **Metrics** (AUC, precision, recall, F1, etc.) and **visualizations** (ROC curves, random AUC distribution)

All modules are **library code** (import only). There are no script entry points in this directory; use [`examples/baseline_model_example.py`](../examples/baseline_model_example.py) to run a full evaluation.

## Directory layout

| File | Description |
|------|-------------|
| `evaluator.py` | Main `Evaluator` class: creates splits, runs k-fold or train/test evaluation, aggregates results |
| `kfold.py` | Split creation: `create_kfold_splits`, `create_group_stratified_kfold_splits`, `create_splits_from_excel`, `export_splits_to_csv` |
| `selection.py` | Cohort selection: `SampleSelectionCriteria`, `apply_selection_criteria`, `build_selection_criteria_from_args`, `load_selection_criteria_from_yaml` |
| `metrics.py` | Metric computation: `compute_auc`, `compute_binary_metrics`, `compute_metrics_by_group`, `aggregate_fold_metrics` (extensible registry) |
| `random_baseline.py` | Random baseline: `compute_random_auc_distribution`, `empirical_p_value`, `generate_random_probs`, `report_random_baseline`, `save_random_baseline_distribution`, `z_score` |
| `visualizations.py` | Plotting: `plot_roc_curve`, `plot_random_auc_distribution`, `setup_figure`, `save_figure` |
| `utils.py` | Data handling: `validate_inputs`, `align_data`, `prepare_predictions_df` |
| `types.py` | Data types: `FoldResults`, `KFoldResults`, `TrainTestResults`, `FoldSplit` |
| `__init__.py` | Re-exports public API; see docstring for terminology and naming conventions |

## How to use

**Import from the package** (from repo root or with package installed):

```python
from evaluation import (
    Evaluator,
    create_splits_from_excel,
    export_splits_to_csv,
    SampleSelectionCriteria,
    apply_selection_criteria,
    build_selection_criteria_from_args,
    load_selection_criteria_from_yaml,
    compute_random_auc_distribution,
    empirical_p_value,
    report_random_baseline,
    plot_random_auc_distribution,
)
```

**Run a full evaluation** (synthetic or CSV/Excel data):

```bash
python examples/baseline_model_example.py --model random --output results/example
```

See the main [README](../README.md) §6.4 (Evaluation Framework) and [examples/README.md](../examples/README.md) for dataset selection, Excel-driven splits, and more usage examples.
