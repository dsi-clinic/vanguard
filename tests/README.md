# Tests

Unit and integration tests for the evaluation framework and related code.

## Purpose

These tests ensure correct behavior of:

- **Evaluator** – k-fold and group-stratified splits, train/test evaluation, backward compatibility
- **K-fold / group-stratified splits** – `create_group_stratified_kfold_splits`, site exclusivity, stratification
- **Selection** – `SampleSelectionCriteria`, `apply_selection_criteria`, YAML/CLI criteria building
- **Random baseline** – `compute_random_auc_distribution`, `empirical_p_value`, `z_score`
- **Clinic metadata** – Excel loading and metadata utilities (`src/utils/clinic_metadata.py`)
- **Skeleton 4D** – graph extraction / skeleton4d module behavior

## What's tested

| File | Coverage |
|------|----------|
| `test_evaluator_group_splits.py` | Evaluator with/without groups and stratify_labels; API compatibility |
| `test_group_stratified_kfold.py` | Group-stratified k-fold split creation and site exclusivity |
| `test_selection.py` | Sample selection criteria, apply_selection_criteria, YAML/args |
| `test_random_baseline.py` | Random AUC distribution, empirical p-value, z-score |
| `test_clinic_metadata.py` | Clinic metadata Excel loading and helpers |
| `test_skeleton4d.py` | Skeleton4d extraction logic |

## How to run

From the repository root with the project environment active (e.g. `micromamba activate vanguard`):

```bash
# Run all tests (ensure project root is on PYTHONPATH)
PYTHONPATH=. pytest tests/ -v

# Or install the package in development mode and run pytest
pip install -e .
pytest tests/ -v
```

To run only evaluation-related tests:

```bash
PYTHONPATH=. pytest tests/test_evaluator_group_splits.py tests/test_selection.py tests/test_random_baseline.py tests/test_group_stratified_kfold.py -v
```
