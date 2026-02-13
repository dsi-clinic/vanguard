"""Centralized evaluation system for model evaluation with k-fold cross-validation.

Terminology
-----------
- **k-fold cross-validation** : Splitting scheme where data is divided into k folds;
  each fold serves once as the validation set while the rest train. Spell "k-fold"
  with a hyphen in prose.
- **stratum** (plural **strata**) : A subgroup used for reporting (e.g. subtype,
  dataset). Per-stratum metrics are computed when predictions include a stratum
  column (e.g. "stratum" or "subtype").
- **group** : An entity that must not cross folds (e.g. site). Group-stratified
  k-fold ensures each group appears in only one validation fold.
- **stratify_labels** : Labels used to stratify splits so that class or subgroup
  distribution is approximately balanced across folds.

Naming conventions
------------------
- **create_** : Build and return objects or splits (e.g. create_kfold_splits,
  create_splits_from_excel).
- **compute_** : Return metrics or derived values (e.g. compute_binary_metrics,
  compute_random_auc_distribution).
- **plot_** : Generate and save a figure (e.g. plot_roc_curve,
  plot_random_auc_distribution).
- **validate_**, **align_**, **build_**, **generate_**, **save_**, **report_** :
  Used for validation, alignment, building composite keys, generating data,
  saving to disk, and printing reports.
- **run_** (examples only): Execute a full evaluation path (e.g. run_kfold,
  run_train_test).
"""

from evaluation.evaluator import (
    Evaluator,
    FoldResults,
    FoldSplit,
    KFoldResults,
    TrainTestResults,
)
from evaluation.kfold import create_splits_from_excel, export_splits_to_csv
from evaluation.random_baseline import (
    compute_random_auc_distribution,
    empirical_p_value,
    generate_random_probs,
    report_random_baseline,
    save_random_baseline_distribution,
    z_score,
)
from evaluation.visualizations import plot_random_auc_distribution

__all__ = [
    "Evaluator",
    "FoldResults",
    "FoldSplit",
    "KFoldResults",
    "TrainTestResults",
    "create_splits_from_excel",
    "export_splits_to_csv",
    "compute_random_auc_distribution",
    "empirical_p_value",
    "generate_random_probs",
    "plot_random_auc_distribution",
    "report_random_baseline",
    "save_random_baseline_distribution",
    "z_score",
]
