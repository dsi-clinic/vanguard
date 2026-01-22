"""Metric computation functions with extensible registry pattern."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

# Constants
MIN_CLASSES_FOR_BINARY = 2

# Metric registry - maps metric names to computation functions
METRIC_REGISTRY: dict[str, callable] = {}


def compute_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute AUC (ROC-AUC) metric.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels (not used for AUC, but kept for consistency)
    y_prob : np.ndarray, optional
        Predicted probabilities for positive class. Required for AUC.

    Returns:
    -------
    dict[str, float]
        Dictionary with 'auc' key

    Raises:
    ------
    ValueError
        If y_prob is None or if there are insufficient samples
    """
    if y_prob is None:
        raise ValueError("y_prob is required for AUC computation")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Check if we have both classes
    unique_labels = np.unique(y_true)
    if len(unique_labels) < MIN_CLASSES_FOR_BINARY:
        # Return NaN if only one class present
        return {"auc": float("nan")}

    try:
        auc = float(roc_auc_score(y_true, y_prob))
        return {"auc": auc}
    except ValueError:
        # Handle edge cases (e.g., all predictions same)
        return {"auc": float("nan")}


# Register AUC metric
METRIC_REGISTRY["auc"] = compute_auc


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics_to_compute: list[str] | None = None,
) -> dict[str, float]:
    """Compute all binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels
    y_prob : np.ndarray
        Predicted probabilities for positive class
    metrics_to_compute : list[str], optional
        List of metric names to compute. If None, computes all registered metrics.

    Returns:
    -------
    dict[str, float]
        Dictionary of computed metrics
    """
    if metrics_to_compute is None:
        metrics_to_compute = list(METRIC_REGISTRY.keys())

    results = {}
    for metric_name in metrics_to_compute:
        if metric_name in METRIC_REGISTRY:
            metric_func = METRIC_REGISTRY[metric_name]
            try:
                metric_result = metric_func(y_true, y_pred, y_prob)
                # Handle both scalar and dict returns
                if isinstance(metric_result, dict):
                    results.update(metric_result)
                else:
                    results[metric_name] = metric_result
            except Exception:
                # If metric computation fails, set to NaN
                results[metric_name] = float("nan")

    return results


def aggregate_fold_metrics(
    fold_metrics_list: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics across k-fold splits (mean ± std).

    Parameters
    ----------
    fold_metrics_list : list[dict[str, float]]
        List of metric dictionaries, one per fold

    Returns:
    -------
    dict[str, dict[str, float]]
        Dictionary with aggregated metrics. Each metric has:
        - "mean": mean across folds
        - "std": standard deviation across folds
    """
    if not fold_metrics_list:
        return {}

    # Get all metric names
    all_metric_names = set()
    for fold_metrics in fold_metrics_list:
        all_metric_names.update(fold_metrics.keys())

    aggregated = {}
    for metric_name in all_metric_names:
        # Extract values for this metric across folds
        values = []
        for fold_metrics in fold_metrics_list:
            if metric_name in fold_metrics:
                value = fold_metrics[metric_name]
                # Skip NaN values
                if not np.isnan(value):
                    values.append(value)

        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            aggregated[metric_name] = {"mean": mean_val, "std": std_val}
        else:
            # All NaN
            aggregated[metric_name] = {"mean": float("nan"), "std": float("nan")}

    return aggregated
