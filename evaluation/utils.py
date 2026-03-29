"""Utility functions for data handling and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Constants
MIN_CLASSES_FOR_BINARY = 2
MAX_CLASSES_FOR_BINARY = 2


def validate_inputs(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    case_ids: np.ndarray | pd.Series | None = None,
) -> None:
    """Validate input data for consistency.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        Feature matrix.
    y : np.ndarray | pd.Series
        Target labels (must be binary 0/1).
    case_ids : np.ndarray | pd.Series, optional
        Patient IDs for tracking.

    Raises:
    ------
    ValueError
        If lengths differ or y is not binary 0/1.
    """
    # Convert to numpy arrays for validation
    if isinstance(X, pd.DataFrame):
        X_array = X.to_numpy()
        n_samples = len(X)
    else:
        X_array = np.asarray(X)
        n_samples = X_array.shape[0]

    y_array = np.asarray(y)

    # Check shapes
    if len(y_array) != n_samples:
        raise ValueError(
            f"X and y must have same number of samples. "
            f"Got X: {n_samples}, y: {len(y_array)}"
        )

    # Check y is binary
    unique_labels = np.unique(y_array)
    if len(unique_labels) > MAX_CLASSES_FOR_BINARY:
        raise ValueError(
            f"y must be binary classification. Found {len(unique_labels)} unique labels: {unique_labels}"
        )

    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"y must contain only 0 and 1. Found labels: {unique_labels}")

    # Check case_ids if provided
    if case_ids is not None:
        case_ids_array = np.asarray(case_ids)
        if len(case_ids_array) != n_samples:
            raise ValueError(
                f"case_ids must have same length as X and y. "
                f"Got {len(case_ids_array)} expected {n_samples}"
            )


def align_data(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    case_ids: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Align features, labels, and case IDs to ensure consistent indexing.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame
        Feature matrix.
    y : np.ndarray | pd.Series
        Target labels.
    case_ids : np.ndarray | pd.Series, optional
        Case IDs.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        Aligned (X, y, case_ids) as numpy arrays.
    """
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X_array = X.to_numpy()
        index = X.index
    else:
        X_array = np.asarray(X)
        index = np.arange(len(X_array))

    y_array = np.asarray(y)

    # If y is a Series with index, align with X
    if isinstance(y, pd.Series) and isinstance(X, pd.DataFrame):
        y_array = y.loc[index].to_numpy()
    elif isinstance(y, pd.Series):
        # If X is array but y is Series, we can't align by index
        # Just use values
        y_array = y.to_numpy()

    # Handle case_ids
    if case_ids is not None:
        if isinstance(case_ids, pd.Series) and isinstance(X, pd.DataFrame):
            case_ids_array = case_ids.loc[index].to_numpy()
        else:
            case_ids_array = np.asarray(case_ids)
    else:
        case_ids_array = None

    return X_array, y_array, case_ids_array


def prepare_predictions_df(
    case_ids: np.ndarray | pd.Series | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    fold: int | None = None,
) -> pd.DataFrame:
    """Format predictions into a standardized DataFrame.

    Optional helper for building prediction tables used by the evaluator.
    Columns: case_id, y_true, y_pred, y_prob; optionally "fold" for k-fold.

    Parameters
    ----------
    case_ids : np.ndarray | pd.Series | None
        Patient IDs, or None to use integer indices.
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray
        Predicted probabilities for positive class.
    fold : int, optional
        Fold index (for k-fold results); if provided, adds "fold" column.

    Returns:
    -------
    pd.DataFrame
        Standardized predictions table.
    """
    if case_ids is None:
        case_ids = np.arange(len(y_true))

    data = {
        "case_id": np.asarray(case_ids),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_prob": np.asarray(y_prob),
    }

    if fold is not None:
        data["fold"] = fold

    return pd.DataFrame(data)
