"""Result and split types for the evaluation system."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FoldResults:
    """Results from a single k-fold fold."""

    fold_idx: int
    predictions: pd.DataFrame  # columns: case_id, y_true, y_pred, y_prob
    metrics: dict[str, float] | None = None  # Optional pre-computed metrics


@dataclass
class KFoldResults:
    """Aggregated results from k-fold cross-validation."""

    fold_metrics: list[dict[str, float]]
    aggregated_metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame  # columns: case_id, fold, y_true, y_pred, y_prob
    n_splits: int
    model_name: str
    run_name: str | None = None


@dataclass
class TrainTestResults:
    """Results from train/test evaluation."""

    metrics: dict[str, float]
    predictions: pd.DataFrame  # columns: case_id, y_true, y_pred, y_prob
    model_name: str
    run_name: str | None = None
