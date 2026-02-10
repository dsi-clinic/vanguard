"""Adapter utilities for using the centralized evaluation framework with radiomics.

This module provides helper functions to bridge between radiomics data formats
and the evaluation framework, without modifying either system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for evaluation imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import Evaluator, TrainTestResults  # noqa: E402
from evaluation.metrics import compute_binary_metrics  # noqa: E402


def create_evaluator_from_radiomics_data(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    patient_ids_train: np.ndarray | pd.Index,
    model_name: str = "radiomics_model",
    random_state: int = 42,
) -> Evaluator:
    """Create an Evaluator instance from radiomics training data.

    Parameters
    ----------
    X_train : np.ndarray | pd.DataFrame
        Training features
    y_train : np.ndarray | pd.Series
        Training labels
    patient_ids_train : np.ndarray | pd.Index
        Patient IDs
    model_name : str
        Name for the model (used in output paths)
    random_state : int
        Random seed

    Returns
    -------
    Evaluator
        Configured evaluator instance
    """
    return Evaluator(
        X=X_train,
        y=y_train,
        patient_ids=patient_ids_train,
        model_name=model_name,
        random_state=random_state,
    )


def create_train_test_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray | pd.Index,
    model_name: str = "radiomics_model",
    stratum: np.ndarray | None = None,
) -> TrainTestResults:
    """Create TrainTestResults from radiomics predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities
    patient_ids : np.ndarray | pd.Index
        Patient IDs
    model_name : str
        Model name
    stratum : np.ndarray | None
        Optional stratum (e.g., tumor_subtype) for subgroup analysis

    Returns
    -------
    TrainTestResults
        Results object that can be saved with evaluator.save_results()
    """
    # Compute metrics using evaluation framework
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    # Create predictions DataFrame
    predictions = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    )

    if stratum is not None:
        predictions["stratum"] = stratum

    return TrainTestResults(
        metrics=metrics,
        predictions=predictions,
        model_name=model_name,
        run_name="test",
    )


def save_evaluation_results(
    evaluator: Evaluator,
    results: TrainTestResults,
    output_dir: Path,
) -> None:
    """Save evaluation results using the centralized framework.

    This generates:
    - metrics.json
    - predictions.csv
    - roc_curve.png
    - pr_curve.png
    - calibration_curve.png

    Parameters
    ----------
    evaluator : Evaluator
        Evaluator instance
    results : TrainTestResults
        Results to save
    output_dir : Path
        Output directory
    """
    evaluator.save_results(results, output_dir)
    print(f"Evaluation results saved to: {output_dir}")
