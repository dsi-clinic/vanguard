"""Centralized evaluation system for model evaluation with k-fold cross-validation."""

from evaluation.evaluator import (
    Evaluator,
    FoldSplit,
    FoldResults,
    KFoldResults,
    TrainTestResults,
)

__all__ = [
    "Evaluator",
    "FoldSplit",
    "FoldResults",
    "KFoldResults",
    "TrainTestResults",
]
