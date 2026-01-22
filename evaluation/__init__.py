"""Centralized evaluation system for model evaluation with k-fold cross-validation."""

from evaluation.evaluator import (
    Evaluator,
    FoldResults,
    FoldSplit,
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
