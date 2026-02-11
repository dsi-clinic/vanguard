"""Tests for Evaluator integration with group-stratified splits."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.evaluator import Evaluator


@pytest.fixture
def synthetic_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Generate synthetic data for evaluator tests."""
    n_samples = 30
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)
    patient_ids = np.array([f"patient_{i:04d}" for i in range(n_samples)])
    groups = np.array(["SiteA"] * 10 + ["SiteB"] * 10 + ["SiteC"] * 10)
    stratify_labels = np.array(["Type1"] * 15 + ["Type2"] * 15)

    return X, y, patient_ids, groups, stratify_labels


# API Compatibility Tests


def test_evaluator_backwards_compatible(synthetic_data) -> None:  # noqa: ANN001
    """Calling create_kfold_splits() without groups/stratify_labels uses existing behavior."""
    X, y, patient_ids, groups, stratify_labels = synthetic_data

    evaluator = Evaluator(
        X, y, patient_ids=patient_ids, model_name="test", random_state=42
    )

    # Standard call (no groups/stratify_labels) - should use existing behavior
    splits = evaluator.create_kfold_splits(n_splits=3, stratify=True, shuffle=True)

    n_splits = 3  # noqa: PLR2004
    assert len(splits) == n_splits
    for split in splits:
        assert hasattr(split, "fold_idx")
        assert hasattr(split, "train_indices")
        assert hasattr(split, "val_indices")


def test_evaluator_with_groups_only(synthetic_data) -> None:  # noqa: ANN001
    """Providing groups but not stratify_labels → should use standard stratified splitting."""
    X, y, patient_ids, groups, stratify_labels = synthetic_data

    evaluator = Evaluator(
        X, y, patient_ids=patient_ids, model_name="test", random_state=42
    )

    # If only groups provided (without stratify_labels), should fall back to standard behavior
    # Actually, based on implementation, if only groups provided without stratify_labels,
    # it should use standard stratified splitting on y
    splits = evaluator.create_kfold_splits(n_splits=3, groups=groups)

    n_splits = 3  # noqa: PLR2004
    assert len(splits) == n_splits


def test_evaluator_with_both(synthetic_data) -> None:  # noqa: ANN001
    """Providing both groups and stratify_labels → uses group-stratified splitting."""
    X, y, patient_ids, groups, stratify_labels = synthetic_data

    evaluator = Evaluator(
        X, y, patient_ids=patient_ids, model_name="test", random_state=42
    )

    splits = evaluator.create_kfold_splits(
        n_splits=3, groups=groups, stratify_labels=stratify_labels
    )

    n_splits = 3  # noqa: PLR2004
    assert len(splits) == n_splits

    # Verify site exclusivity
    from evaluation.kfold import validate_site_exclusivity

    # Convert splits to dict format for validation
    split_dicts = [
        {
            "fold_idx": split.fold_idx,
            "val_indices": split.val_indices,
        }
        for split in splits
    ]

    is_valid = validate_site_exclusivity(split_dicts, groups)
    assert is_valid, "Sites should not cross folds"


def test_evaluator_with_report(synthetic_data) -> None:  # noqa: ANN001
    """Providing return_report=True returns report dictionary."""
    X, y, patient_ids, groups, stratify_labels = synthetic_data

    evaluator = Evaluator(
        X, y, patient_ids=patient_ids, model_name="test", random_state=42
    )

    splits, report = evaluator.create_kfold_splits(
        n_splits=3,
        groups=groups,
        stratify_labels=stratify_labels,
        return_report=True,
    )

    n_splits = 3  # noqa: PLR2004
    assert len(splits) == n_splits
    assert isinstance(report, dict)
    assert "per_fold_site_counts" in report
    assert "per_fold_stratum_counts" in report
    assert "warnings" in report
    assert "infeasible_constraints" in report


def test_evaluator_stratify_labels_only(synthetic_data) -> None:  # noqa: ANN001
    """Providing only stratify_labels (no groups) → uses standard stratified splitting on those labels."""
    X, y, patient_ids, groups, stratify_labels = synthetic_data

    evaluator = Evaluator(
        X, y, patient_ids=patient_ids, model_name="test", random_state=42
    )

    splits = evaluator.create_kfold_splits(n_splits=3, stratify_labels=stratify_labels)

    n_splits = 3  # noqa: PLR2004
    assert len(splits) == n_splits
    # Should use stratify_labels for stratification instead of y
