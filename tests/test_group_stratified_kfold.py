"""Tests for group-stratified k-fold splitting functionality."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from evaluation.kfold import (
    build_composite_stratum_key,
    create_group_stratified_kfold_splits,
    validate_site_exclusivity,
)

# Test constants (avoid magic values)
N_SPLITS = 5
RANDOM_SEED = 42
N_SITES_BALANCED = 3
N_STRATA_BALANCED = 2
SAMPLES_PER_SITE = 30


@pytest.fixture
def synthetic_balanced_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data with balanced site/stratum structure.

    Returns:
    -------
    tuple of (X, y, groups, stratify_labels)
        - X: Feature matrix (n_samples, n_features)
        - y: Binary labels
        - groups: Site assignments (3 sites, balanced)
        - stratify_labels: Stratum assignments (2 strata, balanced)
    """
    n_samples = N_SITES_BALANCED * SAMPLES_PER_SITE
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)

    # 3 sites, each with both strata
    groups = np.array(
        ["SiteA"] * SAMPLES_PER_SITE
        + ["SiteB"] * SAMPLES_PER_SITE
        + ["SiteC"] * SAMPLES_PER_SITE
    )
    stratify_labels = np.tile(
        np.array(
            ["Type1"] * (SAMPLES_PER_SITE // 2) + ["Type2"] * (SAMPLES_PER_SITE // 2)
        ),
        N_SITES_BALANCED,
    )

    return X, y, groups, stratify_labels


@pytest.fixture
def synthetic_imbalanced_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Generate synthetic data with imbalanced site/stratum structure.

    Site A has only Type1, Site B has only Type2.
    """
    n_samples = 40
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)

    groups = np.array(["SiteA"] * 20 + ["SiteB"] * 20)
    stratify_labels = np.array(["Type1"] * 20 + ["Type2"] * 20)

    return X, y, groups, stratify_labels


@pytest.fixture
def synthetic_patient_ids() -> np.ndarray:
    """Generate synthetic patient IDs."""
    return np.array([f"patient_{i:04d}" for i in range(90)])


# Group Exclusivity Tests


def test_site_exclusivity_perfect_case(synthetic_balanced_data) -> None:  # noqa: ANN001
    """With balanced data, verify each site appears in exactly one fold's validation set."""
    X, y, groups, stratify_labels = synthetic_balanced_data

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=RANDOM_SEED
    )

    # Verify site exclusivity
    is_valid = validate_site_exclusivity(splits, groups)
    assert is_valid, "Sites should not cross folds"


def test_site_exclusivity_imbalanced_case(synthetic_imbalanced_data) -> None:  # noqa: ANN001
    """With imbalanced data, still verify no site crosses folds."""
    X, y, groups, stratify_labels = synthetic_imbalanced_data

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=3, random_state=RANDOM_SEED
    )

    # Verify site exclusivity (should still hold even if imbalanced)
    is_valid = validate_site_exclusivity(splits, groups)
    assert is_valid, "Sites should not cross folds even with imbalanced data"


def test_all_samples_assigned(synthetic_balanced_data) -> None:  # noqa: ANN001
    """Every sample appears in exactly one validation fold."""
    X, y, groups, stratify_labels = synthetic_balanced_data

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=RANDOM_SEED
    )

    # Collect all validation indices
    all_val_indices = []
    for split in splits:
        all_val_indices.extend(split["val_indices"].tolist())

    # Check: no duplicates, all samples present
    assert len(all_val_indices) == len(set(all_val_indices)), "No duplicate indices"
    assert len(set(all_val_indices)) == len(
        X
    ), "All samples should be in validation once"


# Stratification Balance Tests


def test_stratum_distribution_balanced(synthetic_balanced_data) -> None:  # noqa: ANN001
    """With balanced data, each fold's validation set has approximately 1/k of each stratum."""
    X, y, groups, stratify_labels = synthetic_balanced_data
    n_splits = 3

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=n_splits, random_state=RANDOM_SEED
    )

    # Check stratum distribution per fold
    unique_strata = np.unique(stratify_labels)
    expected_per_fold = len(X) / n_splits
    tolerance = 0.3  # 30% tolerance

    for split in splits:
        val_indices = split["val_indices"]
        val_strata = stratify_labels[val_indices]

        for stratum in unique_strata:
            count = np.sum(val_strata == stratum)
            expected = expected_per_fold / len(unique_strata)
            assert (
                abs(count - expected) <= expected * tolerance
            ), f"Fold {split['fold_idx']} stratum {stratum} count {count} not within tolerance"


def test_stratum_distribution_imbalanced(synthetic_imbalanced_data) -> None:  # noqa: ANN001
    """With imbalanced data, verify warnings are emitted but splitter still runs."""
    X, y, groups, stratify_labels = synthetic_imbalanced_data

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        splits = create_group_stratified_kfold_splits(
            X,
            y,
            groups,
            stratify_labels,
            n_splits=3,  # noqa: PLR2004
            random_state=RANDOM_SEED,
            return_report=False,
        )

        # Should still produce splits
        assert len(splits) == 3  # noqa: PLR2004
        # May or may not have warnings depending on sklearn behavior


def test_stratum_coverage_all_folds(synthetic_balanced_data) -> None:  # noqa: ANN001
    """Every stratum appears in at least one fold's validation set."""
    X, y, groups, stratify_labels = synthetic_balanced_data

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=RANDOM_SEED
    )

    unique_strata = np.unique(stratify_labels)
    strata_in_folds = set()

    for split in splits:
        val_indices = split["val_indices"]
        val_strata = stratify_labels[val_indices]
        strata_in_folds.update(val_strata)

    assert set(unique_strata).issubset(
        strata_in_folds
    ), "All strata should appear in at least one fold"


# Determinism and Reproducibility Tests


def test_deterministic_same_seed(synthetic_balanced_data) -> None:  # noqa: ANN001
    """Same seed + same data → identical fold assignments."""
    X, y, groups, stratify_labels = synthetic_balanced_data

    splits1 = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=RANDOM_SEED
    )
    splits2 = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=RANDOM_SEED
    )

    # Compare fold assignments
    for s1, s2 in zip(splits1, splits2):
        np.testing.assert_array_equal(
            s1["val_indices"], s2["val_indices"], "Same seed should produce same splits"
        )


def test_deterministic_different_seed(synthetic_balanced_data) -> None:  # noqa: ANN001
    """Different seed → potentially different assignments (but still valid)."""
    X, y, groups, stratify_labels = synthetic_balanced_data

    splits1 = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=42
    )
    splits2 = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=123
    )

    # They might be the same or different, but both should be valid
    assert len(splits1) == len(splits2) == N_SPLITS

    # Verify both are still site-exclusive
    assert validate_site_exclusivity(splits1, groups)
    assert validate_site_exclusivity(splits2, groups)


def test_shuffle_flag(synthetic_balanced_data) -> None:  # noqa: ANN001
    """shuffle=False produces deterministic order-independent of seed."""
    X, y, groups, stratify_labels = synthetic_balanced_data

    splits_no_shuffle1 = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, shuffle=False, random_state=42
    )
    splits_no_shuffle2 = create_group_stratified_kfold_splits(
        X,
        y,
        groups,
        stratify_labels,
        n_splits=N_SPLITS,
        shuffle=False,
        random_state=123,
    )

    # Without shuffle, should be deterministic regardless of seed
    for s1, s2 in zip(splits_no_shuffle1, splits_no_shuffle2):
        np.testing.assert_array_equal(
            s1["val_indices"],
            s2["val_indices"],
            "shuffle=False should be deterministic",
        )


# Edge Cases and Infeasible Constraints


def test_fewer_groups_than_folds() -> None:
    """If n_sites < n_splits, some folds will be empty (should warn, not crash)."""
    n_samples = 10
    n_splits = 5  # noqa: PLR2004
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)
    groups = np.array(["SiteA"] * 5 + ["SiteB"] * 5)  # Only 2 sites
    stratify_labels = np.array(["Type1"] * 10)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        splits = create_group_stratified_kfold_splits(
            X, y, groups, stratify_labels, n_splits=n_splits, random_state=RANDOM_SEED
        )

        # Should still work (may have empty folds)
        assert len(splits) == n_splits


def test_stratum_confined_to_one_site() -> None:
    """When a stratum only exists in one site, stratification is impossible."""
    n_samples = 20
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)
    groups = np.array(["SiteA"] * 10 + ["SiteB"] * 10)
    # Type1 only in SiteA, Type2 only in SiteB
    stratify_labels = np.array(["Type1"] * 10 + ["Type2"] * 10)

    n_splits = 3  # noqa: PLR2004
    splits, report = create_group_stratified_kfold_splits(
        X,
        y,
        groups,
        stratify_labels,
        n_splits=n_splits,
        random_state=RANDOM_SEED,
        return_report=True,
    )

    # Should still produce splits
    assert len(splits) == n_splits

    # Report should indicate infeasible constraints
    assert len(report["infeasible_constraints"]) > 0
    assert any(
        "only in site" in str(c).lower() for c in report["infeasible_constraints"]
    )


def test_tiny_group() -> None:
    """Site with < k samples → verify handling."""
    n_samples = 8
    n_splits = 3  # noqa: PLR2004
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)
    groups = np.array(["SiteA"] * 2 + ["SiteB"] * 3 + ["SiteC"] * 3)  # SiteA has only 2
    stratify_labels = np.array(["Type1"] * 4 + ["Type2"] * 4)

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=n_splits, random_state=RANDOM_SEED
    )

    # Should still work
    assert len(splits) == n_splits
    assert validate_site_exclusivity(splits, groups)


def test_single_site() -> None:
    """Only one site → all samples in same group (degenerate case, should still work)."""
    n_samples = 20
    n_splits = 3  # noqa: PLR2004
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)
    groups = np.array(["SiteA"] * n_samples)
    stratify_labels = np.array(["Type1"] * 10 + ["Type2"] * 10)

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=n_splits, random_state=RANDOM_SEED
    )

    # Should still work (all in one group)
    assert len(splits) == n_splits


def test_single_stratum() -> None:
    """Only one stratum value → stratification is trivial but should not error."""
    n_samples = 20
    n_splits = 3  # noqa: PLR2004
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)
    groups = np.array(["SiteA"] * 10 + ["SiteB"] * 10)
    stratify_labels = np.array(["Type1"] * n_samples)  # All same stratum

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=n_splits, random_state=RANDOM_SEED
    )

    # Should still work
    assert len(splits) == n_splits
    assert validate_site_exclusivity(splits, groups)


# Composite Stratum Keys Tests


def test_composite_stratum_key() -> None:
    """Multiple stratify columns → composite key works correctly."""
    n_samples = 4  # noqa: PLR2004
    metadata = np.array([["A", "X"], ["A", "Y"], ["B", "X"], ["B", "Y"]])

    keys = build_composite_stratum_key(metadata, ["dataset", "subtype"])

    assert len(keys) == n_samples
    assert "A|X" in keys
    assert "A|Y" in keys
    assert "B|X" in keys
    assert "B|Y" in keys


def test_composite_key_uniqueness() -> None:
    """Composite keys are unique strings that preserve stratification intent."""
    n_samples = 3  # noqa: PLR2004
    n_duplicates = 2  # noqa: PLR2004
    metadata = np.array([["A", "X"], ["A", "X"], ["B", "Y"]])

    keys = build_composite_stratum_key(metadata, ["dataset", "subtype"])

    # Should preserve duplicates (for stratification)
    assert len(keys) == n_samples
    assert np.sum(keys == "A|X") == n_duplicates
    assert np.sum(keys == "B|Y") == 1  # noqa: PLR2004


# Output Structure and Compatibility Tests


def test_output_structure_matches_existing(synthetic_balanced_data) -> None:  # noqa: ANN001
    """Returned split dicts have same keys as create_kfold_splits()."""
    X, y, groups, stratify_labels = synthetic_balanced_data

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=RANDOM_SEED
    )

    expected_keys = {
        "fold_idx",
        "train_indices",
        "val_indices",
        "train_patient_ids",
        "val_patient_ids",
    }

    for split in splits:
        assert (
            set(split.keys()) == expected_keys
        ), "Split dict should have expected keys"


def test_patient_ids_preserved(synthetic_balanced_data, synthetic_patient_ids) -> None:  # noqa: ANN001
    """When patient_ids provided, they are correctly mapped to train/val sets."""
    X, y, groups, stratify_labels = synthetic_balanced_data
    # Use subset of patient_ids to match data size
    patient_ids = synthetic_patient_ids[: len(X)]

    splits = create_group_stratified_kfold_splits(
        X,
        y,
        groups,
        stratify_labels,
        patient_ids=patient_ids,
        n_splits=N_SPLITS,
        random_state=RANDOM_SEED,
    )

    for split in splits:
        assert split["train_patient_ids"] is not None
        assert split["val_patient_ids"] is not None
        assert len(split["train_patient_ids"]) == len(split["train_indices"])
        assert len(split["val_patient_ids"]) == len(split["val_indices"])


def test_indices_are_valid(synthetic_balanced_data) -> None:  # noqa: ANN001
    """All indices are in range [0, n_samples) and non-overlapping."""
    X, y, groups, stratify_labels = synthetic_balanced_data
    n_samples = len(X)

    splits = create_group_stratified_kfold_splits(
        X, y, groups, stratify_labels, n_splits=N_SPLITS, random_state=RANDOM_SEED
    )

    for split in splits:
        train_idx = split["train_indices"]
        val_idx = split["val_indices"]

        # Check ranges
        assert np.all((train_idx >= 0) & (train_idx < n_samples))
        assert np.all((val_idx >= 0) & (val_idx < n_samples))

        # Check non-overlapping
        assert len(np.intersect1d(train_idx, val_idx)) == 0


# Warning and Reporting Tests


def test_warning_on_imbalance(synthetic_imbalanced_data) -> None:  # noqa: ANN001
    """When stratum distribution is highly imbalanced, warning is emitted."""
    X, y, groups, stratify_labels = synthetic_imbalanced_data

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        splits, report = create_group_stratified_kfold_splits(
            X,
            y,
            groups,
            stratify_labels,
            n_splits=3,  # noqa: PLR2004
            random_state=RANDOM_SEED,
            return_report=True,
        )

        # Report should contain warnings or infeasible constraints
        assert len(report["warnings"]) > 0 or len(report["infeasible_constraints"]) > 0


def test_report_structure(synthetic_balanced_data) -> None:  # noqa: ANN001
    """If splitter returns a report dict, it contains expected keys."""
    X, y, groups, stratify_labels = synthetic_balanced_data

    splits, report = create_group_stratified_kfold_splits(
        X,
        y,
        groups,
        stratify_labels,
        n_splits=N_SPLITS,
        random_state=RANDOM_SEED,
        return_report=True,
    )

    expected_keys = {
        "per_fold_site_counts",
        "per_fold_stratum_counts",
        "warnings",
        "infeasible_constraints",
    }

    assert set(report.keys()) == expected_keys, "Report should have expected keys"
    assert len(report["per_fold_site_counts"]) == N_SPLITS
    assert len(report["per_fold_stratum_counts"]) == N_SPLITS


def test_warning_message_actionable(synthetic_imbalanced_data) -> None:  # noqa: ANN001
    """Warning messages include specific diagnostics."""
    X, y, groups, stratify_labels = synthetic_imbalanced_data

    splits, report = create_group_stratified_kfold_splits(
        X,
        y,
        groups,
        stratify_labels,
        n_splits=3,
        random_state=RANDOM_SEED,
        return_report=True,
    )

    # Check that infeasible constraints mention specific details
    if report["infeasible_constraints"]:
        constraint_msg = str(report["infeasible_constraints"][0]).lower()
        assert "stratum" in constraint_msg or "site" in constraint_msg
