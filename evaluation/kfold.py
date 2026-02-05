"""K-fold cross-validation split generation and aggregation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


@dataclass
class SplitConfig:
    """Configuration for group-stratified k-fold splitting.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    random_state : int, default=42
        Random seed for reproducibility
    shuffle : bool, default=True
        Whether to shuffle data before splitting
    group_col : str, default="site"
        Column name or identifier for grouping (e.g., "site").
        Groups will not cross folds (site-exclusive splits).
    stratify_cols : list[str], default=["dataset"]
        List of column names to use for stratification.
        If multiple columns, they will be combined into a composite key.
    pinned_groups : dict[str, int] | None, default=None
        Optional mapping of group values to fold indices.
        If provided, these groups will be assigned to specific folds.
        Reserved for future use (not yet implemented).
    """

    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True
    group_col: str = "site"
    stratify_cols: list[str] | None = None
    pinned_groups: dict[str, int] | None = None

    def __post_init__(self: SplitConfig) -> None:
        """Set default stratify_cols if None."""
        if self.stratify_cols is None:
            self.stratify_cols = ["dataset"]


def create_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray | None = None,
    n_splits: int = 5,
    stratify: bool = True,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[dict[str, np.ndarray | int]]:
    """Create k-fold splits and return them as a list of dictionaries.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    patient_ids : np.ndarray, optional
        Patient IDs for tracking
    n_splits : int, default=5
        Number of folds
    stratify : bool, default=True
        Whether to use stratified k-fold (maintains class distribution)
    shuffle : bool, default=True
        Whether to shuffle data before splitting
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    -------
    list[dict[str, np.ndarray | int]]
        List of dictionaries, one per fold, each containing:
        - "fold_idx": fold number (0-indexed)
        - "train_indices": indices for training
        - "val_indices": indices for validation
        - "train_patient_ids": patient IDs for training (if available)
        - "val_patient_ids": patient IDs for validation (if available)
    """
    # Choose splitter
    # When shuffle=False, random_state has no effect and sklearn raises an error
    # So we only pass random_state when shuffle=True
    splitter_kwargs = {
        "n_splits": n_splits,
        "shuffle": shuffle,
    }
    if shuffle:
        splitter_kwargs["random_state"] = random_state

    if stratify:
        splitter = StratifiedKFold(**splitter_kwargs)
    else:
        splitter = KFold(**splitter_kwargs)

    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        split_dict = {
            "fold_idx": fold_idx,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }

        # Add patient IDs if available
        if patient_ids is not None:
            split_dict["train_patient_ids"] = patient_ids[train_idx]
            split_dict["val_patient_ids"] = patient_ids[val_idx]
        else:
            split_dict["train_patient_ids"] = None
            split_dict["val_patient_ids"] = None

        splits.append(split_dict)

    return splits


def build_composite_stratum_key(
    metadata_df: np.ndarray | None,
    stratify_cols: list[str] | None = None,
    separator: str = "|",
) -> np.ndarray:
    """Build composite stratum key from multiple columns.

    Parameters
    ----------
    metadata_df : np.ndarray | None
        Array of stratum values, shape (n_samples, n_stratify_cols).
        Each row corresponds to one sample, each column to one stratify column.
        If 1D array, treated as single column.
    stratify_cols : list[str] | None, optional
        List of column names (for reference/documentation).
        If None and metadata_df is 2D, inferred from number of columns.
    separator : str, default="|"
        Separator to use when combining multiple stratum values.

    Returns:
    -------
    np.ndarray
        Array of composite stratum keys, shape (n_samples,).
        Each element is a string combining all stratum values for that sample.

    Examples:
    --------
    >>> metadata = np.array([["A", "X"], ["A", "Y"], ["B", "X"]])
    >>> build_composite_stratum_key(metadata, ["dataset", "subtype"])
    array(['A|X', 'A|Y', 'B|X'], dtype='<U3')
    >>> single_col = np.array(["A", "B", "A"])
    >>> build_composite_stratum_key(single_col)
    array(['A', 'B', 'A'], dtype='<U1')
    """
    if metadata_df is None or len(metadata_df) == 0:
        raise ValueError("metadata_df cannot be None or empty")

    metadata_array = np.asarray(metadata_df)

    # Handle 1D case (single column)
    if metadata_array.ndim == 1:
        return np.asarray([str(val) for val in metadata_array])

    # Handle 2D case (multiple columns)
    n_samples, n_cols = metadata_array.shape

    if n_cols == 1:
        # Single column in 2D array: return as 1D string array
        return np.asarray([str(val) for val in metadata_array[:, 0]])

    # Multiple columns: combine with separator
    composite_keys = []
    for row in metadata_array:
        key_parts = [str(val) for val in row]
        composite_keys.append(separator.join(key_parts))

    return np.array(composite_keys, dtype=object)


def validate_site_exclusivity(
    splits: list[dict[str, np.ndarray | int]],
    groups: np.ndarray,
) -> bool:
    """Verify that no group (e.g., site) appears in multiple validation folds.

    Parameters
    ----------
    splits : list[dict[str, np.ndarray | int]]
        List of split dictionaries, each with "val_indices" key.
    groups : np.ndarray
        Group assignments for each sample (e.g., site names).

    Returns:
    -------
    bool
        True if site exclusivity is maintained (no site in multiple val folds).
    """
    # Track which groups appear in which folds
    group_to_folds: dict[str, set[int]] = {}

    for split in splits:
        fold_idx = split["fold_idx"]
        val_indices = split["val_indices"]

        # Get unique groups in this fold's validation set
        val_groups = np.unique(groups[val_indices])

        for group in val_groups:
            group_str = str(group)
            if group_str not in group_to_folds:
                group_to_folds[group_str] = set()
            group_to_folds[group_str].add(fold_idx)

    # Check: each group should appear in at most one fold's validation set
    violations = [
        (group, folds) for group, folds in group_to_folds.items() if len(folds) > 1
    ]

    if violations:
        for group, folds in violations:
            warnings.warn(
                f"Group '{group}' appears in multiple validation folds: {sorted(folds)}. "
                "This violates site exclusivity.",
                UserWarning,
                stacklevel=2,
            )
        return False

    return True


def generate_split_report(
    splits: list[dict[str, np.ndarray | int]],
    groups: np.ndarray,
    stratify_labels: np.ndarray,
) -> dict:
    """Generate a report on split distribution across folds.

    Parameters
    ----------
    splits : list[dict[str, np.ndarray | int]]
        List of split dictionaries, each with "val_indices" key.
    groups : np.ndarray
        Group assignments for each sample (e.g., site names).
    stratify_labels : np.ndarray
        Stratum labels for each sample (for stratification reporting).

    Returns:
    -------
    dict
        Report dictionary with keys:
        - "per_fold_site_counts": dict mapping fold_idx -> dict of site -> count
        - "per_fold_stratum_counts": dict mapping fold_idx -> dict of stratum -> count
        - "warnings": list of warning messages
        - "infeasible_constraints": list of constraint violations
    """
    n_splits = len(splits)
    per_fold_site_counts: dict[int, dict[str, int]] = {}
    per_fold_stratum_counts: dict[int, dict[str, int]] = {}
    warnings_list: list[str] = []
    infeasible_constraints: list[str] = []

    # Count distributions per fold
    for split in splits:
        fold_idx = split["fold_idx"]
        val_indices = split["val_indices"]

        # Site counts
        val_groups = groups[val_indices]
        unique_groups, group_counts = np.unique(val_groups, return_counts=True)
        per_fold_site_counts[fold_idx] = {
            str(group): int(count) for group, count in zip(unique_groups, group_counts)
        }

        # Stratum counts
        val_strata = stratify_labels[val_indices]
        unique_strata, stratum_counts = np.unique(val_strata, return_counts=True)
        per_fold_stratum_counts[fold_idx] = {
            str(stratum): int(count)
            for stratum, count in zip(unique_strata, stratum_counts)
        }

    # Check for imbalances and infeasible constraints
    # 1. Check if any stratum is confined to a single site
    unique_strata = np.unique(stratify_labels)
    unique_groups = np.unique(groups)

    for stratum in unique_strata:
        stratum_mask = stratify_labels == stratum
        stratum_groups = np.unique(groups[stratum_mask])

        if len(stratum_groups) == 1:
            infeasible_constraints.append(
                f"Stratum '{stratum}' exists only in site '{stratum_groups[0]}'. "
                "Cannot stratify across folds without splitting this site."
            )

    # 2. Check for highly imbalanced distributions
    expected_samples_per_fold = len(groups) / n_splits
    tolerance = 0.3  # 30% tolerance for imbalance

    for fold_idx, stratum_counts in per_fold_stratum_counts.items():
        total_in_fold = sum(stratum_counts.values())
        if (
            abs(total_in_fold - expected_samples_per_fold)
            > expected_samples_per_fold * tolerance
        ):
            warnings_list.append(
                f"Fold {fold_idx} has {total_in_fold} samples "
                f"(expected ~{expected_samples_per_fold:.1f})"
            )

    # 3. Check if any fold is empty
    for fold_idx, site_counts in per_fold_site_counts.items():
        if not site_counts:
            warnings_list.append(f"Fold {fold_idx} has no samples in validation set")

    return {
        "per_fold_site_counts": per_fold_site_counts,
        "per_fold_stratum_counts": per_fold_stratum_counts,
        "warnings": warnings_list,
        "infeasible_constraints": infeasible_constraints,
    }


def create_group_stratified_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    stratify_labels: np.ndarray,
    patient_ids: np.ndarray | None = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    validate_exclusivity: bool = True,
    return_report: bool = False,
) -> list[dict[str, np.ndarray | int]] | tuple[list[dict[str, np.ndarray | int]], dict]:
    """Create k-fold splits with group exclusivity and stratification.

    Uses StratifiedGroupKFold to ensure:
    - Groups (e.g., sites) do not cross folds (site-exclusive)
    - Stratum distribution is approximately balanced across folds

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (used only for shape validation)
    y : np.ndarray
        Target labels (used only for shape validation)
    groups : np.ndarray
        Group assignments for each sample (e.g., site names).
        Must have same length as X.
    stratify_labels : np.ndarray
        Stratum labels for stratification (e.g., subtype/dataset).
        Must have same length as X.
    patient_ids : np.ndarray, optional
        Patient IDs for tracking
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=True
        Whether to shuffle data before splitting
    random_state : int, default=42
        Random seed for reproducibility
    validate_exclusivity : bool, default=True
        Whether to validate and warn if groups cross folds
    return_report : bool, default=False
        If True, also return a report dictionary with distribution statistics

    Returns:
    -------
    list[dict[str, np.ndarray | int]] | tuple[list[dict[str, np.ndarray | int]], dict]
        List of dictionaries, one per fold, each containing:
        - "fold_idx": fold number (0-indexed)
        - "train_indices": indices for training
        - "val_indices": indices for validation
        - "train_patient_ids": patient IDs for training (if available)
        - "val_patient_ids": patient IDs for validation (if available)
        If return_report=True, returns tuple of (splits, report_dict).

    Raises:
    ------
    ValueError
        If input arrays have inconsistent lengths or if constraints are invalid.
    """
    # Validate inputs
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(f"X and y must have same length. Got {n_samples} vs {len(y)}")
    if len(groups) != n_samples:
        raise ValueError(
            f"groups must have same length as X. Got {len(groups)} vs {n_samples}"
        )
    if len(stratify_labels) != n_samples:
        raise ValueError(
            f"stratify_labels must have same length as X. "
            f"Got {len(stratify_labels)} vs {n_samples}"
        )

    # Check if we have enough groups for the number of splits
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        warnings.warn(
            f"Only {len(unique_groups)} unique groups but {n_splits} folds requested. "
            f"Some folds may be empty.",
            UserWarning,
            stacklevel=2,
        )

    # Create splitter
    # When shuffle=False, random_state has no effect and sklearn raises an error
    # So we only pass random_state when shuffle=True
    splitter_kwargs = {
        "n_splits": n_splits,
        "shuffle": shuffle,
    }
    if shuffle:
        splitter_kwargs["random_state"] = random_state

    splitter = StratifiedGroupKFold(**splitter_kwargs)

    # Generate splits
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(X, stratify_labels, groups)
    ):
        split_dict = {
            "fold_idx": fold_idx,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }

        # Add patient IDs if available
        if patient_ids is not None:
            split_dict["train_patient_ids"] = patient_ids[train_idx]
            split_dict["val_patient_ids"] = patient_ids[val_idx]
        else:
            split_dict["train_patient_ids"] = None
            split_dict["val_patient_ids"] = None

        splits.append(split_dict)

    # Validate site exclusivity
    if validate_exclusivity:
        is_valid = validate_site_exclusivity(splits, groups)
        if not is_valid:
            # Warning already emitted by validate_site_exclusivity
            pass

    # Generate report if requested
    if return_report:
        report = generate_split_report(splits, groups, stratify_labels)

        # Print warnings if any
        if report["warnings"]:
            for warning_msg in report["warnings"]:
                warnings.warn(warning_msg, UserWarning, stacklevel=2)

        if report["infeasible_constraints"]:
            for constraint_msg in report["infeasible_constraints"]:
                warnings.warn(
                    f"Infeasible constraint: {constraint_msg}",
                    UserWarning,
                    stacklevel=2,
                )

        return splits, report

    return splits
