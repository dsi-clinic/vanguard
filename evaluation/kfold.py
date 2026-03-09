"""K-fold cross-validation split generation.

Provides standard stratified k-fold and group-stratified k-fold (group-exclusive
folds with stratification), plus helpers for Excel-driven splits.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)


@dataclass
class FoldSplit:
    """Represents a single fold split (indices and optional patient IDs)."""

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_patient_ids: np.ndarray | None = None
    val_patient_ids: np.ndarray | None = None


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

    Uses stratified k-fold by default so class distribution is maintained
    across folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels (used for stratification when stratify=True).
    patient_ids : np.ndarray, optional
        Patient IDs for tracking.
    n_splits : int, default=5
        Number of folds.
    stratify : bool, default=True
        Whether to use stratified k-fold (maintains class distribution).
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    -------
    list[dict[str, np.ndarray | int]]
        One dict per fold with fold_idx, train_indices, val_indices,
        train_patient_ids, val_patient_ids (latter two optional).
    """
    # Choose splitter
    if stratify:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
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
    """Verify that no group appears in multiple validation folds.

    Parameters
    ----------
    splits : list[dict[str, np.ndarray | int]]
        List of split dictionaries, each with "val_indices" key.
    groups : np.ndarray
        Group assignments for each sample (e.g. site).

    Returns:
    -------
    bool
        True if group exclusivity is maintained (each group in at most one
        validation fold).
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
                "This violates group exclusivity.",
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
        Group assignments for each sample (e.g. site).
    stratify_labels : np.ndarray
        Stratification labels for each sample (stratum values for reporting).

    Returns:
    -------
    dict
        Keys: per_fold_site_counts, per_fold_stratum_counts, warnings,
        infeasible_constraints.
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
                f"Stratum '{stratum}' exists only in group '{stratum_groups[0]}'. "
                "Cannot stratify across folds without splitting this group."
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

    Uses StratifiedGroupKFold so that groups do not cross folds and
    stratum distribution is approximately balanced across folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (used for shape only).
    y : np.ndarray
        Target labels (used for shape only).
    groups : np.ndarray
        Group assignments per sample (e.g. site). Same length as X.
    stratify_labels : np.ndarray
        Stratification labels per sample (e.g. subtype/dataset). Same length as X.
    patient_ids : np.ndarray, optional
        Patient IDs for tracking.
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int, default=42
        Random seed for reproducibility.
    validate_exclusivity : bool, default=True
        Whether to validate and warn if groups cross folds.
    return_report : bool, default=False
        If True, also return a report dict with per-fold counts and warnings.

    Returns:
    -------
    list[dict] or tuple[list[dict], dict]
        List of split dicts (fold_idx, train_indices, val_indices,
        train_patient_ids, val_patient_ids). If return_report=True, returns
        (splits, report).

    Raises:
    ------
    ValueError
        If input lengths differ or constraints are invalid.
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


def _create_splits_from_excel_core(
    excel_path: Path | str,
    patient_ids: np.ndarray | pd.Series | None = None,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    id_col: str = "patient_id",
    group_col: str = "site",
    stratify_cols: list[str] | None = None,
    validate_exclusivity: bool = True,
    return_report: bool = False,
    verbose: bool = True,
    selection_criteria: object | None = None,
) -> tuple[
    list[dict[str, np.ndarray | int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Load Excel metadata, build annotations, and create group-stratified k-fold splits.

    Internal core used by create_splits_from_excel() and export_splits_to_csv().
    When patient_ids is provided, aligns to those IDs and returns splits with indices
    into the provided array; when None, uses all patients from Excel.

    Returns:
    -------
    splits : list of split dicts (fold_idx, train_indices, val_indices, train_patient_ids, val_patient_ids)
    patient_ids : np.ndarray used for splitting (subset of input when provided)
    groups : np.ndarray group labels
    stratify_labels : np.ndarray stratum labels
    report : dict with per_fold_site_counts, per_fold_stratum_counts, warnings, infeasible_constraints
    """
    from evaluation.selection import apply_selection_criteria
    from src.utils.clinic_metadata import (
        align_metadata_to_patient_ids,
        build_split_annotations,
        load_clinic_metadata_excel,
    )

    excel_path = Path(excel_path)
    if stratify_cols is None:
        stratify_cols = ["dataset"]

    if verbose:
        print(f"Loading metadata from {excel_path}...")
    metadata_df = load_clinic_metadata_excel(excel_path)
    if verbose:
        print(f"  Loaded {len(metadata_df)} rows")

    if selection_criteria is not None:
        metadata_df = apply_selection_criteria(
            metadata_df,
            selection_criteria,
            dataset_col="dataset",
            site_col=group_col,
            verbose=verbose,
        )

    if verbose:
        print(f"Building annotations (group={group_col}, stratify={stratify_cols})...")
    annotations = build_split_annotations(
        metadata_df,
        id_col=id_col,
        group_col=group_col,
        stratify_cols=stratify_cols,
    )
    if verbose:
        print(f"  Built annotations for {len(annotations)} patients")
        print(f"  Unique groups: {sorted(annotations['group'].unique())}")
        print(f"  Unique strata: {sorted(annotations['stratum_key'].unique())}")

    # When model provides patient_ids, Excel is the cohort definition: labels/features
    # must cover every patient_id in the Excel file.
    if patient_ids is not None:
        excel_patient_ids = annotations[id_col].astype(str).unique()
        model_ids_set = {str(pid) for pid in np.asarray(patient_ids).ravel()}
        missing = [pid for pid in excel_patient_ids if pid not in model_ids_set]
        if missing:
            n_missing = len(missing)
            n_excel = len(excel_patient_ids)
            sample = sorted(missing)[:10]
            raise ValueError(
                "When using Excel metadata, labels and features must include every "
                "patient_id in the Excel file. "
                f"{n_missing}/{n_excel} Excel patient_ids are missing from your data. "
                f"First missing (up to 10): {sample}"
            )

    if patient_ids is None:
        patient_ids_work = annotations[id_col].to_numpy()
        groups, stratify_labels = align_metadata_to_patient_ids(
            annotations, patient_ids_work, id_col=id_col, warn_missing=False
        )
        valid_mask = ~(pd.isna(groups) | pd.isna(stratify_labels))
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            if verbose:
                print(
                    f"  Warning: Dropping {n_invalid} rows with missing group/stratum"
                )
            patient_ids_work = patient_ids_work[valid_mask]
            groups = groups[valid_mask].copy()
            stratify_labels = stratify_labels[valid_mask].copy()
        original_indices = None
        original_patient_ids = None
    else:
        original_patient_ids = np.asarray(patient_ids)
        groups, stratify_labels = align_metadata_to_patient_ids(
            annotations, original_patient_ids, id_col=id_col, warn_missing=True
        )
        valid_mask = ~(pd.isna(groups) | pd.isna(stratify_labels))
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            if verbose:
                print(
                    f"  Warning: Dropping {n_invalid} rows with missing group/stratum"
                )
        patient_ids_work = original_patient_ids[valid_mask]
        groups = groups[valid_mask]
        stratify_labels = stratify_labels[valid_mask]
        original_indices = np.where(valid_mask)[0]

    if len(patient_ids_work) == 0:
        raise ValueError(
            "No patients with valid group/stratum. Check Excel and patient_ids."
        )

    n_samples = len(patient_ids_work)
    X_dummy = np.zeros((n_samples, 1))
    y_dummy = np.zeros(n_samples, dtype=int)

    if verbose:
        print(f"Generating {n_splits}-fold splits (random_state={random_state})...")
    result = create_group_stratified_kfold_splits(
        X=X_dummy,
        y=y_dummy,
        groups=groups,
        stratify_labels=stratify_labels,
        patient_ids=patient_ids_work,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        validate_exclusivity=validate_exclusivity,
        return_report=True,
    )
    splits, report = result

    if original_indices is not None:
        for split in splits:
            split["train_indices"] = original_indices[split["train_indices"]]
            split["val_indices"] = original_indices[split["val_indices"]]
            split["train_patient_ids"] = original_patient_ids[split["train_indices"]]
            split["val_patient_ids"] = original_patient_ids[split["val_indices"]]
        patient_ids_return = original_patient_ids
    else:
        patient_ids_return = patient_ids_work

    return splits, patient_ids_return, groups, stratify_labels, report


def create_splits_from_excel(
    excel_path: Path | str,
    patient_ids: np.ndarray | pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    id_col: str = "patient_id",
    group_col: str = "site",
    stratify_cols: list[str] | None = None,
    validate_exclusivity: bool = True,
    return_report: bool = False,
    selection_criteria: object | None = None,
) -> list[FoldSplit] | tuple[list[FoldSplit], dict]:
    """Create group-stratified k-fold splits from Excel metadata, aligned to model patient IDs.

    Models call this to get splits without writing/reading CSV. Returns FoldSplit objects
    with train_indices/val_indices into the provided patient_ids array.

    When using Excel metadata, the Excel file defines the cohort: you must provide
    labels and features for every patient_id in the Excel. If any Excel patient_id
    is missing from patient_ids, a ValueError is raised.

    Parameters
    ----------
    excel_path : Path | str
        Path to Excel file with clinic metadata.
    patient_ids : np.ndarray | pd.Series
        Model's patient IDs (same order as X, y). Splits will be aligned to these.
    n_splits : int, default=5
        Number of folds.
    random_state : int, default=42
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    id_col : str, default="patient_id"
        Column name for patient IDs in Excel.
    group_col : str, default="site"
        Column name for grouping (e.g., site). Groups do not cross folds.
    stratify_cols : list[str] | None, default=None
        Columns for stratification (default: ["dataset"]).
    validate_exclusivity : bool, default=True
        Whether to validate site exclusivity.
    return_report : bool, default=False
        If True, also return report dict.
    selection_criteria : SampleSelectionCriteria | dict | None, default=None
        If provided, filter metadata to included datasets/sites/tumor types/etc.
        before building splits. Uses AND logic across criteria.

    Returns:
    -------
    list[FoldSplit] or tuple[list[FoldSplit], dict]
        Fold splits with indices into the provided patient_ids array.
    """
    splits, _, _, _, report = _create_splits_from_excel_core(
        excel_path=excel_path,
        patient_ids=patient_ids,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        id_col=id_col,
        group_col=group_col,
        stratify_cols=stratify_cols,
        validate_exclusivity=validate_exclusivity,
        return_report=True,
        verbose=False,
        selection_criteria=selection_criteria,
    )
    fold_splits = [
        FoldSplit(
            fold_idx=s["fold_idx"],
            train_indices=s["train_indices"],
            val_indices=s["val_indices"],
            train_patient_ids=s.get("train_patient_ids"),
            val_patient_ids=s.get("val_patient_ids"),
        )
        for s in splits
    ]
    if return_report:
        return fold_splits, report
    return fold_splits


def export_splits_to_csv(
    excel_path: Path | str,
    output_path: Path | str,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    id_col: str = "patient_id",
    group_col: str = "site",
    stratify_cols: list[str] | None = None,
    validate_exclusivity: bool = True,
    return_report: bool = False,
    verbose: bool = True,
    selection_criteria: object | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Generate group-stratified k-fold splits from Excel and write to CSV.

    Used by the export_splits CLI script. Uses all patients from Excel.

    Parameters
    ----------
    excel_path : Path | str
        Path to Excel file with clinic metadata.
    output_path : Path | str
        Path where splits CSV will be written.
    n_splits, random_state, shuffle, id_col, group_col, stratify_cols,
    validate_exclusivity : same as create_splits_from_excel.
    return_report : bool, default=False
        If True, return (DataFrame, report).
    verbose : bool, default=True
        Whether to print progress and report.

    Returns:
    -------
    pd.DataFrame or tuple[pd.DataFrame, dict]
        DataFrame with columns patient_id, fold_idx, site, stratum_key.
    """
    output_path = Path(output_path)
    splits, patient_ids_return, groups, stratify_labels, report = (
        _create_splits_from_excel_core(
            excel_path=excel_path,
            patient_ids=None,
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle,
            id_col=id_col,
            group_col=group_col,
            stratify_cols=stratify_cols,
            validate_exclusivity=validate_exclusivity,
            return_report=True,
            verbose=verbose,
            selection_criteria=selection_criteria,
        )
    )

    if verbose:
        print("\nSplit distribution summary:")
        for fold_idx in range(n_splits):
            site_counts = report["per_fold_site_counts"][fold_idx]
            stratum_counts = report["per_fold_stratum_counts"][fold_idx]
            n_val = sum(stratum_counts.values())
            print(f"  Fold {fold_idx}: {n_val} samples")
            print(f"    Sites: {dict(site_counts)}")
            print(f"    Strata: {dict(stratum_counts)}")
        if report["warnings"]:
            print("\nWarnings:")
            for w in report["warnings"]:
                print(f"  - {w}")
        if report["infeasible_constraints"]:
            print("\nInfeasible constraints:")
            for c in report["infeasible_constraints"]:
                print(f"  - {c}")

    rows = []
    for split in splits:
        fold_idx = split["fold_idx"]
        for pid, vi in zip(split["val_patient_ids"], split["val_indices"]):
            rows.append(
                {
                    id_col: pid,
                    "fold_idx": fold_idx,
                    group_col: groups[vi],
                    "stratum_key": stratify_labels[vi],
                }
            )

    splits_df = pd.DataFrame(rows)
    splits_df = splits_df.sort_values([id_col, "fold_idx"]).reset_index(drop=True)

    if verbose:
        print(f"\nWriting splits to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(output_path, index=False)
    if verbose:
        print(f"  Wrote {len(splits_df)} rows ({n_splits} folds)")
        print(f"  Columns: {list(splits_df.columns)}")

    if return_report:
        return splits_df, report
    return splits_df
