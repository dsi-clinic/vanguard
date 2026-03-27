"""Utilities for loading and processing clinic metadata from Excel files."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.kfold import build_composite_stratum_key


def load_clinic_metadata_excel(
    excel_path: Path,
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    """Load clinic metadata from an Excel file.

    Parameters
    ----------
    excel_path : Path
        Path to the Excel file (.xlsx format).
    sheet_name : str | int | None, default=0
        Sheet name or index to read. If None, reads first sheet.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the metadata with all columns from Excel.

    Raises:
    ------
    FileNotFoundError
        If the Excel file does not exist.
    ValueError
        If the file cannot be read or is invalid.

    Examples:
    --------
    >>> from pathlib import Path
    >>> df = load_clinic_metadata_excel(Path("metadata.xlsx"))
    >>> print(df.columns)
    Index(['case_id', 'site', 'dataset', ...], dtype='object')
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    try:
        excel_df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            engine="openpyxl",
        )
    except Exception as e:
        raise ValueError(f"Failed to read Excel file {excel_path}: {e}") from e

    if excel_df.empty:
        raise ValueError(f"Excel file {excel_path} is empty or has no data")

    return excel_df


def get_case_ids_from_excel(
    excel_path: Path | str,
    *,
    id_col: str = "case_id",
    sheet_name: str | int | None = 0,
) -> np.ndarray:
    """Load an Excel metadata file and return the list of patient IDs (for cohort definition).

    Useful when you need the Excel-defined cohort (e.g. to generate synthetic data
    for those IDs) before creating splits.

    Parameters
    ----------
    excel_path : Path or str
        Path to the Excel file.
    id_col : str, default="case_id"
        Column name containing patient IDs.
    sheet_name : str or int or None, default=0
        Sheet to read (passed to load_clinic_metadata_excel).

    Returns:
    -------
    np.ndarray
        One-dimensional array of patient IDs as strings, in Excel row order.
    """
    excel_path = Path(excel_path)
    excel_df = load_clinic_metadata_excel(excel_path, sheet_name=sheet_name)
    if id_col not in excel_df.columns:
        raise ValueError(
            f"Excel file {excel_path} has no column {id_col!r}. "
            f"Available columns: {list(excel_df.columns)}"
        )
    return excel_df[id_col].astype(str).to_numpy()


def build_split_annotations(
    metadata_df: pd.DataFrame,
    *,
    id_col: str = "case_id",
    group_col: str = "site",
    stratify_cols: list[str] | None = None,
    separator: str = "|",
) -> pd.DataFrame:
    """Build split annotations (group and stratum keys) from metadata DataFrame.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame containing metadata with patient IDs, site, and stratum columns.
    id_col : str, default="case_id"
        Column name for patient/sample IDs.
    group_col : str, default="site"
        Column name for grouping (e.g., site). Groups will not cross folds.
    stratify_cols : list[str] | None, default=None
        List of column names to use for stratification.
        If None, defaults to ["dataset"].
        If multiple columns, they will be combined into a composite key.
    separator : str, default="|"
        Separator to use when combining multiple stratum columns.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns:
        - `id_col`: Patient/sample IDs (from input)
        - "group": Group assignments (from `group_col`)
        - "stratum_key": Stratum keys (single column or composite)

    Raises:
    ------
    ValueError
        If required columns are missing or if data is invalid.

    Examples:
    --------
    >>> df = pd.DataFrame({
    ...     "case_id": ["P1", "P2", "P3"],
    ...     "site": ["SiteA", "SiteB", "SiteA"],
    ...     "dataset": ["DS1", "DS1", "DS2"]
    ... })
    >>> annotations = build_split_annotations(df)
    >>> print(annotations.columns)
    Index(['case_id', 'group', 'stratum_key'], dtype='object')
    """
    if stratify_cols is None:
        stratify_cols = ["dataset"]

    # Validate required columns exist
    required_cols = [id_col, group_col] + stratify_cols
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(metadata_df.columns)}"
        )

    # Extract group column
    groups = metadata_df[group_col].copy()

    # Handle nulls in group column
    if groups.isna().any():
        n_nulls = groups.isna().sum()
        warnings.warn(
            f"Found {n_nulls} null values in group column '{group_col}'. "
            "These rows will be dropped.",
            UserWarning,
            stacklevel=2,
        )
        # Drop rows with null groups
        valid_mask = ~groups.isna()
        metadata_df = metadata_df[valid_mask].copy()
        groups = groups[valid_mask]

    # Build stratum key
    if len(stratify_cols) == 1:
        # Single column: use directly
        stratum_key = metadata_df[stratify_cols[0]].astype(str)
    else:
        # Multiple columns: build composite key
        stratum_values = metadata_df[stratify_cols].to_numpy()
        stratum_key_array = build_composite_stratum_key(
            stratum_values, stratify_cols, separator=separator
        )
        stratum_key = pd.Series(stratum_key_array, index=metadata_df.index)

    # Handle nulls in stratum columns
    if stratum_key.isna().any() or (stratum_key == "nan").any():
        n_nulls = (stratum_key.isna() | (stratum_key == "nan")).sum()
        warnings.warn(
            f"Found {n_nulls} null/invalid values in stratum columns. "
            "These rows will be dropped.",
            UserWarning,
            stacklevel=2,
        )
        # Drop rows with null stratum keys
        valid_mask = ~(stratum_key.isna() | (stratum_key == "nan"))
        metadata_df = metadata_df[valid_mask].copy()
        groups = groups[valid_mask]
        stratum_key = stratum_key[valid_mask]

    # Build result DataFrame
    result = pd.DataFrame(
        {
            id_col: metadata_df[id_col],
            "group": groups,
            "stratum_key": stratum_key,
        }
    )

    return result


def align_metadata_to_case_ids(
    annotations_df: pd.DataFrame,
    case_ids: np.ndarray | pd.Series,
    *,
    id_col: str = "case_id",
    warn_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Align metadata annotations to a case_ids array.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        DataFrame from `build_split_annotations()` with columns:
        `id_col`, "group", "stratum_key".
    case_ids : np.ndarray | pd.Series
        Array of patient IDs to align with.
    id_col : str, default="case_id"
        Column name for patient IDs in annotations_df.
    warn_missing : bool, default=True
        Whether to warn if some case_ids are missing from annotations.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (groups, stratify_labels) arrays, aligned to case_ids.
        Missing patients will have NaN values (or raise error if all missing).

    Raises:
    ------
    ValueError
        If no case_ids match the annotations, or if alignment fails.

    Examples:
    --------
    >>> annotations = pd.DataFrame({
    ...     "case_id": ["P1", "P2", "P3"],
    ...     "group": ["SiteA", "SiteB", "SiteA"],
    ...     "stratum_key": ["DS1", "DS1", "DS2"]
    ... })
    >>> case_ids = np.array(["P1", "P2", "P3", "P4"])
    >>> groups, strata = align_metadata_to_case_ids(annotations, case_ids)
    >>> print(groups)
    ['SiteA' 'SiteB' 'SiteA' nan]
    """
    case_ids_array = np.asarray(case_ids)

    # Create index from annotations
    annotations_indexed = annotations_df.set_index(id_col)

    # Align: get groups and stratum_keys for each case_id
    aligned_groups = []
    aligned_strata = []

    missing_patients = []

    for pid in case_ids_array:
        pid_str = str(pid)
        if pid_str in annotations_indexed.index:
            aligned_groups.append(annotations_indexed.loc[pid_str, "group"])
            aligned_strata.append(annotations_indexed.loc[pid_str, "stratum_key"])
        else:
            aligned_groups.append(np.nan)
            aligned_strata.append(np.nan)
            missing_patients.append(pid_str)

    groups_array = np.array(aligned_groups, dtype=object)
    strata_array = np.array(aligned_strata, dtype=object)

    # Warn about missing patients
    if missing_patients and warn_missing:
        n_missing = len(missing_patients)
        n_total = len(case_ids_array)
        warnings.warn(
            f"{n_missing}/{n_total} case_ids not found in annotations. "
            f"First few missing: {missing_patients[:5]}",
            UserWarning,
            stacklevel=2,
        )

    # Check if we have any valid data
    valid_mask = ~(pd.isna(groups_array) | pd.isna(strata_array))
    if not valid_mask.any():
        raise ValueError(
            "No case_ids matched the annotations. "
            "Check that case_id values match between data and metadata."
        )

    return groups_array, strata_array
