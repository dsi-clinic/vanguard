"""Sample selection and filtering for evaluation cohort definition.

This module provides criteria and helpers to restrict evaluation runs to subsets
of the metadata (e.g. specific datasets, sites, tumor types, unilateral/bilateral).
Criteria are combined with AND logic across different dimensions; within a dimension
(e.g. datasets), OR logic applies (IN semantics).

Usage:
    >>> from evaluation.selection import (
    ...     SampleSelectionCriteria,
    ...     apply_selection_criteria,
    ...     build_selection_criteria_from_args,
    ... )
    >>> criteria = SampleSelectionCriteria(datasets=["ISPY2", "DUKE"])
    >>> filtered = apply_selection_criteria(metadata_df, criteria)
    >>> # Or from CLI args:
    >>> criteria = build_selection_criteria_from_args(args)
    >>> # Or from YAML config:
    >>> criteria = load_selection_criteria_from_yaml(Path("docs/eval_selection_example.yaml"))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Column name aliases for laterality (unilateral/bilateral)
# bilateral_mri: 0 = unilateral, 1 = bilateral (Excel format)
LATERALITY_COLUMN_ALIASES = ("bilateral", "laterality", "bilateral_mri")


def _resolve_laterality_column(df: pd.DataFrame) -> str | None:
    """Return the first present laterality column name, or None."""
    for col in LATERALITY_COLUMN_ALIASES:
        if col in df.columns:
            return col
    return None


@dataclass
class SampleSelectionCriteria:
    """Structured criteria for filtering metadata before evaluation.

    All fields are optional. Omit or set to None for no filter on that dimension.
    Criteria are combined with AND logic; within datasets/sites/tumor_types, OR applies.

    Parameters
    ----------
    datasets : list[str] | None
        Include only these datasets (e.g. ["iSpy2", "Duke"]).
    sites : list[str] | None
        Include only these sites.
    tumor_types : list[str] | None
        Include only these tumor subtypes.
    tumor_type_col : str, default="subtype"
        Column name for tumor types (subtype, tumor_subtype, etc.).
    unilateral_only : bool, default=False
        Include only unilateral cases.
    bilateral_only : bool, default=False
        Include only bilateral cases. Mutually exclusive with unilateral_only.
    laterality_col : str | None, default=None
        Column name for laterality. If None, first of ("bilateral", "laterality") in df.
    column_filters : dict[str, list] | None
        Generic column -> allowed values for extensibility.
    """

    datasets: list[str] | None = None
    sites: list[str] | None = None
    tumor_types: list[str] | None = None
    tumor_type_col: str = "subtype"
    unilateral_only: bool = False
    bilateral_only: bool = False
    laterality_col: str | None = None
    column_filters: dict[str, list] | None = None

    def __post_init__(self: SampleSelectionCriteria) -> None:
        """Validate that only one of unilateral_only or bilateral_only is set"""
        if self.unilateral_only and self.bilateral_only:
            raise ValueError(
                "unilateral_only and bilateral_only are mutually exclusive"
            )


def apply_selection_criteria(
    metadata_df: pd.DataFrame,
    criteria: SampleSelectionCriteria | dict | None,
    *,
    dataset_col: str = "dataset",
    site_col: str = "site",
    verbose: bool = False,
) -> pd.DataFrame:
    """Filter metadata by selection criteria with AND logic across dimensions.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Full metadata from Excel or other source.
    criteria : SampleSelectionCriteria | dict | None
        Selection criteria. If None or empty, returns unfiltered DataFrame.
        If dict: {"dataset": ["iSpy2"], "site": ["SiteA"]} format (column -> allowed values).
    dataset_col : str, default="dataset"
        Column name for dataset.
    site_col : str, default="site"
        Column name for site.
    verbose : bool, default=False
        If True, print how many rows were excluded per criterion.

    Notes:
    -----
    If a requested selection column (dataset, site, tumor type, laterality, or
    any column_filters key) is missing from the metadata, that criterion is
    skipped without raising an error; runs may be unfiltered for that criterion.

    Returns:
    -------
    pd.DataFrame
        Filtered metadata (subset of rows).
    """
    if criteria is None:
        return metadata_df

    # Convert dict to criteria-like structure for unified handling
    if isinstance(criteria, dict):
        if not criteria:
            return metadata_df
        # Build column_filters from dict; handle special keys
        c = criteria
        criteria_obj = SampleSelectionCriteria(
            datasets=c.get("datasets") if "datasets" in c else None,
            sites=c.get("sites") if "sites" in c else None,
            tumor_types=c.get("tumor_types") if "tumor_types" in c else None,
            unilateral_only=c.get("unilateral_only", False),
            bilateral_only=c.get("bilateral_only", False),
            column_filters={
                k: v
                for k, v in c.items()
                if k
                not in (
                    "datasets",
                    "sites",
                    "tumor_types",
                    "unilateral_only",
                    "bilateral_only",
                )
                and isinstance(v, list)
            },
        )
    else:
        criteria_obj = criteria

    mask = pd.Series(True, index=metadata_df.index)

    def apply_col_filter(col: str, allowed: list, label: str) -> None:
        nonlocal mask
        if col not in metadata_df.columns:
            if verbose:
                print(f"  Selection: column {col!r} not in metadata, skipping")
            return
        before = mask.sum()
        mask = mask & metadata_df[col].astype(str).isin([str(v) for v in allowed])
        excluded = before - mask.sum()
        if verbose and excluded > 0:
            print(
                f"  Selection: {label} -> {int(mask.sum())} kept ({int(excluded)} excluded)"
            )

    if criteria_obj.datasets is not None and len(criteria_obj.datasets) > 0:
        apply_col_filter(dataset_col, criteria_obj.datasets, "datasets")

    if criteria_obj.sites is not None and len(criteria_obj.sites) > 0:
        apply_col_filter(site_col, criteria_obj.sites, "sites")

    if criteria_obj.tumor_types is not None and len(criteria_obj.tumor_types) > 0:
        apply_col_filter(
            criteria_obj.tumor_type_col,
            criteria_obj.tumor_types,
            "tumor_types",
        )

    if criteria_obj.unilateral_only or criteria_obj.bilateral_only:
        lat_col = criteria_obj.laterality_col or _resolve_laterality_column(metadata_df)
        if lat_col is None:
            if verbose:
                print(
                    "  Selection: no laterality column found (bilateral/laterality/bilateral_mri), skipping"
                )
        else:
            before = mask.sum()
            if lat_col == "bilateral_mri":
                # bilateral_mri: 0 = unilateral, 1 = bilateral
                def _is_unilateral(v: object) -> bool:
                    try:
                        return int(float(v)) == 0
                    except (TypeError, ValueError):
                        return False

                def _is_bilateral(v: object) -> bool:
                    try:
                        return int(float(v)) == 1
                    except (TypeError, ValueError):
                        return False

                if criteria_obj.unilateral_only:
                    mask = mask & metadata_df[lat_col].apply(_is_unilateral)
                else:
                    mask = mask & metadata_df[lat_col].apply(_is_bilateral)
            else:
                if criteria_obj.unilateral_only:
                    mask = mask & (
                        metadata_df[lat_col].astype(str).str.lower() == "unilateral"
                    )
                else:
                    mask = mask & (
                        metadata_df[lat_col].astype(str).str.lower() == "bilateral"
                    )
            excluded = before - mask.sum()
            if verbose and excluded > 0:
                label = "unilateral" if criteria_obj.unilateral_only else "bilateral"
                print(
                    f"  Selection: {label}_only -> {int(mask.sum())} kept ({int(excluded)} excluded)"
                )

    if criteria_obj.column_filters:
        for col, allowed in criteria_obj.column_filters.items():
            apply_col_filter(col, allowed, f"column_filters[{col}]")

    result = metadata_df.loc[mask].copy()
    if verbose:
        n_kept = len(result)
        n_excluded = len(metadata_df) - n_kept
        if n_excluded > 0:
            print(f"  Selection: {n_kept} rows kept ({n_excluded} excluded)")
    return result


def build_selection_criteria_from_args(
    args: object,
    *,
    datasets_attr: str = "datasets",
    sites_attr: str = "sites",
    tumor_types_attr: str = "tumor_types",
    unilateral_only_attr: str = "unilateral_only",
    bilateral_only_attr: str = "bilateral_only",
) -> SampleSelectionCriteria | None:
    """Build SampleSelectionCriteria from CLI-style args object.

    Returns None if no selection args are set (no filtering).

    Parameters
    ----------
    args : object
        Namespace or object with attributes (e.g. argparse.Namespace).
    datasets_attr : str, default="datasets"
        Attribute name for datasets list.
    sites_attr : str, default="sites"
        Attribute name for sites list.
    tumor_types_attr : str, default="tumor_types"
        Attribute name for tumor types list.
    unilateral_only_attr : str, default="unilateral_only"
        Attribute name for unilateral_only flag.
    bilateral_only_attr : str, default="bilateral_only"
        Attribute name for bilateral_only flag.

    Returns:
    -------
    SampleSelectionCriteria | None
        Criteria if any selection is specified, else None.
    """
    datasets = getattr(args, datasets_attr, None)
    sites = getattr(args, sites_attr, None)
    tumor_types = getattr(args, tumor_types_attr, None)
    unilateral_only = bool(getattr(args, unilateral_only_attr, False))
    bilateral_only = bool(getattr(args, bilateral_only_attr, False))

    # Normalize to lists (CLI nargs="+" yields list; default None)
    if datasets is not None and not isinstance(datasets, list):
        datasets = [datasets] if datasets else None
    if sites is not None and not isinstance(sites, list):
        sites = [sites] if sites else None
    if tumor_types is not None and not isinstance(tumor_types, list):
        tumor_types = [tumor_types] if tumor_types else None

    has_any = (
        (datasets and len(datasets) > 0)
        or (sites and len(sites) > 0)
        or (tumor_types and len(tumor_types) > 0)
        or unilateral_only
        or bilateral_only
    )
    if not has_any:
        return None

    return SampleSelectionCriteria(
        datasets=datasets or None,
        sites=sites or None,
        tumor_types=tumor_types or None,
        unilateral_only=unilateral_only,
        bilateral_only=bilateral_only,
    )


def load_selection_criteria_from_yaml(
    config_path: Path | str,
) -> SampleSelectionCriteria | None:
    """Load selection criteria from a YAML config file.

    Expects a top-level "selection" key. All fields are optional.
    Returns None if the selection key is absent or empty.

    Parameters
    ----------
    config_path : Path | str
        Path to YAML file (e.g. docs/eval_selection_example.yaml).

    Returns:
    -------
    SampleSelectionCriteria | None
        Criteria if "selection" key exists and has content, else None.

    Raises:
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If YAML is invalid or selection fields have wrong types.

    Example:
    -------
    Config file (docs/eval_selection_example.yaml)::

        selection:
          datasets: [iSpy2, Duke]
          sites: null
          tumor_types: [luminal, triple_negative]
          unilateral_only: false
          bilateral_only: false
          column_filters:
            some_custom_col: [A, B]
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        data = yaml.safe_load(f)

    if data is None:
        return None

    selection = data.get("selection")
    if selection is None or not isinstance(selection, dict):
        return None

    sel = selection

    def _to_str_list(val: object, name: str) -> list[str] | None:
        if val is None:
            return None
        if not isinstance(val, list):
            raise ValueError(
                f"selection.{name} must be a list of strings, got {type(val).__name__}"
            )
        result = [str(v) for v in val]
        return result if result else None

    def _to_bool(val: object, name: str) -> bool:
        if val is None:
            return False
        if not isinstance(val, bool):
            raise ValueError(
                f"selection.{name} must be a boolean, got {type(val).__name__}"
            )
        return val

    datasets = _to_str_list(sel.get("datasets"), "datasets")
    sites = _to_str_list(sel.get("sites"), "sites")
    tumor_types = _to_str_list(sel.get("tumor_types"), "tumor_types")
    unilateral_only = _to_bool(sel.get("unilateral_only"), "unilateral_only")
    bilateral_only = _to_bool(sel.get("bilateral_only"), "bilateral_only")

    column_filters = None
    cf_raw = sel.get("column_filters")
    if cf_raw is not None:
        if not isinstance(cf_raw, dict):
            raise ValueError(
                "selection.column_filters must be a dict of column -> list, "
                f"got {type(cf_raw).__name__}"
            )
        column_filters = {}
        for k, v in cf_raw.items():
            if not isinstance(v, list):
                raise ValueError(
                    f"selection.column_filters[{k!r}] must be a list, got {type(v).__name__}"
                )
            column_filters[str(k)] = [str(x) for x in v]

    has_any = (
        (datasets and len(datasets) > 0)
        or (sites and len(sites) > 0)
        or (tumor_types and len(tumor_types) > 0)
        or unilateral_only
        or bilateral_only
        or (column_filters and len(column_filters) > 0)
    )
    if not has_any:
        return None

    tumor_type_col = sel.get("tumor_type_col") or "subtype"
    laterality_col = sel.get("laterality_col")

    return SampleSelectionCriteria(
        datasets=datasets,
        sites=sites,
        tumor_types=tumor_types,
        tumor_type_col=str(tumor_type_col),
        unilateral_only=unilateral_only,
        bilateral_only=bilateral_only,
        laterality_col=str(laterality_col) if laterality_col else None,
        column_filters=column_filters,
    )
