#!/usr/bin/env python3
"""Export site-exclusive, subtype-stratified k-fold splits to CSV.

This script reads clinic metadata from Excel, generates group-stratified k-fold
splits (ensuring sites don't cross folds), and exports them to a CSV file that
can be reused across all model training pipelines.

Usage:
    python -m src.utils.export_splits \
        --excel /path/to/clinic_amd_imaging_info.xlsx \
        --output splits.csv \
        --n-splits 5 \
        --group-col site \
        --stratify-cols dataset

    # With composite stratum keys
    python -m src.utils.export_splits \
        --excel metadata.xlsx \
        --output splits.csv \
        --stratify-cols dataset subtype \
        --random-state 42

Examples:
    # Basic usage
    python -m src.utils.export_splits \
        --excel /net/projects2/vanguard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx \
        --output splits.csv \
        --n-splits 5

    # With composite stratum keys
    python -m src.utils.export_splits \
        --excel /net/projects2/vanguard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx \
        --output splits.csv \
        --stratify-cols dataset tumor_subtype \
        --random-state 42

    # Custom configuration
    python -m src.utils.export_splits \
        --excel /net/projects2/vanguard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx \
        --output splits.csv \
        --n-splits 3 \
        --group-col site \
        --stratify-cols dataset \
        --id-col patient_id \
        --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.kfold import create_group_stratified_kfold_splits  # noqa: E402
from src.utils.clinic_metadata import (  # noqa: E402
    align_metadata_to_patient_ids,
    build_split_annotations,
    load_clinic_metadata_excel,
)


def export_splits(
    excel_path: Path,
    output_path: Path,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    id_col: str = "patient_id",
    group_col: str = "site",
    stratify_cols: list[str] | None = None,
    validate_exclusivity: bool = True,
    return_report: bool = False,
) -> pd.DataFrame:
    """Export group-stratified k-fold splits to CSV.

    Parameters
    ----------
    excel_path : Path
        Path to Excel file containing clinic metadata.
    output_path : Path
        Path where splits CSV will be written.
    n_splits : int, default=5
        Number of folds.
    random_state : int, default=42
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    id_col : str, default="patient_id"
        Column name for patient/sample IDs.
    group_col : str, default="site"
        Column name for grouping (e.g., site). Groups will not cross folds.
    stratify_cols : list[str] | None, default=None
        List of column names for stratification.
        If None, defaults to ["dataset"].
        If multiple columns, creates composite keys.
    validate_exclusivity : bool, default=True
        Whether to validate and warn if groups cross folds.
    return_report : bool, default=False
        If True, also return the split report dictionary.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns: patient_id, fold_idx, site, stratum_key.
        If return_report=True, returns tuple of (DataFrame, report_dict).
    """
    # Load metadata from Excel
    print(f"Loading metadata from {excel_path}...")
    metadata_df = load_clinic_metadata_excel(excel_path)
    print(f"  Loaded {len(metadata_df)} rows")

    # Build annotations (group and stratum keys)
    if stratify_cols is None:
        stratify_cols = ["dataset"]

    print(f"Building annotations (group={group_col}, stratify={stratify_cols})...")
    annotations = build_split_annotations(
        metadata_df,
        id_col=id_col,
        group_col=group_col,
        stratify_cols=stratify_cols,
    )
    print(f"  Built annotations for {len(annotations)} patients")
    print(f"  Unique groups: {sorted(annotations['group'].unique())}")
    print(f"  Unique strata: {sorted(annotations['stratum_key'].unique())}")

    # Get patient IDs and align metadata
    patient_ids = annotations[id_col].to_numpy()
    groups, stratify_labels = align_metadata_to_patient_ids(
        annotations, patient_ids, id_col=id_col, warn_missing=False
    )

    # Filter out any NaN values (shouldn't happen, but be safe)
    valid_mask = ~(pd.isna(groups) | pd.isna(stratify_labels))
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  Warning: Dropping {n_invalid} rows with missing group/stratum")
        patient_ids = patient_ids[valid_mask]
        groups = groups[valid_mask]
        stratify_labels = stratify_labels[valid_mask]

    # Create dummy X and y for splitting (we only need indices)
    n_samples = len(patient_ids)
    X_dummy = np.zeros((n_samples, 1))
    y_dummy = np.zeros(n_samples, dtype=int)

    # Generate splits
    print(f"Generating {n_splits}-fold splits (random_state={random_state})...")
    splits, report = create_group_stratified_kfold_splits(
        X=X_dummy,
        y=y_dummy,
        groups=groups,
        stratify_labels=stratify_labels,
        patient_ids=patient_ids,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        validate_exclusivity=validate_exclusivity,
        return_report=True,
    )

    # Print report summary
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
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["infeasible_constraints"]:
        print("\nInfeasible constraints:")
        for constraint in report["infeasible_constraints"]:
            print(f"  - {constraint}")

    # Build output DataFrame
    print("\nBuilding output DataFrame...")
    rows = []
    for split in splits:
        fold_idx = split["fold_idx"]
        val_patient_ids = split["val_patient_ids"]

        # Get group and stratum for each patient in this fold
        for pid in val_patient_ids:
            # Find index of this patient
            pid_idx = np.where(patient_ids == pid)[0]
            if len(pid_idx) > 0:
                idx = pid_idx[0]
                rows.append(
                    {
                        id_col: pid,
                        "fold_idx": fold_idx,
                        group_col: groups[idx],
                        "stratum_key": stratify_labels[idx],
                    }
                )

    splits_df = pd.DataFrame(rows)

    # Sort by fold_idx, then by patient_id for readability
    splits_df = splits_df.sort_values([id_col, "fold_idx"]).reset_index(drop=True)

    # Save to CSV
    print(f"\nWriting splits to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(output_path, index=False)
    print(f"  Wrote {len(splits_df)} rows ({n_splits} folds)")
    print(f"  Columns: {list(splits_df.columns)}")

    if return_report:
        return splits_df, report

    return splits_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--excel",
        type=Path,
        required=True,
        help="Path to Excel file containing clinic metadata (.xlsx format)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for splits CSV file",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before splitting",
    )

    parser.add_argument(
        "--id-col",
        type=str,
        default="patient_id",
        help="Column name for patient/sample IDs (default: patient_id)",
    )

    parser.add_argument(
        "--group-col",
        type=str,
        default="site",
        help="Column name for grouping, e.g., site (default: site)",
    )

    parser.add_argument(
        "--stratify-cols",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Column name(s) for stratification (default: dataset). "
            "If multiple columns, creates composite keys."
        ),
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable validation of site exclusivity",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed split distribution report",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.excel.exists():
        print(f"Error: Excel file not found: {args.excel}", file=sys.stderr)
        sys.exit(1)

    # Export splits
    try:
        splits_df = export_splits(
            excel_path=args.excel,
            output_path=args.output,
            n_splits=args.n_splits,
            random_state=args.random_state,
            shuffle=not args.no_shuffle,
            id_col=args.id_col,
            group_col=args.group_col,
            stratify_cols=args.stratify_cols,
            validate_exclusivity=not args.no_validate,
            return_report=args.report,
        )

        print(f"\n✓ Successfully exported splits to {args.output}")
        print(f"  Total rows: {len(splits_df)}")
        print(f"  Unique patients: {splits_df[args.id_col].nunique()}")
        print(f"  Folds: {sorted(splits_df['fold_idx'].unique())}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
