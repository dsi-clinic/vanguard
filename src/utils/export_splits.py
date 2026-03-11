#!/usr/bin/env python3
"""CLI wrapper to export site-exclusive, subtype-stratified k-fold splits to CSV.

This script calls the evaluation engine (evaluation.kfold.export_splits_to_csv)
to read clinic metadata from Excel, generate group-stratified k-fold splits
(ensuring sites don't cross folds), and write them to a CSV file.

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

    # Dataset selection (CLI or config)
    python -m src.utils.export_splits --excel metadata.xlsx --output splits.csv \
        --datasets iSpy2 Duke
    python -m src.utils.export_splits --excel metadata.xlsx --output splits.csv \
        --config config/eval_selection_example.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.kfold import export_splits_to_csv  # noqa: E402
from evaluation.selection import (  # noqa: E402
    build_selection_criteria_from_args,
    load_selection_criteria_from_yaml,
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
    selection_criteria: object | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Export group-stratified k-fold splits to CSV.

    Wrapper around evaluation.kfold.export_splits_to_csv() for backward compatibility.

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
    pd.DataFrame or tuple[pd.DataFrame, dict]
        DataFrame with columns: patient_id, fold_idx, site, stratum_key.
        If return_report=True, returns tuple of (DataFrame, report_dict).
    """
    return export_splits_to_csv(
        excel_path=excel_path,
        output_path=output_path,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        id_col=id_col,
        group_col=group_col,
        stratify_cols=stratify_cols,
        validate_exclusivity=validate_exclusivity,
        return_report=return_report,
        verbose=True,
        selection_criteria=selection_criteria,
    )


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

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these datasets (e.g. iSpy2 Duke)",
    )
    parser.add_argument(
        "--sites",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these sites",
    )
    parser.add_argument(
        "--tumor-types",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these tumor types/subtypes",
    )
    parser.add_argument(
        "--unilateral-only",
        action="store_true",
        help="Include only unilateral cases (requires laterality column)",
    )
    parser.add_argument(
        "--bilateral-only",
        action="store_true",
        help="Include only bilateral cases (requires laterality column)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config with selection criteria. Used when no CLI selection flags.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.excel.exists():
        print(f"Error: Excel file not found: {args.excel}", file=sys.stderr)
        sys.exit(1)

    # Build selection criteria: CLI flags override; else use --config if set
    selection_criteria = build_selection_criteria_from_args(args)
    if selection_criteria is None and args.config is not None:
        try:
            selection_criteria = load_selection_criteria_from_yaml(args.config)
        except FileNotFoundError as e:
            print(f"Error: Config file not found: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: Invalid config: {e}", file=sys.stderr)
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
            selection_criteria=selection_criteria,
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
