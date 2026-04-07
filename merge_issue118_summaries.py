"""Merge LR and XGB ablation summaries produced for Issue #118."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge Issue #118 ablation_summary CSVs (LR + XGB)."
    )
    parser.add_argument(
        "--lr-summary",
        type=Path,
        required=True,
        help="Path to ablation_summary.csv from the LR run.",
    )
    parser.add_argument(
        "--xgb-summary",
        type=Path,
        required=True,
        help="Path to ablation_summary.csv from the XGB run.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Output path for the combined CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    lr_df = pd.read_csv(args.lr_summary)
    lr_df.insert(0, "model_family", "lr")
    xgb_df = pd.read_csv(args.xgb_summary)
    xgb_df.insert(0, "model_family", "xgb")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([lr_df, xgb_df], ignore_index=True).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
