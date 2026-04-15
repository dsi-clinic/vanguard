#!/usr/bin/env python3
"""Retroactively add Project 1 second-order features to an existing CSV.

Usage::

    python scripts/add_second_order_features.py features_raw.csv -o features_with_second_order.csv

If ``-o`` is omitted the output overwrites the input file.

Best run on ``features_raw.csv`` (before block selection) so that all
intermediate columns like ``per_shell_topology_*`` and ``boundary_crossing_*``
are available.  When run on a block-filtered CSV the graph second-order
features that depend on those orphaned columns will be NaN.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from features.second_order import SECOND_ORDER_COLUMNS, add_second_order_features


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", type=Path, help="path to features CSV")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: overwrite input)",
    )
    args = ap.parse_args()

    features = pd.read_csv(args.input_csv)
    n_before = len(features.columns)
    logging.info(
        "Loaded %s: %d rows x %d columns", args.input_csv, len(features), n_before
    )

    for _, row_series in features.iterrows():
        row_dict = row_series.to_dict()
        add_second_order_features(row_dict)
        for col in SECOND_ORDER_COLUMNS:
            features.loc[row_series.name, col] = row_dict[col]

    n_added = len(features.columns) - n_before
    logging.info(
        "Added %d second-order columns: %s", n_added, list(SECOND_ORDER_COLUMNS)
    )

    out_path = args.output or args.input_csv
    features.to_csv(out_path, index=False)
    logging.info(
        "Wrote %s: %d rows x %d columns", out_path, len(features), len(features.columns)
    )


if __name__ == "__main__":
    main()
