#!/usr/bin/env python3
"""Build the canonical HER2 fold map used by all Phase 1+ models.

The map pins per-case fold assignments so paired-fold metrics across the
Deep Sets and XGBoost pipelines are comparable. By default it operates
on the HER2 \u2229 tabular intersection cohort (n=68) using
StratifiedKFold(n_splits=3, shuffle=True, random_state=42), which
matches the Phase 1 HER2 Deep Sets config.

Usage::

    python scripts/build_her2_fold_map.py \\
        --intersection-cohort data/her2_intersection_case_ids.csv \\
        --output data/fold_map_her2.csv \\
        --n-splits 3 --random-state 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intersection-cohort",
        type=Path,
        default=Path("data/her2_intersection_case_ids.csv"),
        help="CSV with case_id, tumor_subtype, label columns.",
    )
    parser.add_argument(
        "--subtype",
        default="her2_enriched",
        help="Tumor subtype to filter on (column tumor_subtype).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fold_map_her2.csv"),
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_fold_map(
    cohort_df: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Return a (case_id, fold, random_state) DataFrame in input order."""
    required = {"case_id", "label"}
    missing = required.difference(cohort_df.columns)
    if missing:
        raise ValueError(f"cohort_df missing columns {missing}")
    if cohort_df["case_id"].duplicated().any():
        dupes = cohort_df.loc[cohort_df["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"cohort_df has duplicate case_ids: {dupes[:5]}")

    cohort_df = cohort_df.reset_index(drop=True)
    case_ids = cohort_df["case_id"].astype(str).to_numpy()
    y = cohort_df["label"].astype(int).to_numpy()

    fold_assignments = pd.Series(index=range(len(cohort_df)), dtype="Int64")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (_train_idx, val_idx) in enumerate(skf.split(case_ids, y)):
        fold_assignments.iloc[val_idx] = fold_idx

    if fold_assignments.isna().any():
        raise RuntimeError("Unassigned cases in fold map; check StratifiedKFold output")

    out = pd.DataFrame(
        {
            "case_id": case_ids,
            "label": y,
            "fold": fold_assignments.astype(int).to_numpy(),
            "n_splits": n_splits,
            "random_state": random_state,
        }
    )
    return out


def _summarize(fold_map: pd.DataFrame) -> None:
    print("\nFold composition (n_splits=%d, random_state=%d):" % (
        int(fold_map["n_splits"].iloc[0]),
        int(fold_map["random_state"].iloc[0]),
    ))
    summary = (
        fold_map.groupby("fold")
        .agg(n=("case_id", "size"), pos=("label", "sum"))
        .assign(neg=lambda d: d["n"] - d["pos"])
    )
    print(summary.to_string())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    cohort = pd.read_csv(args.intersection_cohort)
    cohort["case_id"] = cohort["case_id"].astype(str)
    if "tumor_subtype" in cohort.columns:
        cohort = cohort[cohort["tumor_subtype"] == args.subtype].copy()
    cohort = cohort[["case_id", "label"]].dropna()
    cohort["label"] = cohort["label"].astype(int)

    if cohort.empty:
        raise SystemExit(
            f"No rows remained after filtering tumor_subtype == {args.subtype!r}"
        )

    fold_map = build_fold_map(
        cohort,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fold_map.to_csv(args.output, index=False)
    print(f"Wrote {len(fold_map)} rows to {args.output}")
    _summarize(fold_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
