#!/usr/bin/env python3
"""Extract and save the train/test fold assignment from the completed resampled run.

Reads predictions.csv from the baseline arm of the resampled ISPY2 ablation
(clinical_plus_tumor_size), which records which fold each patient was assigned
to as the validation set. Writes a CSV with one row per patient:

    case_id, val_fold

This is the "locked split" used to re-run the native-spacing pipeline on the
exact same fold assignment, so that resampled vs. native comparisons differ only
in the centerline source, not in the train/test split.

Also writes a provenance JSON alongside the CSV documenting the source.

Usage (head node, no compute needed):
    cd /ess/home/home1/t-9sbose/vanguard
    micromamba run -n vanguard python scripts/extract_resampled_splits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

RESAMPLED_RUN_DIR = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_resampled_ispy2"
)
SOURCE_ARM = "clinical_plus_tumor_size"
OUTPUT_PATH = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/locked_split_resampled_ispy2.csv"
)
PROVENANCE_PATH = OUTPUT_PATH.with_suffix(".provenance.json")


def main() -> None:
    predictions_path = RESAMPLED_RUN_DIR / "runs" / SOURCE_ARM / "predictions.csv"
    if not predictions_path.exists():
        print(f"ERROR: predictions not found at {predictions_path}", file=sys.stderr)
        sys.exit(1)

    preds = pd.read_csv(predictions_path)
    required_cols = {"case_id", "fold"}
    missing = required_cols - set(preds.columns)
    if missing:
        print(f"ERROR: predictions CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    splits_df = (
        preds[["case_id", "fold"]]
        .rename(columns={"fold": "val_fold"})
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    # Sanity check: each patient should appear in exactly one fold
    dup = splits_df["case_id"].duplicated().sum()
    if dup > 0:
        print(f"WARNING: {dup} duplicate case_ids in predictions (unexpected)", file=sys.stderr)

    n_folds = splits_df["val_fold"].nunique()
    per_fold = splits_df.groupby("val_fold").size().to_dict()
    print(f"Split summary:")
    print(f"  Total patients : {len(splits_df)}")
    print(f"  Number of folds: {n_folds}")
    print(f"  Per-fold counts: {per_fold}")
    print(f"  Source arm     : {SOURCE_ARM}")
    print(f"  Source file    : {predictions_path}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote splits to: {OUTPUT_PATH}")

    provenance = {
        "description": (
            "Locked 5-fold CV split extracted from the resampled ISPY2 ablation run. "
            "Each row maps a case_id to the fold in which it was the validation set. "
            "Used to re-run the native-spacing pipeline on the same fold assignment."
        ),
        "source_run_dir": str(RESAMPLED_RUN_DIR),
        "source_arm": SOURCE_ARM,
        "source_predictions_csv": str(predictions_path),
        "n_patients": int(len(splits_df)),
        "n_folds": int(n_folds),
        "per_fold_counts": {str(k): int(v) for k, v in per_fold.items()},
        "random_state": 42,
        "config_used": str(
            RESAMPLED_RUN_DIR / "runs" / SOURCE_ARM / "config_used.yaml"
        ),
        "column_schema": {
            "case_id": "patient identifier (e.g. ISPY2_102212)",
            "val_fold": "fold index (0..n_folds-1) where this patient was validation",
        },
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2))
    print(f"Wrote provenance to: {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
