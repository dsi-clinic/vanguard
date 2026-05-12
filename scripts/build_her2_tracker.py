#!/usr/bin/env python3
"""Build results/her2_deepsets_tracker.csv with HER2 phase rows.

The shared ``results/deepsets_sweep_tracker.csv`` enforces a fixed
column list via ``scripts/aggregate_deepsets_sweep.py:TRACKER_COLUMNS``
which silently drops extra keys. HER2 work needs ``subgroup`` and
``n_cases`` columns, so we maintain a parallel tracker.

Phase 0 rows added here:
- log(N_points) LR baseline on the n=68 HER2 intersection cohort.
- DS Phase 3 winners (cos_T80, h128_d02_lfocal, h256_d02_lfocal) restricted
  to the n=68 HER2 intersection cohort, per-seed and cross-seed.

Usage::

    python scripts/build_her2_tracker.py \\
        --diagnostics results/her2_phase0_existing_runs.csv \\
        --logn-metrics experiments/her2_phase0/logn_lr/metrics.json \\
        --output results/her2_deepsets_tracker.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRACKER_COLUMNS = [
    "phase",
    "run_id",
    "model_family",
    "subgroup",
    "n_cases",
    "n_folds",
    "seed",
    "mean_fold_auc",
    "std_fold_auc",
    "pooled_auc",
    "pooled_ap",
    "cross_seed_mean_auc",
    "cross_seed_std_auc",
    "fold_map_path",
    "config_path",
    "predictions_path",
    "notes",
]
SUMMARY_FOLD_SENTINEL = -1


def _empty_row() -> dict[str, Any]:
    return {col: None for col in TRACKER_COLUMNS}


def _logn_rows(metrics_path: Path) -> list[dict[str, Any]]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_fold = metrics.get("per_fold_auc", {})
    rows: list[dict[str, Any]] = []
    for fold_idx, auc in per_fold.items():
        row = _empty_row()
        row.update(
            {
                "phase": "phase0d",
                "run_id": f"logn_lr_fold{fold_idx}",
                "model_family": "lr",
                "subgroup": "her2_intersection_n68",
                "n_cases": int(metrics.get("n_cases", 68)),
                "n_folds": int(metrics.get("n_folds", 3)),
                "seed": 42,
                "mean_fold_auc": None,
                "std_fold_auc": None,
                "pooled_auc": float(auc),
                "pooled_ap": metrics.get("per_fold_ap", {}).get(str(fold_idx)),
                "fold_map_path": metrics.get("fold_map_path"),
                "config_path": "scripts/her2_logn_lr_baseline.py",
                "predictions_path": "experiments/her2_phase0/logn_lr/predictions.csv",
                "notes": "one-feature LR on log1p(num_points); per-fold AUC row",
            }
        )
        rows.append(row)

    summary_row = _empty_row()
    summary_row.update(
        {
            "phase": "phase0d",
            "run_id": "logn_lr_summary",
            "model_family": "lr",
            "subgroup": "her2_intersection_n68",
            "n_cases": int(metrics.get("n_cases", 68)),
            "n_folds": int(metrics.get("n_folds", 3)),
            "seed": 42,
            "mean_fold_auc": float(metrics.get("mean_fold_auc")),
            "std_fold_auc": float(metrics.get("std_fold_auc")),
            "pooled_auc": float(metrics.get("pooled_auc")),
            "pooled_ap": float(metrics.get("pooled_ap")),
            "fold_map_path": metrics.get("fold_map_path"),
            "config_path": "scripts/her2_logn_lr_baseline.py",
            "predictions_path": "experiments/her2_phase0/logn_lr/predictions.csv",
            "notes": "one-feature LR on log1p(num_points); summary (mean over folds + pooled)",
        }
    )
    rows.append(summary_row)
    return rows


def _ds_phase3_rows(diagnostics_path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(diagnostics_path)
    df = df[df["subgroup"] == "her2_intersection"]
    df = df[df["source"].isin(["predictions.csv pooled", "predictions.csv recomputed"])]
    rows: list[dict[str, Any]] = []

    pooled = df[df["fold"] == SUMMARY_FOLD_SENTINEL].copy()
    for _, r in pooled.iterrows():
        row = _empty_row()
        row.update(
            {
                "phase": "phase3_prior",
                "run_id": f"{r['config']}__{r['seed']}",
                "model_family": "deepsets",
                "subgroup": "her2_intersection_n68",
                "n_cases": int(r["n_cases"]),
                "n_folds": 5,
                "seed": int(str(r["seed"]).replace("seed", "")),
                "mean_fold_auc": None,
                "std_fold_auc": None,
                "pooled_auc": float(r["auc"]),
                "pooled_ap": float(r["ap"]),
                "fold_map_path": None,
                "config_path": f"experiments/deepsets_phase3_repeated_cv/{r['config']}/runtime_config.yaml",
                "predictions_path": f"experiments/deepsets_phase3_repeated_cv/{r['config']}/{r['seed']}/train/.../predictions.csv",
                "notes": "Phase 3 repeated-CV winner; HER2 cases restricted to n=68 intersection",
            }
        )
        rows.append(row)

    for config, sub in pooled.groupby("config"):
        aucs = sub["auc"].astype(float).to_numpy()
        cs_row = _empty_row()
        cs_row.update(
            {
                "phase": "phase3_prior",
                "run_id": f"{config}__cross_seed",
                "model_family": "deepsets",
                "subgroup": "her2_intersection_n68",
                "n_cases": int(sub["n_cases"].iloc[0]),
                "n_folds": 5,
                "seed": "cross",
                "mean_fold_auc": None,
                "std_fold_auc": None,
                "pooled_auc": None,
                "pooled_ap": None,
                "cross_seed_mean_auc": float(np.mean(aucs)),
                "cross_seed_std_auc": float(np.std(aucs)),
                "fold_map_path": None,
                "config_path": f"experiments/deepsets_phase3_repeated_cv/{config}/runtime_config.yaml",
                "predictions_path": None,
                "notes": f"Cross-seed summary across {len(aucs)} seeds",
            }
        )
        rows.append(cs_row)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("results/her2_phase0_existing_runs.csv"),
    )
    parser.add_argument(
        "--logn-metrics",
        type=Path,
        default=Path("experiments/her2_phase0/logn_lr/metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/her2_deepsets_tracker.csv"),
    )
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    new_rows = _logn_rows(args.logn_metrics) + _ds_phase3_rows(args.diagnostics)
    new_df = pd.DataFrame(new_rows, columns=TRACKER_COLUMNS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.append and args.output.exists():
        existing = pd.read_csv(args.output)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"Appended {len(new_df)} rows; total {len(combined)} in {args.output}")
    else:
        new_df.to_csv(args.output, index=False)
        print(f"Wrote {len(new_df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
