#!/usr/bin/env python3
"""Append the Phase 0e XGB rows to results/her2_deepsets_tracker.csv.

Reads experiments/her2_phase0/xgb_vessel_all/metrics.json and writes one
per-fold row plus one summary row in the canonical tracker schema.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def _empty() -> dict[str, object | None]:
    return {col: None for col in TRACKER_COLUMNS}


def main() -> int:
    metrics_path = Path("experiments/her2_phase0/xgb_vessel_all/metrics.json")
    tracker_path = Path("results/her2_deepsets_tracker.csv")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    new_rows: list[dict[str, object | None]] = []
    for fold_idx, auc in metrics["per_fold_auc"].items():
        row = _empty()
        row.update(
            {
                "phase": "phase0e",
                "run_id": f"xgb_vessel_all_fold{fold_idx}",
                "model_family": "xgb",
                "subgroup": "her2_intersection_n68",
                "n_cases": int(metrics["n_cases"]),
                "n_folds": int(metrics["n_folds"]),
                "seed": int(metrics["xgb_params"]["random_state"]),
                "pooled_auc": float(auc),
                "pooled_ap": metrics["per_fold_ap"].get(str(fold_idx)),
                "fold_map_path": metrics["fold_map_path"],
                "config_path": "scripts/her2_xgb_baseline.py",
                "predictions_path": "experiments/her2_phase0/xgb_vessel_all/predictions.csv",
                "notes": "XGB vessel_all, HER2-only training (n=68 intersection); per-fold row",
            }
        )
        new_rows.append(row)

    summary = _empty()
    summary.update(
        {
            "phase": "phase0e",
            "run_id": "xgb_vessel_all_summary",
            "model_family": "xgb",
            "subgroup": "her2_intersection_n68",
            "n_cases": int(metrics["n_cases"]),
            "n_folds": int(metrics["n_folds"]),
            "seed": int(metrics["xgb_params"]["random_state"]),
            "mean_fold_auc": float(metrics["mean_fold_auc"]),
            "std_fold_auc": float(metrics["std_fold_auc"]),
            "pooled_auc": float(metrics["pooled_auc"]),
            "pooled_ap": float(metrics["pooled_ap"]),
            "fold_map_path": metrics["fold_map_path"],
            "config_path": "scripts/her2_xgb_baseline.py",
            "predictions_path": "experiments/her2_phase0/xgb_vessel_all/predictions.csv",
            "notes": (
                f"XGB vessel_all, HER2-only training (n=68 intersection); "
                f"summary. Compare to existing full-cohort 5-fold HER2-stratum "
                f"AUC 0.6758 from results/model_family_robustness_ispy2_subtype_summary.csv."
            ),
        }
    )
    new_rows.append(summary)

    new_df = pd.DataFrame(new_rows, columns=TRACKER_COLUMNS)
    existing = pd.read_csv(tracker_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(tracker_path, index=False)
    print(f"Appended {len(new_df)} XGB rows to {tracker_path} (total {len(combined)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
