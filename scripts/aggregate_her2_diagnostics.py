#!/usr/bin/env python3
"""Phase 0a aggregator for HER2 Deep Sets diagnostics.

Walks ``experiments/deepsets_phase3_repeated_cv/`` (or any user-supplied
sweep root), reads ``validation_summary.by_group.her2_enriched.{auc,ap}``
from each ``metrics.json``, and re-computes per-fold HER2 metrics from
the per-case ``predictions.csv`` so paired-fold statistics are usable
downstream.

Output: a long-format CSV with one row per (config, seed, fold) plus
a separate per-config / per-seed summary row (``fold == -1``).

Usage::

    python scripts/aggregate_her2_diagnostics.py \\
        --sweep-root experiments/deepsets_phase3_repeated_cv \\
        --output results/her2_phase0_existing_runs.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HER2_STRATUM_VALUE = "her2_enriched"
MIN_CLASSES_FOR_AUC = 2
SUMMARY_FOLD_SENTINEL = -1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("experiments/deepsets_phase3_repeated_cv"),
        help="Directory whose immediate subdirectories are configs, each "
        "containing seed* subdirs with a train/<run>/<exp>/ output tree.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/her2_phase0_existing_runs.csv"),
        help="Destination CSV.",
    )
    parser.add_argument(
        "--intersection-cohort",
        type=Path,
        default=None,
        help="Optional CSV with a ``case_id`` column. If provided, an extra "
        "set of rows with ``subgroup='her2_intersection'`` is appended for "
        "the HER2 \u2229 tabular cohort.",
    )
    return parser.parse_args()


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < MIN_CLASSES_FOR_AUC:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < MIN_CLASSES_FOR_AUC:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _find_metrics_and_predictions(seed_dir: Path) -> tuple[Path, Path] | None:
    """Locate the (metrics.json, predictions.csv) pair beneath a seed dir.

    The tree is ``seed*/train/<runstamp>/<experiment_name>/{metrics.json,predictions.csv}``.
    """
    metrics_candidates = list(seed_dir.rglob("metrics.json"))
    preds_candidates = list(seed_dir.rglob("predictions.csv"))
    if not metrics_candidates or not preds_candidates:
        return None
    metrics_candidates.sort()
    preds_candidates.sort()
    return metrics_candidates[0], preds_candidates[0]


def _aggregate_summary_from_json(
    metrics_path: Path, config_name: str, seed_name: str
) -> dict[str, float | str | int]:
    """Read pre-computed HER2 aggregate AUC/AP from metrics.json."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    vs = metrics.get("validation_summary", {}) or {}
    by_group = vs.get("by_group", {}) or {}
    her2 = by_group.get(HER2_STRATUM_VALUE, {}) or {}
    overall = vs.get("overall", {}) or {}
    return {
        "config": config_name,
        "seed": seed_name,
        "fold": SUMMARY_FOLD_SENTINEL,
        "subgroup": HER2_STRATUM_VALUE,
        "n_cases": int(her2.get("n", 0)) if "n" in her2 else None,
        "auc": float(her2.get("auc", float("nan"))),
        "ap": float(her2.get("ap", float("nan"))),
        "overall_auc": float(overall.get("auc", float("nan"))),
        "overall_ap": float(overall.get("ap", float("nan"))),
        "source": "metrics.json/validation_summary",
    }


def _per_fold_her2_rows(
    predictions_path: Path,
    config_name: str,
    seed_name: str,
    subgroup_label: str,
    case_filter: set[str] | None,
) -> list[dict[str, float | str | int]]:
    """Compute per-fold HER2 AUC/AP from a predictions.csv.

    If ``case_filter`` is given, predictions are restricted to that
    case_id set first (used for the HER2 \u2229 tabular intersection cohort).
    """
    df = pd.read_csv(predictions_path)
    required = {"case_id", "y_true", "y_prob", "fold", "stratum"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{predictions_path} missing columns {missing}; have {list(df.columns)}"
        )

    df["case_id"] = df["case_id"].astype(str)
    her2_df = df[df["stratum"] == HER2_STRATUM_VALUE].copy()
    if case_filter is not None:
        her2_df = her2_df[her2_df["case_id"].isin(case_filter)]
    if her2_df.empty:
        return []

    rows: list[dict[str, float | str | int]] = []
    for fold_idx, fold_df in her2_df.groupby("fold"):
        y_true = fold_df["y_true"].to_numpy()
        y_prob = fold_df["y_prob"].to_numpy()
        rows.append(
            {
                "config": config_name,
                "seed": seed_name,
                "fold": int(fold_idx),
                "subgroup": subgroup_label,
                "n_cases": int(len(fold_df)),
                "auc": _safe_auc(y_true, y_prob),
                "ap": _safe_ap(y_true, y_prob),
                "overall_auc": float("nan"),
                "overall_ap": float("nan"),
                "source": "predictions.csv recomputed",
            }
        )

    y_true_all = her2_df["y_true"].to_numpy()
    y_prob_all = her2_df["y_prob"].to_numpy()
    rows.append(
        {
            "config": config_name,
            "seed": seed_name,
            "fold": SUMMARY_FOLD_SENTINEL,
            "subgroup": subgroup_label,
            "n_cases": int(len(her2_df)),
            "auc": _safe_auc(y_true_all, y_prob_all),
            "ap": _safe_ap(y_true_all, y_prob_all),
            "overall_auc": float("nan"),
            "overall_ap": float("nan"),
            "source": "predictions.csv pooled",
        }
    )
    return rows


def collect_rows(
    sweep_root: Path,
    intersection_case_ids: set[str] | None,
) -> pd.DataFrame:
    """Walk the sweep root and return one long-format DataFrame of HER2 metrics."""
    if not sweep_root.exists():
        raise FileNotFoundError(f"--sweep-root {sweep_root} does not exist")

    all_rows: list[dict[str, float | str | int]] = []
    for config_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        config_name = config_dir.name
        for seed_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            seed_name = seed_dir.name
            pair = _find_metrics_and_predictions(seed_dir)
            if pair is None:
                logging.warning(
                    "Skipping %s/%s: no metrics.json+predictions.csv pair found",
                    config_name,
                    seed_name,
                )
                continue
            metrics_path, predictions_path = pair

            all_rows.append(
                _aggregate_summary_from_json(metrics_path, config_name, seed_name)
            )
            all_rows.extend(
                _per_fold_her2_rows(
                    predictions_path=predictions_path,
                    config_name=config_name,
                    seed_name=seed_name,
                    subgroup_label=HER2_STRATUM_VALUE,
                    case_filter=None,
                )
            )
            if intersection_case_ids is not None:
                all_rows.extend(
                    _per_fold_her2_rows(
                        predictions_path=predictions_path,
                        config_name=config_name,
                        seed_name=seed_name,
                        subgroup_label="her2_intersection",
                        case_filter=intersection_case_ids,
                    )
                )

    return pd.DataFrame(all_rows)


def _load_intersection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"--intersection-cohort {path} does not exist")
    df = pd.read_csv(path)
    if "case_id" not in df.columns:
        raise ValueError(f"{path} missing required 'case_id' column")
    return set(df["case_id"].astype(str).unique())


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows collected.")
        return
    summary = df[df["fold"] == SUMMARY_FOLD_SENTINEL].copy()
    summary = summary[summary["source"].str.contains("pooled|validation_summary")]
    print("\nPer-config / per-seed HER2 AUC summary (fold == -1 rows):")
    print(
        summary[
            ["config", "seed", "subgroup", "n_cases", "auc", "ap", "source"]
        ]
        .sort_values(["config", "seed", "subgroup", "source"])
        .to_string(index=False)
    )

    print("\nCross-seed mean by (config, subgroup) from validation_summary rows:")
    cross = (
        summary[summary["source"] == "metrics.json/validation_summary"]
        .groupby(["config", "subgroup"])["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print(cross.to_string(index=False))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    intersection = _load_intersection(args.intersection_cohort)
    df = collect_rows(args.sweep_root, intersection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    _print_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
