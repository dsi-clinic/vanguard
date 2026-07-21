"""Phase 5 analysis: paired arm1-vs-arm2 delta-AUC, fold variance, arm3 ceiling.

Per the approved harmonized-single-breast-dataset plan (LAB_NOTEBOOK.md
2026-07-14/15): the primary statistic is paired delta-AUC = AUC(arm2) -
AUC(arm1), paired by (fold, seed) -- not a naive mean-of-means comparison --
since folds are frozen/shared between arms 1 and 2 (identical 1332-case
cohort, `analysis/assert_arm_parity.py` confirmed) and per-fold variance is
exactly the nuisance factor pairing cancels out. Arm 3 (native-unilateral-
only) is a ceiling/sanity-check only, on a different, smaller, non-matched
cohort -- reported separately, not pooled into the primary statistic.

Usage::

    python -m analysis.harmonized_single_breast_analysis \
        --root experiments/harmonized_single_breast_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

_ARMS = (
    "gnn_voxel_mixed_baseline",
    "gnn_voxel_single_breast",
    "gnn_voxel_unilateral_only",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/harmonized_single_breast_v1")
    )
    return parser.parse_args()


def load_per_fold_metrics(root: Path) -> pd.DataFrame:
    """Collect (arm, seed, fold, auc, ap) rows from every metrics.json under root."""
    rows = []
    for arm in _ARMS:
        arm_dir = root / arm
        if not arm_dir.exists():
            raise FileNotFoundError(
                f"No results directory for arm {arm!r} under {root}"
            )
        for seed_dir in sorted(arm_dir.glob("seed_*")):
            seed = int(seed_dir.name.removeprefix("seed_"))
            (metrics_path,) = seed_dir.glob(f"*/{arm}/metrics.json")
            metrics = json.loads(metrics_path.read_text())
            for fold_row in metrics["per_fold_metrics"]:
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "fold": fold_row["fold"],
                        "auc": fold_row["auc"],
                        "ap": fold_row["ap"],
                    }
                )
    metrics_df = pd.DataFrame(rows)
    missing = [
        (arm, seed)
        for arm in _ARMS
        for seed in metrics_df["seed"].unique()
        if not ((metrics_df["arm"] == arm) & (metrics_df["seed"] == seed)).any()
    ]
    if missing:
        raise ValueError(f"Missing (arm, seed) combinations: {missing}")
    return metrics_df


def paired_arm1_vs_arm2(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Delta-AUC = arm2 - arm1, paired by (fold, seed)."""
    arm1 = metrics_df[metrics_df["arm"] == "gnn_voxel_mixed_baseline"].set_index(
        ["seed", "fold"]
    )["auc"]
    arm2 = metrics_df[metrics_df["arm"] == "gnn_voxel_single_breast"].set_index(
        ["seed", "fold"]
    )["auc"]
    paired = pd.DataFrame({"auc_arm1_mixed": arm1, "auc_arm2_single": arm2})
    paired["delta"] = paired["auc_arm2_single"] - paired["auc_arm1_mixed"]
    return paired.reset_index()


def fold_variance_by_arm(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Std of fold AUC within each (arm, seed), then averaged across seeds."""
    per_seed_std = metrics_df.groupby(["arm", "seed"])["auc"].std().rename("fold_std")
    return per_seed_std.groupby("arm").agg(["mean", "std", "count"])


def main() -> None:
    """Load all runs, compute the paired statistic and fold variance, apply the stop rule."""
    args = _parse_args()
    metrics_df = load_per_fold_metrics(args.root)

    print("=== Per-arm aggregated AUC (mean +/- std across seed x fold) ===")
    print(
        metrics_df.groupby("arm")["auc"].agg(["mean", "std", "count"]).loc[list(_ARMS)]
    )
    print()

    print("=== Paired arm2-vs-arm1 delta-AUC, by (seed, fold) ===")
    paired = paired_arm1_vs_arm2(metrics_df)
    print(paired.to_string(index=False))
    print()
    n_positive = int((paired["delta"] > 0).sum())
    n_total = len(paired)
    mean_delta = paired["delta"].mean()
    print(f"Positive deltas: {n_positive}/{n_total}, mean delta = {mean_delta:.4f}")
    stat, p_value = wilcoxon(paired["delta"])
    print(
        f"Wilcoxon signed-rank: statistic={stat:.2f}, p={p_value:.4f} (directional evidence, n={n_total} is thin for a formal claim)"
    )
    print()

    print(
        "=== Fold-to-fold AUC variance by arm (std of fold AUC within each seed, then averaged) ==="
    )
    variance = fold_variance_by_arm(metrics_df)
    print(variance)
    print()

    arm1_std = variance.loc["gnn_voxel_mixed_baseline", "mean"]
    arm2_std = variance.loc["gnn_voxel_single_breast", "mean"]
    variance_shrank = arm2_std < arm1_std
    print(
        f"Arm1 mean fold-std: {arm1_std:.4f}, Arm2 mean fold-std: {arm2_std:.4f} (shrank: {variance_shrank})"
    )
    print()

    print("=== Arm 3 (native-unilateral-only) ceiling check -- NOT case-matched ===")
    arm3_mean = metrics_df[metrics_df["arm"] == "gnn_voxel_unilateral_only"][
        "auc"
    ].mean()
    arm2_mean = metrics_df[metrics_df["arm"] == "gnn_voxel_single_breast"]["auc"].mean()
    arm1_mean = metrics_df[metrics_df["arm"] == "gnn_voxel_mixed_baseline"][
        "auc"
    ].mean()
    print(f"Arm1 (mixed) mean AUC:       {arm1_mean:.4f}")
    print(f"Arm2 (single-breast) mean AUC: {arm2_mean:.4f}")
    print(f"Arm3 (unilateral-only) mean AUC (ceiling): {arm3_mean:.4f}")
    print()

    print("=== Stop rule ===")
    consistently_positive = n_positive >= int(
        0.8 * n_total
    )  # informal "most" bar, not a hard p<0.05 gate
    if consistently_positive and variance_shrank:
        print(
            "PROCEED: paired delta is consistently positive and fold variance shrank -- "
            "rerun the 3-arm matrix for segment and junction node modes."
        )
    else:
        print(
            "STOP: paired delta is not consistently positive across seeds/folds, "
            "and/or fold-to-fold variance did not shrink. Do not proceed to segment/junction "
            "representations -- write up the negative result and revisit other hypotheses "
            "(e.g. site/dataset confound, per configs/gnn_segment_smoke_site_confound_check.yaml)."
        )


if __name__ == "__main__":
    main()
