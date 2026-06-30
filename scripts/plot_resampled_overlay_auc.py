#!/usr/bin/env python3
"""Overlay resampled AUC results on the Q3 baseline AUC summary plot.

Shows each arm as a horizontal row with:
  - Blue circle + error bar: Q3 baseline AUC mean ± 1 std
  - Orange dots: individual fold AUCs from the resampled run (5 folds)
  - Orange diamond: resampled AUC mean

This makes it easy to see where the resampled result falls within the
variance of the Q3 baseline.

Usage:
    python scripts/plot_resampled_overlay_auc.py \
        --baseline-csv results/independent_signal_q3_summary.csv \
        --resampled-fold-csv /ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/ablation_fold_auc.csv \
        --resampled-summary-csv /ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/ablation_summary.csv \
        --out results/resampled_vs_q3_auc_overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


ARM_ORDER = [
    "clinical_plus_tumor_size",
    "clinical_plus_tumor_size_plus_morph",
    "clinical_plus_tumor_size_plus_graph",
    "clinical_plus_tumor_size_plus_kinematic",
    "clinical_plus_tumor_size_plus_graph_plus_kinematic",
    "clinical_plus_tumor_size_plus_vessel_all",
]

ARM_LABELS = {
    "clinical_plus_tumor_size": "clinical\n+ tumor_size",
    "clinical_plus_tumor_size_plus_morph": "+ morph",
    "clinical_plus_tumor_size_plus_graph": "+ graph",
    "clinical_plus_tumor_size_plus_kinematic": "+ kinematic",
    "clinical_plus_tumor_size_plus_graph_plus_kinematic": "+ graph\n+ kinematic",
    "clinical_plus_tumor_size_plus_vessel_all": "+ vessel_all",
}

BASELINE_COLOR = "#4C72B0"   # blue
RESAMPLED_COLOR = "#DD8452"  # orange


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("results/independent_signal_q3_summary.csv"),
    )
    p.add_argument(
        "--resampled-fold-csv",
        type=Path,
        default=Path(
            "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
            "/independent_signal_resampled_ispy2/ablation_fold_auc.csv"
        ),
    )
    p.add_argument(
        "--resampled-summary-csv",
        type=Path,
        default=Path(
            "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
            "/independent_signal_resampled_ispy2/ablation_summary.csv"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/resampled_vs_q3_auc_overlay.png"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load baseline summary (mean ± std per arm) ---
    base_df = pd.read_csv(args.baseline_csv)
    if "arm_name" not in base_df.columns:
        # try alternative column names
        for col in ("run", "name"):
            if col in base_df.columns:
                base_df = base_df.rename(columns={col: "arm_name"})
                break
    base_df = base_df.set_index("arm_name")

    # --- Load resampled fold-level AUC ---
    fold_df = pd.read_csv(args.resampled_fold_csv)
    if "arm_name" not in fold_df.columns:
        for col in ("run_name", "name"):
            if col in fold_df.columns:
                fold_df = fold_df.rename(columns={col: "arm_name"})
                break

    # --- Load resampled summary (mean per arm) ---
    res_df = pd.read_csv(args.resampled_summary_csv)
    if "arm_name" not in res_df.columns:
        for col in ("run_name", "name"):
            if col in res_df.columns:
                res_df = res_df.rename(columns={col: "arm_name"})
                break
    res_df = res_df.set_index("arm_name")

    # Filter to arms present in both datasets
    arms = [a for a in ARM_ORDER if a in base_df.index and a in res_df.index]
    n = len(arms)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(13, max(4.0, 0.9 * n)))

    for i, arm in enumerate(arms):
        y = float(y_pos[i])

        # --- Q3 baseline: dot + error bar ---
        base_mean = float(base_df.loc[arm, "auc_mean"])
        base_std = float(base_df.loc[arm, "auc_std"]) if "auc_std" in base_df.columns else 0.0

        ax.errorbar(
            base_mean, y + 0.15,
            xerr=base_std,
            fmt="o",
            color=BASELINE_COLOR,
            ecolor=BASELINE_COLOR,
            elinewidth=2.0,
            capsize=5,
            markersize=9,
            zorder=4,
            alpha=0.9,
        )

        # --- Resampled: mean as diamond + error bar ---
        res_mean = float(res_df.loc[arm, "auc_mean"])
        res_std = float(res_df.loc[arm, "auc_std"]) if "auc_std" in res_df.columns else 0.0
        ax.errorbar(
            res_mean, y - 0.15,
            xerr=res_std,
            fmt="D",
            color=RESAMPLED_COLOR,
            ecolor=RESAMPLED_COLOR,
            elinewidth=2.0,
            capsize=5,
            markersize=7,
            zorder=6,
            alpha=0.9,
        )

        # --- Text labels ---
        ax.text(
            0.995, y + 0.15,
            f"{base_mean:.3f} ± {base_std:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=8, color=BASELINE_COLOR,
        )
        ax.text(
            0.995, y - 0.15,
            f"{res_mean:.3f} ± {res_std:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=8, color=RESAMPLED_COLOR,
        )

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [ARM_LABELS.get(a, a) for a in arms],
        fontsize=10,
    )
    ax.set_ylim(-0.6, n - 0.4)

    # X-axis
    all_aucs = [float(base_df.loc[a, "auc_mean"]) for a in arms]
    all_stds = [float(base_df.loc[a, "auc_std"]) if "auc_std" in base_df.columns else 0.0 for a in arms]
    res_means = [float(res_df.loc[a, "auc_mean"]) for a in arms]
    res_stds = [float(res_df.loc[a, "auc_std"]) if "auc_std" in res_df.columns else 0.0 for a in arms]

    x_min = min(
        min(m - s for m, s in zip(all_aucs, all_stds)),
        min(m - s for m, s in zip(res_means, res_stds)),
    ) - 0.02
    x_max = max(
        max(m + s for m, s in zip(all_aucs, all_stds)),
        max(m + s for m, s in zip(res_means, res_stds)),
    ) + 0.04
    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)
    ax.set_xlim(x_min, x_max)

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xlabel("Validation AUC (5-fold CV)", fontsize=12)
    ax.set_title(
        "Q3 Baseline vs. Resampled AUC by Feature Arm (ISPY2, LR)",
        fontsize=13, fontweight="bold", pad=10,
    )

    ax.xaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color=BASELINE_COLOR, label="Q3 baseline (mean ± 1 std, old code)"),
        mpatches.Patch(color=RESAMPLED_COLOR, label="Resampled (◆ mean ± 1 std, new code)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
