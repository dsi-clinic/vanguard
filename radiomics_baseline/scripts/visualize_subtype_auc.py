#!/usr/bin/env python3
"""Grouped bar chart comparing test AUC by molecular subtype across models.

Reads metrics.json from each rerun output directory and plots overall + per-
subtype test AUCs side-by-side.

Usage
-----
    python scripts/visualize_subtype_auc.py \
        --output-dir outputs \
        --fig-path figures/subtype_auc_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Model display names (short) and their output directory basenames.
MODELS = [
    ("mRMR-20\n(kinetic, corr0.90)", "rerun_bin100_kinsubonly_mrmr20"),
    ("mRMR-20\n(all, corr0.80)", "rerun_bin100_all_mrmr20_corr080"),
    ("kBest-60\n(kinetic, corr0.95)", "rerun_bin100_kinsubonly_kbest60_corr095"),
    ("kBest-50\n(kinetic, bin8, corr0.95)", "rerun_bin8_kinsubonly_kbest50_corr095"),
    ("mRMR-20\n(all, corr0.70)", "rerun_bin100_all_mrmr20_corr070"),
    ("kBest-40\n(kinetic, corr0.80)", "rerun_bin100_kinsubonly_kbest40_corr080"),
]

SUBTYPES = ["her2_enriched", "luminal_a", "luminal_b", "triple_negative"]
SUBTYPE_LABELS = ["HER2-enriched", "Luminal A", "Luminal B", "Triple-negative"]
SUBTYPE_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#8491B4"]
OVERALL_COLOR = "#3C5488"


def load_metrics(output_dir: Path) -> list[dict | None]:
    """Load metrics.json for each model; return None if missing."""
    results = []
    for _, dirname in MODELS:
        mpath = output_dir / dirname / "training" / "metrics.json"
        if mpath.exists():
            results.append(json.loads(mpath.read_text()))
        else:
            results.append(None)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Parent directory containing rerun_* model outputs",
    )
    parser.add_argument(
        "--fig-path",
        default="figures/subtype_auc_comparison.png",
        help="Path for output figure",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_path = Path(args.fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    all_metrics = load_metrics(output_dir)

    n_models = len(MODELS)
    n_groups = len(SUBTYPES) + 1  # overall + 4 subtypes
    x = np.arange(n_models)
    bar_width = 0.15
    offsets = np.arange(n_groups) - (n_groups - 1) / 2

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot overall bars
    overall_aucs = []
    for m in all_metrics:
        overall_aucs.append(m["auc_test"] if m else 0)

    bars = ax.bar(
        x + offsets[0] * bar_width,
        overall_aucs,
        bar_width,
        label="Overall",
        color=OVERALL_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars, overall_aucs):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )

    # Plot per-subtype bars
    for si, (subtype_key, subtype_label, color) in enumerate(
        zip(SUBTYPES, SUBTYPE_LABELS, SUBTYPE_COLORS)
    ):
        vals = []
        for m in all_metrics:
            if m and subtype_key in m.get("auc_test_by_subtype", {}):
                vals.append(m["auc_test_by_subtype"][subtype_key])
            else:
                vals.append(0)

        bars = ax.bar(
            x + offsets[si + 1] * bar_width,
            vals,
            bar_width,
            label=subtype_label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )

    # Reference line at chance
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(n_models - 0.5, 0.505, "chance", fontsize=8, color="grey", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in MODELS], fontsize=9)
    ax.set_ylabel("Test AUC", fontsize=12)
    ax.set_title(
        "pCR Prediction: Test AUC by Molecular Subtype Across Top Models",
        fontsize=13,
        pad=12,
    )
    ax.set_ylim(0.4, 0.82)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved subtype AUC comparison → {fig_path}")


if __name__ == "__main__":
    main()
