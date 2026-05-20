"""Generate a presentation-quality bar chart comparing AUC across model families.

Reads the model family matrix summary CSV and produces a grouped bar chart
with error bars (±1 std) for each (feature arm, model family) combination.

Usage:
    python scripts/plot_model_auc_comparison.py [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Friendly labels for feature arms and model families
ARM_LABELS = {
    "clinical_plus_tumor_size": "Clinical + Tumor Size",
    "clinical_plus_tumor_size_plus_kinematic": "Clinical + Tumor Size\n+ Kinematic",
    "clinical_plus_tumor_size_plus_vessel_all": "Clinical + Tumor Size\n+ Vessel (All)",
}

MODEL_LABELS = {
    "lr": "Linear (Elastic Net)",
    "rf": "Random Forest",
    "xgb": "Nonlinear (XGBoost)",
}

# Colour palette – one colour per model family
MODEL_COLORS = {
    "lr": "#4C72B0",
    "rf": "#55A868",
    "xgb": "#C44E52",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the AUC comparison chart."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("results/model_family_matrix_ispy2_summary.csv"),
        help="Path to summary CSV with columns: arm_name, model_family, auc_mean, auc_std",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/model_auc_comparison.png"),
        help="Output path for the chart (PNG).",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["lr", "xgb"],
        help="Model families to include (default: lr xgb).",
    )
    return ap.parse_args()


def plot_auc_comparison(
    summary: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> None:
    """Create a grouped bar chart of AUC (mean ± std) per arm and model."""
    summary = summary[summary["model_family"].isin(models)].copy()
    if summary.empty:
        raise ValueError(f"No rows for models {models}")

    # Friendly labels
    summary["arm_label"] = (
        summary["arm_name"].map(ARM_LABELS).fillna(summary["arm_name"])
    )
    summary["model_label"] = (
        summary["model_family"].map(MODEL_LABELS).fillna(summary["model_family"])
    )

    arms = summary["arm_label"].unique()
    n_arms = len(arms)
    n_models = len(models)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(max(8, n_arms * 2.2), 5))

    bar_width = 0.28
    x = np.arange(n_arms)

    for i, model_key in enumerate(models):
        subset = summary[summary["model_family"] == model_key].set_index("arm_label")
        means = [subset.loc[a, "auc_mean"] if a in subset.index else 0 for a in arms]
        stds = [subset.loc[a, "auc_std"] if a in subset.index else 0 for a in arms]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=stds,
            capsize=4,
            color=MODEL_COLORS.get(model_key, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
            label=MODEL_LABELS.get(model_key, model_key),
            zorder=3,
        )
        # Value labels on bars
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.005,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=11)
    ax.set_ylabel("AUC (mean ± std)", fontsize=13)
    ax.set_title(
        "Linear vs Nonlinear Model AUC Comparison",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0.4, max(summary["auc_mean"] + summary["auc_std"]) + 0.08)
    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label="Random (0.5)",
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # High-level conclusion annotation
    best_row = summary.loc[summary["auc_mean"].idxmax()]
    conclusion = (
        "Both linear and nonlinear models perform comparably,\n"
        "suggesting the predictive signal is largely linear.\n"
        f"Best: {MODEL_LABELS.get(best_row['model_family'], best_row['model_family'])} "
        f"on {best_row['arm_label']} (AUC = {best_row['auc_mean']:.3f})"
    )
    ax.text(
        0.98,
        0.03,
        conclusion,
        transform=ax.transAxes,
        fontsize=9.5,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f0f0f0",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart to {output_path}")


def main() -> None:
    """Run the AUC comparison chart generation."""
    args = parse_args()
    summary = pd.read_csv(args.csv)
    plot_auc_comparison(summary, args.models, args.out)


if __name__ == "__main__":
    main()
