#!/usr/bin/env python3
"""Bar chart of 5-fold CV AUC by molecular subtype for the best model.

Computes per-fold per-subtype AUC from the saved CV predictions to get
proper standard deviation error bars.

Usage
-----
    python scripts/visualize_best_model_subtype.py \
        --model-dir outputs/rerun_bin100_kinsubonly_mrmr20 \
        --fig-path figures/best_model_subtype_auc.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SUBTYPES = ["her2_enriched", "luminal_a", "luminal_b", "triple_negative"]
SUBTYPE_LABELS = ["HER2-\nenriched", "Luminal A", "Luminal B", "Triple-\nnegative"]
MIN_CLASSES_FOR_AUC = 2


def compute_fold_subtype_aucs(predictions_path: Path) -> dict[str, list[float]]:
    """Compute AUC per fold per subtype from CV predictions CSV."""
    pred_df = pd.read_csv(predictions_path)
    folds = sorted(pred_df["fold"].unique())

    result: dict[str, list[float]] = {s: [] for s in SUBTYPES}
    result["overall"] = []

    for fold in folds:
        fold_df = pred_df[pred_df["fold"] == fold]

        # Overall
        if len(fold_df["y_true"].unique()) >= MIN_CLASSES_FOR_AUC:
            result["overall"].append(
                roc_auc_score(fold_df["y_true"], fold_df["y_prob"])
            )

        # Per subtype
        for subtype in SUBTYPES:
            sub_df = fold_df[fold_df["subtype"] == subtype]
            if (
                len(sub_df) >= MIN_CLASSES_FOR_AUC
                and len(sub_df["y_true"].unique()) >= MIN_CLASSES_FOR_AUC
            ):
                result[subtype].append(
                    roc_auc_score(sub_df["y_true"], sub_df["y_prob"])
                )

    return result


def main() -> None:
    """Render the best-model subtype AUC bar chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="outputs/rerun_bin100_kinsubonly_mrmr20",
        help="Model output directory containing training/cv/predictions.csv",
    )
    parser.add_argument(
        "--fig-path",
        default="figures/best_model_subtype_auc.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title (two lines separated by \\n)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    fig_path = Path(args.fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_path = model_dir / "training" / "cv" / "predictions.csv"
    fold_aucs = compute_fold_subtype_aucs(predictions_path)

    groups = ["overall"] + SUBTYPES
    group_labels = ["Overall"] + SUBTYPE_LABELS
    means = [np.mean(fold_aucs[g]) if fold_aucs[g] else 0 for g in groups]
    stds = [np.std(fold_aucs[g]) if fold_aucs[g] else 0 for g in groups]

    x = np.arange(len(groups))
    colors = ["#3C5488", "#E64B35", "#4DBBD5", "#00A087", "#8491B4"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars = ax.bar(
        x,
        means,
        0.55,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        error_kw={"linewidth": 1.5, "color": "0.3"},
    )

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.008,
            f"{m:.3f} \u00b1 {s:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(groups) - 0.4, 0.505, "chance", fontsize=8, color="grey", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_ylim(0.35, 0.95)
    if args.title:
        title = args.title.replace("\\n", "\n")
    else:
        # Derive a reasonable title from the model directory name
        dirname = model_dir.name
        if "all" in dirname:
            feat_desc = "DCE + kinetic + subtraction"
        elif "kinsubonly" in dirname:
            feat_desc = "kinetic + subtraction"
        else:
            feat_desc = "radiomics"
        title = (
            f"Best Model (mRMR-20, {feat_desc}, bin100)\n"
            "5-Fold CV AUC by Molecular Subtype"
        )
    ax.set_title(title, fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
