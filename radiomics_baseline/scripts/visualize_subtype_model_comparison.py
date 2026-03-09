#!/usr/bin/env python3
"""Bar chart comparing CV AUC of subtype-specific models vs the overall model.

Computes per-fold AUC from saved CV predictions for proper error bars.

Usage
-----
    python scripts/visualize_subtype_model_comparison.py \
        --output-dir outputs \
        --fig-path figures/subtype_model_cv_auc_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OVERALL_DIR = "rerun_bin100_kinsubonly_mrmr20"

# (label, dirname, color)
# Groups: Overall | ISPY2-only subtypes
# All-sites subtypes (no harm) | All-sites + ComBat
# Subtype-specific models, each trained on all available data for that subtype.
# Luminal A/B only exist in ISPY2, so their ISPY2 models already use all data.
MODELS = [
    ("HER2-\nenriched", "rerun_bin100_kinsubonly_mrmr20_her2_enriched_allsites"),
    ("Luminal A", "rerun_bin100_kinsubonly_mrmr20_luminal_a"),
    ("Luminal B", "rerun_bin100_kinsubonly_mrmr20_luminal_b"),
    ("Triple-\nnegative", "rerun_bin100_kinsubonly_mrmr20_triple_negative_allsites"),
]

BAR_COLOR = "#800000"


def per_fold_auc(predictions_path: Path) -> list[float]:
    """Compute AUC per CV fold from predictions CSV."""
    df = pd.read_csv(predictions_path)
    aucs = []
    for fold in sorted(df["fold"].unique()):
        fold_df = df[df["fold"] == fold]
        if len(fold_df["y_true"].unique()) >= 2:
            aucs.append(roc_auc_score(fold_df["y_true"], fold_df["y_prob"]))
    return aucs


def load_model(output_dir: Path, dirname: str) -> tuple[float, float, int]:
    """Return (mean_auc, std_auc, n_train) from a model directory.

    Tries per-fold predictions first (for proper std), falls back to
    metrics.json aggregates.
    """
    pred_path = output_dir / dirname / "training" / "cv" / "predictions.csv"
    metrics_path = output_dir / dirname / "training" / "metrics.json"

    if pred_path.exists():
        aucs = per_fold_auc(pred_path)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
    elif metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        mean_auc = m["auc_train_cv"]
        std_auc = m["auc_train_cv_std"]
    else:
        return 0.0, 0.0, 0

    # Get sample count from metrics
    n = 0
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        n = m.get("n_samples", 0)

    return mean_auc, std_auc, n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Parent directory containing model outputs",
    )
    parser.add_argument(
        "--fig-path",
        default="figures/subtype_model_cv_auc_comparison.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_path = Path(args.fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all models, skipping any that don't exist yet
    labels = []
    means = []
    stds = []
    ns = []

    for label, dirname in MODELS:
        m, s, n = load_model(output_dir, dirname)
        if m == 0.0 and s == 0.0 and n == 0:
            print(f"  [SKIP] {dirname} — not found")
            continue
        labels.append(label)
        means.append(m)
        stds.append(s)
        ns.append(n)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.3), 5.5))

    bars = ax.bar(
        x,
        means,
        0.6,
        yerr=stds,
        capsize=5,
        color=BAR_COLOR,
        edgecolor="white",
        linewidth=0.5,
        error_kw={"linewidth": 1.5, "color": "0.3"},
    )

    for bar, m, s, n in zip(bars, means, stds, ns):
        y_top = bar.get_height() + s + 0.008
        label_text = f"{m:.3f} \u00b1 {s:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_top,
            label_text,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(labels) - 0.3, 0.505, "chance", fontsize=8, color="grey", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("5-Fold CV AUC", fontsize=12)
    ax.set_ylim(0.3, 0.9)
    ax.set_title(
        "pCR Prediction: Subtype-Specific Models",
        fontsize=13,
        pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
