"""Visualization functions using seaborn for enhanced aesthetics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# Constants
MIN_CLASSES_FOR_BINARY = 2

# Visualization registry - maps visualization names to plotting functions
VISUALIZATION_REGISTRY: dict[str, callable] = {}


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    """Plot ROC curve with AUC using seaborn.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities for positive class
    output_path : Path
        Path to save the plot
    title : str, default="ROC Curve"
        Plot title
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Check if we have both classes
    unique_labels = np.unique(y_true)
    if len(unique_labels) < MIN_CLASSES_FOR_BINARY:
        # Can't plot ROC with only one class
        return

    # Set seaborn style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Plot using seaborn
    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f"AUC = {auc:.3f}")

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random")

    # Styling
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# Register ROC curve visualization
VISUALIZATION_REGISTRY["roc_curve"] = plot_roc_curve


def plot_random_auc_distribution(
    auc_values: list[float] | np.ndarray,
    output_path: Path | str,
    observed_auc: float | None = None,
    title: str = "Random baseline AUC distribution",
) -> None:
    """Plot histogram of random-model AUCs; optional vertical line at observed AUC.

    Standalone function (not in VISUALIZATION_REGISTRY). Call with auc_values
    from compute_random_auc_distribution and optional observed_auc for comparison.

    Parameters
    ----------
    auc_values : list of float or np.ndarray
        AUCs from the null distribution.
    output_path : Path or str
        Path to save the plot.
    observed_auc : float, optional
        AUC of the real model; if provided, a vertical line is drawn.
    title : str, default="Random baseline AUC distribution"
        Plot title.
    """
    output_path = Path(output_path)
    arr = np.asarray(auc_values)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(arr, ax=ax, bins=min(50, max(10, arr.size // 20)), stat="density")

    if observed_auc is not None and not np.isnan(observed_auc):
        ax.axvline(
            observed_auc, color="red", linestyle="--", linewidth=2, label="Model AUC"
        )

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr)) if arr.size > 1 else 0.0
    ax.set_xlabel("AUC", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    text = f"mean = {mean_val:.3f}, std = {std_val:.3f}, n = {len(arr)}"
    if observed_auc is not None and not np.isnan(observed_auc):
        from evaluation.random_baseline import empirical_p_value, z_score

        z = z_score(observed_auc, mean_val, std_val)
        p_val = empirical_p_value(arr.tolist(), observed_auc)
        text += f"\nz = {z:.3f}, P(random ≥ observed) = {p_val:.4f}"
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes, fontsize=10, verticalalignment="top"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
