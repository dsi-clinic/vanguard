"""Visualization functions using seaborn with a default plot theme."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# Constants
MIN_CLASSES_FOR_BINARY = 2

# Default plot theme (single place to change figure size, DPI, style for all plots)
DEFAULT_FIGURE_SIZE = (8, 6)
DEFAULT_DPI = 200
DEFAULT_STYLE = "whitegrid"
PLOT_THEME = {
    "figure_size": DEFAULT_FIGURE_SIZE,
    "dpi": DEFAULT_DPI,
    "style": DEFAULT_STYLE,
    "fontsize_label": 12,
    "fontsize_title": 14,
    "grid_alpha": 0.3,
}


def setup_figure(
    figsize: tuple[float, float] | None = None,
    style: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure and axes with the default plot theme applied.

    Parameters
    ----------
    figsize : tuple[float, float], optional
        (width, height) in inches. Defaults to PLOT_THEME["figure_size"].
    style : str, optional
        Seaborn style (e.g. "whitegrid"). Defaults to PLOT_THEME["style"].

    Returns:
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes ready for plotting.
    """
    if figsize is None:
        figsize = PLOT_THEME["figure_size"]
    if style is None:
        style = PLOT_THEME["style"]
    sns.set_style(style)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig: plt.Figure, output_path: Path, dpi: int | None = None) -> None:
    """Save figure to path using theme DPI and close it."""
    if dpi is None:
        dpi = PLOT_THEME["dpi"]
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
        return

    fig, ax = setup_figure()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random")

    fs_label = PLOT_THEME["fontsize_label"]
    fs_title = PLOT_THEME["fontsize_title"]
    ax.set_xlabel("False Positive Rate", fontsize=fs_label)
    ax.set_ylabel("True Positive Rate", fontsize=fs_label)
    ax.set_title(title, fontsize=fs_title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=PLOT_THEME["grid_alpha"])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    save_figure(fig, output_path)


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

    fig, ax = setup_figure()
    sns.histplot(arr, ax=ax, bins=min(50, max(10, arr.size // 20)), stat="density")

    if observed_auc is not None and not np.isnan(observed_auc):
        ax.axvline(
            observed_auc, color="red", linestyle="--", linewidth=2, label="Model AUC"
        )

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr)) if arr.size > 1 else 0.0
    fs_label = PLOT_THEME["fontsize_label"]
    fs_title = PLOT_THEME["fontsize_title"]
    ax.set_xlabel("AUC", fontsize=fs_label)
    ax.set_ylabel("Density", fontsize=fs_label)
    ax.set_title(title, fontsize=fs_title, fontweight="bold")
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
    ax.grid(True, alpha=PLOT_THEME["grid_alpha"])
    ax.set_xlim([0, 1])

    plt.tight_layout()
    save_figure(fig, output_path)
