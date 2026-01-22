"""Visualization functions using seaborn for enhanced aesthetics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# Visualization registry - maps visualization names to plotting functions
VISUALIZATION_REGISTRY: dict[str, callable] = {}


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    """
    Plot ROC curve with AUC using seaborn.
    
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
    if len(unique_labels) < 2:
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
