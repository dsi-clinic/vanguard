"""Shared plotting functions for the GNN ``graph_qc.csv`` confound audit.

Pure functions over DataFrames (no ``VanguardCenterlineDataset`` dependency),
so both the build-time writer (``gnn/data_loader.py::_write_graph_qc``, which
already has everything in memory) and the post-training writer
(``gnn/train.py``, which has ``predictions.csv``) can call the same plotting
code without importing each other. ``gnn/plot_graph_qc.py`` is the CLI that
reloads a cache from disk and calls these for ad-hoc/manual regeneration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.visualizations import PLOT_THEME, save_figure, setup_figure

GRAPH_QC_PLOTS_DIRNAME = "graph_qc_plots"
_PALETTE = sns.color_palette("colorblind")
_JITTER_RNG = np.random.default_rng(0)
_JITTER_WIDTH = 0.08


def _palette_for(categories: pd.Series) -> dict[object, tuple]:
    """Assign a fixed-order categorical color to every distinct value."""
    ordered = sorted(categories.unique(), key=str)
    return {cat: _PALETTE[i % len(_PALETTE)] for i, cat in enumerate(ordered)}


def _legend_handles(
    colors: dict[object, tuple], counts: dict[object, str] | None = None
) -> list[plt.Line2D]:
    """Build legend handles, optionally appending an ``n=...`` string per category.

    ``counts`` is required in every call site below -- every group shown on a
    graph_qc plot must say explicitly how many observations back it, since
    with only a handful of graphs a "distribution" can silently mean n=1.
    """
    handles = []
    for cat, color in colors.items():
        label = str(cat) if counts is None else f"{cat} ({counts[cat]})"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color,
                label=label,
                markersize=8,
            )
        )
    return handles


def plot_num_nodes_vs_group(
    qc: pd.DataFrame,
    x_col: str,
    color_col: str,
    out_path: Path,
    title: str,
) -> None:
    """Scatter num_nodes against a categorical x column, colored by another.

    Both axes are categorical/discrete with very few graphs (a handful per
    smoke build, hundreds for a full build), so a jittered scatter is more
    legible than a boxplot with 1-2 points per box.
    """
    fig, ax = setup_figure()
    categories = sorted(qc[x_col].astype(str).unique())
    positions = {cat: i for i, cat in enumerate(categories)}
    colors = _palette_for(qc[color_col])
    counts = {cat: f"n={n} graphs" for cat, n in qc[color_col].value_counts().items()}

    for _, row in qc.iterrows():
        x = positions[str(row[x_col])] + _JITTER_RNG.uniform(
            -_JITTER_WIDTH, _JITTER_WIDTH
        )
        ax.scatter(
            x,
            row["num_nodes"],
            color=colors[row[color_col]],
            s=90,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel("num_nodes")
    ax.set_title(
        f"{title} (n={len(qc)} graphs total)",
        fontsize=PLOT_THEME["fontsize_title"],
        fontweight="bold",
    )
    ax.legend(
        handles=_legend_handles(colors, counts),
        title=color_col,
        loc="upper right",
        fontsize=9,
    )
    ax.grid(True, alpha=PLOT_THEME["grid_alpha"])
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_prediction_vs_num_nodes(
    predictions: pd.DataFrame, out_path: Path, pcr_col: str = "pcr"
) -> None:
    """Scatter predicted probability against graph size, colored by true pcr."""
    fig, ax = setup_figure()
    colors = _palette_for(predictions[pcr_col])
    counts = {
        cat: f"n={n} cases" for cat, n in predictions[pcr_col].value_counts().items()
    }

    for _, row in predictions.iterrows():
        ax.scatter(
            row["num_nodes"],
            row["y_prob"],
            color=colors[row[pcr_col]],
            s=90,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("num_nodes")
    ax.set_ylabel("predicted P(pcr)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Prediction vs num_nodes (n={len(predictions)} cases total)",
        fontsize=PLOT_THEME["fontsize_title"],
        fontweight="bold",
    )
    ax.legend(
        handles=_legend_handles(colors, counts),
        title=pcr_col,
        loc="upper right",
        fontsize=9,
    )
    ax.grid(True, alpha=PLOT_THEME["grid_alpha"])
    plt.tight_layout()
    save_figure(fig, out_path)


def plot_feature_distributions(
    node_df: pd.DataFrame, node_features: list[str], group_col: str, out_path: Path
) -> None:
    """Node-level density histograms of every feature, split by ``group_col``.

    Unfilled step outlines (not filled areas): with ``fill=True``, overlapping
    bins from different groups visually read as *stacked* rather than
    independently-overlaid densities, which misrepresents ``common_norm=False``
    per-group normalization.

    Each line here pools every *node* (voxel) across every graph in that
    group -- with only a handful of graphs, a group can be a single graph's
    thousands of (non-independent, spatially autocorrelated) nodes, not a
    between-case distribution. Both the number of graphs and the number of
    nodes behind each line are put in the legend explicitly so that's never
    left implicit.
    """
    colors = _palette_for(node_df[group_col])
    hue_order = sorted(colors, key=str)
    n_graphs = node_df.groupby(group_col)["case_id"].nunique()
    n_nodes = node_df.groupby(group_col).size()
    counts = {
        cat: f"n={n_graphs[cat]} graphs, {n_nodes[cat]} nodes" for cat in hue_order
    }

    fig, axes = plt.subplots(
        1, len(node_features), figsize=(6 * len(node_features), 5), squeeze=False
    )
    axes = axes[0]
    for ax, name in zip(axes, node_features, strict=True):
        sns.histplot(
            data=node_df,
            x=name,
            hue=group_col,
            hue_order=hue_order,
            palette=colors,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            linewidth=2,
            bins=30,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{name} by {group_col}", fontsize=PLOT_THEME["fontsize_title"])
        ax.legend(
            handles=_legend_handles(colors, counts),
            title=group_col,
            loc="upper right",
            fontsize=8,
        )
        ax.grid(True, alpha=PLOT_THEME["grid_alpha"])
    plt.tight_layout()
    save_figure(fig, out_path)


def write_build_time_plots(
    qc: pd.DataFrame, node_df: pd.DataFrame, node_features: list[str], out_dir: Path
) -> None:
    """Write the 4 QC plots derivable at build time (no predictions needed).

    ``num_nodes_vs_pcr.png``, ``num_nodes_vs_dataset.png``,
    ``feature_distributions_by_dataset.png``, ``feature_distributions_by_pcr.png``.
    ``prediction_vs_num_nodes.png`` is not included here -- it needs a trained
    model's predictions, which don't exist at build time (see
    ``write_prediction_plot``, called from ``gnn/train.py``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_num_nodes_vs_group(
        qc,
        x_col="pcr",
        color_col="dataset",
        out_path=out_dir / "num_nodes_vs_pcr.png",
        title="num_nodes vs pcr",
    )
    plot_num_nodes_vs_group(
        qc,
        x_col="dataset",
        color_col="pcr",
        out_path=out_dir / "num_nodes_vs_dataset.png",
        title="num_nodes vs dataset",
    )
    plot_feature_distributions(
        node_df,
        node_features,
        group_col="dataset",
        out_path=out_dir / "feature_distributions_by_dataset.png",
    )
    plot_feature_distributions(
        node_df,
        node_features,
        group_col="pcr",
        out_path=out_dir / "feature_distributions_by_pcr.png",
    )


def write_prediction_plot(
    predictions: pd.DataFrame, case_num_nodes: pd.DataFrame, out_dir: Path
) -> None:
    """Write ``prediction_vs_num_nodes.png``, merging predictions with num_nodes.

    ``case_num_nodes`` just needs ``case_id`` and ``num_nodes`` columns -- a
    loaded ``graph_qc.csv`` or a training run's ``cohort_df`` both work.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = predictions.merge(
        case_num_nodes[["case_id", "num_nodes"]], on="case_id", how="inner"
    )
    plot_prediction_vs_num_nodes(merged, out_dir / "prediction_vs_num_nodes.png")
