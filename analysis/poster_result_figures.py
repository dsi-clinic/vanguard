"""Render the two results figures for the symposium poster, one per sub-question.

The poster asks one question in two parts, and each part gets its own figure:

  q1_signal.png       Is there signal in the blood vessels? Pooled out-of-fold
                      AUC with bootstrap 95% CIs for the vessel models and for
                      the negative control (the two node features that carry no
                      contrast-enhancement information), against chance.
  q2_graph_delta.png  Does the graph representation add anything? Paired
                      bootstrap AUC differences -- graph model minus the tabular
                      summary of the same features, and each structural graph
                      mode minus the voxel graph -- against zero.

Both read the committed bootstrap outputs rather than recomputing anything, so
the figures cannot drift from the numbers in the experiment READMEs.

Usage:
    python -m analysis.poster_result_figures \
        --experiments-root experiments \
        --out-dir presentations/symposium_poster_2026-08
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

# One colour for "a real vessel model", one contrasting colour reserved for the
# negative control -- the only place on either figure where colour marks a
# different kind of thing rather than decoration.
MODEL = "#101820"
CONTROL = "#B1B3B3"
REFERENCE = "#C5050C"

LABEL_SIZE = 22
TICK_SIZE = 20

# Rows shown on the sub-question 1 figure, in poster order, keyed by the model
# names used in all_models_auc_ci.csv.
Q1_ROWS = (
    ("GNN junction", "Vessel graph GNN\n(junction)"),
    ("GNN voxel 7feat (baseline)", "Vessel graph GNN\n(voxel)"),
    ("tabular 7-means logreg", "Vessel kinetics\n(tabular means)"),
    ("GNN voxel 2feat", "Negative control\n(no enhancement features)"),
)

Q2_ROWS = (
    (
        "gnn_baseline_minus_tabular_7means_logreg",
        "Graph GNN  −  tabular\n(same features, same folds)",
    ),
    ("gnn_junction_minus_voxel", "Junction graph  −  voxel graph"),
    ("gnn_segment_minus_voxel", "Segment graph  −  voxel graph"),
)


def _forest(
    ax: Axes,
    labels: list[str],
    centres: list[float],
    los: list[float],
    his: list[float],
    colors: list[str],
    reference: float,
) -> None:
    """Horizontal point-and-interval plot with a dashed reference line."""
    ys = range(len(labels))
    ax.axvline(reference, color=REFERENCE, ls="--", lw=3, zorder=0)
    for y, centre, lo, hi, color in zip(ys, centres, los, his, colors):
        ax.plot([lo, hi], [y, y], color=color, lw=6, solid_capstyle="round")
        ax.plot([centre], [y], marker="o", ms=18, color=color)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=LABEL_SIZE)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _q1(auc_csv: Path, out_path: Path) -> None:
    """Sub-question 1: do the vessel models clear chance?"""
    table = pd.read_csv(auc_csv).set_index("model")
    rows = [table.loc[key] for key, _ in Q1_ROWS]
    labels = [label for _, label in Q1_ROWS]
    colors = [CONTROL if "control" in label.lower() else MODEL for label in labels]

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    _forest(
        ax,
        labels,
        [r["pooled_oof_auc"] for r in rows],
        [r["ci_lo"] for r in rows],
        [r["ci_hi"] for r in rows],
        colors,
        reference=0.5,
    )
    ax.set_xlabel(
        "Pooled out-of-fold AUC (95% bootstrap CI), N = 179", fontsize=LABEL_SIZE
    )
    ax.text(
        0.5,
        -0.66,
        "chance",
        color=REFERENCE,
        fontsize=LABEL_SIZE,
        ha="center",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def _q2(gate_csv: Path, mode_csv: Path, out_path: Path) -> None:
    """Sub-question 2: does any graph representation beat its tabular twin?"""
    gate = pd.read_csv(gate_csv).iloc[0]
    modes = pd.read_csv(mode_csv).set_index("comparison")
    deltas = {
        "gnn_baseline_minus_tabular_7means_logreg": gate,
        "gnn_junction_minus_voxel": modes.loc["gnn_junction_minus_voxel"],
        "gnn_segment_minus_voxel": modes.loc["gnn_segment_minus_voxel"],
    }
    rows = [deltas[key] for key, _ in Q2_ROWS]
    labels = [label for _, label in Q2_ROWS]

    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    _forest(
        ax,
        labels,
        [r["delta_auc"] for r in rows],
        [r["ci_lo"] for r in rows],
        [r["ci_hi"] for r in rows],
        [MODEL] * len(rows),
        reference=0.0,
    )
    ax.set_xlabel("Paired ΔAUC (95% bootstrap CI)", fontsize=LABEL_SIZE)
    ax.text(
        0.0,
        -0.56,
        "no gain",
        color=REFERENCE,
        fontsize=LABEL_SIZE,
        ha="center",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    """Write both poster results figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _q1(
        args.experiments_root / "uchicago_all_models" / "all_models_auc_ci.csv",
        args.out_dir / "q1_signal.png",
    )
    _q2(
        args.experiments_root
        / "uchicago_tabular_bar_floored"
        / "gate_paired_delta.csv",
        args.experiments_root
        / "uchicago_graph_mode_floored"
        / "graph_mode_paired_delta.csv",
        args.out_dir / "q2_graph_delta.png",
    )


if __name__ == "__main__":
    main()
