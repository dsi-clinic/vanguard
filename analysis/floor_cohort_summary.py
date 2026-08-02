"""Cohort-level view of the kinetic baseline floor, across every case.

Reads the ``summary.csv`` that ``analysis.floor_activation_map`` writes (one row
per case) and renders the two questions the per-case figures cannot answer:

    how many voxels does the floor touch in a typical image?
    would a max over the precontrast frames replace the floor entirely?

Writes ``cohort_affected_voxel_histogram.png`` and
``cohort_operator_comparison.png`` beside the summary.

Reads a CSV only -- light enough for the head node.

Usage:
    python -m analysis.floor_cohort_summary --summary-csv <out-dir>/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_BINS = 40
_MEAN_COLOR = "steelblue"
_MAX_COLOR = "tab:orange"


def _hist_zero_separated(
    ax: plt.Axes,
    series: list[np.ndarray],
    colors: list[str],
    labels: list[str | None],
) -> None:
    """Histogram that keeps "zero affected voxels" as its own bar.

    A case with no affected voxels is a categorically different outcome from one
    with a few, so it gets a standalone bar separated by a gap and a divider,
    never a shared first bin. The binned range starts at the smallest nonzero
    value; overlaid series split the zero slot side by side.
    """
    nonzero = [v[v > 0] for v in series]
    pooled = np.concatenate(nonzero)
    edges = np.linspace(pooled.min(), pooled.max(), _BINS + 1)
    width = edges[1] - edges[0]
    zero_center = edges[0] - 2 * width
    sub = width / len(series)

    for i, (v, color, label) in enumerate(zip(series, colors, labels)):
        alpha = 0.75 if len(series) > 1 else 1.0
        ax.hist(v[v > 0], bins=edges, color=color, alpha=alpha, label=label)
        n_zero = int((v == 0).sum())
        left = zero_center - width / 2 + i * sub
        ax.bar(left + sub / 2, n_zero, width=sub, color=color, alpha=alpha)
        ax.annotate(
            str(n_zero),
            (left + sub / 2, n_zero),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    ax.axvline(edges[0] - width, color="gray", lw=0.8, ls=":")
    ticks = matplotlib.ticker.MaxNLocator(nbins=6).tick_values(edges[0], edges[-1])
    ticks = [t for t in ticks if edges[0] <= t <= edges[-1]]
    ax.set_xticks([zero_center, *ticks])
    ax.set_xticklabels(["0", *(f"{t:g}" for t in ticks)])
    ax.set_xlim(zero_center - 1.5 * width, edges[-1] + width)


def main() -> None:
    """Render both cohort figures from a per-case summary CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()
    render(args.summary_csv)


def render(summary_csv: Path) -> None:
    """Write the cohort histogram and the mean-vs-max operator comparison."""
    summary = pd.read_csv(summary_csv)
    out_dir = summary_csv.parent
    _affected_histogram(summary, out_dir / "cohort_affected_voxel_histogram.png")
    _operator_comparison(summary, out_dir / "cohort_operator_comparison.png")


def _affected_histogram(summary: pd.DataFrame, out_path: Path) -> None:
    """How many vessel voxels the floor touches per image, over the cohort.

    One case is one observation, so the two panels integrate to the cohort size.
    The absolute count sets the scale of the intervention; the percentage panel
    is what says whether the floor stays a tail correction in every image or
    swallows a meaningful share of the vasculature in some of them.
    """
    counts = summary["support_floor_active"].to_numpy()
    pcts = summary["support_floor_active_pct"].to_numpy()
    n_cases = len(summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _hist_zero_separated(axes[0], [counts], [_MEAN_COLOR], [None])
    axes[0].set_xlabel("floor-affected vessel-support voxels in the image")
    axes[0].set_ylabel("number of cases")
    axes[0].set_title(
        f"median {np.median(counts):.0f}, "
        f"p90 {np.percentile(counts, 90):.0f}, max {counts.max():.0f}"
    )
    _hist_zero_separated(axes[1], [pcts], [_MEAN_COLOR], [None])
    axes[1].set_xlabel("floor-affected voxels (% of vessel support)")
    axes[1].set_ylabel("number of cases")
    axes[1].set_title(
        f"median {np.median(pcts):.2f}%, "
        f"p90 {np.percentile(pcts, 90):.2f}%, max {pcts.max():.2f}%"
    )
    fig.suptitle(
        f"How much of each image does the baseline floor touch? — {n_cases} cases"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _operator_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    """Mean vs max precontrast baseline: does max remove the need for the floor?

    Left: the same per-case affected-voxel distribution under both baseline
    operators. Right: the paired per-case counts -- every point below the
    identity line is a case where switching the operator alone shrinks the
    problem, and points on the x axis are cases max fixes outright.
    """
    mean_counts = summary["support_floor_active"].to_numpy()
    max_counts = summary["support_floor_active_maxop"].to_numpy()
    n_cases = len(summary)
    n_solved = int((max_counts == 0).sum())
    total_mean = int(mean_counts.sum())
    total_max = int(max_counts.sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _hist_zero_separated(
        axes[0],
        [mean_counts, max_counts],
        [_MEAN_COLOR, _MAX_COLOR],
        ["mean baseline (current)", "max baseline"],
    )
    axes[0].set_xlabel("floor-affected vessel-support voxels in the image")
    axes[0].set_ylabel("number of cases")
    axes[0].set_title(
        f"cohort total {total_mean} -> {total_max} voxels "
        f"({100 * (total_mean - total_max) / max(total_mean, 1):.0f}% rescued)"
    )
    axes[0].legend(fontsize=8)

    lim = max(mean_counts.max(), max_counts.max(), 1) * 1.1
    axes[1].plot([0, lim], [0, lim], color="gray", lw=0.8, ls="--", label="no change")
    axes[1].scatter(mean_counts, max_counts, s=14, color=_MEAN_COLOR, linewidths=0)
    axes[1].set_xlim(-0.02 * lim, lim)
    axes[1].set_ylim(-0.02 * lim, lim)
    axes[1].set_xlabel("affected voxels — mean baseline (current)")
    axes[1].set_ylabel("affected voxels — max baseline")
    axes[1].set_title(f"{n_solved} / {n_cases} cases need no floor at all under max")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Could a max over the precontrast frames replace the floor? — {n_cases} cases"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"wrote {out_path} (cohort affected voxels {total_mean} mean -> "
        f"{total_max} max; {n_solved}/{n_cases} cases fully resolved by max)"
    )


if __name__ == "__main__":
    main()
