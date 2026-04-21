#!/usr/bin/env python3
"""Plot feature distributions grouped by pCR vs non-pCR.

Produces two multi-panel figures:
  1. First-order features  — the source columns that feed into second-order derivations
  2. Second-order features — the 15 engineered ratios / contrasts

Each panel shows overlapping histograms (with KDE) for pCR=1 vs pCR=0,
plus a Mann–Whitney U p-value annotation.

Usage::

    python scripts/plot_pcr_distributions.py \
        experiments/second_order_features/features_engineered_labeled.csv \
        -o experiments/second_order_features
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from features.second_order import SECOND_ORDER_COLUMNS

# Features whose source data was never computed into the JSON payloads and
# therefore require a full re-extraction from raw imaging data.
NEEDS_REEXTRACTION: frozenset[str] = frozenset({
    "kinematic_crossing_early_fraction",
})

# First-order source columns that are used to derive the second-order features.
# These are the direct inputs referenced in second_order.py.
FIRST_ORDER_SOURCE_COLUMNS: tuple[str, ...] = (
    # tumor_size sources
    "tumor_size_tumor_voxels",
    "tumor_size_tumor_shell_r0_2_voxels",
    "tumor_size_tumor_shell_r4_8_voxels",
    # morph sources
    "morph_seg_length_std",
    "morph_seg_length_mean",
    "morph_seg_length_sum",
    "morph_radius_mean_std",
    "morph_radius_mean_mean",
    "morph_bifurcation_count",
    # representative first-order graph/morph/kinematic features
    "morph_seg_tortuosity_mean",
    "morph_seg_volume_mean",
    "morph_curvature_mean_mean",
    "graph_skeleton_voxels",
    "graph_nodes",
    "graph_edges",
)


def _plot_distribution_grid(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    title: str,
    out_path: Path,
    *,
    max_cols: int = 4,
    all_expected_cols: list[str] | None = None,
) -> None:
    """Plot a grid of distribution panels, one per feature, colored by label.

    Parameters
    ----------
    all_expected_cols : list[str] | None
        The full list of expected columns (before filtering to those present in
        *df*).  When provided, columns that were expected but absent from *df*
        are shown as grayed-out panels with an explanatory note.
    """
    missing_cols = []
    if all_expected_cols is not None:
        missing_cols = [c for c in all_expected_cols if c not in df.columns]
    display_cols = list(feature_cols) + missing_cols

    n_features = len(display_cols)
    n_cols = min(max_cols, n_features) if n_features else 1
    n_rows = math.ceil(n_features / n_cols) if n_features else 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )
    if n_features <= 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    group_0 = df[df[label_col] == 0]
    group_1 = df[df[label_col] == 1]

    for idx, col in enumerate(display_cols):
        row_idx, col_idx = divmod(idx, n_cols)
        ax = axes[row_idx, col_idx]

        if col in missing_cols:
            ax.set_facecolor("#f0f0f0")
            ax.text(
                0.5, 0.5,
                "source column not\npresent in input CSV",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic",
            )
            ax.set_title(col, fontsize=9, color="#999999")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        vals_0 = group_0[col].dropna()
        vals_1 = group_1[col].dropna()

        if vals_0.empty and vals_1.empty:
            ax.set_facecolor("#f0f0f0")
            if col in NEEDS_REEXTRACTION:
                reason = "requires re-extraction\nfrom raw imaging data"
            else:
                reason = "all values NaN\n(source columns may be\nmissing from pipeline)"
            ax.text(
                0.5, 0.5, reason,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic",
            )
            ax.set_title(col, fontsize=9, color="#999999")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        bins = min(40, max(10, int(np.sqrt(len(vals_0) + len(vals_1)))))

        ax.hist(vals_0, bins=bins, alpha=0.5, density=True,
                label=f"non-pCR (n={len(vals_0)})", color="#3274A1", edgecolor="white", linewidth=0.3)
        ax.hist(vals_1, bins=bins, alpha=0.5, density=True,
                label=f"pCR (n={len(vals_1)})", color="#E1812C", edgecolor="white", linewidth=0.3)

        # KDE overlay
        for vals, color in [(vals_0, "#3274A1"), (vals_1, "#E1812C")]:
            if len(vals) > 5:
                try:
                    sns.kdeplot(vals, ax=ax, color=color, linewidth=1.5, warn_singular=False)
                except Exception:  # noqa: BLE001
                    pass

        # Mann-Whitney U test
        if len(vals_0) >= 3 and len(vals_1) >= 3:
            try:
                _, pval = stats.mannwhitneyu(vals_0, vals_1, alternative="two-sided")
                significance = ""
                if pval < 0.001:
                    significance = " ***"
                elif pval < 0.01:
                    significance = " **"
                elif pval < 0.05:
                    significance = " *"
                ax.text(
                    0.97, 0.95,
                    f"p={pval:.3g}{significance}",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                )
            except Exception:  # noqa: BLE001
                pass

        # Truncate long column names for readability
        short_name = col.replace("tumor_size_", "ts_").replace("kinematic_", "kin_").replace("morph_", "m_").replace("graph_", "g_")
        ax.set_title(short_name, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused axes
    for idx in range(n_features, n_rows * n_cols):
        row_idx, col_idx = divmod(idx, n_cols)
        axes[row_idx, col_idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", out_path)


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", type=Path, help="Path to features_engineered_labeled.csv")
    ap.add_argument(
        "-o", "--outdir", type=Path, default=None,
        help="Output directory (default: same as input CSV)",
    )
    ap.add_argument(
        "--label-col", type=str, default="pcr", help="Label column name",
    )
    args = ap.parse_args()

    outdir = args.outdir or args.input_csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    logging.info("Loaded %s: %d rows x %d cols", args.input_csv, len(df), len(df.columns))
    logging.info(
        "Label distribution: %s",
        df[args.label_col].value_counts().to_dict(),
    )

    # --- Second-order features ---
    second_order_present = [c for c in SECOND_ORDER_COLUMNS if c in df.columns]
    _plot_distribution_grid(
        df, second_order_present, args.label_col,
        title="Second-Order Feature Distributions: pCR vs non-pCR",
        out_path=outdir / "distributions_second_order_by_pcr.png",
        all_expected_cols=list(SECOND_ORDER_COLUMNS),
    )

    # --- First-order source features ---
    first_order_present = [c for c in FIRST_ORDER_SOURCE_COLUMNS if c in df.columns]
    _plot_distribution_grid(
        df, first_order_present, args.label_col,
        title="First-Order Source Feature Distributions: pCR vs non-pCR",
        out_path=outdir / "distributions_first_order_by_pcr.png",
        all_expected_cols=list(FIRST_ORDER_SOURCE_COLUMNS),
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
