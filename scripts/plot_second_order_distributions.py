#!/usr/bin/env python3
"""Plot distribution histograms and summary statistics for second-order
engineered features and their first-order sources.

Usage (from ``vanguard/``)::

    PYTHONPATH=. python scripts/plot_second_order_distributions.py \
        experiments/second_order_features/features_engineered_labeled.csv \
        -o experiments/second_order_features/

Reads the labeled features CSV produced by ``tabular_cohort.py`` and writes:

- ``distributions_second_order.png``
- ``distributions_first_order_sources.png``
- ``summary_second_order.csv``
- ``summary_first_order_sources.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.second_order import SECOND_ORDER_COLUMNS

# ── first-order source columns (every row.get() target in second_order.py) ──

FIRST_ORDER_SOURCE_COLUMNS: tuple[str, ...] = (
    # morph inputs
    "morph_seg_length_std",
    "morph_seg_length_mean",
    "morph_radius_mean_std",
    "morph_radius_mean_mean",
    "morph_bifurcation_count",
    "morph_seg_length_sum",
    # per-shell topology
    "per_shell_topology_inside_tumor_shell_length_mm",
    "per_shell_topology_shell_0_2mm_shell_length_mm",
    "per_shell_topology_shell_5_10mm_shell_length_mm",
    "per_shell_topology_shell_10_20mm_shell_length_mm",
    "per_shell_topology_inside_tumor_shell_volume_burden_mm3",
    "per_shell_topology_shell_0_2mm_shell_volume_burden_mm3",
    "per_shell_topology_shell_5_10mm_shell_volume_burden_mm3",
    "per_shell_topology_shell_10_20mm_shell_volume_burden_mm3",
    "per_shell_topology_inside_tumor_bifurcation_count",
    "per_shell_topology_shell_0_2mm_bifurcation_count",
    "per_shell_topology_inside_tumor_node_count",
    "per_shell_topology_shell_0_2mm_node_count",
    # graph totals / boundary / burden
    "graph_totals_node_count",
    "boundary_crossing_crossing_length_mm",
    "tumor_burden_inside_or_near_length_mm",
    # kinematic shell kinetics — time-to-enhancement
    "kinematic_shell_kinetics_inside_tumor_time_to_enhancement_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_shell_0_2mm_time_to_enhancement_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_shell_5_10mm_time_to_enhancement_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_shell_10_20mm_time_to_enhancement_hurdle_value_given_signal_median",
    # kinematic shell kinetics — peak enhancement
    "kinematic_shell_kinetics_inside_tumor_peak_enhancement_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_shell_0_2mm_peak_enhancement_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_shell_5_10mm_peak_enhancement_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_shell_10_20mm_peak_enhancement_hurdle_value_given_signal_median",
    # kinematic shell kinetics — washin / washout slopes
    "kinematic_shell_kinetics_inside_tumor_washin_slope_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_shell_0_2mm_washin_slope_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_inside_tumor_washout_slope_hurdle_value_given_signal_median",
    "kinematic_shell_kinetics_shell_0_2mm_washout_slope_hurdle_value_given_signal_median",
    # kinematic arrival delay
    "kinematic_arrival_delay_vs_reference_near_tumor_segments_hurdle_value_given_signal_sd",
    "kinematic_arrival_delay_vs_reference_near_tumor_segments_hurdle_value_given_signal_mean",
)

# ── name abbreviation for plot titles ───────────────────────────────────────

_ABBREVIATIONS: list[tuple[str, str]] = [
    ("per_shell_topology_", "pst_"),
    ("kinematic_shell_kinetics_", "ksk_"),
    ("kinematic_arrival_delay_vs_reference_near_tumor_segments_", "kin_arr_delay_ref_near_"),
    ("kinematic_", "kin_"),
    ("boundary_crossing_", "bc_"),
    ("tumor_burden_", "tb_"),
    ("graph_totals_", "gt_"),
    ("graph_", "g_"),
    ("morph_", "m_"),
    ("_hurdle_value_given_signal_", "_hvgs_"),
]


def _abbreviate(name: str) -> str:
    """Shorten a feature name for plot titles."""
    for long, short in _ABBREVIATIONS:
        name = name.replace(long, short)
    return name


# ── summary table ───────────────────────────────────────────────────────────

def _summarize(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Descriptive statistics per feature, matching the original CSV schema."""
    n_total = len(df)
    rows: list[dict] = []
    for col in columns:
        present = col in df.columns
        vals = df[col].dropna() if present else pd.Series(dtype=float)
        n = len(vals)
        if n > 0:
            rows.append({
                "feature": col,
                "present": True,
                "n_valid": n,
                "pct_valid": f"{100 * n / n_total:.1f}%",
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "q25": vals.quantile(0.25),
                "median": vals.median(),
                "q75": vals.quantile(0.75),
                "max": vals.max(),
            })
        else:
            rows.append({
                "feature": col,
                "present": present,
                "n_valid": 0,
                "pct_valid": "0.0%",
                "mean": "",
                "std": "",
                "min": "",
                "q25": "",
                "median": "",
                "q75": "",
                "max": "",
            })
    return pd.DataFrame(rows)


# ── plotting ────────────────────────────────────────────────────────────────

def _plot_distributions(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    title: str,
    output_path: Path,
    color: str,
    n_cols: int = 3,
) -> None:
    """Histogram grid clipped to 1st–99th percentile."""
    plot_cols = [c for c in columns if c in df.columns and df[c].notna().any()]
    n_plots = len(plot_cols)
    if n_plots == 0:
        return
    n_rows = -(-n_plots // n_cols)  # ceil division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, col in enumerate(plot_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        vals = df[col].dropna()
        lo, hi = np.percentile(vals, [1, 99])
        clipped = vals[(vals >= lo) & (vals <= hi)]
        ax.hist(clipped, bins=30, color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(_abbreviate(col), fontsize=9)
        ax.annotate(
            f"n={len(vals)}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
            color="0.4",
        )

    for idx in range(n_plots, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_csv", type=Path, help="features_engineered_labeled.csv")
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: same as input CSV)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    outdir = args.output_dir or args.input_csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    _summarize(df, SECOND_ORDER_COLUMNS).to_csv(
        outdir / "summary_second_order.csv", index=False,
    )
    _summarize(df, FIRST_ORDER_SOURCE_COLUMNS).to_csv(
        outdir / "summary_first_order_sources.csv", index=False,
    )

    _plot_distributions(
        df,
        SECOND_ORDER_COLUMNS,
        "second-order feature distributions (1st-99th pctl)",
        outdir / "distributions_second_order.png",
        color="tab:blue",
    )
    _plot_distributions(
        df,
        FIRST_ORDER_SOURCE_COLUMNS,
        "first-order source feature distributions (1st-99th pctl)",
        outdir / "distributions_first_order_sources.png",
        color="tab:orange",
    )


if __name__ == "__main__":
    main()
