#!/usr/bin/env python3
"""Three-way AUC comparison: Q3 baseline / resampled / native locked.

Each arm is a horizontal row. Three markers per row:
  - Blue  circle  : Q3 baseline mean ± std (old code, native, free split)
  - Orange diamond: Resampled mean ± std   (curr code, resampled, free split)
  - Green square  : Native locked mean ± std (curr code, native, SAME split)

A vertical dashed line marks the clinical+tumor_size baseline of each run
(same color) so you can see deltas at a glance.

The bottom panel shows the head-to-head delta: native_locked − resampled,
with a zero reference line. Positive = native is better (resampling hurts).

Usage:
    micromamba run -n vanguard python scripts/plot_locked_split_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Q3_CSV = PROJECT_ROOT / "results" / "independent_signal_q3_summary.csv"
RESAMPLED_CSV = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_resampled_ispy2/ablation_summary.csv"
)
NATIVE_LOCKED_CSV = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_native_locked_ispy2/ablation_summary.csv"
)
RESAMPLED_FOLD_CSV = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_resampled_ispy2/ablation_fold_auc.csv"
)
NATIVE_LOCKED_FOLD_CSV = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_native_locked_ispy2/ablation_fold_auc.csv"
)

OUT_PATH = PROJECT_ROOT / "results" / "locked_split_auc_comparison.png"

# ── Layout ────────────────────────────────────────────────────────────────────

ARM_ORDER = [
    "clinical_plus_tumor_size",
    "clinical_plus_tumor_size_plus_morph",
    "clinical_plus_tumor_size_plus_graph",
    "clinical_plus_tumor_size_plus_kinematic",
    "clinical_plus_tumor_size_plus_graph_plus_kinematic",
    "clinical_plus_tumor_size_plus_vessel_all",
]

ARM_LABELS = {
    "clinical_plus_tumor_size": "clinical + tumor_size",
    "clinical_plus_tumor_size_plus_morph": "+ morph",
    "clinical_plus_tumor_size_plus_graph": "+ graph",
    "clinical_plus_tumor_size_plus_kinematic": "+ kinematic",
    "clinical_plus_tumor_size_plus_graph_plus_kinematic": "+ graph + kinematic",
    "clinical_plus_tumor_size_plus_vessel_all": "+ vessel_all",
}

RUNS = [
    ("q3",     Q3_CSV,           "Q3 baseline  (old code · native · split unknown)",        "tab:blue",   "o", 8),
    ("resamp", RESAMPLED_CSV,    "Resampled    (curr code · resampled · split = source) ┐", "tab:orange", "D", 7),
    ("native", NATIVE_LOCKED_CSV,"Native       (curr code · native   · same split)      ┘", "tab:green",  "s", 7),
]


def _load_summary(path: Path, key: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    df = pd.read_csv(path)
    arm_col = "arm_name" if "arm_name" in df.columns else "run_name"
    return df.rename(columns={arm_col: "arm_name"})[
        ["arm_name", "auc_mean", "auc_std"]
    ].assign(key=key)


def _load_folds(path: Path, key: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    arm_col = "arm_name" if "arm_name" in df.columns else "run_name"
    return df.rename(columns={arm_col: "arm_name"})[["arm_name", "auc"]].assign(key=key)


def main() -> None:
    summaries = {}
    for key, path, _, _, _, _ in RUNS:
        df = _load_summary(path, key)
        if df is not None:
            summaries[key] = df.set_index("arm_name")

    n_arms = len(ARM_ORDER)
    y_pos = np.arange(n_arms)

    fig, (ax_main, ax_delta) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.subplots_adjust(wspace=0.05)

    # ── Main panel: AUC per arm ───────────────────────────────────────────────
    offsets = {"q3": -0.18, "resamp": 0.0, "native": 0.18}

    for key, path, label, color, marker, ms in RUNS:
        if key not in summaries:
            continue
        df = summaries[key]

        means, stds = [], []
        for arm in ARM_ORDER:
            if arm in df.index:
                means.append(float(df.loc[arm, "auc_mean"]))
                stds.append(float(df.loc[arm, "auc_std"] or 0.0))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        ax_main.errorbar(
            means,
            y_pos + offsets[key],
            xerr=stds,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            markersize=ms,
            label=label,
            zorder=3,
        )

    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=10)
    ax_main.set_xlabel("Validation AUC", fontsize=11)
    ax_main.set_title(
        "ISPY2 Independent-Signal Ablation\nLocked-Split Comparison",
        fontsize=12, fontweight="bold",
    )
    ax_main.set_xlim(0.50, 0.70)
    ax_main.grid(axis="x", linestyle=":", alpha=0.4)
    ax_main.invert_yaxis()

    legend_handles = []
    for key, _, label, color, marker, ms in RUNS:
        if key in summaries:
            legend_handles.append(
                plt.Line2D(
                    [0], [0], marker=marker, color="w",
                    markerfacecolor=color, markeredgecolor=color,
                    markersize=ms, label=label,
                )
            )
    ax_main.legend(
        handles=legend_handles, loc="lower right", fontsize=8.5, framealpha=0.92,
        handletextpad=0.6, borderpad=0.8,
    )

    # ── Delta panel: native_locked − resampled ────────────────────────────────
    if "native" in summaries and "resamp" in summaries:
        deltas = []
        for arm in ARM_ORDER:
            n = summaries["native"]
            r = summaries["resamp"]
            if arm in n.index and arm in r.index:
                deltas.append(float(n.loc[arm, "auc_mean"]) - float(r.loc[arm, "auc_mean"]))
            else:
                deltas.append(np.nan)

        colors_delta = ["tab:green" if d >= 0 else "tab:orange" for d in deltas]
        ax_delta.barh(
            y_pos, deltas, color=colors_delta, alpha=0.75, height=0.5,
        )
        ax_delta.axvline(0, color="black", linewidth=0.9, linestyle="-")
        ax_delta.set_yticks(y_pos)
        ax_delta.set_yticklabels([], fontsize=0)
        ax_delta.set_xlabel("Native − Resampled\n(same split)", fontsize=9)
        ax_delta.set_title("Δ AUC\n(spacing effect)", fontsize=10, fontweight="bold")
        ax_delta.grid(axis="x", linestyle=":", alpha=0.4)
        ax_delta.invert_yaxis()

        x_abs = max(abs(d) for d in deltas if not np.isnan(d))
        pad = 0.04
        x_min, x_max = -(x_abs + pad), (x_abs + pad)
        ax_delta.set_xlim(x_min, x_max)

        # Shaded half-backgrounds to signal direction
        ax_delta.axvspan(0, x_max, color="tab:green", alpha=0.07, zorder=0)
        ax_delta.axvspan(x_min, 0, color="tab:orange", alpha=0.07, zorder=0)

        # Direction labels inside the panel, near the top
        ax_delta.text(
            x_max * 0.55, 0.2, "Native\nBetter",
            ha="center", va="top", fontsize=9,
            color="tab:green", fontweight="bold",
        )
        ax_delta.text(
            x_min * 0.55, 0.2, "Resamp\nBetter",
            ha="center", va="top", fontsize=9,
            color="tab:orange", fontweight="bold",
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
