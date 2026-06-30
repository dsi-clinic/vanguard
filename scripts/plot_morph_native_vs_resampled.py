#!/usr/bin/env python3
"""Distribution grid: morph features, native vs. resampled, split by pCR status.

Four KDE curves per subplot:
  - Native   pCR=1  (solid green)
  - Native   pCR=0  (dashed green)
  - Resampled pCR=1 (solid orange)
  - Resampled pCR=0 (dashed orange)

Features sorted by KS statistic (native-all vs. resampled-all), most affected first.
Bold subplot title = KS > 0.15.

Output:
    /home/t-9sbose/vanguard_qc_pngs/morph_features_native_vs_resampled.png

Usage:
    micromamba run -n vanguard python scripts/plot_morph_native_vs_resampled.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────

NATIVE_CSV = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_native_locked_ispy2/features_full_labeled.csv"
)
RESAMPLED_CSV = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_resampled_ispy2/features_full_labeled.csv"
)
OUT_DIR = Path("/home/t-9sbose/vanguard_qc_pngs")
OUT_PATH = OUT_DIR / "morph_features_native_vs_resampled.png"

# ── Feature list ──────────────────────────────────────────────────────────────

MORPH_FEATURES = [
    "morph_bifurcation_count",
    "morph_seg_dup_fraction",
    "morph_seg_length_mean",
    "morph_seg_length_std",
    "morph_seg_length_max",
    "morph_seg_tortuosity_mean",
    "morph_seg_tortuosity_std",
    "morph_seg_tortuosity_max",
    "morph_seg_volume_mean",
    "morph_seg_volume_std",
    "morph_seg_volume_max",
    "morph_curvature_mean_mean",
    "morph_curvature_mean_std",
    "morph_curvature_mean_max",
    "morph_radius_mean_mean",
    "morph_radius_mean_std",
    "morph_radius_mean_max",
    "morph_bif_angle_mean",
    "morph_bif_angle_std",
    "morph_bif_angle_max",
    "morph_seg_unique_per_skeleton_voxel",
    "morph_bifurcation_per_segment",
    "morph_seg_length_cv",
    "morph_radius_cv",
    "morph_bifurcation_density_per_length",
]

FEATURE_LABELS = {c: c.replace("morph_", "").replace("_", " ") for c in MORPH_FEATURES}

N_COLS = 5
N_ROWS = 5

COLOR_NATIVE    = "tab:green"
COLOR_RESAMPLED = "tab:orange"


def _kde(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = values[~np.isnan(values)]
    if len(values) < 5:
        return np.zeros_like(x_grid)
    try:
        return stats.gaussian_kde(values, bw_method="scott")(x_grid)
    except Exception:
        return np.zeros_like(x_grid)


def main() -> None:
    print("Loading feature tables...")
    cols = ["case_id", "pcr"] + MORPH_FEATURES
    native    = pd.read_csv(NATIVE_CSV,    usecols=cols)
    resampled = pd.read_csv(RESAMPLED_CSV, usecols=cols)

    # Rank features by KS statistic (collapsed across pCR)
    ks_stats: dict[str, float] = {}
    for feat in MORPH_FEATURES:
        n = native[feat].dropna().to_numpy()
        r = resampled[feat].dropna().to_numpy()
        ks_stats[feat] = float(stats.ks_2samp(n, r).statistic) if (len(n) > 1 and len(r) > 1) else 0.0

    sorted_feats = sorted(MORPH_FEATURES, key=lambda f: ks_stats[f], reverse=True)

    print("\nTop 5 most affected (KS, native vs. resampled):")
    for f in sorted_feats[:5]:
        print(f"  {f:<45} KS={ks_stats[f]:.3f}")

    # Split by pCR
    n_pcr1  = native[native["pcr"] == 1]
    n_pcr0  = native[native["pcr"] == 0]
    r_pcr1  = resampled[resampled["pcr"] == 1]
    r_pcr0  = resampled[resampled["pcr"] == 0]
    print(f"\nNative   pCR=1: {len(n_pcr1)}, pCR=0: {len(n_pcr0)}")
    print(f"Resampled pCR=1: {len(r_pcr1)}, pCR=0: {len(r_pcr0)}")

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(22, 18))
    axes_flat = axes.flatten()

    for ax_idx, feat in enumerate(sorted_feats):
        ax = axes_flat[ax_idx]

        groups = {
            "native_1":   n_pcr1[feat].dropna().to_numpy(),
            "native_0":   n_pcr0[feat].dropna().to_numpy(),
            "resamp_1":   r_pcr1[feat].dropna().to_numpy(),
            "resamp_0":   r_pcr0[feat].dropna().to_numpy(),
        }

        all_vals = np.concatenate(list(groups.values()))
        lo = np.nanpercentile(all_vals, 1)
        hi = np.nanpercentile(all_vals, 99)
        if lo >= hi:
            lo, hi = all_vals.min(), all_vals.max()
        x_grid = np.linspace(lo, hi, 200)

        styles = {
            "native_1": (COLOR_NATIVE,    "-",  0.75, 1.6),
            "native_0": (COLOR_NATIVE,    "--", 0.40, 1.2),
            "resamp_1": (COLOR_RESAMPLED, "-",  0.75, 1.6),
            "resamp_0": (COLOR_RESAMPLED, "--", 0.40, 1.2),
        }

        for key, vals in groups.items():
            color, ls, alpha, lw = styles[key]
            clipped = np.clip(vals, lo, hi)
            kde = _kde(clipped, x_grid)
            ax.fill_between(x_grid, kde, alpha=0.10, color=color)
            ax.plot(x_grid, kde, color=color, linestyle=ls, linewidth=lw, alpha=alpha + 0.1)

        ks = ks_stats[feat]
        weight = "bold" if ks > 0.15 else "normal"
        ax.set_title(f"{FEATURE_LABELS[feat]}\nKS={ks:.3f}", fontsize=7.5, fontweight=weight, pad=3)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)
        ax.spines[["top", "right", "left"]].set_visible(False)

    for ax in axes_flat[len(sorted_feats):]:
        ax.set_visible(False)

    # Legend: source (color) × pCR (line style)
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_NATIVE,    linestyle="-",  linewidth=2, label="Native · pCR=1"),
        mlines.Line2D([], [], color=COLOR_NATIVE,    linestyle="--", linewidth=2, label="Native · pCR=0"),
        mlines.Line2D([], [], color=COLOR_RESAMPLED, linestyle="-",  linewidth=2, label="Resampled · pCR=1"),
        mlines.Line2D([], [], color=COLOR_RESAMPLED, linestyle="--", linewidth=2, label="Resampled · pCR=0"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        fontsize=10,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.suptitle(
        "Morph Feature Distributions by Source & pCR Status  (ISPY2, n=808)\n"
        "Sorted by KS statistic (native vs. resampled) — most affected first  ·  bold = KS > 0.15",
        fontsize=12, fontweight="bold", y=1.05,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
