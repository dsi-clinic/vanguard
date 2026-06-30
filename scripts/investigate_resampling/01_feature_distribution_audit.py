#!/usr/bin/env python3
"""
Feature distribution audit for the resampled ISPY2 features.

For each feature block (morph, graph, kinematic), computes per-feature stats
(mean, std, min, max, pct_nan) and generates distribution histograms. Flags
any features with high NaN rates or suspicious value ranges.

Outputs (written to --out-dir):
  block_stats.csv            — per-block aggregate stats
  feature_nan_rates.csv      — per-feature NaN rates, sorted descending
  morph_distributions.png    — histograms of top-variance morph features
  graph_distributions.png    — histograms of top-variance graph features
  kinematic_distributions.png— histograms of top-variance kinematic features
  clinical_tumor_check.txt   — sanity check on clinical + tumor_size cols

Usage:
  python scripts/investigate_resampling/01_feature_distribution_audit.py \
      --features-csv /ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/features_full_labeled.csv \
      --out-dir /ess/scratch/scratch1/t-9sbose/resampled_investigation/01_feature_distribution
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Each block is identified by its column-name prefix in features_full_labeled.csv
BLOCK_PREFIXES = {
    "morph": "morph_",
    "graph": "graph_",
    "kinematic": "kinematic_",
}

# Clinical and tumor_size columns don't have a single clean prefix — we identify
# them by exclusion: anything that isn't morph/graph/kinematic/pcr/case_id/site
VESSEL_PREFIXES = ("morph_", "graph_", "kinematic_")
META_COLS = {"case_id", "pcr", "site", "dataset"}


def get_block_cols(df: pd.DataFrame, prefix: str) -> list:
    return [c for c in df.columns if c.startswith(prefix)]


def compute_block_stats(df: pd.DataFrame, block_name: str, cols: list) -> dict:
    """Aggregate stats for one block: NaN rates, value ranges."""
    sub = df[cols].select_dtypes(include="number")  # skip string/object columns
    if sub.empty:
        return {"block": block_name, "n_features": len(cols), "note": "no numeric cols"}
    nan_rates = sub.isna().mean()
    return {
        "block": block_name,
        "n_features": len(cols),
        "pct_nan_mean": round(float(nan_rates.mean() * 100), 2),
        "pct_nan_max": round(float(nan_rates.max() * 100), 2),
        "n_all_nan_cols": int((nan_rates == 1.0).sum()),
        "n_high_nan_cols": int((nan_rates > 0.05).sum()),  # >5% missing
        "value_mean": round(float(sub.mean().mean()), 4),
        "value_std": round(float(sub.std().mean()), 4),
        "value_min": round(float(sub.min().min()), 4),
        "value_max": round(float(sub.max().max()), 4),
    }


def plot_block_histograms(
    df: pd.DataFrame, cols: list, title: str, out_path: Path, n_show: int = 16
) -> None:
    """Plot histograms for the top-n_show highest-variance features in a block."""
    variances = df[cols].var().dropna()
    if variances.empty:
        print(f"  [skip plot] no finite-variance columns for {title}")
        return
    top_cols = list(variances.nlargest(n_show).index)

    ncols_grid = 4
    nrows_grid = (len(top_cols) + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, 3.5 * nrows_grid))
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(top_cols):
        ax = axes_flat[i]
        vals = df[col].dropna()
        ax.hist(vals.values, bins=30, color="steelblue", alpha=0.75, edgecolor="none")
        short_name = col.replace("morph_", "").replace("graph_", "").replace("kinematic_", "")
        ax.set_title(short_name, fontsize=7, pad=3)
        nan_pct = df[col].isna().mean() * 100
        ax.set_xlabel(f"n={len(vals)}  NaN={nan_pct:.1f}%", fontsize=6)
        ax.tick_params(labelsize=6)

    for j in range(len(top_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, required=True,
                        help="Path to features_full_labeled.csv from the ML run")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.features_csv} ...")
    df = pd.read_csv(args.features_csv)
    print(f"  Shape: {df.shape}  ({df.shape[0]} patients, {df.shape[1]} columns)")

    # --- PCR label check (sanity) ---
    if "pcr" in df.columns:
        vc = df["pcr"].value_counts()
        print(f"  PCR label distribution: {dict(vc)}")
    else:
        print("  WARNING: 'pcr' column not found!")

    # --- Per-feature NaN audit ---
    nan_rates = (df.isna().mean() * 100).round(2)
    nan_df = nan_rates.reset_index()
    nan_df.columns = ["feature", "pct_nan"]
    nan_df = nan_df.sort_values("pct_nan", ascending=False)
    nan_df.to_csv(args.out_dir / "feature_nan_rates.csv", index=False)

    high_nan = nan_df[nan_df["pct_nan"] > 5]
    print(f"\n  Features with >5% NaN: {len(high_nan)} / {len(nan_df)}")
    if len(high_nan) > 0:
        print("  Top 15 high-NaN features:")
        for _, row in high_nan.head(15).iterrows():
            print(f"    {row['feature']}: {row['pct_nan']:.1f}%")

    # --- Per-block stats ---
    print("\n  Per-block statistics:")
    block_rows = []
    for block_name, prefix in BLOCK_PREFIXES.items():
        cols = get_block_cols(df, prefix)
        if not cols:
            print(f"  WARNING: no {block_name} columns found (prefix '{prefix}')")
            continue
        stats = compute_block_stats(df, block_name, cols)
        block_rows.append(stats)
        print(
            f"  {block_name:12s}: {stats['n_features']:4d} features | "
            f"NaN mean={stats['pct_nan_mean']:5.1f}% max={stats['pct_nan_max']:5.1f}% | "
            f"all-NaN cols={stats['n_all_nan_cols']} | "
            f"value range [{stats['value_min']:.3g}, {stats['value_max']:.3g}]"
        )

    # Clinical / tumor_size cols are everything not in a vessel block and not meta
    all_vessel_cols = set(
        c for c in df.columns if any(c.startswith(p) for p in VESSEL_PREFIXES)
    )
    clinical_tumor_cols = [
        c for c in df.columns if c not in all_vessel_cols and c not in META_COLS
    ]
    if clinical_tumor_cols:
        stats = compute_block_stats(df, "clinical+tumor_size", clinical_tumor_cols)
        block_rows.append(stats)
        print(
            f"  {'clinical+tumor_size':12s}: {stats['n_features']:4d} features | "
            f"NaN mean={stats['pct_nan_mean']:5.1f}% max={stats['pct_nan_max']:5.1f}%"
        )
        # Write a text summary of clinical/tumor cols for manual inspection
        with open(args.out_dir / "clinical_tumor_check.txt", "w") as fh:
            fh.write("Clinical + tumor_size columns:\n")
            for col in sorted(clinical_tumor_cols):
                nan_pct = df[col].isna().mean() * 100
                vals = df[col].dropna()
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                if is_numeric:
                    fh.write(
                        f"  {col}: n={len(vals)} non-null ({nan_pct:.1f}% NaN), "
                        f"mean={vals.mean():.4g}, std={vals.std():.4g}, "
                        f"min={vals.min():.4g}, max={vals.max():.4g}\n"
                    )
                else:
                    top_vals = vals.value_counts().head(3).index.tolist()
                    fh.write(
                        f"  {col}: n={len(vals)} non-null ({nan_pct:.1f}% NaN), "
                        f"type=categorical, top_values={top_vals}\n"
                    )
        print(f"  Saved clinical_tumor_check.txt")

    pd.DataFrame(block_rows).to_csv(args.out_dir / "block_stats.csv", index=False)
    print(f"\n  Saved block_stats.csv")

    # --- Distribution histograms ---
    print("\n  Generating distribution histograms ...")
    for block_name, prefix in BLOCK_PREFIXES.items():
        cols = get_block_cols(df, prefix)
        if not cols:
            continue
        out_path = args.out_dir / f"{block_name}_distributions.png"
        plot_block_histograms(
            df, cols,
            f"{block_name.upper()} block — top-16 features by variance (resampled ISPY2, n={df.shape[0]})",
            out_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
