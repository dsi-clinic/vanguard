#!/usr/bin/env python3
"""
Baseline vs. resampled AUC comparison plot.

Reads the previous group's published baseline AUC table and the new resampled
run's ablation_summary.csv, joins them on arm_name, and produces a grouped
bar chart showing both sets of results side by side. Also writes a CSV delta
table suitable for copying into a lab notebook or presentation.

Outputs (written to --out-dir):
  auc_comparison.png     — grouped bar chart (baseline vs resampled, ±1 std)
  auc_delta_table.csv    — joined table with delta column

Usage:
  python scripts/investigate_resampling/03_auc_comparison_plot.py \
      --baseline-csv results/independent_signal_q3_summary.csv \
      --resampled-csv /ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/ablation_summary.csv \
      --out-dir /ess/scratch/scratch1/t-9sbose/resampled_investigation/03_auc_comparison
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# How the arm names in the baseline CSV map to the resampled CSV.
# The baseline uses short names; the resampled run uses verbose names.
ARM_LABEL_MAP = {
    "clinical_plus_tumor_size":                        "clinical\n+ tumor_size",
    "clinical_plus_tumor_size_plus_morph":             "+ morph",
    "clinical_plus_tumor_size_plus_graph":             "+ graph",
    "clinical_plus_tumor_size_plus_kinematic":         "+ kinematic",
    "clinical_plus_tumor_size_plus_graph_plus_kinematic": "+ graph\n+ kinematic",
    "clinical_plus_tumor_size_plus_vessel_all":        "+ vessel_all",
}

ARM_ORDER = [
    "clinical_plus_tumor_size",
    "clinical_plus_tumor_size_plus_morph",
    "clinical_plus_tumor_size_plus_graph",
    "clinical_plus_tumor_size_plus_kinematic",
    "clinical_plus_tumor_size_plus_graph_plus_kinematic",
    "clinical_plus_tumor_size_plus_vessel_all",
]


def load_baseline(path: Path) -> pd.DataFrame:
    """Load baseline CSV. Handles both 'arm_name' and 'run' as the arm identifier."""
    df = pd.read_csv(path)
    # Normalize arm name column
    for candidate in ("arm_name", "run", "name"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "arm_name"})
            break
    # Normalize AUC columns
    for auc_col in ("auc_mean", "auc", "val_auc", "mean_auc"):
        if auc_col in df.columns:
            df = df.rename(columns={auc_col: "auc_mean_baseline"})
            break
    for std_col in ("auc_std", "std_auc", "se_auc"):
        if std_col in df.columns:
            df = df.rename(columns={std_col: "auc_std_baseline"})
            break
    if "auc_std_baseline" not in df.columns:
        df["auc_std_baseline"] = float("nan")
    return df[["arm_name", "auc_mean_baseline", "auc_std_baseline"]].copy()


def load_resampled(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for candidate in ("arm_name", "run", "name"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "arm_name"})
            break
    for auc_col in ("auc_mean", "auc", "val_auc"):
        if auc_col in df.columns:
            df = df.rename(columns={auc_col: "auc_mean_resampled"})
            break
    for std_col in ("auc_std", "std_auc"):
        if std_col in df.columns:
            df = df.rename(columns={std_col: "auc_std_resampled"})
            break
    if "auc_std_resampled" not in df.columns:
        df["auc_std_resampled"] = float("nan")
    return df[["arm_name", "auc_mean_resampled", "auc_std_resampled"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--resampled-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_base = load_baseline(args.baseline_csv)
    df_res = load_resampled(args.resampled_csv)

    merged = pd.merge(df_base, df_res, on="arm_name", how="outer")
    merged["delta"] = merged["auc_mean_resampled"] - merged["auc_mean_baseline"]
    merged["delta"] = merged["delta"].round(4)

    # Sort by the canonical arm order
    arm_order_idx = {arm: i for i, arm in enumerate(ARM_ORDER)}
    merged["_sort"] = merged["arm_name"].map(arm_order_idx).fillna(999)
    merged = merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    # Pretty labels
    merged["label"] = merged["arm_name"].map(ARM_LABEL_MAP).fillna(merged["arm_name"])

    merged.to_csv(args.out_dir / "auc_delta_table.csv", index=False)
    print("AUC comparison table:")
    print(merged[["arm_name", "auc_mean_baseline", "auc_mean_resampled", "delta"]].to_string(index=False))

    # --- Grouped bar chart ---
    n = len(merged)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    base_auc = merged["auc_mean_baseline"].to_numpy(dtype=float)
    base_std = merged["auc_std_baseline"].fillna(0).to_numpy(dtype=float)
    res_auc = merged["auc_mean_resampled"].to_numpy(dtype=float)
    res_std = merged["auc_std_resampled"].fillna(0).to_numpy(dtype=float)

    bars_base = ax.bar(
        x - width / 2, base_auc, width,
        yerr=base_std, capsize=4,
        color="steelblue", alpha=0.85, label="Baseline (native segmentations)",
        error_kw={"elinewidth": 1.2, "ecolor": "gray"},
    )
    bars_res = ax.bar(
        x + width / 2, res_auc, width,
        yerr=res_std, capsize=4,
        color="darkorange", alpha=0.85, label="Resampled segmentations",
        error_kw={"elinewidth": 1.2, "ecolor": "gray"},
    )

    # Annotate delta above each pair
    for i, (b, r, d) in enumerate(zip(base_auc, res_auc, merged["delta"])):
        if np.isnan(d):
            continue
        color = "green" if d > 0 else "red"
        top = max(b, r) + max(base_std[i], res_std[i]) + 0.005
        ax.text(i, top, f"Δ{d:+.3f}", ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"].tolist(), fontsize=9)
    ax.set_ylabel("Validation AUC (5-fold CV, ±1 std)")
    ax.set_title("Baseline vs. Resampled: pCR prediction AUC by feature arm (ISPY2)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.45, min(1.0, max(np.nanmax(base_auc + base_std),
                                    np.nanmax(res_auc + res_std)) + 0.12))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="chance")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = args.out_dir / "auc_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
