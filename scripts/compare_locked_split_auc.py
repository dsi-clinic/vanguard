#!/usr/bin/env python3
"""Compare AUC tables across the three ISPY2 independent-signal runs.

Produces a side-by-side table showing:
  - Q3 baseline: old code + native data (historical reference)
  - Resampled  : current code + resampled data (free-floating split)
  - Native locked: current code + native data + SAME split as resampled

The delta columns show how much each feature block adds beyond the clinical
baseline, and whether the negative vessel-feature deltas from the resampled
run persist in the native locked run.

If native_locked vs. resampled show the same pattern → resampling is not the
cause of the drop; the code change (second_order.py) is more likely.
If the pattern differs → resampling itself is moving the AUC.

Usage (head node, after native locked run completes):
    cd /ess/home/home1/t-9sbose/vanguard
    micromamba run -n vanguard python scripts/compare_locked_split_auc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# ── Input paths ──────────────────────────────────────────────────────────────

Q3_BASELINE = PROJECT_ROOT / "results" / "independent_signal_q3_summary.csv"

RESAMPLED = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_resampled_ispy2/ablation_summary.csv"
)

NATIVE_LOCKED = Path(
    "/ess/scratch/scratch1/t-9sbose/vanguard_experiments"
    "/independent_signal_native_locked_ispy2/ablation_summary.csv"
)

# Display names for the table columns
RUNS = [
    ("q3_baseline", Q3_BASELINE, "Q3 baseline\n(old code, native, free split)"),
    ("resampled", RESAMPLED, "Resampled\n(curr code, resampled, free split)"),
    ("native_locked", NATIVE_LOCKED, "Native locked\n(curr code, native, SAME split)"),
]

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


def _load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  MISSING: {label}")
        print(f"    expected at: {path}")
        return None
    df = pd.read_csv(path)
    arm_col = "arm_name" if "arm_name" in df.columns else "run_name"
    return df[[arm_col, "auc_mean", "auc_std"]].rename(
        columns={arm_col: "arm_name", "auc_mean": label, "auc_std": f"{label}_std"}
    )


def _delta_col(df: pd.DataFrame, col: str) -> pd.Series:
    """AUC delta vs. the clinical+tumor_size baseline arm."""
    baseline_mask = df["arm_name"] == "clinical_plus_tumor_size"
    if baseline_mask.sum() == 0:
        return pd.Series([float("nan")] * len(df), index=df.index)
    baseline = float(df.loc[baseline_mask, col].iloc[0])
    return df[col] - baseline


def main() -> None:
    print("=" * 72)
    print("ISPY2 Independent-Signal AUC Comparison — Locked Split")
    print("=" * 72)
    print()
    print("Loading results...")

    frames = []
    available = []
    for key, path, label in RUNS:
        df = _load(path, key)
        if df is not None:
            frames.append(df)
            available.append((key, label))
            print(f"  OK: {label.splitlines()[0]} ({len(df)} arms)")
        else:
            print(f"  (skipped)")

    if not frames:
        print("\nNo results found. Run the experiments first.")
        sys.exit(1)

    # Merge all available runs on arm_name
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="arm_name", how="outer")

    # Reorder to canonical arm order
    merged["_order"] = merged["arm_name"].map(
        {arm: i for i, arm in enumerate(ARM_ORDER)}
    )
    merged = merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    merged["label"] = merged["arm_name"].map(ARM_LABELS).fillna(merged["arm_name"])

    # ── Main AUC table ───────────────────────────────────────────────────────
    print()
    print("─" * 72)
    print("AUC MEAN (± std across 5 folds)")
    print("─" * 72)

    header_parts = [f"{'Arm':<30}"]
    for key, label in available:
        header_parts.append(f"{label.splitlines()[0]:>16}")
    print("  ".join(header_parts))
    print("  ".join(["─" * 30] + ["─" * 16] * len(available)))

    for _, row in merged.iterrows():
        parts = [f"{row['label']:<30}"]
        for key, _ in available:
            if key in row and not pd.isna(row[key]):
                std_col = f"{key}_std"
                std = row.get(std_col, float("nan"))
                std_str = f"±{std:.3f}" if not pd.isna(std) else ""
                parts.append(f"{row[key]:.3f} {std_str:>7}")
            else:
                parts.append(f"{'—':>16}")
        print("  ".join(parts))

    # ── Delta table (vs. clinical+tumor_size baseline) ───────────────────────
    print()
    print("─" * 72)
    print("DELTA vs. clinical+tumor_size baseline (same run's baseline)")
    print("─" * 72)

    for key, label in available:
        if key in merged.columns:
            merged[f"{key}_delta"] = _delta_col(merged, key)

    header_parts = [f"{'Arm':<30}"]
    for key, label in available:
        header_parts.append(f"{label.splitlines()[0]:>14}")
    print("  ".join(header_parts))
    print("  ".join(["─" * 30] + ["─" * 14] * len(available)))

    for _, row in merged.iterrows():
        parts = [f"{row['label']:<30}"]
        for key, _ in available:
            dcol = f"{key}_delta"
            if dcol in row and not pd.isna(row[dcol]):
                val = row[dcol]
                sign = "+" if val >= 0 else ""
                parts.append(f"{sign}{val:+.3f}".rjust(14))
            else:
                parts.append(f"{'—':>14}")
        print("  ".join(parts))

    # ── Head-to-head: resampled vs. native_locked ────────────────────────────
    if "resampled" in merged.columns and "native_locked" in merged.columns:
        print()
        print("─" * 72)
        print("RESAMPLED vs. NATIVE LOCKED (same split, different centerlines)")
        print("Positive = native_locked is better; Negative = resampled is better")
        print("─" * 72)

        merged["resamp_vs_native"] = merged["native_locked"] - merged["resampled"]

        header = f"  {'Arm':<30}  {'Resampled':>10}  {'Native Locked':>14}  {'Delta':>8}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for _, row in merged.iterrows():
            r = row.get("resampled", float("nan"))
            n = row.get("native_locked", float("nan"))
            d = row.get("resamp_vs_native", float("nan"))
            r_str = f"{r:.3f}" if not pd.isna(r) else "—"
            n_str = f"{n:.3f}" if not pd.isna(n) else "—"
            d_str = f"{d:+.3f}" if not pd.isna(d) else "—"
            print(f"  {row['label']:<30}  {r_str:>10}  {n_str:>14}  {d_str:>8}")

        # Interpretation hint
        non_baseline = merged.loc[
            merged["arm_name"] != "clinical_plus_tumor_size", "resamp_vs_native"
        ].dropna()
        if len(non_baseline) > 0:
            print()
            consistently_negative = (non_baseline < 0).all()
            consistently_positive = (non_baseline > 0).all()
            if consistently_negative:
                print(
                    "  Interpretation: native_locked < resampled for ALL vessel arms.\n"
                    "  → Resampling is not degrading vessel features; the drop seen in\n"
                    "    the resampled run is not due to the spacing change.\n"
                    "    Look at code changes (second_order.py) as the likely cause."
                )
            elif consistently_positive:
                print(
                    "  Interpretation: native_locked > resampled for ALL vessel arms.\n"
                    "  → Resampled centerlines produce weaker vessel features than native.\n"
                    "    The spacing change degrades the vessel signal."
                )
            else:
                print(
                    "  Interpretation: mixed pattern — some arms favor resampled, some native.\n"
                    "  → Effect is feature-block specific; investigate arm by arm."
                )

    # ── Save merged table ────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "results" / "locked_split_auc_comparison.csv"
    save_cols = ["arm_name", "label"] + [
        col
        for col in merged.columns
        if col not in {"arm_name", "label", "_order"}
    ]
    merged[[c for c in save_cols if c in merged.columns]].to_csv(out_path, index=False)
    print()
    print(f"Saved full comparison table to: {out_path}")


if __name__ == "__main__":
    main()
