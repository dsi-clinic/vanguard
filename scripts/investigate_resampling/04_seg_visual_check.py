#!/usr/bin/env python3
"""
Visual spot-check: native vs. resampled vessel segmentation MIPs.

For a set of ISPY2 cases, loads the native vessel segmentation NPZ and the
resampled NPZ at a single timepoint, generates maximum-intensity projections
(XY, XZ, YZ views), and saves side-by-side comparison PNGs. Also checks for
data integrity issues (NaN values, out-of-range intensities, suspiciously empty
or full masks after resampling).

This script is COMPUTE HEAVY (loads large NPZ arrays) and must be run via
Slurm, not on the head node. The launcher script submits it automatically.

Outputs (written to --out-dir):
  {case_id}_timepoint{t}_native_vs_resampled.png  — side-by-side MIPs
  integrity_report.csv                             — per-case NaN/range/coverage checks

Usage (via Slurm — see submit_resampling_investigation.sh):
  python scripts/investigate_resampling/04_seg_visual_check.py \
      --native-seg-root /gpfs/data/karczmar-lab/workspaces/saritbose/vessel_segmentations \
      --resampled-seg-root /gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations \
      --cohort ISPY2 \
      --case-ids ISPY2_100899 ISPY2_102011 ISPY2_102212 ISPY2_103301 ISPY2_103651 \
      --timepoint 0 \
      --out-dir /ess/scratch/scratch1/t-9sbose/resampled_investigation/04_seg_visual_check
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_seg_npz(path: Path) -> np.ndarray:
    """Load a vessel segmentation NPZ. Returns a 3-D float32 array (Z, Y, X)."""
    data = np.load(str(path))
    # NPZ may have different key names depending on how it was saved
    keys = list(data.keys())
    arr = data[keys[0]]  # take the first (and usually only) array
    return arr.astype(np.float32)


def max_project(volume: np.ndarray, axis: int) -> np.ndarray:
    """Max-intensity projection along the given axis (0=Z→XY, 1=Y→XZ, 2=X→YZ)."""
    return np.nanmax(volume, axis=axis)


def check_integrity(arr: np.ndarray, label: str) -> dict:
    """Return a dict of integrity metrics for one segmentation volume."""
    return {
        "label": label,
        "shape": str(arr.shape),
        "has_nan": bool(np.any(np.isnan(arr))),
        "n_nan": int(np.sum(np.isnan(arr))),
        "min_val": float(np.nanmin(arr)),
        "max_val": float(np.nanmax(arr)),
        "pct_nonzero": float(np.nansum(arr > 0.5) / arr.size * 100),
        "mean_val": float(np.nanmean(arr)),
    }


def plot_comparison(
    native_arr: np.ndarray,
    resampled_arr: np.ndarray,
    case_id: str,
    timepoint: int,
    out_path: Path,
) -> None:
    """Side-by-side 3-view MIP comparison: native (top row) vs resampled (bottom row)."""
    views = [
        ("XY (axial)", 0),    # max along Z axis → see XY plane
        ("XZ (coronal)", 1),  # max along Y axis → see XZ plane
        ("YZ (sagittal)", 2), # max along X axis → see YZ plane
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"{case_id}  |  timepoint {timepoint}\nTop: native    Bottom: resampled",
        fontweight="bold", fontsize=11
    )

    for col_idx, (view_name, axis) in enumerate(views):
        nat_proj = max_project(native_arr, axis)
        res_proj = max_project(resampled_arr, axis)

        vmax = max(nat_proj.max(), res_proj.max(), 1e-6)

        for row_idx, (proj, label) in enumerate([
            (nat_proj, f"Native  {native_arr.shape}"),
            (res_proj, f"Resampled  {resampled_arr.shape}"),
        ]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(proj, cmap="hot", vmin=0, vmax=vmax, origin="lower",
                           aspect="auto")
            title = view_name if row_idx == 0 else ""
            ax.set_title(f"{title}\n{label}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-seg-root", type=Path, required=True,
                        help="Root of native vessel segmentations: <root>/<cohort>/<case>/images/")
    parser.add_argument("--resampled-seg-root", type=Path, required=True,
                        help="Root of resampled vessel segmentations")
    parser.add_argument("--cohort", type=str, default="ISPY2")
    parser.add_argument("--case-ids", nargs="+", required=True,
                        help="List of case IDs to spot-check")
    parser.add_argument("--timepoint", type=int, default=0,
                        help="Which timepoint index (0-5) to compare")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    t = args.timepoint
    integrity_rows = []

    for case_id in args.case_ids:
        print(f"\nProcessing {case_id} ...")

        # Locate segmentation files
        native_path = (
            args.native_seg_root / args.cohort / case_id / "images"
            / f"{case_id}_{t:04d}_vessel_segmentation.npz"
        )
        resampled_path = (
            args.resampled_seg_root / args.cohort / case_id / "images"
            / f"{case_id}_{t:04d}_vessel_segmentation.npz"
        )

        if not native_path.exists():
            print(f"  SKIP: native not found at {native_path}")
            integrity_rows.append({
                "case_id": case_id, "timepoint": t, "status": "native_missing"
            })
            continue
        if not resampled_path.exists():
            print(f"  SKIP: resampled not found at {resampled_path}")
            integrity_rows.append({
                "case_id": case_id, "timepoint": t, "status": "resampled_missing"
            })
            continue

        try:
            print(f"  Loading native   {native_path} ...")
            native_arr = load_seg_npz(native_path)
            print(f"  Loading resampled {resampled_path} ...")
            resampled_arr = load_seg_npz(resampled_path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            integrity_rows.append({
                "case_id": case_id, "timepoint": t, "status": f"load_error: {e}"
            })
            continue

        # Integrity checks
        for label, arr in [("native", native_arr), ("resampled", resampled_arr)]:
            row = check_integrity(arr, label)
            row.update({"case_id": case_id, "timepoint": t})
            integrity_rows.append(row)
            print(
                f"  {label}: shape={arr.shape}  NaN={row['has_nan']}  "
                f"range=[{row['min_val']:.3f}, {row['max_val']:.3f}]  "
                f"nonzero={row['pct_nonzero']:.1f}%"
            )
            if row["has_nan"]:
                print(f"  WARNING: {row['n_nan']} NaN values in {label} segmentation!")
            if row["pct_nonzero"] < 0.01:
                print(f"  WARNING: {label} segmentation is nearly empty!")
            if row["pct_nonzero"] > 50:
                print(f"  WARNING: {label} segmentation is >50% non-zero (unexpected)")

        # MIP comparison plot
        out_path = args.out_dir / f"{case_id}_timepoint{t}_native_vs_resampled.png"
        plot_comparison(native_arr, resampled_arr, case_id, t, out_path)
        print(f"  Saved {out_path}")

    # Write integrity report
    import pandas as pd
    report_df = pd.DataFrame(integrity_rows)
    report_df.to_csv(args.out_dir / "integrity_report.csv", index=False)
    print(f"\nSaved integrity_report.csv ({len(integrity_rows)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
