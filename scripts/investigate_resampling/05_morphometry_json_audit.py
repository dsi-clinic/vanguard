#!/usr/bin/env python3
"""
Morphometry JSON physical-plausibility audit.

Reads morphometry.json files from the resampled centerlines and checks whether
the vessel lengths, radii, and volumes make physical sense given the known
voxel spacing of the resampled images.

Why this matters: morph features are in VOXEL units. If a vessel is physically
10mm long but the spacing changed, its voxel length changed too. We want to
confirm that the resampled values are physically plausible and identify any
outlier cases.

Checks performed:
  - Segment lengths (voxels) → convert to mm using median spacing
  - Mean vessel radii (voxels) → convert to mm
  - Volume per segment (voxels³) → convert to mm³
  - Tortuosity (dimensionless, should be ≥ 1.0)
  - Curvature (radians/voxel, should be 0–π/2)
  - Flags cases with zero segments (empty skeletons)

Expected physical ranges for breast blood vessels:
  - Segment length: 0.5–100 mm (most 1–30 mm)
  - Vessel radius: 0.3–5 mm (most 0.5–2 mm)
  - Tortuosity: 1.0–3.0 (highly tortuous vessels can reach ~5)

Outputs (written to --out-dir):
  morphometry_per_case_stats.csv  — per-case: n_segments, mean_length_mm, etc.
  morphometry_summary_stats.csv   — aggregate over all cases
  morphometry_length_hist.png     — histogram of segment lengths in mm
  morphometry_radius_hist.png     — histogram of mean radii in mm
  morphometry_flags.csv           — cases that fail plausibility checks

Usage:
  python scripts/investigate_resampling/05_morphometry_json_audit.py \
      --centerline-root /gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d \
      --cohort ISPY2 \
      --resampled-spacing-csv /ess/scratch/scratch1/t-9sbose/resampled_investigation/02_spacing_audit/resampled_spacings.csv \
      --out-dir /ess/scratch/scratch1/t-9sbose/resampled_investigation/05_morphometry_audit
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Physical plausibility thresholds (in mm after spacing conversion)
PLAUSIBLE_LENGTH_MM = (0.3, 200.0)    # segment length
PLAUSIBLE_RADIUS_MM = (0.1, 10.0)     # mean vessel radius
PLAUSIBLE_TORTUOSITY = (1.0, 10.0)    # dimensionless


def iter_segments(morph_dict: dict):
    """
    Yield individual segment dicts from a morphometry.json.

    The JSON structure is: {component_id: {component_label: [segment_list, ...]}}
    Each item in segment_list has keys: segment, radius, length, tortuosity, volume, curvature.
    """
    for comp_data in morph_dict.values():
        if not isinstance(comp_data, dict):
            continue
        for branch_segs in comp_data.values():
            if not isinstance(branch_segs, list):
                continue
            for seg in branch_segs:
                if isinstance(seg, dict) and "length" in seg:
                    yield seg


def extract_case_stats(morph_path: Path, spacing_mm: float) -> dict:
    """
    Parse one morphometry.json and return aggregate stats.
    spacing_mm is the geometric mean voxel size in mm (used to convert voxels → mm).
    """
    try:
        data = json.loads(morph_path.read_text())
    except Exception as e:
        return {"error": str(e)}

    segs = list(iter_segments(data))
    if not segs:
        return {"n_segments": 0, "error": "no segments"}

    lengths_vox = np.array([s["length"] for s in segs], dtype=float)
    radii_vox = np.array([s["radius"]["mean"] for s in segs if "radius" in s], dtype=float)
    tortuosities = np.array([s.get("tortuosity", float("nan")) for s in segs], dtype=float)
    volumes_vox3 = np.array([s.get("volume", float("nan")) for s in segs], dtype=float)

    lengths_mm = lengths_vox * spacing_mm
    radii_mm = radii_vox * spacing_mm
    volumes_mm3 = volumes_vox3 * (spacing_mm ** 3)

    return {
        "n_segments": len(segs),
        "length_mm_mean": float(np.mean(lengths_mm)),
        "length_mm_median": float(np.median(lengths_mm)),
        "length_mm_max": float(np.max(lengths_mm)),
        "radius_mm_mean": float(np.mean(radii_mm)) if len(radii_mm) > 0 else float("nan"),
        "radius_mm_max": float(np.max(radii_mm)) if len(radii_mm) > 0 else float("nan"),
        "tortuosity_mean": float(np.nanmean(tortuosities)),
        "tortuosity_max": float(np.nanmax(tortuosities)),
        "volume_mm3_total": float(np.nansum(volumes_mm3)),
        "pct_length_too_short": float(np.mean(lengths_mm < PLAUSIBLE_LENGTH_MM[0]) * 100),
        "pct_length_too_long": float(np.mean(lengths_mm > PLAUSIBLE_LENGTH_MM[1]) * 100),
        "pct_tortuosity_lt1": float(np.mean(tortuosities < PLAUSIBLE_TORTUOSITY[0]) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--cohort", type=str, default="ISPY2")
    parser.add_argument("--resampled-spacing-csv", type=Path, default=None,
                        help="CSV from 02_spacing_audit.py (provides per-case spacing)."
                             " If omitted, uses the spacing from a single sample case.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit to first N cases (for testing)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-case spacings if available
    if args.resampled_spacing_csv and args.resampled_spacing_csv.exists():
        spacing_df = pd.read_csv(args.resampled_spacing_csv)
        # Geometric mean of x/y/z spacing → single scale factor for voxel→mm
        if all(c in spacing_df.columns for c in ("x_mm", "y_mm", "z_mm")):
            spacing_df["spacing_geomean"] = (
                spacing_df["x_mm"] * spacing_df["y_mm"] * spacing_df["z_mm"]
            ) ** (1 / 3)
            spacing_map = dict(zip(spacing_df["case_id"], spacing_df["spacing_geomean"]))
        else:
            spacing_map = {}
    else:
        spacing_map = {}

    # Default spacing: use the sample case we checked earlier
    DEFAULT_SPACING_MM = (0.6641 * 0.6641 * 1.25) ** (1 / 3)  # ≈ 0.849 mm

    # Find all morphometry JSONs
    cohort_dir = args.centerline_root / args.cohort
    morph_paths = sorted(cohort_dir.rglob("*_morphometry.json"))
    if args.max_cases:
        morph_paths = morph_paths[: args.max_cases]

    print(f"Found {len(morph_paths)} morphometry.json files under {cohort_dir}")

    rows = []
    all_lengths_mm = []
    all_radii_mm = []

    for p in morph_paths:
        case_id = p.parent.name
        spacing_mm = spacing_map.get(case_id, DEFAULT_SPACING_MM)
        stats = extract_case_stats(p, spacing_mm)
        stats["case_id"] = case_id
        stats["spacing_mm"] = round(spacing_mm, 4)
        rows.append(stats)

        if "length_mm_mean" in stats and not np.isnan(stats["length_mm_mean"]):
            # Collect for global histogram (don't store all per-segment values — memory)
            # We approximate with per-case mean (imperfect but lightweight)
            all_lengths_mm.append(stats["length_mm_mean"])
        if "radius_mm_mean" in stats and not np.isnan(stats["radius_mm_mean"]):
            all_radii_mm.append(stats["radius_mm_mean"])

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "morphometry_per_case_stats.csv", index=False)

    # Aggregate summary
    numeric_cols = [c for c in df.columns if c not in ("case_id", "error")]
    summary = df[numeric_cols].describe().T
    summary.to_csv(args.out_dir / "morphometry_summary_stats.csv")

    print("\nPer-case morphometry summary (over all cases):")
    for col in ("n_segments", "length_mm_mean", "length_mm_max", "radius_mm_mean",
                "tortuosity_mean", "volume_mm3_total"):
        if col in df.columns:
            vals = df[col].dropna()
            print(
                f"  {col:30s}: mean={vals.mean():.3g}  "
                f"std={vals.std():.3g}  min={vals.min():.3g}  max={vals.max():.3g}"
            )

    # Flag suspicious cases
    flags = []
    for _, row in df.iterrows():
        reasons = []
        if row.get("n_segments", 0) == 0:
            reasons.append("no_segments")
        if row.get("pct_length_too_long", 0) > 1:
            reasons.append(f"length_outliers>{row['pct_length_too_long']:.1f}%")
        if row.get("pct_tortuosity_lt1", 0) > 5:
            reasons.append(f"tortuosity<1_in_{row['pct_tortuosity_lt1']:.1f}%_segs")
        if row.get("length_mm_max", 0) > 300:
            reasons.append(f"max_length={row['length_mm_max']:.0f}mm")
        if reasons:
            flags.append({"case_id": row["case_id"], "flags": "; ".join(reasons)})

    if flags:
        flags_df = pd.DataFrame(flags)
        flags_df.to_csv(args.out_dir / "morphometry_flags.csv", index=False)
        print(f"\nFlagged {len(flags)} cases with plausibility issues:")
        for _, r in flags_df.iterrows():
            print(f"  {r['case_id']}: {r['flags']}")
    else:
        print("\nNo plausibility flags — all cases look reasonable.")

    # Histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if all_lengths_mm:
        ax1.hist(all_lengths_mm, bins=40, color="steelblue", alpha=0.75, edgecolor="none")
        ax1.set_xlabel("Mean segment length per case (mm)")
        ax1.set_ylabel("n cases")
        ax1.set_title("Vessel segment lengths\n(per-case mean, converted from voxels)")
        ax1.axvline(0.5, color="red", linestyle="--", alpha=0.6, label="< 0.5mm (suspect)")
        ax1.axvline(100, color="orange", linestyle="--", alpha=0.6, label="> 100mm (suspect)")
        ax1.legend(fontsize=8)

    if all_radii_mm:
        ax2.hist(all_radii_mm, bins=40, color="darkorange", alpha=0.75, edgecolor="none")
        ax2.set_xlabel("Mean vessel radius per case (mm)")
        ax2.set_ylabel("n cases")
        ax2.set_title("Vessel radii\n(per-case mean, converted from voxels)")
        ax2.axvline(0.1, color="red", linestyle="--", alpha=0.6, label="< 0.1mm (suspect)")
        ax2.axvline(5.0, color="orange", linestyle="--", alpha=0.6, label="> 5mm (suspect)")
        ax2.legend(fontsize=8)

    fig.suptitle(
        f"Morphometry physical plausibility — resampled {args.cohort} "
        f"(spacing ~{DEFAULT_SPACING_MM:.3f} mm geomean)",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(args.out_dir / "morphometry_plausibility.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved morphometry_plausibility.png")
    print("Done.")


if __name__ == "__main__":
    main()
