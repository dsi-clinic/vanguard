#!/usr/bin/env python3
"""
Voxel-spacing audit: compare native vs. resampled tumor masks.

This is important because the morph feature block computes vessel lengths and
volumes in VOXEL UNITS (not mm). If resampling changed the voxel spacing, then
morph feature values will be numerically different even for physically identical
vessels. We want to document:
  1. What spacing the resampled masks actually have (are they all the same?)
  2. How that differs from the native masks
  3. What the implied scale factor is for morph features (voxels → mm conversion)

Outputs (written to --out-dir):
  resampled_spacings.csv   — (x_mm, y_mm, z_mm, vol_vox3) for all resampled masks
  native_spacings.csv      — same for a sample of native masks
  spacing_summary.txt      — human-readable comparison + morph scale implications
  spacing_distributions.png— histograms of x/y/z spacing, native vs resampled

Usage:
  python scripts/investigate_resampling/02_spacing_audit.py \
      --resampled-mask-dir /gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks \
      --native-mask-dir /ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/segmentations/expert \
      --out-dir /ess/scratch/scratch1/t-9sbose/resampled_investigation/02_spacing_audit
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
    def read_spacing(nii_path: Path):
        """Return (x_mm, y_mm, z_mm) spacing from a NIfTI/NRRD header (no pixel data loaded)."""
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(nii_path))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()  # only reads header, not voxels — fast
        return reader.GetSpacing(), reader.GetSize()
except ImportError:
    import nibabel as nib
    def read_spacing(nii_path: Path):
        img = nib.load(str(nii_path))
        hdr = img.header
        spacings = tuple(float(x) for x in hdr.get_zooms()[:3])
        size = tuple(int(x) for x in hdr.get_data_shape()[:3])
        return spacings, size


def collect_spacings(mask_dir: Path, max_cases: int = None) -> pd.DataFrame:
    """Read spacing+size from all .nii.gz masks in a directory."""
    paths = sorted(mask_dir.glob("*.nii.gz"))
    if max_cases is not None:
        paths = paths[:max_cases]
    rows = []
    for p in paths:
        try:
            spacing, size = read_spacing(p)
            rows.append({
                "case_id": p.stem.replace(".nii", ""),
                "x_mm": round(spacing[0], 4),
                "y_mm": round(spacing[1], 4),
                "z_mm": round(spacing[2], 4),
                "nx": size[0],
                "ny": size[1],
                "nz": size[2],
                "voxel_vol_mm3": round(spacing[0] * spacing[1] * spacing[2], 4),
            })
        except Exception as e:
            rows.append({"case_id": p.stem, "error": str(e)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resampled-mask-dir", type=Path, required=True)
    parser.add_argument("--native-mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--native-sample-n", type=int, default=50,
                        help="Number of native masks to sample (reading all 808 is slower)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resampled masks (all of them) ---
    print(f"Reading resampled mask headers from {args.resampled_mask_dir} ...")
    df_res = collect_spacings(args.resampled_mask_dir)
    df_res_ok = df_res[~df_res.get("error", pd.Series(dtype=str)).notna()].copy() if "error" in df_res.columns else df_res.copy()
    df_res.to_csv(args.out_dir / "resampled_spacings.csv", index=False)
    print(f"  {len(df_res)} resampled masks read")

    # --- Native masks (sample) ---
    print(f"Reading native mask headers (sample n={args.native_sample_n}) ...")
    df_nat = collect_spacings(args.native_mask_dir, max_cases=args.native_sample_n)
    df_nat_ok = df_nat[~df_nat.get("error", pd.Series(dtype=str)).notna()].copy() if "error" in df_nat.columns else df_nat.copy()
    df_nat.to_csv(args.out_dir / "native_spacings.csv", index=False)
    print(f"  {len(df_nat)} native masks read")

    # --- Summary text ---
    lines = []
    lines.append("=" * 60)
    lines.append("VOXEL SPACING AUDIT: RESAMPLED vs NATIVE TUMOR MASKS")
    lines.append("=" * 60)

    for label, df_ok in [("RESAMPLED", df_res_ok), ("NATIVE (sample)", df_nat_ok)]:
        lines.append(f"\n{label} (n={len(df_ok)}):")
        for dim in ("x_mm", "y_mm", "z_mm"):
            if dim not in df_ok.columns:
                continue
            vals = df_ok[dim].dropna()
            lines.append(
                f"  {dim}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                f"min={vals.min():.4f}  max={vals.max():.4f}"
            )
        if "voxel_vol_mm3" in df_ok.columns:
            v = df_ok["voxel_vol_mm3"].dropna()
            lines.append(f"  voxel vol: mean={v.mean():.4f} mm³")
        # Show unique spacing values (there should be few distinct ones)
        if all(c in df_ok.columns for c in ("x_mm", "y_mm", "z_mm")):
            combos = df_ok[["x_mm", "y_mm", "z_mm"]].dropna().value_counts().head(5)
            lines.append(f"  Most common spacings:")
            for idx, cnt in combos.items():
                lines.append(f"    {tuple(idx)}  → {cnt} cases")

    # Compute implied scale factor for morph features
    if "z_mm" in df_res_ok.columns and "z_mm" in df_nat_ok.columns:
        z_res = df_res_ok["z_mm"].median()
        z_nat = df_nat_ok["z_mm"].median()
        scale_z = z_res / z_nat if z_nat > 0 else float("nan")
        lines.append("\n" + "=" * 60)
        lines.append("MORPH FEATURE SCALE IMPLICATION")
        lines.append("=" * 60)
        lines.append(
            f"\nMorphometry features (morph block) are in VOXEL units, not mm."
        )
        lines.append(
            f"Z-spacing changed: native median={z_nat:.4f}mm → resampled={z_res:.4f}mm"
        )
        lines.append(
            f"Scale factor (resampled/native) for Z-voxel counts: {scale_z:.4f}x"
        )
        if scale_z < 1.0:
            lines.append(
                f"  → A vessel that was {1/scale_z:.1f} voxels long (native Z) is now "
                f"1 voxel long (resampled Z). Z-oriented vessels appear SHORTER in voxels."
            )
        else:
            lines.append(
                f"  → Resampled Z-spacing is COARSER than native. "
                f"Z-oriented vessels appear LONGER in voxels."
            )
        lines.append(
            "\nIMPACT: StandardScaler normalizes per-column mean/std within each "
            "training fold, so a global scale shift in morph values is absorbed. "
            "However, if spacing varies WITHIN the native cohort (multi-site), "
            "resampling homogenizes the distribution — the relative patient ordering "
            "changes and the morph block's discriminative signal may increase or "
            "decrease depending on whether site-spacing confounded the native features."
        )

    summary_text = "\n".join(lines)
    print(summary_text)
    (args.out_dir / "spacing_summary.txt").write_text(summary_text)
    print(f"\nSaved spacing_summary.txt")

    # --- Distribution plots ---
    dims = ["x_mm", "y_mm", "z_mm"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, dim in zip(axes, dims):
        if dim in df_res_ok.columns:
            ax.hist(df_res_ok[dim].dropna(), bins=20, alpha=0.6,
                    color="steelblue", label=f"resampled (n={len(df_res_ok)})", edgecolor="none")
        if dim in df_nat_ok.columns:
            ax.hist(df_nat_ok[dim].dropna(), bins=20, alpha=0.6,
                    color="darkorange", label=f"native sample (n={len(df_nat_ok)})", edgecolor="none")
        ax.set_xlabel(f"{dim} (mm)")
        ax.set_ylabel("n masks")
        ax.set_title(dim)
        ax.legend(fontsize=8)
    fig.suptitle("Tumor mask voxel spacing: native vs resampled", fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out_dir / "spacing_distributions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved spacing_distributions.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
