#!/usr/bin/env python3
"""STEP 3: build the two 4-version comparison figures for DUKE_001.

This is a *lightweight* script: it only ever loads a few timepoints per version
and works slice-by-slice, so it is safe to run on the head node (no GPU, no
Slurm).

What each panel shows
---------------------
  * Top row    : the early post-contrast raw MRI (timepoint 0001) with the
                 vessel mask overlaid in 30%-opacity red.
  * Bottom row : the subtraction image  (timepoint 0002 - timepoint 0000), i.e.
                 the SECOND post-contrast phase minus pre-contrast, again with
                 the vessel mask overlaid.

Choosing the slice: ONE physical z for all four versions
--------------------------------------------------------
Earlier we picked each version's own best slice, which meant the four columns
sat at different anatomical heights.  Now we instead:
  1. Find the axial slice with the most vessel signal in the ORIGINAL version
     (Version 4) - our reference.
  2. Convert that slice index into a physical position in millimetres
     (index x z-spacing, measured from the shared volume origin).
  3. For every version, choose the slice index closest to that SAME physical
     position.  Because the four versions have different z spacing, the same
     physical location lands on a different index in each, but it is the same
     anatomy - so the columns are now directly comparable.

Both figures are 4 columns (versions) x 2 rows (raw / subtraction).  The
"matched" figure additionally resizes every panel to a common pixel grid; for
axial views this is essentially a no-op because we never resample in-plane.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to a file, no display
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# (folder name, human-readable column label).  Order = left-to-right columns.
VERSIONS = [
    ("version1_z2.0mm", "Version 1: z=2.0mm\n(thick, OOD)"),
    ("version2_z1.4mm", "Version 2: z=1.4mm\n(upsampled)"),
    ("version3_z1.0mm", "Version 3: z=1.0mm\n(upsampled)"),
    ("version4_original", "Version 4: original\n(~1.1mm)"),
]

# The version we trust as ground truth for choosing the comparison slice.
REFERENCE_VERSION = "version4_original"

# Timepoints (file suffixes).  0000 = pre-contrast, 0001 = early post-contrast,
# 0002 = second post-contrast phase.
TP_PRE = "0000"          # pre-contrast (subtraction baseline)
TP_RAW_MASK = "0001"     # early post-contrast: shown raw + provides the mask
TP_SUB_POST = "0002"     # SECOND post-contrast phase: used for the subtraction


def to_model_orient(sitk_array: np.ndarray) -> np.ndarray:
    """Reproduce the geometric reshuffle batch_segmentation applies before
    inference, so the image axes match the saved vessel mask.

    SimpleITK hands back arrays indexed (z, y, x).  The model pipeline does
    swapaxes(0,2) -> swapaxes(0,1) -> reverse axis 0, giving (y, x, z) with the
    axial-slice (z) axis last.
    """
    return np.swapaxes(np.swapaxes(sitk_array, 0, 2), 0, 1)[::-1]


def load_oriented(nii_path: Path) -> tuple[np.ndarray, float]:
    """Read a NIfTI; return (array in model orientation float32, z-spacing mm)."""
    img = sitk.ReadImage(str(nii_path))
    z_spacing = float(img.GetSpacing()[2])
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return to_model_orient(arr), z_spacing


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a vessel probability volume (y, x, z) from a saved .npz."""
    return np.load(mask_path)["vessel"].astype(np.float32)


def normalize_to_255(img2d: np.ndarray) -> np.ndarray:
    """Clip to the 1st-99th percentile and rescale to 0-255 (uint8).

    Clipping at percentiles (rather than min/max) throws away a few extreme
    outlier pixels that would otherwise wash out the contrast of everything else.
    """
    lo, hi = np.percentile(img2d, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = (np.clip(img2d, lo, hi) - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def resize_nearest(arr2d: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a 2-D array to out_shape, using only numpy."""
    in_h, in_w = arr2d.shape
    out_h, out_w = out_shape
    if (in_h, in_w) == (out_h, out_w):
        return arr2d
    row_idx = np.linspace(0, in_h - 1, out_h).round().astype(int)
    col_idx = np.linspace(0, in_w - 1, out_w).round().astype(int)
    return arr2d[row_idx][:, col_idx]


def mask_npz_path(exp_root: Path, folder: str, case_id: str, tp: str) -> Path:
    source = case_id.split("_")[0]  # "DUKE_001" -> "DUKE"
    return (
        exp_root / "segmentations" / folder / source / case_id / "images"
        / f"{case_id}_{tp}_vessel_segmentation.npz"
    )


def reference_physical_z(exp_root: Path, case_id: str) -> float:
    """Physical z (mm from origin) of the best vessel slice in the reference.

    We sum the vessel probability over each axial slice of the reference
    version's early-post-contrast mask, take the slice with the most signal, and
    multiply its index by the reference z-spacing to get a physical location.
    """
    mask = load_mask(mask_npz_path(exp_root, REFERENCE_VERSION, case_id, TP_RAW_MASK))
    # We also need the reference z-spacing; read it from the matching volume.
    ref_vol = exp_root / "volumes" / REFERENCE_VERSION / case_id / f"{case_id}_{TP_RAW_MASK}.nii.gz"
    z_spacing = float(sitk.ReadImage(str(ref_vol)).GetSpacing()[2])

    per_slice_signal = mask.sum(axis=(0, 1))
    best_z = int(np.argmax(per_slice_signal))
    phys_z = best_z * z_spacing
    print(f"  reference ({REFERENCE_VERSION}): best slice z={best_z}/{mask.shape[2]}, "
          f"z-spacing={z_spacing} mm  ->  physical z = {phys_z:.1f} mm")
    return phys_z


def gather_version(exp_root: Path, folder: str, case_id: str, target_phys_z: float,
                   mask_thresh: float):
    """Load everything for one version and slice it at the shared physical z."""
    vol_dir = exp_root / "volumes" / folder / case_id
    pre_path = vol_dir / f"{case_id}_{TP_PRE}.nii.gz"
    raw_path = vol_dir / f"{case_id}_{TP_RAW_MASK}.nii.gz"
    sub_path = vol_dir / f"{case_id}_{TP_SUB_POST}.nii.gz"
    mask_path = mask_npz_path(exp_root, folder, case_id, TP_RAW_MASK)

    for p in (pre_path, raw_path, sub_path, mask_path):
        if not p.exists():
            print(f"  [WARN] missing for {folder}: {p}")
            return None

    raw, z_spacing = load_oriented(raw_path)       # early post-contrast (0001)
    pre, _ = load_oriented(pre_path)               # pre-contrast (0000)
    sub_post, _ = load_oriented(sub_path)          # second post-contrast (0002)
    subtraction = sub_post - pre                   # tp2 - tp0

    mask = load_mask(mask_path)
    if mask.shape != raw.shape:
        print(f"  [WARN] {folder}: mask shape {mask.shape} != image {raw.shape}")
        return None

    n_z = raw.shape[2]
    # Same physical location -> nearest slice index for THIS version's spacing.
    k = int(round(target_phys_z / z_spacing))
    k = max(0, min(k, n_z - 1))

    raw_slice = normalize_to_255(raw[:, :, k])
    sub_slice = normalize_to_255(subtraction[:, :, k])
    mask_slice = mask[:, :, k] > mask_thresh

    print(f"  {folder}: z-spacing={z_spacing} mm, {n_z} slices, "
          f"chosen slice k={k} (physical {k*z_spacing:.1f} mm), "
          f"vessel voxels on slice={int(mask_slice.sum())}")
    return {"raw": raw_slice, "sub": sub_slice, "mask": mask_slice, "k": k}


def draw_panel(ax, base_uint8: np.ndarray, mask_bool: np.ndarray) -> None:
    """Show a grayscale slice with the vessel mask overlaid in 30%-opacity red."""
    ax.imshow(base_uint8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    overlay = np.zeros((*mask_bool.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0                             # red channel
    overlay[..., 3] = np.where(mask_bool, 0.30, 0.0)  # alpha (opacity)
    ax.imshow(overlay, interpolation="nearest")
    ax.set_axis_off()


def build_figure(panels, labels, out_path: Path, match_resolution: bool) -> None:
    """Render the 4x2 comparison figure and save it."""
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8.5))

    if match_resolution:
        target = (
            max(p["raw"].shape[0] for p in panels if p),
            max(p["raw"].shape[1] for p in panels if p),
        )

    for col, (panel, label) in enumerate(zip(panels, labels)):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        ax_top.set_title(label, fontsize=11)

        if panel is None:
            for ax in (ax_top, ax_bot):
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_axis_off()
            continue

        raw, sub, mask = panel["raw"], panel["sub"], panel["mask"]
        if match_resolution:
            raw = resize_nearest(raw, target)
            sub = resize_nearest(sub, target)
            mask = resize_nearest(mask.astype(np.uint8), target).astype(bool)

        draw_panel(ax_top, raw, mask)
        draw_panel(ax_bot, sub, mask)

    fig.text(0.015, 0.72, "raw MRI\n(early post-contrast)", rotation=90,
             va="center", ha="center", fontsize=11)
    fig.text(0.015, 0.28, "subtraction\n(tp2 - tp0)", rotation=90,
             va="center", ha="center", fontsize=11)

    fig.suptitle("Vessel Segmentation Quality vs. Z Spacing - DUKE_001\n"
                 "(all columns at the same physical z)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-root",
        default=str(Path.home() / "vanguard_qc_pngs" / "resampling_experiment"),
        help="Experiment root holding volumes/ and segmentations/.",
    )
    parser.add_argument("--case-id", default="DUKE_001")
    parser.add_argument("--mask-thresh", type=float, default=0.5,
                        help="Probability above which a voxel counts as vessel.")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    labels = [label for _, label in VERSIONS]

    print("Choosing the shared comparison slice:")
    target_phys_z = reference_physical_z(exp_root, args.case_id)

    print("\nGathering panels for each version (same physical z):")
    panels = [
        gather_version(exp_root, folder, args.case_id, target_phys_z, args.mask_thresh)
        for folder, _ in VERSIONS
    ]

    print("\nBuilding figures:")
    build_figure(panels, labels, exp_root / "comparison_native_resolution.png",
                 match_resolution=False)
    build_figure(panels, labels, exp_root / "comparison_matched_resolution.png",
                 match_resolution=True)

    print("\nSTEP 3 complete.")


if __name__ == "__main__":
    main()
