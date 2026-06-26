#!/usr/bin/env python3
"""STEP 3 (sagittal): build two 4-version SAGITTAL comparison figures for DUKE_001.

Why a sagittal view?  The earlier axial figures could not show the z-spacing
difference, because an axial slice lies in the in-plane (x, y) plane that we
never resample.  A SAGITTAL slice is a y-by-z plane, so the through-plane z axis
is now one of the two in-plane axes of the picture.  That means the coarse 2.0mm
spacing of Version 1 shows up directly as a blocky, low-resolution image, while
Versions 2->3->4 get progressively sharper.

This script is lightweight (a few timepoints, slice-by-slice) so it runs on the
head node: no GPU, no Slurm.

Per version it:
  1. Subtraction image = second post-contrast (0002) - pre-contrast (0000).
     (Per your confirmed choice we use 0002, not 0001, for the subtraction.)
  2. Sagittal slice = fix an x index, giving a (y, z) plane.  We pick the x index
     whose vessel mask carries the most total signal, and use that one index for
     every panel of that version.
  3. Independently normalize the raw and subtraction sagittal slices (clip to the
     1st-99th percentile, scale to 0-255).

Two figures (4 version columns x 2 rows: raw / subtraction), vessel mask overlaid
in 30%-opacity red:
  * comparison_sagittal_native_resolution.png  - native pixel grids, drawn at
    true physical proportions, so Version 1 looks visibly blocky (the key figure).
  * comparison_sagittal_matched_resolution.png - every panel resampled to a common
    pixel grid so anatomy is directly comparable regardless of z resolution.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image

VERSIONS = [
    ("version1_z2.0mm", "Version 1: z=2.0mm\n(thick, OOD)"),
    ("version2_z1.4mm", "Version 2: z=1.4mm\n(upsampled)"),
    ("version3_z1.0mm", "Version 3: z=1.0mm\n(upsampled)"),
    ("version4_original", "Version 4: original\n(~1.0mm)"),
]

TP_PRE = "0000"          # pre-contrast (subtraction baseline)
TP_RAW_MASK = "0001"     # early post-contrast: shown raw + provides the mask
TP_SUB_POST = "0002"     # second post-contrast phase: used for the subtraction


def to_model_orient(sitk_array: np.ndarray) -> np.ndarray:
    """Match batch_segmentation's geometric reshuffle: (z,y,x) -> (y, x, z)."""
    return np.swapaxes(np.swapaxes(sitk_array, 0, 2), 0, 1)[::-1]


def load_oriented(nii_path: Path) -> tuple[np.ndarray, float, float]:
    """Return (array (y,x,z) float32, y-spacing mm, z-spacing mm)."""
    img = sitk.ReadImage(str(nii_path))
    sx, sy, sz = img.GetSpacing()  # (x, y, z)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return to_model_orient(arr), float(sy), float(sz)


def load_mask(mask_path: Path) -> np.ndarray:
    return np.load(mask_path)["vessel"].astype(np.float32)


def normalize_to_255(img2d: np.ndarray) -> np.ndarray:
    """Clip to 1st-99th percentile, rescale to 0-255 uint8."""
    lo, hi = np.percentile(img2d, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip(img2d, lo, hi) - lo) / (hi - lo) * 255.0


def mask_npz_path(exp_root: Path, folder: str, case_id: str, tp: str) -> Path:
    source = case_id.split("_")[0]
    return (
        exp_root / "segmentations" / folder / source / case_id / "images"
        / f"{case_id}_{tp}_vessel_segmentation.npz"
    )


def gather_version(exp_root: Path, folder: str, case_id: str, mask_thresh: float):
    """Load one version, pick its best sagittal x, return (y,z) slices."""
    vol_dir = exp_root / "volumes" / folder / case_id
    pre_path = vol_dir / f"{case_id}_{TP_PRE}.nii.gz"
    raw_path = vol_dir / f"{case_id}_{TP_RAW_MASK}.nii.gz"
    sub_path = vol_dir / f"{case_id}_{TP_SUB_POST}.nii.gz"
    mask_path = mask_npz_path(exp_root, folder, case_id, TP_RAW_MASK)

    for p in (pre_path, raw_path, sub_path, mask_path):
        if not p.exists():
            print(f"  [WARN] missing for {folder}: {p}")
            return None

    raw, y_sp, z_sp = load_oriented(raw_path)   # (y, x, z)
    pre, _, _ = load_oriented(pre_path)
    sub_post, _, _ = load_oriented(sub_path)
    subtraction = sub_post - pre                 # tp2 - tp0

    mask = load_mask(mask_path)
    if mask.shape != raw.shape:
        print(f"  [WARN] {folder}: mask {mask.shape} != image {raw.shape}")
        return None

    # Sagittal slice = fix x (axis 1).  Total vessel signal per x = sum over (y, z).
    per_x_signal = mask.sum(axis=(0, 2))
    best_x = int(np.argmax(per_x_signal))

    # Each panel is a (y, z) plane: rows = y (in-plane), cols = z (through-plane).
    raw_slice = normalize_to_255(raw[:, best_x, :]).astype(np.uint8)
    sub_slice = normalize_to_255(subtraction[:, best_x, :]).astype(np.uint8)
    mask_slice = mask[:, best_x, :] > mask_thresh

    print(f"  {folder}: z-spacing={z_sp} mm, plane shape (y,z)={raw_slice.shape}, "
          f"best sagittal x={best_x}/{mask.shape[1]}, "
          f"vessel voxels on slice={int(mask_slice.sum())}")
    return {
        "raw": raw_slice, "sub": sub_slice, "mask": mask_slice,
        "y_sp": y_sp, "z_sp": z_sp, "best_x": best_x,
    }


def pil_resize(arr2d: np.ndarray, out_hw: tuple[int, int], smooth: bool) -> np.ndarray:
    """Resize a 2-D array to (H, W). Bilinear for images, nearest for masks."""
    out_h, out_w = out_hw
    # Pillow moved these constants under Image.Resampling; fall back for old PIL.
    filters = getattr(Image, "Resampling", Image)
    resample = filters.BILINEAR if smooth else filters.NEAREST
    # PIL.resize takes (width, height).
    im = Image.fromarray(arr2d).resize((out_w, out_h), resample)
    return np.asarray(im)


def draw_panel(ax, base_uint8, mask_bool, y_sp, z_sp) -> None:
    """Draw a (y, z) sagittal slice with vessel mask in 30% red.

    We set the display extent to physical millimetres and aspect='equal' so the
    panel keeps true anatomical proportions; with nearest-neighbour interpolation
    a coarse z grid (few columns) therefore looks blocky rather than being hidden.
    Cols = z (width = n_z * z_sp mm), rows = y (height = n_y * y_sp mm).
    """
    n_y, n_z = base_uint8.shape
    extent = [0, n_z * z_sp, 0, n_y * y_sp]  # [left, right, bottom, top] in mm
    ax.imshow(base_uint8, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=extent, aspect="equal", origin="lower")
    overlay = np.zeros((n_y, n_z, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = np.where(mask_bool, 0.30, 0.0)
    ax.imshow(overlay, interpolation="nearest", extent=extent, aspect="equal",
              origin="lower")
    ax.set_axis_off()


def build_figure(panels, labels, out_path, match_resolution) -> None:
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))

    if match_resolution:
        # Common pixel grid = largest n_y and largest n_z across versions.
        target_hw = (
            max(p["raw"].shape[0] for p in panels if p),
            max(p["raw"].shape[1] for p in panels if p),
        )
        # Common physical mm box so every (resized) panel is drawn at one scale.
        common_y_mm = max(p["raw"].shape[0] * p["y_sp"] for p in panels if p)
        common_z_mm = max(p["raw"].shape[1] * p["z_sp"] for p in panels if p)

    for col, (panel, label) in enumerate(zip(panels, labels)):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        ax_top.set_title(label, fontsize=11)

        if panel is None:
            for ax in (ax_top, ax_bot):
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_axis_off()
            continue

        raw, sub, mask = panel["raw"], panel["sub"], panel["mask"]
        y_sp, z_sp = panel["y_sp"], panel["z_sp"]

        if match_resolution:
            raw = pil_resize(raw, target_hw, smooth=True)
            sub = pil_resize(sub, target_hw, smooth=True)
            mask = pil_resize(mask.astype(np.uint8), target_hw, smooth=False).astype(bool)
            # After matching, derive an effective spacing so the common physical
            # box is filled identically by every column.
            y_sp = common_y_mm / target_hw[0]
            z_sp = common_z_mm / target_hw[1]

        draw_panel(ax_top, raw, mask, y_sp, z_sp)
        draw_panel(ax_bot, sub, mask, y_sp, z_sp)

    fig.text(0.015, 0.72, "raw MRI\n(early post-contrast)", rotation=90,
             va="center", ha="center", fontsize=11)
    fig.text(0.015, 0.28, "subtraction\n(tp2 - tp0)", rotation=90,
             va="center", ha="center", fontsize=11)

    fig.suptitle("Vessel Segmentation Quality vs. Z Spacing — DUKE_001 (Sagittal View)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-root",
        default=str(Path.home() / "vanguard_qc_pngs" / "resampling_experiment"),
    )
    parser.add_argument("--case-id", default="DUKE_001")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    labels = [label for _, label in VERSIONS]

    print("Gathering sagittal panels for each version:")
    panels = [
        gather_version(exp_root, folder, args.case_id, args.mask_thresh)
        for folder, _ in VERSIONS
    ]

    print("\nBuilding figures:")
    build_figure(panels, labels,
                 exp_root / "comparison_sagittal_native_resolution.png",
                 match_resolution=False)
    build_figure(panels, labels,
                 exp_root / "comparison_sagittal_matched_resolution.png",
                 match_resolution=True)

    print("\nSTEP 3 (sagittal) complete.")


if __name__ == "__main__":
    main()
