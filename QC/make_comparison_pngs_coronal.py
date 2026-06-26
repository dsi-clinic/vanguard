#!/usr/bin/env python3
"""STEP 3 (coronal): two 4-version CORONAL comparison figures for DUKE_001.

A coronal slice fixes y (the anterior-posterior axis), giving an (x, z) plane
where x is left-right and z is the through-plane (resampled) axis.  Like the
sagittal view, z is now IN the plane of the image, so the coarse z=2.0mm
spacing of Version 1 appears visibly blocky while Version 4 (native) is sharp.
The coronal view gives a different anatomical perspective — you see both breasts
and the chest wall simultaneously — compared to the single-breast sagittal view.

Subtraction = second post-contrast (0002) - pre-contrast (0000).
Raw-MRI row shown on early post-contrast (0001).
Best coronal y chosen per-version as the y slice with the most vessel signal.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image

VERSIONS = [
    ("version1_z2.0mm",   "Version 1: z=2.0mm\n(thick, OOD)"),
    ("version2_z1.4mm",   "Version 2: z=1.4mm\n(upsampled)"),
    ("version3_z1.0mm",   "Version 3: z=1.0mm\n(upsampled)"),
    ("version4_original", "Version 4: original\n(~1.0mm)"),
]

TP_PRE       = "0000"
TP_RAW_MASK  = "0001"
TP_SUB_POST  = "0002"


def to_model_orient(arr: np.ndarray) -> np.ndarray:
    """(z,y,x) -> (y,x,z) matching batch_segmentation's preprocessing."""
    return np.swapaxes(np.swapaxes(arr, 0, 2), 0, 1)[::-1]


def load_oriented(nii_path: Path) -> tuple[np.ndarray, float, float, float]:
    """Return (array (y,x,z) float32, x-spacing mm, y-spacing mm, z-spacing mm)."""
    img = sitk.ReadImage(str(nii_path))
    sx, sy, sz = img.GetSpacing()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return to_model_orient(arr), float(sx), float(sy), float(sz)


def load_mask(path: Path) -> np.ndarray:
    return np.load(path)["vessel"].astype(np.float32)


def normalize_to_255(img2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img2d, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return ((np.clip(img2d, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)


def mask_npz_path(exp_root: Path, folder: str, case_id: str, tp: str) -> Path:
    source = case_id.split("_")[0]
    return (
        exp_root / "segmentations" / folder / source / case_id / "images"
        / f"{case_id}_{tp}_vessel_segmentation.npz"
    )


def gather_version(exp_root: Path, folder: str, case_id: str, mask_thresh: float,
                   fixed_y: int | None = None):
    """Load one version, find the best coronal y, return (x,z) slices."""
    vol_dir = exp_root / "volumes" / folder / case_id
    pre_path = vol_dir / f"{case_id}_{TP_PRE}.nii.gz"
    raw_path = vol_dir / f"{case_id}_{TP_RAW_MASK}.nii.gz"
    sub_path = vol_dir / f"{case_id}_{TP_SUB_POST}.nii.gz"
    mask_path = mask_npz_path(exp_root, folder, case_id, TP_RAW_MASK)

    for p in (pre_path, raw_path, sub_path, mask_path):
        if not p.exists():
            print(f"  [WARN] missing for {folder}: {p}")
            return None

    # Array shape is (y, x, z).
    raw,      x_sp, y_sp, z_sp = load_oriented(raw_path)
    pre,      *_                = load_oriented(pre_path)
    sub_post, *_                = load_oriented(sub_path)
    subtraction = sub_post - pre          # tp2 - tp0

    mask = load_mask(mask_path)
    if mask.shape != raw.shape:
        print(f"  [WARN] {folder}: mask {mask.shape} != image {raw.shape}")
        return None

    # Coronal: fix y (axis 0). Use fixed_y if given, otherwise pick argmax.
    if fixed_y is not None:
        best_y = max(0, min(int(fixed_y), mask.shape[0] - 1))
    else:
        per_y_signal = mask.sum(axis=(1, 2))
        best_y = int(np.argmax(per_y_signal))

    # Coronal panel is (x, z) — rows=x (left-right), cols=z (through-plane).
    raw_slice = normalize_to_255(raw[best_y, :, :])   # shape (x, z)
    sub_slice = normalize_to_255(subtraction[best_y, :, :])
    mask_slice = mask[best_y, :, :] > mask_thresh

    print(f"  {folder}: z-spacing={z_sp:.3f} mm, panel (x,z)={raw_slice.shape}, "
          f"best coronal y={best_y}/{mask.shape[0]}, "
          f"vessel voxels={int(mask_slice.sum())}")
    return {
        "raw": raw_slice, "sub": sub_slice, "mask": mask_slice,
        "x_sp": x_sp, "z_sp": z_sp,
    }


def pil_resize(arr2d: np.ndarray, out_hw: tuple[int, int], smooth: bool) -> np.ndarray:
    out_h, out_w = out_hw
    filters = getattr(Image, "Resampling", Image)
    resample = filters.BILINEAR if smooth else filters.NEAREST
    return np.asarray(Image.fromarray(arr2d).resize((out_w, out_h), resample))


def draw_panel(ax, base_uint8, mask_bool, x_sp, z_sp) -> None:
    """Draw an (x, z) coronal slice at true physical proportions.

    Rows = x (left-right, in-plane), cols = z (through-plane, the resampled axis).
    Version 1's coarser z-grid produces fewer, wider columns -> visibly blocky.
    """
    n_x, n_z = base_uint8.shape
    extent = [0, n_z * z_sp, 0, n_x * x_sp]   # [left, right, bottom, top] mm
    ax.imshow(base_uint8, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=extent, aspect="equal", origin="lower")
    overlay = np.zeros((n_x, n_z, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = np.where(mask_bool, 0.30, 0.0)
    ax.imshow(overlay, interpolation="nearest", extent=extent, aspect="equal",
              origin="lower")
    ax.set_axis_off()


def build_figure(panels, labels, out_path: Path, match_resolution: bool) -> None:
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))

    if match_resolution:
        target_hw = (
            max(p["raw"].shape[0] for p in panels if p),
            max(p["raw"].shape[1] for p in panels if p),
        )
        common_x_mm = max(p["raw"].shape[0] * p["x_sp"] for p in panels if p)
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
        x_sp, z_sp = panel["x_sp"], panel["z_sp"]

        if match_resolution:
            raw  = pil_resize(raw,  target_hw, smooth=True)
            sub  = pil_resize(sub,  target_hw, smooth=True)
            mask = pil_resize(mask.astype(np.uint8), target_hw, smooth=False).astype(bool)
            x_sp = common_x_mm / target_hw[0]
            z_sp = common_z_mm / target_hw[1]

        draw_panel(ax_top, raw,  mask, x_sp, z_sp)
        draw_panel(ax_bot, sub,  mask, x_sp, z_sp)

    fig.text(0.015, 0.72, "raw MRI\n(early post-contrast)", rotation=90,
             va="center", ha="center", fontsize=11)
    fig.text(0.015, 0.28, "subtraction\n(tp2 - tp0)", rotation=90,
             va="center", ha="center", fontsize=11)
    fig.suptitle("Vessel Segmentation Quality vs. Z Spacing — DUKE_001 (Coronal View)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root",
        default=str(Path(os.environ.get("VANGUARD_QC_OUTPUT", str(Path.home() / "vanguard_qc_pngs"))) / "resampling_experiment"))
    parser.add_argument("--case-id", default="DUKE_001")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--fixed-y", type=int, default=None,
                        help="Fix the coronal y index for all versions instead of per-version argmax.")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    labels = [label for _, label in VERSIONS]

    print("Gathering coronal panels for each version:")
    panels = [
        gather_version(exp_root, folder, args.case_id, args.mask_thresh,
                       fixed_y=args.fixed_y)
        for folder, _ in VERSIONS
    ]

    print("\nBuilding figures:")
    build_figure(panels, labels,
                 exp_root / "comparison_coronal_native_resolution.png",
                 match_resolution=False)
    build_figure(panels, labels,
                 exp_root / "comparison_coronal_matched_resolution.png",
                 match_resolution=True)

    print("\nSTEP 3 (coronal) complete.")


if __name__ == "__main__":
    main()
