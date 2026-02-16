#!/usr/bin/env python3
"""Visualize DCE phases and kinetic parameter maps for a patient.

Produces a 2-row figure:
  - Top row:    Raw DCE phases (grayscale, shared intensity scale, mask contour)
  - Bottom row: Kinetic parameter maps (per-map colormap, mask contour)

Both rows are cropped to an expanded tumor bounding box on the max-area z-slice.

Usage
-----
    python visualize_kinetic_maps.py \
        --images-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
        --masks-dir  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
        --pid DUKE_001 \
        --outdir radiomics_baseline/figures
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Helpers (same pattern as visualize_dce_tumor_bbox.py)
# ---------------------------------------------------------------------------

_PHASE_RE = re.compile(r"_(\d{4})\.nii(?:\.gz)?$")

KINETIC_MAP_NAMES = ["E_early", "E_peak", "slope_in", "slope_out", "AUC"]

# Colormaps: signed maps get diverging, unsigned get sequential
_CMAP = {
    "E_early": "inferno",
    "E_peak": "inferno",
    "slope_in": "inferno",
    "slope_out": "coolwarm",
    "AUC": "inferno",
    "t_peak_voxel": "viridis",
}


def read_img(path: Path) -> np.ndarray:
    """Read a 3D NIfTI image and return it as a float32 NumPy array (z, y, x)."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def read_mask(path: Path) -> np.ndarray:
    """Read a 3D mask image and return a uint8 array (z, y, x) in {0, 1}."""
    m = sitk.ReadImage(str(path))
    return (sitk.GetArrayFromImage(m) > 0).astype(np.uint8)


def tumor_z_with_max_area(mask: np.ndarray) -> int:
    """Return the z-index of the slice with the largest nonzero mask area."""
    areas = (mask > 0).sum(axis=(1, 2))
    if areas.max() == 0:
        raise ValueError("Mask appears empty.")
    return int(np.argmax(areas))


def bbox_2d_from_mask(mask2d: np.ndarray) -> tuple[int, int, int, int]:
    """Compute a tight 2D bounding box (ymin, ymax, xmin, xmax)."""
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0:
        raise ValueError("2D mask slice is empty.")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def expand_bbox(
    ymin: int, ymax: int, xmin: int, xmax: int,
    expand: float, H: int, W: int,
) -> tuple[int, int, int, int]:
    """Expand a 2D bbox by a factor, clamped to image bounds."""
    cy, cx = 0.5 * (ymin + ymax), 0.5 * (xmin + xmax)
    h, w = (ymax - ymin + 1) * expand, (xmax - xmin + 1) * expand
    return (
        int(max(0, np.floor(cy - 0.5 * h))),
        int(min(H - 1, np.ceil(cy + 0.5 * h))),
        int(max(0, np.floor(cx - 0.5 * w))),
        int(min(W - 1, np.ceil(cx + 0.5 * w))),
    )


def build_phase_paths(images_dir: Path, pid: str) -> list[Path]:
    """Collect existing raw phase paths for a patient, sorted by phase index."""
    patient_dir = images_dir / pid
    paths: list[tuple[int, Path]] = []
    for p in sorted(patient_dir.glob(f"{pid}_????.nii.gz")):
        m = _PHASE_RE.search(p.name)
        if m:
            paths.append((int(m.group(1)), p))
    paths.sort(key=lambda x: x[0])
    return [p for _, p in paths]


def build_kinetic_map_paths(images_dir: Path, pid: str) -> list[tuple[str, Path]]:
    """Return (name, path) for each kinetic map that exists."""
    patient_dir = images_dir / pid
    result: list[tuple[str, Path]] = []
    for name in KINETIC_MAP_NAMES:
        path = patient_dir / f"{pid}_kinetic_{name}.nii.gz"
        if path.exists():
            result.append((name, path))
    # Also check optional t_peak_voxel
    tp = patient_dir / f"{pid}_kinetic_t_peak_voxel.nii.gz"
    if tp.exists():
        result.append(("t_peak_voxel", tp))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize DCE phases + kinetic maps for one patient.",
    )
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--masks-dir", required=True, type=Path)
    ap.add_argument("--pid", required=True, type=str)
    ap.add_argument("--zoom-factor", type=float, default=2.0)
    ap.add_argument("--outdir", type=Path, default=Path("radiomics_baseline/figures"))
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    pid = args.pid

    # Load mask and pick z-slice
    mask_path = args.masks_dir / f"{pid}.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask3d = read_mask(mask_path)
    z = tumor_z_with_max_area(mask3d)
    mask2d = mask3d[z]

    # Bounding box
    ymin, ymax, xmin, xmax = bbox_2d_from_mask(mask2d)

    # Phase paths
    phase_paths = build_phase_paths(args.images_dir, pid)
    if not phase_paths:
        raise FileNotFoundError(f"No phase files for {pid}")
    first_img = read_img(phase_paths[0])
    H, W = first_img.shape[1:]
    eymin, eymax, exmin, exmax = expand_bbox(
        ymin, ymax, xmin, xmax, args.zoom_factor, H, W,
    )

    # Crop function
    def crop(vol: np.ndarray) -> np.ndarray:
        return vol[z, eymin : eymax + 1, exmin : exmax + 1]

    mask_crop = crop(mask3d.astype(np.float32))

    # Load phase crops
    phase_crops = []
    for pth in phase_paths:
        phase_crops.append(crop(read_img(pth)))

    # Load kinetic map crops
    kmap_entries = build_kinetic_map_paths(args.images_dir, pid)
    if not kmap_entries:
        print(f"[WARN] No kinetic maps found for {pid}. Run generate_kinetic_maps.py first.",
              file=sys.stderr)

    kmap_crops: list[tuple[str, np.ndarray]] = []
    for name, pth in kmap_entries:
        kmap_crops.append((name, crop(read_img(pth))))

    # Figure layout
    n_phases = len(phase_crops)
    n_maps = len(kmap_crops)
    n_cols = max(n_phases, n_maps)
    n_rows = 2 if n_maps > 0 else 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=160,
        squeeze=False,
    )

    # Global intensity range for raw phases
    gmin = float(min(np.min(c) for c in phase_crops))
    gmax = float(max(np.max(c) for c in phase_crops))
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        gmin, gmax = None, None

    # Row 1: raw DCE phases
    for i, (pth, img_crop) in enumerate(zip(phase_paths, phase_crops)):
        ax = axes[0, i]
        ax.imshow(img_crop, cmap="gray", vmin=gmin, vmax=gmax)
        ax.contour(mask_crop, levels=[0.5], colors="red", linewidths=1.5)
        m = _PHASE_RE.search(pth.name)
        label = f"t={int(m.group(1))}" if m else pth.stem
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    # Hide unused columns in row 1
    for i in range(n_phases, n_cols):
        axes[0, i].axis("off")

    # Row 2: kinetic maps
    if n_rows == 2:
        for i, (name, kimg) in enumerate(kmap_crops):
            ax = axes[1, i]
            cmap = _CMAP.get(name, "inferno")
            # For slope_out, center colormap at 0
            if name == "slope_out":
                vabs = max(abs(np.nanmin(kimg)), abs(np.nanmax(kimg)), 1e-6)
                im = ax.imshow(kimg, cmap=cmap, vmin=-vabs, vmax=vabs)
            else:
                im = ax.imshow(kimg, cmap=cmap)
            ax.contour(mask_crop, levels=[0.5], colors="red", linewidths=1.5)
            ax.set_title(name, fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused columns in row 2
        for i in range(n_maps, n_cols):
            axes[1, i].axis("off")

    fig.suptitle(
        f"{pid} — DCE phases & kinetic maps (z={z})",
        y=0.98, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = outdir / f"{pid}_kinetic_maps.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {outpath}")


if __name__ == "__main__":
    main()
