#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize DCE-MRI phases (0000..0005) with tumor mask and a zoomed-out crop.

- Loads phases {pid}/{pid}_0000.nii.gz ... {pid}/{pid}_0005.nii.gz
- Uses one global [vmin, vmax] across phases to preserve enhancement
- Selects the tumor slice with max mask area (consistent z across phases)
- Crops to an expanded "breast" box around the tumor (configurable)
- Overlays tumor mask contour in red
- Saves to radiomics_baseline/figures/{pid}_dce_tumor_zoom.png

Example Usage:

python radiomics_baseline/visualize_dce_tumor_bbox.py \
  --pid DUKE_001 \
  --images-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks-dir  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --zoom-factor 2.0 \
  --outdir radiomics_baseline/figures

"""

import argparse, sys, re
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def read_img(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    return arr.astype(np.float32)

def read_mask(path: Path) -> np.ndarray:
    m = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(m) > 0
    return a.astype(np.uint8)  # (z, y, x) in {0,1}

def tumor_z_with_max_area(mask: np.ndarray) -> int:
    areas = (mask > 0).sum(axis=(1, 2))
    if areas.max() == 0:
        raise ValueError("Mask appears empty—no positive voxels found.")
    return int(np.argmax(areas))

def bbox_2d_from_mask(mask2d: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0:
        raise ValueError("2D mask slice is empty; cannot compute bbox.")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

def expand_bbox(ymin, ymax, xmin, xmax, expand: float, H: int, W: int):
    cy = 0.5 * (ymin + ymax)
    cx = 0.5 * (xmin + xmax)
    h = (ymax - ymin + 1) * expand
    w = (xmax - xmin + 1) * expand
    nymin = int(max(0, np.floor(cy - 0.5 * h)))
    nymax = int(min(H - 1, np.ceil(cy + 0.5 * h)))
    nxmin = int(max(0, np.floor(cx - 0.5 * w)))
    nxmax = int(min(W - 1, np.ceil(cx + 0.5 * w)))
    return nymin, nymax, nxmin, nxmax

def build_phase_paths(images_dir: Path, pid: str) -> list[Path]:
    paths = []
    for p in range(6):  # exactly 0000..0005
        candidate = images_dir / f"{pid}" / f"{pid}_{p:04d}.nii.gz"
        if candidate.exists():
            paths.append(candidate)
        else:
            print(f"[WARN] missing phase {p:04d}: {candidate}", file=sys.stderr)
    if not paths:
        raise FileNotFoundError("No phases (0000..0005) found for this PID.")
    return paths

_phase_re = re.compile(r"_(\d{4})\.nii(?:\.gz)?$")  # robust for .nii or .nii.gz
def parse_phase_number(path: Path) -> int | None:
    m = _phase_re.search(path.name)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--masks-dir",  required=True, type=Path)
    ap.add_argument("--pid",        required=True, type=str)
    ap.add_argument("--zoom-factor", type=float, default=2.0,
                    help=">1.0 zooms OUT (more context). 2.0 is a good start.")
    ap.add_argument("--outdir", type=Path, default=Path("radiomics_baseline/figures"))
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- load mask & choose slice ----
    mask_path = args.masks_dir / f"{args.pid}.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask3d = read_mask(mask_path)
    z = tumor_z_with_max_area(mask3d)
    mask2d = mask3d[z]

    # tumor bbox on this slice, then expand to show more context
    ymin, ymax, xmin, xmax = bbox_2d_from_mask(mask2d)

    # derive H, W from first phase
    phase_paths = build_phase_paths(args.images_dir, args.pid)
    first_img = read_img(phase_paths[0])
    H, W = first_img.shape[1:]
    eymin, eymax, exmin, exmax = expand_bbox(ymin, ymax, xmin, xmax,
                                             expand=args.zoom_factor, H=H, W=W)

    # ---- load available phases, crop consistently, compute global min/max ----
    crops = []
    for pth in phase_paths:
        vol = read_img(pth)
        if z < 0 or z >= vol.shape[0]:
            raise IndexError(f"Chosen z={z} outside volume depth for {pth.name}.")
        crops.append(vol[z, eymin:eymax+1, exmin:exmax+1])

    gmin = float(min(np.min(c) for c in crops))
    gmax = float(max(np.max(c) for c in crops))
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        gmin = gmax = None  # fallback

    # ---- plot ----
    n = len(crops)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), dpi=160)
    if n == 1:
        axes = [axes]

    mask_crop = mask2d[eymin:eymax+1, exmin:exmax+1]

    for ax, pth, img in zip(axes, phase_paths, crops):
        ax.imshow(img, cmap="gray", vmin=gmin, vmax=gmax)
        ax.contour(mask_crop, levels=[0.5], colors="red", linewidths=1.5)
        phase = parse_phase_number(pth)
        ax.set_title(f"t={phase}" if phase is not None else pth.name)
        ax.axis("off")

    fig.suptitle(f"{args.pid} — DCE phases with tumor mask (zoom={args.zoom_factor}×)", y=0.98)
    outpath = outdir / f"{args.pid}_dce_tumor_zoom.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved: {outpath}")

if __name__ == "__main__":
    main()
