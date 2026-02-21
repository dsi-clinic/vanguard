#!/usr/bin/env python3
"""Generate all maps (kinetic + subtraction) for one patient and save 3 diagnostic PNGs.

Outputs saved to {outdir}/:
  {pid}_fig1_dce_phases.png       – Raw DCE phases with segmentation contour
  {pid}_fig2_kinetic_maps.png     – Kinetic parameter maps (E_early, E_peak, ...)
  {pid}_fig3_subtraction_maps.png – Subtraction images (wash_in, wash_out)

Usage
-----
    python visualize_patient_maps.py [--pid DUKE_001] [--overwrite]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# Make generate_kinetic_maps importable from the same scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_kinetic_maps import (  # noqa: E402
    generate_maps_for_pid,
    KINETIC_MAP_NAMES,
    SUBTRACTION_MAP_NAMES,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

IMAGES_DIR  = Path("/net/projects2/vanguard/MAMA-MIA-syn60868042/images")
MASKS_DIR   = Path("/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert")
KINETIC_DIR = Path("/home/summe/vanguard/radiomics_baseline/kinetic_maps")
FIGURES_DIR = Path("/home/summe/vanguard/radiomics_baseline/figures")
ZOOM_FACTOR = 2.5
DPI         = 160

_PHASE_RE = re.compile(r"_(\d{4})\.nii(?:\.gz)?$")

# Colormap config: (cmap_name, center_at_zero)
_CMAP: dict[str, tuple[str, bool]] = {
    "E_early":      ("inferno",  False),
    "E_peak":       ("inferno",  False),
    "slope_in":     ("inferno",  False),
    "slope_out":    ("coolwarm", True),
    "AUC":          ("inferno",  False),
    "t_peak_voxel": ("viridis",  False),
    "wash_in":      ("plasma",   False),
    "wash_out":     ("coolwarm", True),
}

# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def read_nifti(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def read_mask(path: Path) -> np.ndarray:
    return (sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0).astype(np.uint8)


def peak_z(mask3d: np.ndarray) -> int:
    areas = (mask3d > 0).sum(axis=(1, 2))
    if areas.max() == 0:
        raise ValueError("Mask is empty.")
    return int(np.argmax(areas))


def bbox2d(mask2d: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask2d > 0)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def expand_bbox(
    ymin: int, ymax: int, xmin: int, xmax: int, factor: float, H: int, W: int
) -> tuple[int, int, int, int]:
    cy, cx = 0.5 * (ymin + ymax), 0.5 * (xmin + xmax)
    h = (ymax - ymin + 1) * factor
    w = (xmax - xmin + 1) * factor
    return (
        int(max(0, np.floor(cy - 0.5 * h))),
        int(min(H - 1, np.ceil(cy + 0.5 * h))),
        int(max(0, np.floor(cx - 0.5 * w))),
        int(min(W - 1, np.ceil(cx + 0.5 * w))),
    )

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def add_map_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    img_crop: np.ndarray,
    mask_crop: np.ndarray,
    name: str,
    cmap: str,
    center_at_zero: bool,
) -> None:
    """Plot one map panel with auto-scaled colormap (inside-mask only) + contour."""
    tumor_vals = img_crop[mask_crop > 0]
    if tumor_vals.size == 0 or not np.any(np.isfinite(tumor_vals)):
        vmin, vmax = None, None
    elif center_at_zero:
        vabs = max(abs(float(np.nanmin(tumor_vals))), abs(float(np.nanmax(tumor_vals))), 1e-6)
        vmin, vmax = -vabs, vabs
    else:
        vmin = float(np.nanmin(tumor_vals))
        vmax = float(np.nanmax(tumor_vals))

    im = ax.imshow(img_crop, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(mask_crop, levels=[0.5], colors="red", linewidths=1.2)
    ax.set_title(name, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid",        default="DUKE_001")
    ap.add_argument("--overwrite",  action="store_true",
                    help="Force regeneration of maps even if they already exist.")
    ap.add_argument("--zoom-factor", type=float, default=ZOOM_FACTOR)
    ap.add_argument("--outdir",     type=Path, default=FIGURES_DIR)
    args = ap.parse_args()

    pid      = args.pid
    outdir   = args.outdir
    zoom     = args.zoom_factor
    pid_dir  = KINETIC_DIR / pid
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate kinetic + subtraction maps
    # ------------------------------------------------------------------
    print(f"[INFO] Generating maps for {pid} ...")
    result = generate_maps_for_pid(
        pid=pid,
        images_dir=str(IMAGES_DIR),
        masks_dir=str(MASKS_DIR),
        mask_pattern="{pid}.nii.gz",
        output_dir=str(KINETIC_DIR),
        generate_tpeak_voxel=True,
        generate_subtraction=True,
        overwrite=args.overwrite,
    )
    if result["status"] == "error":
        print(f"[ERROR] Map generation failed: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Status={result['status']}, maps_generated={result['maps_generated']}")

    # ------------------------------------------------------------------
    # 2. Shared crop geometry
    # ------------------------------------------------------------------
    mask3d   = read_mask(MASKS_DIR / f"{pid}.nii.gz")
    z        = peak_z(mask3d)
    mask2d   = mask3d[z]
    ymin, ymax, xmin, xmax = bbox2d(mask2d)

    first_phase_vol = read_nifti(IMAGES_DIR / pid / f"{pid}_0000.nii.gz")
    H, W = first_phase_vol.shape[1], first_phase_vol.shape[2]
    eymin, eymax, exmin, exmax = expand_bbox(ymin, ymax, xmin, xmax, zoom, H, W)

    def crop(vol3d: np.ndarray) -> np.ndarray:
        return vol3d[z, eymin : eymax + 1, exmin : exmax + 1]

    mask_crop = crop(mask3d.astype(np.float32))

    # ------------------------------------------------------------------
    # 3. Figure 1 — DCE phases with segmentation
    # ------------------------------------------------------------------
    phase_paths = sorted(
        [p for p in (IMAGES_DIR / pid).glob(f"{pid}_????.nii.gz")
         if _PHASE_RE.search(p.name)],
        key=lambda p: int(_PHASE_RE.search(p.name).group(1)),  # type: ignore[union-attr]
    )
    phase_crops = [
        (int(_PHASE_RE.search(p.name).group(1)), crop(read_nifti(p)))  # type: ignore[union-attr]
        for p in phase_paths
    ]

    gmin = float(min(np.nanmin(c) for _, c in phase_crops))
    gmax = float(max(np.nanmax(c) for _, c in phase_crops))

    n = len(phase_crops)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), dpi=DPI, squeeze=False)
    for ax, (t, crop_arr) in zip(axes[0], phase_crops):
        ax.imshow(crop_arr, cmap="gray", vmin=gmin, vmax=gmax)
        ax.contour(mask_crop, levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title(f"t={t:04d}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{pid}  —  DCE phases  (z={z}, zoom={zoom}×)", fontsize=11, y=1.01)
    fig.tight_layout()
    out1 = outdir / f"{pid}_fig1_dce_phases.png"
    fig.savefig(out1, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"[INFO] Saved: {out1}")

    # ------------------------------------------------------------------
    # 4. Figure 2 — Kinetic maps
    # ------------------------------------------------------------------
    kmap_entries: list[tuple[str, np.ndarray]] = []
    for name in KINETIC_MAP_NAMES:
        p = pid_dir / f"{pid}_kinetic_{name}.nii.gz"
        if p.exists():
            kmap_entries.append((name, crop(read_nifti(p))))
    tp = pid_dir / f"{pid}_kinetic_t_peak_voxel.nii.gz"
    if tp.exists():
        kmap_entries.append(("t_peak_voxel", crop(read_nifti(tp))))

    n = len(kmap_entries)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), dpi=DPI, squeeze=False)
    for ax, (name, kimg) in zip(axes[0], kmap_entries):
        cmap, center = _CMAP.get(name, ("inferno", False))
        add_map_panel(fig, ax, kimg, mask_crop, name, cmap, center)
    fig.suptitle(f"{pid}  —  Kinetic maps  (z={z})", fontsize=11, y=1.01)
    fig.tight_layout()
    out2 = outdir / f"{pid}_fig2_kinetic_maps.png"
    fig.savefig(out2, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"[INFO] Saved: {out2}")

    # ------------------------------------------------------------------
    # 5. Figure 3 — Subtraction images
    # ------------------------------------------------------------------
    sub_entries: list[tuple[str, np.ndarray]] = []
    for name in SUBTRACTION_MAP_NAMES:
        p = pid_dir / f"{pid}_subtraction_{name}.nii.gz"
        if p.exists():
            sub_entries.append((name, crop(read_nifti(p))))

    if not sub_entries:
        print("[WARN] No subtraction maps found; skipping figure 3.", file=sys.stderr)
    else:
        n = len(sub_entries)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), dpi=DPI, squeeze=False)
        for ax, (name, simg) in zip(axes[0], sub_entries):
            cmap, center = _CMAP.get(name, ("coolwarm", True))
            add_map_panel(fig, ax, simg, mask_crop, name, cmap, center)
        fig.suptitle(f"{pid}  —  Subtraction images  (z={z})", fontsize=11, y=1.01)
        fig.tight_layout()
        out3 = outdir / f"{pid}_fig3_subtraction_maps.png"
        fig.savefig(out3, bbox_inches="tight", dpi=DPI)
        plt.close(fig)
        print(f"[INFO] Saved: {out3}")


if __name__ == "__main__":
    main()
