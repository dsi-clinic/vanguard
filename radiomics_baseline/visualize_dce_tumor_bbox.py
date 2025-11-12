#!/usr/bin/env python3
"""
Visualize DCE-MRI phases for one patient with a tumor bounding box.

- loads N DCE phases (0001, 0002, …)
- tries to get a tight tumor box from the segmentation mask
- falls back to JSON breast_coordinates if mask is empty/missing
- normalizes all phases with the same min/max
- draws the box on every phase
- saves to radiomics_baseline/figures by default

Example Code:
python radiomics_baseline/visualize_dce_tumor_bbox.py \
  --pid ISPY2_491779 \
  --images-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks-dir  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --json-dir   /net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files \
  --image-patterns "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz,{pid}/{pid}_0003.nii.gz,{pid}/{pid}_0004.nii.gz"


"""

import argparse, json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def load_sitk(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return sitk.ReadImage(str(path))


def mask_bbox_from_seg(mask_img: sitk.Image):
    """Get tight bbox from nonzero mask voxels. Returns (zmin,zmax,ymin,ymax,xmin,xmax) or None."""
    arr = sitk.GetArrayFromImage(mask_img)  # (Z,Y,X)
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return None
    z_min, z_max = int(nz[0].min()), int(nz[0].max())
    y_min, y_max = int(nz[1].min()), int(nz[1].max())
    x_min, x_max = int(nz[2].min()), int(nz[2].max())
    return z_min, z_max, y_min, y_max, x_min, x_max


def bbox_from_json(json_path: Path):
    """Fallback bbox from patient json (1-based → 0-based)."""
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    coords = data.get("primary_lesion", {}).get("breast_coordinates")
    if not coords:
        return None
    x_min = coords["x_min"] - 1
    x_max = coords["x_max"] - 1
    y_min = coords["y_min"] - 1
    y_max = coords["y_max"] - 1
    z_min = coords["z_min"] - 1
    z_max = coords["z_max"] - 1
    return z_min, z_max, y_min, y_max, x_min, x_max


def load_slice(img: sitk.Image, z: int) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    z = max(0, min(z, arr.shape[0] - 1))
    return arr[z, :, :]


def normalize_many(slices):
    stacked = np.stack(slices, axis=0)
    vmin = np.percentile(stacked, 1)
    vmax = np.percentile(stacked, 99)
    out = []
    for s in slices:
        ss = np.clip((s - vmin) / (vmax - vmin + 1e-8), 0, 1)
        out.append(ss)
    return out


def main():
    # default outdir = radiomics_baseline/figures (folder of this script + "figures")
    default_outdir = Path(__file__).resolve().parent / "figures"

    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--json-dir", required=True)
    ap.add_argument(
        "--image-patterns",
        required=True,
        help='comma-separated patterns, e.g. "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz"',
    )
    ap.add_argument("--outdir", default=str(default_outdir))
    args = ap.parse_args()

    pid = args.pid
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    patterns = [p.strip() for p in args.image_patterns.split(",") if p.strip()]

    mask_path = Path(args.masks_dir) / f"{pid}.nii.gz"
    mask_img = load_sitk(mask_path)
    tumor_bbox = mask_bbox_from_seg(mask_img)

    if tumor_bbox is None:
        json_path = Path(args.json_dir) / f"{pid}.json"
        tumor_bbox = bbox_from_json(json_path)

    if tumor_bbox is None:
        raise RuntimeError("Could not determine tumor/breast bounding box from mask or JSON")

    z_min, z_max, y_min, y_max, x_min, x_max = tumor_bbox
    z_mid = (z_min + z_max) // 2

    slice_imgs = []
    titles = []
    for pat in patterns:
        img_path = Path(args.images_dir) / pat.format(pid=pid)
        img = load_sitk(img_path)
        sl = load_slice(img, z_mid)
        slice_imgs.append(sl)
        titles.append(img_path.name)

    slice_norm = normalize_many(slice_imgs)

    n = len(slice_norm)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, img2d, title in zip(axes, slice_norm, titles):
        ax.imshow(img2d, cmap="gray", vmin=0, vmax=1)

        rect_x = x_min
        rect_y = y_min
        rect_w = x_max - x_min + 1
        rect_h = y_max - y_min + 1
        ax.add_patch(
            plt.Rectangle(
                (rect_x, rect_y),
                rect_w,
                rect_h,
                edgecolor="red",
                fill=False,
                linewidth=2,
            )
        )

        ax.set_title(f"{title}\nz={z_mid}")
        ax.axis("off")

    fig.suptitle(f"{pid} – DCE phases with tumor bbox", y=0.98)
    fig.tight_layout()

    out_path = outdir / f"{pid}_dce_tumor_bbox.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    main()
