"""Module for comparing ground truth and deep learning vessel segmentations."""

from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
from skimage.morphology import skeletonize
from skimage.transform import resize

NDIM_THRESHOLD: int = 4
PROB_THRESHOLD: float = 0.2
P_ID: str = "041"

BASE_DIR: Path = Path("/net/projects2/vanguard")
GT_PATH: Path = (
    BASE_DIR / f"gt_masks/Segmentation_Masks_NRRD/Breast_MRI_{P_ID}/"
    f"Segmentation_Breast_MRI_{P_ID}_Dense_and_Vessels.seg.nrrd"
)
DL_PATH: Path = (
    BASE_DIR / f"vessel_segmentations/DUKE/DUKE_{P_ID}/"
    f"images/DUKE_{P_ID}_0000_vessel_segmentation.npz"
)


def compare_vessels(gt_p: str | Path, dl_p: str | Path) -> None:
    """Compare ground truth and deep learning vessel segmentations.

    This function reads a 3D NRRD ground truth mask and an NPZ AI segmentation,
    normalizes their orientations, computes the skeleton of the GT, and saves
    a comparative visualization.

    Args:
        gt_p: Path to the ground truth NRRD file.
        dl_p: Path to the deep learning NPZ file.
    """
    try:
        gt_data, _ = nrrd.read(str(gt_p))
    except FileNotFoundError:
        print(f"Error: Could not find ground truth file at {gt_p}")
        return

    gt_3d: np.ndarray = (
        gt_data[0, :, :, :] if gt_data.ndim == NDIM_THRESHOLD else gt_data
    )
    v_gt: np.ndarray = (gt_3d == 1).astype(np.float32)

    with np.load(str(dl_p)) as data:
        v_dl_prob: np.ndarray = data["vessel"]

    v_dl: np.ndarray = v_dl_prob.transpose(1, 0, 2)
    v_dl = (v_dl > PROB_THRESHOLD).astype(np.float32)
    v_dl = resize(
        v_dl,
        v_gt.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    v_dl = np.flipud(v_dl)

    v_gt_centerline: np.ndarray = skeletonize(v_gt.astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.max(v_gt, axis=2), cmap="Blues")
    axes[0].set_title(f"Radiologist GT (Patient {P_ID})")

    axes[1].imshow(np.max(v_dl, axis=2), cmap="Reds")
    axes[1].set_title("AI Fixed Orientation")

    axes[2].imshow(np.max(v_dl, axis=2), cmap="gray", alpha=0.3)
    axes[2].imshow(np.max(v_gt_centerline, axis=2), cmap="hot")
    axes[2].set_title("Overlay: AI vs GT Centerline")

    output_fn: str = f"vessel_output_{P_ID}.png"
    plt.savefig(output_fn)
    plt.close(fig)
    print(f"Output saved to {output_fn}")


if __name__ == "__main__":
    if GT_PATH.exists() and DL_PATH.exists():
        compare_vessels(GT_PATH, DL_PATH)
    else:
        print(f"Error: Files not found at {GT_PATH} or {DL_PATH}")
