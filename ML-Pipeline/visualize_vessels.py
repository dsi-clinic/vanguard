"""Module for comparing ground truth and deep learning vessel segmentations."""

from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
from skimage.morphology import skeletonize
from skimage.transform import resize

NDIM_THRESHOLD = 4
PROB_THRESHOLD = 0.2
P_ID = "041"

GT_PATH = Path(
    f"/net/projects2/vanguard/gt_masks/Segmentation_Masks_NRRD/Breast_MRI_{P_ID}/"
    f"Segmentation_Breast_MRI_{P_ID}_Dense_and_Vessels.seg.nrrd"
)
DL_PATH = Path(
    f"/net/projects2/vanguard/vessel_segmentations/DUKE/DUKE_{P_ID}/"
    f"images/DUKE_{P_ID}_0000_vessel_segmentation.npz"
)


def compare_vessels(gt_p: str | Path, dl_p: str | Path) -> None:
    """Compare ground truth and deep learning vessel segmentations.

    Args:
        gt_p: Path to ground truth segmentation.
        dl_p: Path to deep learning segmentation.
    """
    gt_data, _ = nrrd.read(str(gt_p))
    print(f"Original GT Shape: {gt_data.shape}")

    gt_3d = gt_data[0, :, :, :] if gt_data.ndim == NDIM_THRESHOLD else gt_data
    v_gt = (gt_3d == 1).astype(np.float32)

    with np.load(str(dl_p)) as data:
        v_dl_prob = data["vessel"]

    v_dl = v_dl_prob.transpose(1, 0, 2)
    v_dl = (v_dl > PROB_THRESHOLD).astype(np.float32)
    v_dl = resize(v_dl, v_gt.shape, order=0, preserve_range=True, anti_aliasing=False)
    v_dl = np.flipud(v_dl)

    v_gt_centerline = skeletonize(v_gt.astype(np.uint8))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(np.max(v_gt, axis=2), cmap="Blues")
    plt.title(f"Radiologist GT (Patient {P_ID})")

    plt.subplot(1, 3, 2)
    plt.imshow(np.max(v_dl, axis=2), cmap="Reds")
    plt.title("AI Fixed Orientation")

    plt.subplot(1, 3, 3)
    plt.imshow(np.max(v_dl, axis=2), cmap="gray", alpha=0.3)
    plt.imshow(np.max(v_gt_centerline, axis=2), cmap="hot")
    plt.title("Overlay: AI vs GT Centerline")

    output_fn = f"vessel_output_{P_ID}.png"
    plt.savefig(output_fn)
    print(f"Output saved to {output_fn}")


if __name__ == "__main__":
    if GT_PATH.exists() and DL_PATH.exists():
        compare_vessels(GT_PATH, DL_PATH)
    else:
        print("Error: Files not found.")
