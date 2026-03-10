"""Module for computing Perivascular Density (PVD) features from vessel segmentations."""

from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np
from scipy import ndimage

DIMENSION_THRESHOLD = 3
DEFAULT_SPACING = [1, 1, 1]


def get_pvd_feature(
    cl_path: str | Path,
    tumor_mask_path: str | Path,
    mm_dilation: int = 10,
) -> float:
    """Compute PVD by counting vessel voxels within a dilated shell around a tumor.

    Args:
        cl_path: Path to the .npz vessel segmentation file.
        tumor_mask_path: Path to the .nii.gz or .nrrd tumor mask.
        mm_dilation: Dilation distance in mm.

    Returns:
        The count of vessel voxels within the peritumoral shell, or NaN on error.
    """
    path_cl = Path(cl_path)
    path_mask = Path(tumor_mask_path)

    try:
        cl_file = np.load(path_cl)
        cl_data = cl_file[cl_file.files[0]]
    except (FileNotFoundError, KeyError, OSError) as e:
        print(f"Error reading centerline {path_cl.name}: {e}")
        return np.nan

    try:
        if path_mask.suffix in [".nii", ".gz"]:
            img = nib.load(path_mask)
            tumor_data = img.get_fdata()
            spacing = img.header.get_zooms()[:3]
        else:
            tumor_data, header = nrrd.read(path_mask)
            spacing = (
                [np.linalg.norm(v) for v in header["space directions"] if v is not None]
                if "space directions" in header
                and header["space directions"] is not None
                else header.get("spacings", DEFAULT_SPACING)
            )
    except Exception as e:
        print(f"Error reading mask {path_mask.name}: {e}")
        return np.nan

    tumor_data = np.squeeze(tumor_data)
    if tumor_data.ndim > DIMENSION_THRESHOLD:
        tumor_data = tumor_data[..., 0]

    cl_binary = (cl_data > 0).astype(np.uint8)
    tumor_binary = (tumor_data > 0).astype(np.uint8)

    if cl_binary.shape != tumor_binary.shape:
        raise ValueError(
            f"Shape mismatch: Centerline {cl_binary.shape} vs Mask {tumor_binary.shape}"
        )

    its = int(mm_dilation / spacing[0])
    dilated_tumor = ndimage.binary_dilation(tumor_binary, iterations=its)
    shell_mask = np.logical_and(dilated_tumor, np.logical_not(tumor_binary))

    pvd_count = float(np.sum(np.logical_and(cl_binary, shell_mask)))

    return pvd_count
