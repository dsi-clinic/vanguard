"""Physical-grid operations used by Vanguard DCE preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from preprocessing.dicom import DicomGeometry

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])
HALF_VOXEL = 0.5


def sitk_image(array_zyx: np.ndarray, geometry: DicomGeometry) -> sitk.Image:
    """Attach exact DICOM LPS geometry to a ZYX array."""
    array = np.asarray(array_zyx)
    if tuple(array.shape) != geometry.shape_zyx:
        raise ValueError(f"array shape {array.shape} != geometry {geometry.shape_zyx}")
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(geometry.spacing_xyz_mm)
    image.SetOrigin(geometry.origin_lps_mm)
    image.SetDirection(geometry.direction_lps)
    return image


def isotropic_geometry(
    geometry: DicomGeometry,
    *,
    spacing_mm: float = 1.0,
) -> DicomGeometry:
    """Create a source-aligned isotropic grid without losing LPS coordinates."""
    if not spacing_mm > 0:
        raise ValueError("spacing_mm must be positive")
    native_size_xyz = np.asarray(geometry.shape_zyx[::-1], dtype=np.float64)
    native_spacing = np.asarray(geometry.spacing_xyz_mm, dtype=np.float64)
    size_xyz = np.rint(native_size_xyz * native_spacing / spacing_mm).astype(int)
    return DicomGeometry(
        series_instance_uid=geometry.series_instance_uid,
        frame_of_reference_uid=geometry.frame_of_reference_uid,
        shape_zyx=tuple(int(value) for value in size_xyz[::-1]),
        spacing_xyz_mm=(spacing_mm, spacing_mm, spacing_mm),
        origin_lps_mm=geometry.origin_lps_mm,
        direction_lps=geometry.direction_lps,
        slice_thickness_mm=spacing_mm,
    )


def resample_to_geometry(
    array_zyx: np.ndarray,
    source: DicomGeometry,
    target: DicomGeometry,
    *,
    nearest: bool = False,
) -> np.ndarray:
    """Resample between grids in one physical frame, honoring both directions."""
    moving = sitk_image(np.asarray(array_zyx, dtype=np.float32), source)
    reference = sitk_image(np.zeros(target.shape_zyx, dtype=np.float32), target)
    interpolator = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    output = sitk.Resample(
        moving,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        0.0,
        sitk.sitkFloat32,
    )
    return np.asarray(sitk.GetArrayFromImage(output), dtype=np.float32)


def nifti_affine_ras(geometry: DicomGeometry) -> np.ndarray:
    """Build a NIfTI XYZ-index-to-RAS affine from DICOM LPS geometry."""
    direction = np.asarray(geometry.direction_lps, dtype=np.float64).reshape(3, 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = LPS_TO_RAS @ direction @ np.diag(geometry.spacing_xyz_mm)
    affine[:3, 3] = LPS_TO_RAS @ np.asarray(geometry.origin_lps_mm)
    return affine


def save_nifti_xyz(
    path: str | Path, array_xyz: np.ndarray, geometry: DicomGeometry
) -> None:
    """Save an XYZ array with its true RAS affine."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        np.ascontiguousarray(array_xyz, dtype=np.float32),
        nifti_affine_ras(geometry),
    )
    image.set_qform(image.affine, code=1)
    image.set_sform(image.affine, code=1)
    nib.save(image, str(target))


def physical_points_from_zyx(
    indices_zyx: np.ndarray,
    geometry: DicomGeometry,
) -> np.ndarray:
    """Convert ZYX voxel centers to physical LPS points."""
    indices_xyz = np.asarray(indices_zyx, dtype=np.float64)[:, ::-1]
    direction = np.asarray(geometry.direction_lps).reshape(3, 3)
    return (
        np.asarray(geometry.origin_lps_mm)
        + (direction @ (indices_xyz * np.asarray(geometry.spacing_xyz_mm)).T).T
    )


def continuous_xyz_from_physical(
    points_lps: np.ndarray,
    geometry: DicomGeometry,
) -> np.ndarray:
    """Convert physical LPS points to continuous XYZ indices."""
    direction = np.asarray(geometry.direction_lps).reshape(3, 3)
    return (
        direction.T @ (np.asarray(points_lps) - np.asarray(geometry.origin_lps_mm)).T
    ).T / np.asarray(geometry.spacing_xyz_mm)


def rasterize_skeleton_identity(
    skeleton_zyx: np.ndarray,
    source: DicomGeometry,
    target: DicomGeometry,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize source skeleton centers onto a target grid in the same frame."""
    indices = np.argwhere(np.asarray(skeleton_zyx, dtype=bool))
    points = physical_points_from_zyx(indices, source)
    continuous_xyz = continuous_xyz_from_physical(points, target)
    size_xyz = np.asarray(target.shape_zyx[::-1])
    inside = np.all(continuous_xyz >= -HALF_VOXEL, axis=1) & np.all(
        continuous_xyz < size_xyz - HALF_VOXEL, axis=1
    )
    rounded = np.rint(continuous_xyz[inside]).astype(int)
    rounded = np.clip(rounded, 0, size_xyz - 1)
    mapped = np.zeros(target.shape_zyx, dtype=np.uint8)
    mapped[rounded[:, 2], rounded[:, 1], rounded[:, 0]] = 1
    metrics = {
        "source_skeleton_points": int(indices.shape[0]),
        "points_inside_target": int(inside.sum()),
        "fraction_inside_target": float(inside.mean()) if inside.size else 0.0,
        "unique_target_voxels": int(mapped.sum()),
        "rasterization_collisions": int(inside.sum() - mapped.sum()),
    }
    return mapped, metrics
