"""Definitions and extraction helpers for the tumor-size feature block."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from features._common import safe_float

BLOCK_NAME = "tumor_size"
METADATA_COLUMNS = {
    "tumor_mask_exists",
    "tumor_mask_loaded",
}


def matches_column(column: str) -> bool:
    """Return whether a column belongs to the tumor-size block."""
    return column.startswith("tumor_size_") or column in METADATA_COLUMNS


def resolve_tumor_mask_path(
    *,
    case_id: str,
    dataset_name: str,
    tumor_mask_root: Path,
    tumor_mask_pattern: str,
) -> Path | None:
    """Resolve tumor-mask path using common filename/layout fallbacks."""
    patterns = [
        tumor_mask_pattern,
        "{case_id}.nii.gz",
        "{case_id}.nii",
        "{case_id}.nrrd",
        "{case_id}.npy",
    ]
    roots = [tumor_mask_root, tumor_mask_root / dataset_name]
    seen: set[Path] = set()

    for pattern in patterns:
        for root in roots:
            try:
                filename = pattern.format(case_id=case_id, dataset=dataset_name)
            except Exception:  # noqa: BLE001
                continue
            candidate = root / filename
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate

    return None


def load_tumor_mask_zyx(
    tumor_mask_path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    threshold: float,
) -> np.ndarray:
    """Load tumor segmentation as a 3D bool mask aligned to zyx."""
    path_lower = str(tumor_mask_path).lower()

    if path_lower.endswith(".npy"):
        arr = np.load(tumor_mask_path)
        if arr.ndim < 3:
            raise ValueError(
                f"Unsupported tumor npy ndim={arr.ndim}: {tumor_mask_path}"
            )
        if arr.ndim > 3:
            lead_axes = tuple(range(arr.ndim - 3))
            arr = np.any(arr > threshold, axis=lead_axes).astype(np.uint8)
    else:
        if not (
            path_lower.endswith(".nii")
            or path_lower.endswith(".nii.gz")
            or path_lower.endswith(".nrrd")
        ):
            raise ValueError(f"Unsupported tumor mask format: {tumor_mask_path}")
        import SimpleITK as sitk

        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(tumor_mask_path)))

    candidates = [
        np.asarray(arr),
        np.transpose(np.asarray(arr), (1, 2, 0)),
        np.transpose(np.asarray(arr), (2, 1, 0)),
    ]
    selected = next(
        (c for c in candidates if tuple(int(v) for v in c.shape) == expected_shape_zyx),
        None,
    )
    if selected is None:
        raise ValueError(
            "Tumor mask shape mismatch: "
            f"expected {expected_shape_zyx}, got {np.asarray(arr).shape} "
            f"for {tumor_mask_path}"
        )

    mask = np.asarray(selected > threshold, dtype=bool)
    if not np.any(mask):
        raise ValueError(f"Tumor mask is empty after thresholding: {tumor_mask_path}")
    return mask


def parse_tumor_radii(values: Any) -> list[int]:
    """Parse and sanitize tumor-radius list from config."""
    if values is None:
        return [0, 2, 4, 8]
    if isinstance(values, (int, float, np.integer, np.floating)):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return [0, 2, 4, 8]

    radii: set[int] = set()
    for value in values:
        maybe = safe_float(value)
        if maybe is None:
            continue
        radii.add(max(0, int(round(maybe))))

    if not radii:
        return [0, 2, 4, 8]
    if 0 not in radii:
        radii.add(0)
    return sorted(radii)


def build_local_tumor_context(
    *,
    case_id: str,
    dataset_name: str,
    centerline_path: Path,
    tumor_mask_root: Path,
    tumor_mask_pattern: str,
    tumor_threshold: float,
    tumor_radii_voxels: list[int],
) -> dict[str, Any]:
    """Load and cache tumor-local arrays once for block-specific feature extractors."""
    context: dict[str, Any] = {
        "available": False,
        "tumor_mask_exists": 0.0,
        "tumor_mask_loaded": 0.0,
    }

    tumor_mask_path = resolve_tumor_mask_path(
        case_id=case_id,
        dataset_name=dataset_name,
        tumor_mask_root=tumor_mask_root,
        tumor_mask_pattern=tumor_mask_pattern,
    )
    if tumor_mask_path is None:
        return context

    context["tumor_mask_exists"] = 1.0

    try:
        skeleton = np.load(centerline_path).astype(bool, copy=False)
        tumor_mask = load_tumor_mask_zyx(
            tumor_mask_path,
            expected_shape_zyx=tuple(int(v) for v in skeleton.shape),
            threshold=tumor_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        logging.debug("Tumor feature extraction failed for %s: %s", case_id, exc)
        return context

    context["tumor_mask_loaded"] = 1.0
    context["available"] = True
    context["skeleton_voxels"] = float(np.count_nonzero(skeleton))
    context["tumor_voxels"] = float(np.count_nonzero(tumor_mask))
    context["overlap_tumor"] = float(np.count_nonzero(skeleton & tumor_mask))

    max_radius = max(tumor_radii_voxels) if tumor_radii_voxels else 0
    tz, ty, tx = np.nonzero(tumor_mask)
    z0 = max(int(tz.min()) - max_radius - 1, 0)
    y0 = max(int(ty.min()) - max_radius - 1, 0)
    x0 = max(int(tx.min()) - max_radius - 1, 0)
    z1 = min(int(tz.max()) + max_radius + 2, tumor_mask.shape[0])
    y1 = min(int(ty.max()) + max_radius + 2, tumor_mask.shape[1])
    x1 = min(int(tx.max()) + max_radius + 2, tumor_mask.shape[2])

    tumor_crop = tumor_mask[z0:z1, y0:y1, x0:x1]
    skeleton_crop = skeleton[z0:z1, y0:y1, x0:x1]
    context["skeleton_crop"] = skeleton_crop
    context["dist_to_tumor"] = ndimage.distance_transform_edt(~tumor_crop)
    return context


def extract_tumor_size_local_features(
    local_context: dict[str, Any],
    tumor_radii_voxels: list[int],
) -> dict[str, float]:
    """Extract tumor-size block features from local tumor masks."""
    features: dict[str, float] = {
        "tumor_mask_exists": float(local_context.get("tumor_mask_exists", 0.0)),
        "tumor_mask_loaded": float(local_context.get("tumor_mask_loaded", 0.0)),
    }
    if not local_context.get("available", False):
        return features

    tumor_voxels = float(local_context["tumor_voxels"])
    dist_to_tumor = np.asarray(local_context["dist_to_tumor"], dtype=float)
    features["tumor_size_tumor_voxels"] = tumor_voxels

    prev_radius = 0
    for radius in tumor_radii_voxels:
        region = dist_to_tumor <= float(radius)
        region_voxels = float(np.count_nonzero(region))
        region_prefix = f"tumor_r{radius}"
        features[f"tumor_size_{region_prefix}_region_voxels"] = region_voxels

        if radius > prev_radius:
            shell = (dist_to_tumor > float(prev_radius)) & (
                dist_to_tumor <= float(radius)
            )
            shell_voxels = float(np.count_nonzero(shell))
            shell_prefix = f"tumor_shell_r{prev_radius}_{radius}"
            features[f"tumor_size_{shell_prefix}_voxels"] = shell_voxels
        prev_radius = radius

    return features
