#!/usr/bin/env python3
r"""Overlay skeleton + tumor on a DCE-like slice for spatial alignment QA (issue #119).

Run from the repo root with the same environment as the rest of Vanguard
(`micromamba activate vanguard` on the cluster). Requires: numpy, scipy,
matplotlib, PyYAML, SimpleITK.

Example::

    PYTHONPATH=. python scripts/deepsets_alignment_check.py \\
        --case-ids ISPY2_100899,ISPY2_102011,ISPY2_102212

Clinical DCE (default)::

    --dce-root /net/projects2/vanguard/MAMA-MIA-syn60868042/images

Vessel segmentation phases (same grid lineage as tc4d)::

    --use-vessel-segmentation-phases \\
    --vessel-segmentation-root /net/projects2/vanguard/vessel_segmentations/ISPY2
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

# Headless plotting on clusters without a display
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import SimpleITK as sitk

from deepsets_volume_align import align_zyx_volume_to_shape
from load_cohort import load_config

CONTOUR_LEVEL = 0.5

NDIM_3D = 3


def resolve_tumor_mask_path(
    *,
    case_id: str,
    dataset_name: str,
    tumor_mask_root: Path,
    tumor_mask_pattern: str,
) -> Path | None:
    """Resolve tumor-mask path using common filename/layout fallbacks (mirrors ``features.tumor_size``)."""
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
                logging.debug(
                    "Skipping invalid tumor mask pattern %r for case %s",
                    pattern,
                    case_id,
                )
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
    """Load tumor segmentation as a 3D bool mask aligned to zyx (mirrors ``features.tumor_size``)."""
    path_lower = str(tumor_mask_path).lower()

    if path_lower.endswith(".npy"):
        arr = np.load(tumor_mask_path)
        if arr.ndim < NDIM_3D:
            raise ValueError(
                f"Unsupported tumor npy ndim={arr.ndim}: {tumor_mask_path}"
            )
        if arr.ndim > NDIM_3D:
            lead_axes = tuple(range(arr.ndim - NDIM_3D))
            arr = np.any(arr > threshold, axis=lead_axes).astype(np.uint8)
    else:
        if not (
            path_lower.endswith(".nii")
            or path_lower.endswith(".nii.gz")
            or path_lower.endswith(".nrrd")
        ):
            raise ValueError(f"Unsupported tumor mask format: {tumor_mask_path}")
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


OFFSETS_3D = np.array(
    [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ],
    dtype=np.int64,
)
ENDPOINT_DEGREE = 1
CHAIN_DEGREE = 2


def _load_sitk_volume_zyx(
    path: Path, expected_shape_zyx: tuple[int, int, int]
) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    return align_zyx_volume_to_shape(arr, expected_shape_zyx)


def _load_mask_with_spacing(
    tumor_mask_path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    threshold: float,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    mask = load_tumor_mask_zyx(
        tumor_mask_path,
        expected_shape_zyx=expected_shape_zyx,
        threshold=threshold,
    )
    path_lower = str(tumor_mask_path).lower()
    if path_lower.endswith(".npy"):
        spacing_mm_zyx = (1.0, 1.0, 1.0)
    else:
        image = sitk.ReadImage(str(tumor_mask_path))
        spacing_xyz = tuple(float(v) for v in image.GetSpacing())
        spacing_mm_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return mask, spacing_mm_zyx


def _build_signed_distance_mm(
    tumor_mask_zyx: np.ndarray,
    spacing_mm_zyx: tuple[float, float, float],
) -> np.ndarray:
    tumor = np.asarray(tumor_mask_zyx, dtype=bool)
    outside_mm = ndimage.distance_transform_edt(~tumor, sampling=spacing_mm_zyx)
    inside_mm = ndimage.distance_transform_edt(tumor, sampling=spacing_mm_zyx)
    return outside_mm - inside_mm


def _sample_signed_distance_mm(
    xyz_vox: tuple[int, int, int],
    signed_dist_mm: np.ndarray,
) -> float:
    x, y, z = (int(v) for v in xyz_vox)
    z = int(np.clip(z, 0, signed_dist_mm.shape[0] - 1))
    y = int(np.clip(y, 0, signed_dist_mm.shape[1] - 1))
    x = int(np.clip(x, 0, signed_dist_mm.shape[2] - 1))
    return float(signed_dist_mm[z, y, x])


def _build_neighbor_map(
    skeleton_mask_zyx: np.ndarray,
) -> tuple[
    list[tuple[int, int, int]], dict[tuple[int, int, int], list[tuple[int, int, int]]]
]:
    coords_zyx = np.argwhere(skeleton_mask_zyx)
    if coords_zyx.size == 0:
        return [], {}
    coords_xyz = [(int(x), int(y), int(z)) for z, y, x in coords_zyx]
    coord_set = set(coords_xyz)
    neighbors: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for x, y, z in coords_xyz:
        node_neighbors: list[tuple[int, int, int]] = []
        for dz, dy, dx in OFFSETS_3D:
            neighbor = (x + int(dx), y + int(dy), z + int(dz))
            if neighbor in coord_set:
                node_neighbors.append(neighbor)
        neighbors[(x, y, z)] = node_neighbors
    return coords_xyz, neighbors


def _point_mm(
    xyz_vox: tuple[int, int, int], spacing_mm_zyx: tuple[float, float, float]
) -> np.ndarray:
    x, y, z = (float(v) for v in xyz_vox)
    return np.asarray(
        [
            x * float(spacing_mm_zyx[2]),
            y * float(spacing_mm_zyx[1]),
            z * float(spacing_mm_zyx[0]),
        ],
        dtype=np.float32,
    )


def _compute_curvature(
    *,
    xyz_vox: tuple[int, int, int],
    neighbor_xyz: list[tuple[int, int, int]],
    spacing_mm_zyx: tuple[float, float, float],
) -> float:
    point_mm = _point_mm(xyz_vox, spacing_mm_zyx)
    neighbor_vectors = [
        _point_mm(neighbor, spacing_mm_zyx) - point_mm for neighbor in neighbor_xyz
    ]
    valid_vectors = [vec for vec in neighbor_vectors if np.linalg.norm(vec) > 0.0]
    if not valid_vectors:
        return 0.0
    curvature_rad = 0.0
    degree = len(valid_vectors)
    if degree == ENDPOINT_DEGREE:
        return 0.0
    if degree == CHAIN_DEGREE:
        v1 = valid_vectors[0] / np.linalg.norm(valid_vectors[0])
        v2 = valid_vectors[1] / np.linalg.norm(valid_vectors[1])
        curvature_rad = float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return curvature_rad


def _default_dataset_name(config: Any) -> str:
    include = config.feature_toggles.dataset_include
    if include is None:
        raise ValueError("config.feature_toggles.dataset_include is required")
    if isinstance(include, str):
        return str(include)
    if not include:
        raise ValueError("dataset_include must be non-empty")
    return str(include[0])


def _choose_axial_slice_z(tumor_mask_zyx: np.ndarray) -> int:
    areas = tumor_mask_zyx.sum(axis=(1, 2)).astype(int)
    return int(np.argmax(areas))


def _load_vessel_timepoint(
    *,
    vessel_root: Path,
    case_id: str,
    time_index: int,
    expected_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    path = (
        vessel_root
        / case_id
        / "images"
        / f"{case_id}_{time_index:04d}_vessel_segmentation.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing vessel segmentation: {path}")
    loaded = np.load(path, allow_pickle=False)
    if "vessel" not in loaded.files:
        raise ValueError(f"NPZ at {path} has no 'vessel' array")
    arr = np.asarray(loaded["vessel"], dtype=np.float32)
    loaded.close()
    return align_zyx_volume_to_shape(arr, expected_shape_zyx)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the alignment check script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deepsets_ispy2.yaml"),
        help="YAML with centerline_root, tumor_mask_root, toggles.",
    )
    parser.add_argument(
        "--dce-root",
        type=Path,
        default=Path("/net/projects2/vanguard/MAMA-MIA-syn60868042/images"),
        help="Parent of per-case dirs for clinical phase NIfTIs.",
    )
    parser.add_argument(
        "--use-vessel-segmentation-phases",
        action="store_true",
        help="Use tc4d vessel NPZs instead of clinical NIfTI phases.",
    )
    parser.add_argument(
        "--vessel-segmentation-root",
        type=Path,
        default=Path("/net/projects2/vanguard/vessel_segmentations/ISPY2"),
        help="Dataset root containing <case_id>/images/*_vessel_segmentation.npz",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        default="ISPY2_100899,ISPY2_102011,ISPY2_102212",
        help="Comma-separated case IDs.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=1,
        help="Phase index (0001 = first post-contrast when 0000 is pre).",
    )
    parser.add_argument(
        "--slice-z",
        type=int,
        default=None,
        help="Override axial slice index (zyx z); default = max tumor area slice.",
    )
    parser.add_argument(
        "--slice-thickness",
        type=int,
        default=1,
        help="Include skeleton voxels with |z - slice_z| <= this value.",
    )
    parser.add_argument(
        "--color-by",
        choices=("signed_distance_mm", "curvature_rad"),
        default="signed_distance_mm",
        help="Scalar used for point colors.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/deepsets_issue119/figures"),
        help="Directory for PNG outputs.",
    )
    return parser.parse_args()


def _clinical_phase_path(dce_root: Path, case_id: str, time_index: int) -> Path:
    return dce_root / case_id / f"{case_id}_{time_index:04d}.nii.gz"


def main() -> None:
    """Load cohort paths from config and write overlay PNGs per case."""
    args = parse_args()
    config = load_config(args.config)
    toggles = config.feature_toggles
    data_paths = config.data_paths
    dataset_name = _default_dataset_name(config)
    centerline_root = Path(data_paths.centerline_root)
    tumor_mask_root = Path(data_paths.tumor_mask_root)
    tumor_pattern = str(toggles.tumor_mask_file_pattern)
    tumor_threshold = float(toggles.tumor_mask_threshold)
    skeleton_pattern = str(toggles.centerline_file_pattern)

    case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        study_dir = centerline_root / dataset_name / case_id
        skeleton_path = study_dir / skeleton_pattern.format(case_id=case_id)
        if not skeleton_path.exists():
            raise FileNotFoundError(f"Missing skeleton: {skeleton_path}")

        skeleton = np.load(skeleton_path).astype(bool, copy=False)
        shape_zyx = tuple(int(v) for v in skeleton.shape)

        tumor_path = resolve_tumor_mask_path(
            case_id=case_id,
            dataset_name=dataset_name,
            tumor_mask_root=tumor_mask_root,
            tumor_mask_pattern=tumor_pattern,
        )
        if tumor_path is None:
            raise FileNotFoundError(f"No tumor mask for case_id={case_id}")

        tumor_mask, spacing_mm_zyx = _load_mask_with_spacing(
            tumor_path,
            expected_shape_zyx=shape_zyx,
            threshold=tumor_threshold,
        )

        if args.use_vessel_segmentation_phases:
            volume_zyx = _load_vessel_timepoint(
                vessel_root=args.vessel_segmentation_root,
                case_id=case_id,
                time_index=args.time_index,
                expected_shape_zyx=shape_zyx,
            )
            source_tag = "vessel_seg"
        else:
            dce_path = _clinical_phase_path(args.dce_root, case_id, args.time_index)
            if not dce_path.exists():
                raise FileNotFoundError(f"Missing DCE phase: {dce_path}")
            volume_zyx = _load_sitk_volume_zyx(dce_path, shape_zyx)
            source_tag = "clinical"

        slice_z = (
            int(args.slice_z)
            if args.slice_z is not None
            else _choose_axial_slice_z(tumor_mask)
        )
        signed_dist_mm = _build_signed_distance_mm(tumor_mask, spacing_mm_zyx)
        coords_xyz, neighbor_map = _build_neighbor_map(skeleton)

        xs: list[float] = []
        ys: list[float] = []
        colors: list[float] = []
        dz = int(args.slice_thickness)
        for xyz in coords_xyz:
            _, _, z = xyz
            if abs(int(z) - slice_z) > dz:
                continue
            neighbors = neighbor_map[xyz]
            if args.color_by == "curvature_rad":
                val = _compute_curvature(
                    xyz_vox=xyz,
                    neighbor_xyz=neighbors,
                    spacing_mm_zyx=spacing_mm_zyx,
                )
            else:
                val = _sample_signed_distance_mm(xyz, signed_dist_mm)
            xs.append(float(xyz[0]))
            ys.append(float(xyz[1]))
            colors.append(float(val))

        slice_img = np.asarray(volume_zyx[slice_z, :, :], dtype=np.float32)
        mask_slice = np.asarray(tumor_mask[slice_z, :, :], dtype=float)
        finite = slice_img[np.isfinite(slice_img)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = (
                float(np.percentile(finite, 1.0)),
                float(np.percentile(finite, 99.0)),
            )
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(finite.min()), float(finite.max())
                if vmax <= vmin:
                    vmax = vmin + 1.0

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(slice_img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if np.any(mask_slice > CONTOUR_LEVEL):
            ax.contour(mask_slice, levels=[CONTOUR_LEVEL], colors="red", linewidths=1.5)
        if xs:
            sc = ax.scatter(
                xs,
                ys,
                c=colors,
                cmap="plasma" if args.color_by == "signed_distance_mm" else "viridis",
                s=10,
                alpha=0.85,
                edgecolors="none",
            )
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=args.color_by)
        title = (
            f"{case_id} — z={slice_z} — t={args.time_index:04d} — {source_tag} — "
            f"{args.color_by}"
        )
        ax.set_title(title)
        ax.set_xlabel("x (voxel)")
        ax.set_ylabel("y (voxel)")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
        out_path = (
            args.out_dir
            / f"alignment_{source_tag}_{safe}_z{slice_z}_t{args.time_index:04d}.png"
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
