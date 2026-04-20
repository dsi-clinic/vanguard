"""Build tumor-local Deep Sets point clouds from saved centerline outputs."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from clinical_features import get_clinical_features
from features.tumor_size import load_tumor_mask_zyx, resolve_tumor_mask_path
from load_cohort import load_config
from tabular_cohort import _as_optional_bool, load_labels

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
BIFURCATION_DEGREE = 3
FALLBACK_NEAREST_POINT_COUNT = 64
POINT_FEATURE_NAMES = ["curvature_rad"]


def _str_or_empty(val: object) -> str:
    """Coerce to string, but map pandas missing values to ``""``.

    Why: plain ``str(NaN)`` yields the literal ``"nan"``, which survives
    ``dropna()`` downstream and pollutes per-subgroup metrics as a fake group.
    """
    return "" if pd.isna(val) else str(val)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ispy2.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def _load_mask_with_spacing(
    tumor_mask_path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    threshold: float,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load tumor mask and spacing in zyx order."""
    mask = load_tumor_mask_zyx(
        tumor_mask_path,
        expected_shape_zyx=expected_shape_zyx,
        threshold=threshold,
    )
    path_lower = str(tumor_mask_path).lower()
    if path_lower.endswith(".npy"):
        spacing_mm_zyx = (1.0, 1.0, 1.0)
    else:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(tumor_mask_path))
        spacing_xyz = tuple(float(v) for v in image.GetSpacing())
        spacing_mm_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return mask, spacing_mm_zyx


def _compute_local_radius_mm(
    tumor_mask_zyx: np.ndarray,
    spacing_mm_zyx: tuple[float, float, float],
    *,
    radius_floor_mm: float,
    radius_scale: float,
    radius_cap_mm: float | None,
) -> tuple[float, float]:
    """Return tumor-equivalent radius and chosen tumor-local cutoff radius in mm."""
    voxel_volume_mm3 = float(np.prod(np.asarray(spacing_mm_zyx, dtype=float)))
    tumor_volume_mm3 = float(np.count_nonzero(tumor_mask_zyx)) * voxel_volume_mm3
    tumor_equiv_radius_mm = (
        float(((3.0 * tumor_volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0))
        if tumor_volume_mm3 > 0.0
        else 0.0
    )
    local_radius_mm = max(
        float(radius_floor_mm), float(radius_scale) * tumor_equiv_radius_mm
    )
    if radius_cap_mm is not None:
        local_radius_mm = min(local_radius_mm, float(radius_cap_mm))
    return tumor_equiv_radius_mm, local_radius_mm


def _build_signed_distance_mm(
    tumor_mask_zyx: np.ndarray,
    spacing_mm_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Return signed distance to tumor boundary in mm."""
    tumor = np.asarray(tumor_mask_zyx, dtype=bool)
    outside_mm = ndimage.distance_transform_edt(~tumor, sampling=spacing_mm_zyx)
    inside_mm = ndimage.distance_transform_edt(tumor, sampling=spacing_mm_zyx)
    return outside_mm - inside_mm


def _sample_signed_distance_mm(
    xyz_vox: tuple[int, int, int],
    signed_dist_mm: np.ndarray,
) -> float:
    """Sample signed distance for an xyz voxel coordinate."""
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
    """Return skeleton voxel coordinates and their 26-neighborhood adjacency."""
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
    """Convert xyz voxel coordinates into xyz millimeter coordinates."""
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
    """Return a simple pointwise curvature proxy from neighboring skeleton voxels."""
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
    elif degree == CHAIN_DEGREE:
        v1 = valid_vectors[0] / np.linalg.norm(valid_vectors[0])
        v2 = valid_vectors[1] / np.linalg.norm(valid_vectors[1])
        curvature_rad = float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return curvature_rad


def _build_case_set(
    *,
    case_id: str,
    label: int,
    skeleton_mask_zyx: np.ndarray,
    tumor_mask_zyx: np.ndarray,
    spacing_mm_zyx: tuple[float, float, float],
    local_radius_mm: float,
    tumor_equiv_radius_mm: float,
    toy_perfect_feature: bool = False,
    toy_only: bool = False,
) -> dict[str, Any] | None:
    """Build one tumor-local point set for Deep Sets."""
    coords_xyz, neighbor_map = _build_neighbor_map(skeleton_mask_zyx)
    if not coords_xyz:
        return None

    signed_dist_mm = _build_signed_distance_mm(tumor_mask_zyx, spacing_mm_zyx)
    tumor_coords_zyx = np.argwhere(tumor_mask_zyx)
    if tumor_coords_zyx.size == 0:
        return None

    candidate_rows: list[tuple[float, list[float]]] = []
    feature_rows: list[list[float]] = []
    for xyz_vox in coords_xyz:
        signed_distance_mm = _sample_signed_distance_mm(xyz_vox, signed_dist_mm)
        neighbors = neighbor_map[xyz_vox]
        curvature_rad = _compute_curvature(
            xyz_vox=xyz_vox,
            neighbor_xyz=neighbors,
            spacing_mm_zyx=spacing_mm_zyx,
        )
        if toy_only:
            row = [float(label)]
        else:
            row = [float(curvature_rad)]
            if toy_perfect_feature:
                row.append(float(label))
        candidate_rows.append((float(signed_distance_mm), row))
        if signed_distance_mm <= float(local_radius_mm):
            feature_rows.append(row)

    used_fallback = False
    if not feature_rows:
        candidate_rows.sort(key=lambda pair: pair[0])
        feature_rows = [row for _, row in candidate_rows[:FALLBACK_NEAREST_POINT_COUNT]]
        used_fallback = bool(feature_rows)
    if not feature_rows:
        return None

    return {
        "x": torch.tensor(feature_rows, dtype=torch.float32),
        "y": torch.tensor([int(label)], dtype=torch.float32),
        "case_id": str(case_id),
        "feature_names": (
            ["toy_perfect_label"]
            if toy_only
            else list(POINT_FEATURE_NAMES)
            + (["toy_perfect_label"] if toy_perfect_feature else [])
        ),
        "local_radius_mm": float(local_radius_mm),
        "tumor_equiv_radius_mm": float(tumor_equiv_radius_mm),
        "num_points": int(len(feature_rows)),
        "used_fallback_nearest_points": float(used_fallback),
    }


def main() -> None:
    """Build serialized Deep Sets inputs and a manifest CSV."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    output_dir = args.output_dir
    set_dir = output_dir / "sets"
    manifest_parts_dir = output_dir / "manifest_parts"
    set_dir.mkdir(parents=True, exist_ok=True)
    manifest_parts_dir.mkdir(parents=True, exist_ok=True)

    params = config.model_params
    radius_floor_mm = float(params.deepsets_local_radius_floor_mm)
    radius_scale = float(params.deepsets_local_radius_scale)
    radius_cap_raw = params.deepsets_local_radius_cap_mm
    radius_cap_mm = (
        None if radius_cap_raw in {None, "", "none", "null"} else float(radius_cap_raw)
    )

    data_paths = config.data_paths
    toggles = config.feature_toggles
    centerline_root = Path(data_paths.centerline_root)
    tumor_mask_root = Path(data_paths.tumor_mask_root)
    tumor_mask_pattern = str(toggles.tumor_mask_file_pattern)
    tumor_threshold = float(toggles.tumor_mask_threshold)
    skeleton_pattern = str(toggles.centerline_file_pattern)
    dataset_include = toggles.dataset_include
    bilateral_filter = _as_optional_bool(toggles.bilateral_filter)
    toy_perfect_feature = bool(getattr(toggles, "toy_perfect_feature", False))
    toy_only = bool(getattr(toggles, "toy_only", False))
    if toy_perfect_feature or toy_only:
        logging.warning(
            "TOY EXPERIMENT MODE: injecting perfect label feature into every point. "
            "Results will be artificially perfect. Do NOT use for real experiments."
        )

    labels_df = load_labels(
        Path(data_paths.labels_csv),
        data_paths.id_column,
        data_paths.label_column,
    )

    clinical_df = get_clinical_features(config).copy()
    clinical_df["case_id"] = clinical_df["case_id"].astype(str)
    if dataset_include is not None and "dataset" in clinical_df.columns:
        if isinstance(dataset_include, str):
            dataset_include = [dataset_include]
        clinical_df = clinical_df[
            clinical_df["dataset"].astype(str).isin({str(v) for v in dataset_include})
        ].copy()
    if bilateral_filter is not None and "bilateral" in clinical_df.columns:
        parsed_bilateral = clinical_df["bilateral"].map(_as_optional_bool)
        clinical_df = clinical_df[parsed_bilateral == bilateral_filter].copy()

    manifest_source = clinical_df.merge(labels_df, on="case_id", how="inner")
    manifest_source = manifest_source.rename(columns={data_paths.label_column: "label"})
    if num_shards > 1:
        manifest_source = (
            manifest_source.iloc[shard_index::num_shards].copy().reset_index(drop=True)
        )

    total_cases = int(len(manifest_source))
    optional_metadata_cols = [
        c
        for c in ("site", "tumor_subtype", "bilateral")
        if c in manifest_source.columns
    ]
    progress_every = 50
    skipped_missing_centerline = 0
    skipped_missing_tumor_mask = 0
    skipped_empty_case_set = 0
    failed_case_builds = 0
    fallback_case_sets = 0
    rows: list[dict[str, Any]] = []

    logging.info(
        "Starting Deep Sets dataset build for %d cases into %s",
        total_cases,
        output_dir,
    )

    for index, row in enumerate(manifest_source.itertuples(index=False), start=1):
        case_id = str(row.case_id)
        dataset_name = str(getattr(row, "dataset", ""))
        study_dir = centerline_root / dataset_name / case_id
        skeleton_path = study_dir / skeleton_pattern.format(case_id=case_id)
        if not skeleton_path.exists():
            skipped_missing_centerline += 1
            if index % progress_every == 0 or index == total_cases:
                logging.info(
                    "Deep Sets build progress %d/%d: wrote=%d missing_centerline=%d "
                    "missing_tumor_mask=%d empty_case_set=%d failed=%d fallback=%d",
                    index,
                    total_cases,
                    len(rows),
                    skipped_missing_centerline,
                    skipped_missing_tumor_mask,
                    skipped_empty_case_set,
                    failed_case_builds,
                    fallback_case_sets,
                )
            continue

        try:
            skeleton_mask = np.load(skeleton_path).astype(bool, copy=False)
            tumor_mask_path = resolve_tumor_mask_path(
                case_id=case_id,
                dataset_name=dataset_name,
                tumor_mask_root=tumor_mask_root,
                tumor_mask_pattern=tumor_mask_pattern,
            )
            if tumor_mask_path is None:
                skipped_missing_tumor_mask += 1
                if index % progress_every == 0 or index == total_cases:
                    logging.info(
                        "Deep Sets build progress %d/%d: wrote=%d missing_centerline=%d "
                        "missing_tumor_mask=%d empty_case_set=%d failed=%d fallback=%d",
                        index,
                        total_cases,
                        len(rows),
                        skipped_missing_centerline,
                        skipped_missing_tumor_mask,
                        skipped_empty_case_set,
                        failed_case_builds,
                        fallback_case_sets,
                    )
                continue
            tumor_mask_zyx, spacing_mm_zyx = _load_mask_with_spacing(
                tumor_mask_path,
                expected_shape_zyx=tuple(int(v) for v in skeleton_mask.shape),
                threshold=tumor_threshold,
            )
            tumor_equiv_radius_mm, local_radius_mm = _compute_local_radius_mm(
                tumor_mask_zyx,
                spacing_mm_zyx,
                radius_floor_mm=radius_floor_mm,
                radius_scale=radius_scale,
                radius_cap_mm=radius_cap_mm,
            )
            case_set = _build_case_set(
                case_id=case_id,
                label=int(row.label),
                skeleton_mask_zyx=skeleton_mask,
                tumor_mask_zyx=tumor_mask_zyx,
                spacing_mm_zyx=spacing_mm_zyx,
                local_radius_mm=local_radius_mm,
                tumor_equiv_radius_mm=tumor_equiv_radius_mm,
                toy_perfect_feature=toy_perfect_feature,
                toy_only=toy_only,
            )
        except Exception as exc:  # noqa: BLE001
            failed_case_builds += 1
            logging.warning("Failed Deep Sets dataset build for %s: %s", case_id, exc)
            if index % progress_every == 0 or index == total_cases:
                logging.info(
                    "Deep Sets build progress %d/%d: wrote=%d missing_centerline=%d "
                    "missing_tumor_mask=%d empty_case_set=%d failed=%d fallback=%d",
                    index,
                    total_cases,
                    len(rows),
                    skipped_missing_centerline,
                    skipped_missing_tumor_mask,
                    skipped_empty_case_set,
                    failed_case_builds,
                    fallback_case_sets,
                )
            continue

        if case_set is None:
            skipped_empty_case_set += 1
            if index % progress_every == 0 or index == total_cases:
                logging.info(
                    "Deep Sets build progress %d/%d: wrote=%d missing_centerline=%d "
                    "missing_tumor_mask=%d empty_case_set=%d failed=%d fallback=%d",
                    index,
                    total_cases,
                    len(rows),
                    skipped_missing_centerline,
                    skipped_missing_tumor_mask,
                    skipped_empty_case_set,
                    failed_case_builds,
                    fallback_case_sets,
                )
            continue
        fallback_case_sets += int(case_set.get("used_fallback_nearest_points", 0.0))

        set_path = set_dir / f"{case_id}.pt"
        torch.save(case_set, set_path)
        manifest_row = {
            "case_id": case_id,
            "set_path": str(set_path),
            "label": int(row.label),
            "dataset": dataset_name,
            "num_points": int(case_set["num_points"]),
            "local_radius_mm": float(case_set["local_radius_mm"]),
            "tumor_equiv_radius_mm": float(case_set["tumor_equiv_radius_mm"]),
            "used_fallback_nearest_points": int(
                case_set["used_fallback_nearest_points"]
            ),
        }
        for col in optional_metadata_cols:
            manifest_row[col] = _str_or_empty(getattr(row, col, ""))
        rows.append(manifest_row)

        if index % progress_every == 0 or index == total_cases:
            logging.info(
                "Deep Sets build progress %d/%d: wrote=%d missing_centerline=%d "
                "missing_tumor_mask=%d empty_case_set=%d failed=%d fallback=%d",
                index,
                total_cases,
                len(rows),
                skipped_missing_centerline,
                skipped_missing_tumor_mask,
                skipped_empty_case_set,
                failed_case_builds,
                fallback_case_sets,
            )

    manifest_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    if num_shards > 1:
        manifest_path = (
            manifest_parts_dir / f"deepsets_manifest_part_{shard_index:03d}.csv"
        )
    else:
        manifest_path = output_dir / "deepsets_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logging.info(
        "Finished Deep Sets dataset build: wrote=%d total=%d missing_centerline=%d "
        "missing_tumor_mask=%d empty_case_set=%d failed=%d fallback=%d manifest=%s",
        len(manifest_df),
        total_cases,
        skipped_missing_centerline,
        skipped_missing_tumor_mask,
        skipped_empty_case_set,
        failed_case_builds,
        fallback_case_sets,
        manifest_path,
    )


if __name__ == "__main__":
    main()
