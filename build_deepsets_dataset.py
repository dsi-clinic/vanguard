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
from deepsets_volume_align import align_zyx_4d_to_shape
from features.tumor_size import load_tumor_mask_zyx, resolve_tumor_mask_path
from graph_extraction.constants import (
    BIFURCATION_MIN_DEGREE,
    KINETIC_SIGNAL_EPS,
    MIN_KINETIC_TIMEPOINTS,
    NDIM_4D,
)
from graph_extraction.core4d import (
    discover_study_timepoints,
    load_time_series_from_files,
)
from graph_extraction.feature_stats import (
    _arrival_index_from_enhancement,
    _safe_ratio,
    _shell_name_for_signed_distance,
)
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
FALLBACK_NEAREST_POINT_COUNT = 64
_REF_PEAK_EPS = 1e-10

DEEPSETS_FEATURE_BASELINE = "baseline"
DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY = "geometry_topology"
DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY_DYNAMIC = "geometry_topology_dynamic"
VALID_DEEPSETS_POINT_FEATURE_SETS: frozenset[str] = frozenset(
    {
        DEEPSETS_FEATURE_BASELINE,
        DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY,
        DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY_DYNAMIC,
    }
)


def deepsets_point_feature_names(regime: str) -> list[str]:
    """Ordered feature column names for a Deep Sets point-feature regime."""
    if regime == DEEPSETS_FEATURE_BASELINE:
        return ["curvature_rad"]
    geom_topo = [
        "signed_distance_mm",
        "abs_signed_distance_mm",
        "inside_tumor",
        "shell_0_2mm",
        "shell_2_5mm",
        "shell_5_10mm",
        "shell_ge_10mm",
        "degree",
        "is_endpoint",
        "is_chain",
        "is_bifurcation",
        "offset_x_mm",
        "offset_y_mm",
        "offset_z_mm",
        "support_radius_mm",
        "support_radius_available",
    ]
    if regime == DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY:
        return list(geom_topo)
    if regime == DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY_DYNAMIC:
        return list(
            geom_topo
            + [
                "arrival_index_norm",
                "has_arrival",
                "peak_index_norm",
                "peak_enhancement",
                "washin_slope",
                "washout_slope",
                "positive_enhancement_auc",
                "peak_rel_reference",
                "auc_rel_reference",
                "kinetic_signal_ok",
                "reference_ok",
            ]
        )
    raise ValueError(
        f"Unknown deepsets_point_feature_set {regime!r}; expected one of "
        f"{sorted(VALID_DEEPSETS_POINT_FEATURE_SETS)}"
    )


def _outside_shell_one_hot_four(
    signed_distance_mm: float,
) -> tuple[float, float, float, float]:
    """One-hot for outside shells; all zero when inside tumor (signed < 0)."""
    if signed_distance_mm < 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    name = _shell_name_for_signed_distance(float(signed_distance_mm))
    if name == "shell_0_2mm":
        return (1.0, 0.0, 0.0, 0.0)
    if name == "shell_2_5mm":
        return (0.0, 1.0, 0.0, 0.0)
    if name == "shell_5_10mm":
        return (0.0, 0.0, 1.0, 0.0)
    if name in ("shell_10_20mm", "shell_gt20mm"):
        return (0.0, 0.0, 0.0, 1.0)
    return (0.0, 0.0, 0.0, 0.0)


def _tumor_centroid_xyz_mm(
    tumor_mask_zyx: np.ndarray,
    spacing_mm_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Tumor centroid in mm, shape (3,) as x, y, z."""
    tumor_coords_zyx = np.argwhere(np.asarray(tumor_mask_zyx, dtype=bool))
    if tumor_coords_zyx.size == 0:
        return np.zeros(3, dtype=np.float32)
    centroid_zyx = np.mean(tumor_coords_zyx.astype(np.float64), axis=0)
    sz, sy, sx = (
        float(spacing_mm_zyx[0]),
        float(spacing_mm_zyx[1]),
        float(spacing_mm_zyx[2]),
    )
    return np.asarray(
        [centroid_zyx[2] * sx, centroid_zyx[1] * sy, centroid_zyx[0] * sz],
        dtype=np.float32,
    )


def _reference_enhancement_baseline(
    signal_4d: np.ndarray,
    support: np.ndarray,
    tumor: np.ndarray,
) -> tuple[float, float]:
    """Reference peak enhancement and positive AUC (kinematic-style, no breast ref mask)."""
    support = np.asarray(support, dtype=bool)
    tumor = np.asarray(tumor, dtype=bool)
    ref_mask = np.ones_like(support, dtype=bool)
    ref_cand = (~support) & (~tumor)
    if np.any(ref_cand):
        ref_mask = ref_cand
    n_t = int(signal_4d.shape[0])
    time_axis = np.arange(n_t, dtype=float)
    ref_curve = np.asarray(
        [float(np.mean(signal_4d[t][ref_mask])) for t in range(n_t)],
        dtype=float,
    )
    ref_enh = ref_curve - float(ref_curve[0])
    peak = float(np.max(np.maximum(ref_enh, 0.0)))
    auc = float(np.trapz(np.maximum(ref_enh, 0.0), x=time_axis))
    return peak, auc


def _dynamic_features_for_voxel(
    *,
    signal_4d: np.ndarray | None,
    xyz_vox: tuple[int, int, int],
    ref_peak_enh: float,
    ref_auc_pos: float,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Return the 11 dynamic scalars in feature-name order (or zeros if no signal)."""
    if signal_4d is None or signal_4d.ndim != NDIM_4D:
        return (0.0,) * 11
    n_t = int(signal_4d.shape[0])
    kinetic_ok = (
        1.0 if n_t >= MIN_KINETIC_TIMEPOINTS and np.all(np.isfinite(signal_4d)) else 0.0
    )
    if kinetic_ok == 0.0:
        return (0.0,) * 9 + (0.0, 0.0)
    x, y, z = (int(v) for v in xyz_vox)
    z = int(np.clip(z, 0, signal_4d.shape[1] - 1))
    y = int(np.clip(y, 0, signal_4d.shape[2] - 1))
    x = int(np.clip(x, 0, signal_4d.shape[3] - 1))
    curve = np.asarray(signal_4d[:, z, y, x], dtype=float)
    enh = curve - float(curve[0])
    time_axis = np.arange(n_t, dtype=float)
    tte_idx = _arrival_index_from_enhancement(enh)
    arrival_norm = (
        float(tte_idx) / float(max(1, n_t - 1)) if tte_idx is not None else 0.0
    )
    has_arr = 1.0 if tte_idx is not None else 0.0
    peak_idx = int(np.argmax(enh))
    peak_norm = float(peak_idx) / float(max(1, n_t - 1))
    peak_enh = float(np.max(enh))
    start_idx = 0 if tte_idx is None else int(tte_idx)
    washin_den = float(time_axis[peak_idx] - time_axis[start_idx])
    washin_slope = (
        float((enh[peak_idx] - enh[start_idx]) / washin_den)
        if washin_den > 0.0
        else 0.0
    )
    washout_den = float(time_axis[-1] - time_axis[peak_idx])
    washout_slope = (
        float((enh[-1] - enh[peak_idx]) / washout_den) if washout_den > 0.0 else 0.0
    )
    auc = float(np.trapz(np.maximum(enh, 0.0), x=time_axis))
    peak_rel = _safe_ratio(max(peak_enh, 0.0), max(ref_peak_enh, 1e-12))
    auc_rel = _safe_ratio(max(auc, 0.0), max(ref_auc_pos, 1e-12))
    ref_ok = 1.0 if ref_peak_enh > _REF_PEAK_EPS else 0.0
    return (
        arrival_norm,
        has_arr,
        peak_norm,
        peak_enh,
        washin_slope,
        washout_slope,
        auc,
        peak_rel,
        auc_rel,
        kinetic_ok,
        ref_ok,
    )


def _try_load_vessel_4d(
    *,
    vessel_root: Path | None,
    case_id: str,
    expected_shape_zyx: tuple[int, int, int],
) -> np.ndarray | None:
    if vessel_root is None or not str(vessel_root).strip():
        return None
    root = Path(vessel_root)
    if not root.exists():
        return None
    try:
        paths, _ = discover_study_timepoints(root, case_id)
        arr = load_time_series_from_files(paths)
        return align_zyx_4d_to_shape(arr, expected_shape_zyx)
    except (OSError, ValueError) as exc:
        logging.warning("Vessel 4D unavailable for %s: %s", case_id, exc)
        return None


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
    point_feature_set: str,
    support_edt_mm_zyx: np.ndarray | None,
    support_radius_available_scalar: float,
    signal_4d: np.ndarray | None,
    toy_perfect_feature: bool = False,
    toy_only: bool = False,
) -> dict[str, Any] | None:
    """Build one tumor-local point set for Deep Sets."""
    feature_names = deepsets_point_feature_names(point_feature_set)
    coords_xyz, neighbor_map = _build_neighbor_map(skeleton_mask_zyx)
    if not coords_xyz:
        return None

    signed_dist_mm = _build_signed_distance_mm(tumor_mask_zyx, spacing_mm_zyx)
    tumor_coords_zyx = np.argwhere(tumor_mask_zyx)
    if tumor_coords_zyx.size == 0:
        return None

    centroid_xyz_mm = _tumor_centroid_xyz_mm(tumor_mask_zyx, spacing_mm_zyx)
    tumor_b = np.asarray(tumor_mask_zyx, dtype=bool)

    ref_peak = 0.0
    ref_auc = 0.0
    support_for_ref: np.ndarray
    if signal_4d is not None and signal_4d.ndim == NDIM_4D:
        support_for_ref = np.any(signal_4d > KINETIC_SIGNAL_EPS, axis=0)
    else:
        support_for_ref = (
            np.ones_like(tumor_b, dtype=bool)
            if support_edt_mm_zyx is None
            else np.ones_like(tumor_b, dtype=bool)
        )
    if support_edt_mm_zyx is not None:
        support_for_ref = (
            np.asarray(support_edt_mm_zyx > 0.0, dtype=bool) | support_for_ref
        )
    if signal_4d is not None and signal_4d.ndim == NDIM_4D:
        ref_peak, ref_auc = _reference_enhancement_baseline(
            signal_4d, support_for_ref, tumor_b
        )

    kinetic_tp = int(signal_4d.shape[0]) if signal_4d is not None else 0

    def _one_row(xyz_vox: tuple[int, int, int]) -> list[float]:
        signed_distance_mm = _sample_signed_distance_mm(xyz_vox, signed_dist_mm)
        neighbors = neighbor_map[xyz_vox]
        deg = len(neighbors)
        if point_feature_set == DEEPSETS_FEATURE_BASELINE:
            curvature_rad = _compute_curvature(
                xyz_vox=xyz_vox,
                neighbor_xyz=neighbors,
                spacing_mm_zyx=spacing_mm_zyx,
            )
            return [float(curvature_rad)]

        pmm = _point_mm(xyz_vox, spacing_mm_zyx)
        off = pmm - centroid_xyz_mm
        s0, s1, s2, s3 = _outside_shell_one_hot_four(signed_distance_mm)
        sup_r = 0.0
        if support_edt_mm_zyx is not None:
            xv, yv, zv = (int(v) for v in xyz_vox)
            zv = int(np.clip(zv, 0, support_edt_mm_zyx.shape[0] - 1))
            yv = int(np.clip(yv, 0, support_edt_mm_zyx.shape[1] - 1))
            xv = int(np.clip(xv, 0, support_edt_mm_zyx.shape[2] - 1))
            sup_r = float(support_edt_mm_zyx[zv, yv, xv])
        row = [
            float(signed_distance_mm),
            float(abs(signed_distance_mm)),
            1.0 if signed_distance_mm < 0.0 else 0.0,
            s0,
            s1,
            s2,
            s3,
            float(deg),
            1.0 if deg == ENDPOINT_DEGREE else 0.0,
            1.0 if deg == CHAIN_DEGREE else 0.0,
            1.0 if deg >= BIFURCATION_MIN_DEGREE else 0.0,
            float(off[0]),
            float(off[1]),
            float(off[2]),
            sup_r,
            support_radius_available_scalar,
        ]
        if point_feature_set == DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY:
            return row
        dyn = _dynamic_features_for_voxel(
            signal_4d=signal_4d,
            xyz_vox=xyz_vox,
            ref_peak_enh=ref_peak,
            ref_auc_pos=ref_auc,
        )
        return row + list(dyn)

    candidate_rows: list[tuple[float, list[float]]] = []
    feature_rows: list[list[float]] = []
    for xyz_vox in coords_xyz:
        signed_distance_mm = _sample_signed_distance_mm(xyz_vox, signed_dist_mm)
        row = _one_row(xyz_vox)
        if len(row) != len(feature_names):
            raise RuntimeError(
                f"Feature count mismatch for {case_id}: row={len(row)} names={len(feature_names)}"
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
        "feature_names": list(feature_names),
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
        "point_feature_set": str(point_feature_set),
        "kinetic_timepoint_count": int(kinetic_tp),
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
    point_feature_set = str(
        getattr(params, "deepsets_point_feature_set", DEEPSETS_FEATURE_BASELINE)
    )
    if point_feature_set not in VALID_DEEPSETS_POINT_FEATURE_SETS:
        raise ValueError(
            f"Invalid model_params.deepsets_point_feature_set={point_feature_set!r}; "
            f"expected one of {sorted(VALID_DEEPSETS_POINT_FEATURE_SETS)}"
        )
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
    support_pattern = str(
        getattr(
            toggles,
            "deepsets_support_mask_pattern",
            "{case_id}_skeleton_4d_exam_support_mask.npy",
        )
    )
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

    vessel_root_raw = getattr(data_paths, "vessel_segmentation_root", "") or ""
    vessel_root_opt = (
        Path(str(vessel_root_raw)) if str(vessel_root_raw).strip() else None
    )

    logging.info(
        "Starting Deep Sets dataset build for %d cases into %s (point_feature_set=%s)",
        total_cases,
        output_dir,
        point_feature_set,
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
            sk_shape = tuple(int(v) for v in skeleton_mask.shape)
            support_path = study_dir / support_pattern.format(case_id=case_id)
            support_edt_mm_zyx: np.ndarray | None = None
            support_radius_available_scalar = 0.0
            if support_path.exists():
                try:
                    sup_mask = np.load(support_path).astype(bool, copy=False)
                    if tuple(sup_mask.shape) == sk_shape:
                        support_edt_mm_zyx = ndimage.distance_transform_edt(
                            sup_mask,
                            sampling=spacing_mm_zyx,
                        ).astype(np.float32, copy=False)
                        support_radius_available_scalar = 1.0
                except (OSError, ValueError) as sup_exc:
                    logging.warning(
                        "Could not load support mask for %s (%s): %s",
                        case_id,
                        support_path,
                        sup_exc,
                    )
            signal_4d: np.ndarray | None = None
            if point_feature_set == DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY_DYNAMIC:
                signal_4d = _try_load_vessel_4d(
                    vessel_root=vessel_root_opt,
                    case_id=case_id,
                    expected_shape_zyx=sk_shape,
                )
                if signal_4d is not None and tuple(signal_4d.shape[1:]) != sk_shape:
                    logging.warning(
                        "Vessel 4D shape mismatch for %s: %s vs skeleton %s; skipping dynamics",
                        case_id,
                        signal_4d.shape,
                        sk_shape,
                    )
                    signal_4d = None
            case_set = _build_case_set(
                case_id=case_id,
                label=int(row.label),
                skeleton_mask_zyx=skeleton_mask,
                tumor_mask_zyx=tumor_mask_zyx,
                spacing_mm_zyx=spacing_mm_zyx,
                local_radius_mm=local_radius_mm,
                tumor_equiv_radius_mm=tumor_equiv_radius_mm,
                point_feature_set=point_feature_set,
                support_edt_mm_zyx=support_edt_mm_zyx,
                support_radius_available_scalar=support_radius_available_scalar,
                signal_4d=signal_4d,
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
