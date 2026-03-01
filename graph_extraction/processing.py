"""Shared 3D/4D skeleton processing and morphometry helpers."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from graph_extraction.skeleton3d import skeletonize3d
from graph_extraction.skeleton4d import skeletonize4d
from graph_extraction.skeleton_to_graph import (
    assign_component_labels,
    build_vessel_json,
    detect_bifurcations,
    extract_segments,
    obtain_radius_map,
    segments_to_graph,
)
from graph_extraction.visuals import edges_to_segments

DEFAULT_SEGMENTATION_DIR = Path("/net/projects2/vanguard/vessel_segmentations")
NDIM_3D = 3
NDIM_4D = 4


def load_segmentation_array(path: Path) -> np.ndarray:
    """Load a segmentation array from .npy or compressed .npz files."""
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            if "vessel" in data.files:
                return data["vessel"]
            if len(data.files) == 1:
                return data[data.files[0]]
            raise ValueError(
                f"NPZ file {path} has multiple arrays: {data.files}; expected 'vessel'."
            )
        finally:
            data.close()
    return data


_OFFSETS_3D = np.array(
    [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ],
    dtype=np.int64,
)


def extract_single_timepoint_volume(arr: np.ndarray, npy_channel: int) -> np.ndarray:
    """Extract one `(z, y, x)` probability volume from per-timepoint arrays."""
    if arr.ndim == NDIM_3D:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == NDIM_4D:
        if npy_channel < 0 or npy_channel >= arr.shape[0]:
            raise ValueError(
                f"Requested channel {npy_channel} but array has {arr.shape[0]} channels."
            )
        return arr[npy_channel].astype(np.float32, copy=False)
    raise ValueError(
        f"Per-timepoint input must be 3D or channel-first 4D, got shape {arr.shape}"
    )


def _load_array_from_path(path: Path) -> np.ndarray:
    """Load a single array from .npy or .npz. NPZ files use the first stored array."""
    data = np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        keys = list(data.files)
        if not keys:
            raise ValueError(f"NPZ file contains no arrays: {path}")
        return np.asarray(data[keys[0]])
    return np.asarray(data)


def load_time_series_from_files(paths: list[Path], npy_channel: int) -> np.ndarray:
    """Load and stack per-timepoint arrays into `(t, z, y, x)`."""
    volumes: list[np.ndarray] = []
    expected_shape: tuple[int, int, int] | None = None

    for path in paths:
        arr = load_segmentation_array(path)
        vol = extract_single_timepoint_volume(arr, npy_channel=npy_channel)
        if expected_shape is None:
            expected_shape = tuple(int(x) for x in vol.shape)
        elif tuple(vol.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, got {vol.shape} from {path}"
            )
        volumes.append(vol)

    if not volumes:
        raise ValueError("No input files provided.")
    return np.stack(volumes, axis=0).astype(np.float32, copy=False)


def discover_study_timepoints(
    input_dir: Path, study_id: str
) -> tuple[list[Path], list[int]]:
    """Discover and sort timepoint files for one study id.

    Expects layout: input_dir / [SITE] / [STUDY_ID] / images / *.npz
    SITE is parsed from study_id as the first underscore-separated component
    (e.g. ISPY2_202539 -> SITE=ISPY2, STUDY_ID=ISPY2_202539).
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    candidates = sorted(
        p
        for ext in (".npy", ".npz")
        for p in input_dir.rglob(f"*{study_id}*_vessel_segmentation{ext}")
    )
    if not candidates:
        raise ValueError(
            f"No candidate .npz files found for study_id='{study_id}' in {images_dir}"
        )

    patt = re.compile(
        rf"{re.escape(study_id)}_(\d{{4}})_vessel_segmentation\.(npy|npz)$",
        flags=re.IGNORECASE,
    )

    timepoint_pairs: list[tuple[int, Path]] = []
    for path in candidates:
        match = patt.search(path.name)
        if match is not None:
            timepoint_pairs.append((int(match.group(1)), path))

    if not timepoint_pairs:
        example_names = ", ".join(p.name for p in candidates[:5])
        raise ValueError(
            "Found candidate files but none matched the expected timepoint pattern "
            f"for study_id='{study_id}'. First candidates: {example_names}"
        )

    seen: dict[int, Path] = {}
    duplicates: list[str] = []
    for tp, path in sorted(timepoint_pairs, key=lambda x: (x[0], x[1].name)):
        if tp in seen:
            duplicates.append(f"{tp:04d}: {seen[tp].name} | {path.name}")
        else:
            seen[tp] = path

    if duplicates:
        dup_msg = "; ".join(duplicates[:5])
        raise ValueError(
            "Duplicate files found for one or more timepoints. "
            f"Please resolve duplicates. Examples: {dup_msg}"
        )

    ordered = sorted(seen.items(), key=lambda kv: kv[0])
    return [p for _, p in ordered], [tp for tp, _ in ordered]


def baseline_3d_mask(priority_zyx: np.ndarray, threshold_low: float) -> np.ndarray:
    """Run 3D skeleton extraction and return a binary skeleton mask."""
    edges = skeletonize3d(priority_zyx, threshold=threshold_low)
    return edges > 0


def largest_component_3d(mask_zyx: np.ndarray) -> np.ndarray:
    """Keep only the largest 26-connected 3D component."""
    from scipy import ndimage

    if not np.any(mask_zyx):
        return mask_zyx

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, n_comp = ndimage.label(mask_zyx.astype(np.uint8), structure=structure)
    if n_comp <= 1:
        return mask_zyx

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep_label = int(np.argmax(sizes))
    return labels == keep_label


def collapse_4d_to_exam_skeleton(
    mask_4d: np.ndarray,
    min_temporal_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse a 4D manifold to one 3D exam-level skeleton."""
    from scipy import ndimage

    if mask_4d.ndim != NDIM_4D:
        raise ValueError(f"Expected 4D mask (t,z,y,x), got shape {mask_4d.shape}")

    t_dim = int(mask_4d.shape[0])
    if min_temporal_support < 1 or min_temporal_support > t_dim:
        raise ValueError(
            f"min-temporal-support must be in [1, {t_dim}], got {min_temporal_support}"
        )

    support_count = np.count_nonzero(mask_4d, axis=0).astype(np.int32)
    support_mask = support_count >= min_temporal_support
    if not np.any(support_mask):
        raise ValueError(
            "4D collapse produced an empty support mask. "
            "Lower min-temporal-support or adjust pruning thresholds."
        )

    priority = ndimage.distance_transform_edt(support_mask).astype(
        np.float32, copy=False
    )
    exam_skeleton = skeletonize3d(priority, threshold=0.0) > 0
    exam_skeleton = largest_component_3d(exam_skeleton)
    if not np.any(exam_skeleton):
        raise ValueError(
            "Exam-level skeleton is empty after largest-component filtering."
        )

    return exam_skeleton, support_mask, support_count


def mask_to_edges_bitmask(mask_zyx: np.ndarray) -> np.ndarray:
    """Convert a binary skeleton mask to the 26-neighbor edge bitmask format."""
    if mask_zyx.ndim != NDIM_3D:
        raise ValueError(f"Expected 3D mask, got shape {mask_zyx.shape}")

    nodes = mask_zyx.astype(bool, copy=False)
    zdim, ydim, xdim = nodes.shape
    edges = np.zeros(nodes.shape, dtype=np.uint32)

    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                if not nodes[z, y, x]:
                    continue
                for b, (dz, dy, dx) in enumerate(_OFFSETS_3D):
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < zdim and 0 <= ny < ydim and 0 <= nx < xdim:
                        if nodes[nz, ny, nx]:
                            edges[z, y, x] |= np.uint32(1 << b)

    return edges


def build_morphometry_from_skeleton(
    skeleton_mask_zyx: np.ndarray,
    vessel_reference_zyx: np.ndarray,
    output_json_path: Path,
) -> dict[str, int]:
    """Build morphometry JSON from a skeleton mask and vessel reference volume."""
    if skeleton_mask_zyx.ndim != NDIM_3D:
        raise ValueError(f"Skeleton mask must be 3D, got {skeleton_mask_zyx.shape}")
    if vessel_reference_zyx.ndim != NDIM_3D:
        raise ValueError(
            f"Vessel reference must be 3D, got {vessel_reference_zyx.shape}"
        )
    if tuple(skeleton_mask_zyx.shape) != tuple(vessel_reference_zyx.shape):
        raise ValueError(
            "Shape mismatch between skeleton and vessel reference: "
            f"{skeleton_mask_zyx.shape} vs {vessel_reference_zyx.shape}"
        )

    edges = mask_to_edges_bitmask(skeleton_mask_zyx)
    segments = edges_to_segments(edges)
    if segments.size == 0:
        raise ValueError("Skeleton has zero segments; cannot compute morphometry.")

    graph = segments_to_graph(segments)
    if graph.number_of_nodes() == 0:
        raise ValueError("Skeleton graph has zero nodes; cannot compute morphometry.")

    radius_map = obtain_radius_map(vessel_reference_zyx, graph)
    segment_paths = extract_segments(graph)
    bifurcations = detect_bifurcations(graph)
    vessel_labels = assign_component_labels(graph)

    build_vessel_json(
        graph,
        vessel_labels,
        segment_paths,
        radius_map,
        bifurcations,
        output_path=output_json_path,
    )

    return {
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "segment_count": int(len(segment_paths)),
        "component_count": int(len(vessel_labels)),
    }


def process_3d_case(
    *,
    input_file: Path,
    output_dir: Path,
    threshold_low: float,
    npy_channel: int,
    force_skeleton: bool,
    force_features: bool,
) -> dict[str, object]:
    """Run 3D skeleton extraction + morphometry for one input file."""
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem
    skeleton_path = output_dir / f"{stem}_skeleton_3d_mask.npy"
    morphometry_path = output_dir / f"{stem}_morphometry.json"

    skeleton_took = 0.0
    features_took = 0.0

    arr = load_segmentation_array(input_file)
    priority_zyx = extract_single_timepoint_volume(arr, npy_channel=npy_channel)

    if skeleton_path.exists() and not force_skeleton:
        skeleton_mask = np.load(skeleton_path).astype(bool, copy=False)
        skeleton_status = "loaded_existing"
    else:
        t0 = time.perf_counter()
        skeleton_mask = baseline_3d_mask(priority_zyx, threshold_low=threshold_low)
        np.save(skeleton_path, skeleton_mask.astype(np.uint8))
        skeleton_took = float(time.perf_counter() - t0)
        skeleton_status = "computed"

    if morphometry_path.exists() and not force_features:
        feature_stats: dict[str, int] | None = None
        features_status = "loaded_existing"
    else:
        t1 = time.perf_counter()
        feature_stats = build_morphometry_from_skeleton(
            skeleton_mask_zyx=skeleton_mask,
            vessel_reference_zyx=priority_zyx,
            output_json_path=morphometry_path,
        )
        features_took = float(time.perf_counter() - t1)
        features_status = "computed"

    return {
        "mode": "3d",
        "input_file": str(input_file),
        "threshold_low": float(threshold_low),
        "npy_channel": int(npy_channel),
        "skeleton_status": skeleton_status,
        "features_status": features_status,
        "skeleton_voxels": int(np.count_nonzero(skeleton_mask)),
        "skeleton_path": str(skeleton_path),
        "morphometry_path": str(morphometry_path),
        "feature_stats": feature_stats,
        "timing_seconds": {
            "skeleton": skeleton_took,
            "features": features_took,
            "total": float(time.perf_counter() - start),
        },
    }


def process_4d_study(
    *,
    input_dir: Path,
    study_id: str,
    output_dir: Path,
    npy_channel: int,
    threshold_low: float,
    threshold_high: float | None,
    max_temporal_radius: int,
    min_voxels_per_timepoint: int,
    min_anchor_fraction: float,
    min_anchor_voxels: int,
    max_candidates: int | None,
    min_temporal_support: int,
    force_skeleton: bool,
    force_features: bool,
    save_center_manifold_mask: bool,
    verbose: bool = True,
) -> dict[str, object]:
    """Run 4D exam-level skeleton extraction + morphometry for one study."""
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton_path = output_dir / f"{study_id}_skeleton_4d_exam_mask.npy"
    support_path = output_dir / f"{study_id}_skeleton_4d_exam_support_mask.npy"
    manifold_path = output_dir / f"{study_id}_center_manifold_4d_mask.npy"
    morphometry_path = output_dir / f"{study_id}_morphometry.json"

    skeleton_took = 0.0
    features_took = 0.0

    discovered_files: list[Path] | None = None
    discovered_timepoints: list[int] | None = None

    if skeleton_path.exists() and support_path.exists() and not force_skeleton:
        skeleton_mask = np.load(skeleton_path).astype(bool, copy=False)
        support_mask = np.load(support_path).astype(bool, copy=False)
        retained_per_t: list[int] | None = None
        skeleton_status = "loaded_existing"
    else:
        t0 = time.perf_counter()
        discovered_files, discovered_timepoints = discover_study_timepoints(
            input_dir=input_dir,
            study_id=study_id,
        )
        priority_4d = load_time_series_from_files(
            discovered_files,
            npy_channel=npy_channel,
        )

        mask_4d = skeletonize4d(
            priority_4d,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            max_temporal_radius=max_temporal_radius,
            min_voxels_per_timepoint=min_voxels_per_timepoint,
            min_anchor_fraction=min_anchor_fraction,
            min_anchor_voxels=min_anchor_voxels,
            max_candidates=max_candidates,
            verbose=verbose,
        )
        skeleton_mask, support_mask, _ = collapse_4d_to_exam_skeleton(
            mask_4d,
            min_temporal_support=min_temporal_support,
        )

        np.save(skeleton_path, skeleton_mask.astype(np.uint8))
        np.save(support_path, support_mask.astype(np.uint8))
        if save_center_manifold_mask:
            np.save(manifold_path, mask_4d.astype(np.uint8))

        retained_per_t = [int(x) for x in np.count_nonzero(mask_4d, axis=(1, 2, 3))]
        skeleton_took = float(time.perf_counter() - t0)
        skeleton_status = "computed"

    if morphometry_path.exists() and not force_features:
        feature_stats: dict[str, int] | None = None
        features_status = "loaded_existing"
    else:
        t1 = time.perf_counter()
        feature_stats = build_morphometry_from_skeleton(
            skeleton_mask_zyx=skeleton_mask,
            vessel_reference_zyx=support_mask.astype(np.uint8, copy=False),
            output_json_path=morphometry_path,
        )
        features_took = float(time.perf_counter() - t1)
        features_status = "computed"

    return {
        "mode": "4d",
        "study_id": study_id,
        "input_dir": str(input_dir),
        "npy_channel": int(npy_channel),
        "threshold_low": float(threshold_low),
        "threshold_high": None if threshold_high is None else float(threshold_high),
        "max_temporal_radius": int(max_temporal_radius),
        "min_voxels_per_timepoint": int(min_voxels_per_timepoint),
        "min_anchor_fraction": float(min_anchor_fraction),
        "min_anchor_voxels": int(min_anchor_voxels),
        "max_candidates": None if max_candidates is None else int(max_candidates),
        "min_temporal_support": int(min_temporal_support),
        "skeleton_status": skeleton_status,
        "features_status": features_status,
        "skeleton_voxels": int(np.count_nonzero(skeleton_mask)),
        "support_voxels": int(np.count_nonzero(support_mask)),
        "skeleton_path": str(skeleton_path),
        "support_path": str(support_path),
        "manifold_path": str(manifold_path) if save_center_manifold_mask else None,
        "morphometry_path": str(morphometry_path),
        "feature_stats": feature_stats,
        "study_files": None
        if discovered_files is None
        else [str(p) for p in discovered_files],
        "study_timepoints": discovered_timepoints,
        "retained_per_timepoint_4d": retained_per_t,
        "timing_seconds": {
            "skeleton": skeleton_took,
            "features": features_took,
            "total": float(time.perf_counter() - start),
        },
    }
