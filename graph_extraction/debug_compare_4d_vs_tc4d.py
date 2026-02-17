"""Debug runtime for 4d-vs-tc4d comparison.

This module hosts debug-only compare helpers and CLI plumbing. Production
processing should depend on `graph_extraction.tc4d` only.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import numba as nb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from graph_extraction.core4d import (
    DEFAULT_SEGMENTATION_DIR,
    collapse_4d_to_exam_skeleton,
    discover_study_timepoints,
    largest_component_3d,
    load_time_series_from_files,
)
from graph_extraction.tc4d import (
    NDIM_3D,
    NDIM_4D,
    _run_graph_consensus_pipeline,
)
from graph_extraction.vessel_mip import (
    compute_hit_miss_vs_radiologist,
    render_vessel_coverage_mip,
)

_MP4_RENDER_CONTEXT: dict[str, Any] | None = None
NDIM_5D = 5
IO_CACHE_VERSION = "v2"
DEFAULT_IO_CACHE_DIR = Path(".cache/compare_4d")
DEFAULT_RADIOLOGIST_ANNOTATIONS_DIR = Path(
    "/net/projects2/vanguard/Duke-Breast-Cancer-MRI-Supplement-v3"
)
DEFAULT_TUMOR_MASK_DIR = Path(
    "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert"
)
COMPARE_4D_TC4D_VIZ_FLIP_SPEC = "z"
VIZ_FLIP_SPECS = ("none", "z", "y", "x", "zy", "zx", "yx", "zyx")
TUMOR_PERITUMOR_INNER_RADIUS_VOX = 0
TUMOR_PERITUMOR_OUTER_RADIUS_VOX = 10
BREAST_OVERLAY_MAX_POINTS = 160_000
TUMOR_OUTLINE_MAX_POINTS = 90_000

FOURD_BASELINE_PARAMS = {
    "threshold_low": 0.5,
    "threshold_high": 0.85,
    "max_temporal_radius": 1,
    "min_voxels_per_timepoint": 64,
    "min_anchor_fraction": 0.005,
    "min_anchor_voxels": 128,
}


def _cache_key(payload: dict[str, Any]) -> str:
    """Generate stable cache key for JSON-serializable payload."""

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _count_components(mask_zyx: np.ndarray) -> int:
    """Count 26-connected components in a 3D binary mask."""
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    _, n_comp = ndimage.label(mask_zyx.astype(np.uint8), structure=structure)
    return int(n_comp)


def _apply_flip_spec(mask_zyx: np.ndarray, flip_spec: str) -> np.ndarray:
    """Apply axis flips in `z/y/x` notation to a 3D mask."""
    spec = str(flip_spec).strip().lower()
    if spec in ("", "none"):
        return mask_zyx

    axis_by_char = {"z": 0, "y": 1, "x": 2}
    axes: list[int] = []
    seen: set[str] = set()
    for char in spec:
        if char in seen:
            continue
        seen.add(char)
        axis = axis_by_char.get(char)
        if axis is None:
            raise ValueError(
                f"Unsupported flip spec '{flip_spec}'. Supported: {list(VIZ_FLIP_SPECS)}"
            )
        axes.append(axis)
    if not axes:
        return mask_zyx
    return np.flip(mask_zyx, axis=tuple(axes))


def compute_overlap_metrics(
    baseline_4d_exam_zyx: np.ndarray,
    tc4d_exam_zyx: np.ndarray,
) -> dict[str, float | int]:
    """Compute overlap diagnostics between baseline 4d and tc4d exam masks."""
    baseline = np.asarray(baseline_4d_exam_zyx, dtype=bool)
    tc4d = np.asarray(tc4d_exam_zyx, dtype=bool)
    if baseline.shape != tc4d.shape:
        raise ValueError(
            "Overlap shape mismatch: "
            f"{tuple(baseline.shape)} vs {tuple(tc4d.shape)}"
        )

    intersection = int(np.count_nonzero(baseline & tc4d))
    union = int(np.count_nonzero(baseline | tc4d))
    baseline_voxels = int(np.count_nonzero(baseline))
    tc4d_voxels = int(np.count_nonzero(tc4d))
    sum_voxels = baseline_voxels + tc4d_voxels

    dice = (2.0 * intersection / sum_voxels) if sum_voxels > 0 else 0.0
    jaccard = (intersection / union) if union > 0 else 0.0
    baseline_overlap = (intersection / baseline_voxels) if baseline_voxels > 0 else 0.0
    tc4d_overlap = (intersection / tc4d_voxels) if tc4d_voxels > 0 else 0.0

    return {
        "intersection_voxels": intersection,
        "union_voxels": union,
        "dice": float(dice),
        "jaccard": float(jaccard),
        "baseline_4d_overlap_fraction": float(baseline_overlap),
        "tc4d_overlap_fraction": float(tc4d_overlap),
    }

def _git_stdout(project_root: Path, *git_args: str) -> str | None:
    """Return stripped stdout for a git command or ``None`` on failure."""
    try:
        proc = subprocess.run(
            ["git", *git_args],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    return value if value else None

def _collect_git_metadata(project_root: Path) -> dict[str, Any]:
    """Collect lightweight git provenance for run-to-commit traceability."""
    commit = _git_stdout(project_root, "rev-parse", "HEAD")
    commit_short = _git_stdout(project_root, "rev-parse", "--short=12", "HEAD")
    branch = _git_stdout(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty: bool | None = None
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = bool(proc.stdout.strip())
    except Exception:
        dirty = None
    return {
        "project_root": str(project_root),
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "dirty": dirty,
    }


def _cache_paths(
    cache_dir: Path,
    *,
    namespace: str,
    key: str,
) -> tuple[Path, Path]:
    """Return array/meta cache paths for one namespace/key pair."""
    ns_dir = cache_dir / namespace
    return ns_dir / f"{key}.npy", ns_dir / f"{key}.json"


def _cache_load_array(
    cache_dir: Path,
    *,
    namespace: str,
    key: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Load cached array + metadata; return (None, None) on cache miss/bad entry."""
    array_path, meta_path = _cache_paths(cache_dir, namespace=namespace, key=key)
    if not array_path.exists() or not meta_path.exists():
        return None, None

    try:
        arr = np.load(array_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return arr, meta
    except Exception:
        return None, None


def _cache_save_array(
    cache_dir: Path,
    *,
    namespace: str,
    key: str,
    arr: np.ndarray,
    meta: dict[str, Any],
) -> None:
    """Write cached array + metadata atomically."""
    array_path, meta_path = _cache_paths(cache_dir, namespace=namespace, key=key)
    array_path.parent.mkdir(parents=True, exist_ok=True)

    array_tmp = array_path.with_suffix(".npy.tmp")
    with array_tmp.open("wb") as f:
        np.save(f, np.asarray(arr), allow_pickle=False)
    os.replace(array_tmp, array_path)

    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(meta_tmp, meta_path)

def _load_time_series_from_single_npy(
    path: Path,
    layout: str,
    npy_channel: int,
) -> np.ndarray:
    """Load a single NPY into ``(t, z, y, x)`` based on explicit layout."""
    arr = np.load(path)

    if layout == "tzyx":
        if arr.ndim != NDIM_4D:
            raise ValueError(
                f"layout=tzyx expects 4D array, got shape {arr.shape} from {path}"
            )
        return arr.astype(np.float32, copy=False)

    if layout == "ctzyx":
        if arr.ndim != NDIM_5D:
            raise ValueError(
                f"layout=ctzyx expects 5D array, got shape {arr.shape} from {path}"
            )
        if npy_channel < 0 or npy_channel >= arr.shape[0]:
            raise ValueError(
                f"Requested channel {npy_channel} but array has {arr.shape[0]} channels."
            )
        return arr[npy_channel].astype(np.float32, copy=False)

    if layout == "tczyx":
        if arr.ndim != NDIM_5D:
            raise ValueError(
                f"layout=tczyx expects 5D array, got shape {arr.shape} from {path}"
            )
        if npy_channel < 0 or npy_channel >= arr.shape[1]:
            raise ValueError(
                f"Requested channel {npy_channel} but array has {arr.shape[1]} channels."
            )
        return arr[:, npy_channel].astype(np.float32, copy=False)

    raise ValueError(f"Unsupported layout: {layout}")

def _extract_xyz(mask_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return float32 XYZ coordinates for nonzero voxels in a 3D mask."""
    z, y, x = np.nonzero(mask_zyx)
    return (
        x.astype(np.float32, copy=False),
        y.astype(np.float32, copy=False),
        z.astype(np.float32, copy=False),
    )

def _subsample_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly subsample coordinate arrays to at most ``max_points``."""
    if max_points is None or max_points <= 0:
        return x, y, z
    n = x.size
    if n <= max_points:
        return x, y, z
    keep = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return x[keep], y[keep], z[keep]


def _prepare_breast_overlay_points(
    breast_mask_zyx: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Prepare sparse breast boundary points for translucent 3D overlays."""
    from scipy import ndimage

    if breast_mask_zyx is None:
        return None
    breast = np.asarray(breast_mask_zyx, dtype=bool)
    if breast.ndim != NDIM_3D or not np.any(breast):
        return None

    shell = breast & ~ndimage.binary_erosion(
        breast,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0,
    )
    if not np.any(shell):
        shell = breast
    bx, by, bz = _extract_xyz(shell)
    return _subsample_xyz(bx, by, bz, max_points=BREAST_OVERLAY_MAX_POINTS)


def _prepare_tumor_outline_points(
    tumor_mask_zyx: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Prepare sparse tumor-boundary points for 3D overlays."""
    from scipy import ndimage

    if tumor_mask_zyx is None:
        return None
    tumor = np.asarray(tumor_mask_zyx, dtype=bool)
    if tumor.ndim != NDIM_3D or not np.any(tumor):
        return None

    outline = tumor & ~ndimage.binary_erosion(
        tumor,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0,
    )
    if not np.any(outline):
        outline = tumor
    tx, ty, tz = _extract_xyz(outline)
    return _subsample_xyz(tx, ty, tz, max_points=TUMOR_OUTLINE_MAX_POINTS)

def _file_signature(path: Path) -> dict[str, Any]:
    """Return stable file signature fields used for cache invalidation."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def _attach_cache_diag(
    info: dict[str, Any],
    *,
    cache_enabled: bool,
    cache_dir: Path | None,
    cache_namespace: str,
    cache_key_value: str | None,
    cache_hit: bool,
) -> dict[str, Any]:
    """Attach standardized cache diagnostics to an info dictionary."""
    merged = dict(info)
    merged["cache"] = {
        "enabled": cache_enabled,
        "dir": None if cache_dir is None else str(cache_dir),
        "namespace": str(cache_namespace),
        "key": cache_key_value,
        "hit": cache_hit,
    }
    return merged

def _to_breast_mri_case_id(token: str) -> str | None:
    """Convert common study-id tokens into a ``Breast_MRI_XXX`` case id."""
    clean = token.strip()
    if not clean:
        return None

    match_breast = re.search(r"Breast_MRI_(\d+)", clean, flags=re.IGNORECASE)
    if match_breast is not None:
        return f"Breast_MRI_{int(match_breast.group(1)):03d}"

    match_duke = re.search(r"DUKE_(\d+)", clean, flags=re.IGNORECASE)
    if match_duke is not None:
        return f"Breast_MRI_{int(match_duke.group(1)):03d}"

    match_suffix = re.search(r"(\d+)$", clean)
    if match_suffix is None:
        return None

    digits = match_suffix.group(1)
    if len(digits) <= 3:
        return f"Breast_MRI_{int(digits):03d}"

    # Last-3 fallback for ids like ISPY2_202539 -> Breast_MRI_539.
    return f"Breast_MRI_{int(digits[-3:]):03d}"

def _resolve_radiologist_nrrd_root(annotations_dir: Path) -> Path:
    """Resolve `Segmentation_Masks_NRRD` root from a provided base path."""
    candidate = annotations_dir / "Segmentation_Masks_NRRD"
    if candidate.exists():
        return candidate
    return annotations_dir

def _resolve_radiologist_seg_path(
    *,
    nrrd_root: Path,
    case_id: str,
) -> Path | None:
    """Resolve the Dense+Vessels segmentation path for one Breast_MRI case."""
    case_dir = nrrd_root / case_id
    if not case_dir.exists():
        return None

    matches = sorted(case_dir.glob("Segmentation_*_Dense_and_Vessels.seg.nrrd"))
    if not matches:
        return None
    return matches[0]

def _resolve_breast_seg_path(
    *,
    nrrd_root: Path,
    case_id: str,
) -> Path | None:
    """Resolve the Breast segmentation path for one Breast_MRI case."""
    case_dir = nrrd_root / case_id
    if not case_dir.exists():
        return None

    matches = sorted(case_dir.glob("Segmentation_*_Breast.seg.nrrd"))
    if not matches:
        return None
    return matches[0]

def _parse_seg_nrrd_segments(path: Path) -> dict[str, dict[str, int | None]]:
    """Parse Slicer ``.seg.nrrd`` segment metadata keyed by segment name."""
    header_chunks: list[bytes] = []
    with path.open("rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            header_chunks.append(chunk)
            joined = b"".join(header_chunks)
            if b"\n\n" in joined or b"\r\n\r\n" in joined:
                break

    joined = b"".join(header_chunks)
    if b"\n\n" in joined:
        header_bytes = joined.split(b"\n\n", maxsplit=1)[0]
    elif b"\r\n\r\n" in joined:
        header_bytes = joined.split(b"\r\n\r\n", maxsplit=1)[0]
    else:
        header_bytes = joined
    header_text = header_bytes.decode("utf-8", errors="ignore")

    names_by_idx: dict[str, str] = {}
    labels_by_idx: dict[str, int] = {}
    layers_by_idx: dict[str, int] = {}
    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        name_match = re.match(r"^Segment(\d+)_Name:=(.+)$", line)
        if name_match is not None:
            names_by_idx[name_match.group(1)] = name_match.group(2).strip()
            continue

        label_match = re.match(r"^Segment(\d+)_LabelValue:=(-?\d+)$", line)
        if label_match is not None:
            labels_by_idx[label_match.group(1)] = int(label_match.group(2))
            continue

        layer_match = re.match(r"^Segment(\d+)_Layer:=(-?\d+)$", line)
        if layer_match is not None:
            layers_by_idx[layer_match.group(1)] = int(layer_match.group(2))

    segments_by_name: dict[str, dict[str, int | None]] = {}
    for idx, name in names_by_idx.items():
        segments_by_name[name] = {
            "label_value": labels_by_idx.get(idx),
            "layer": layers_by_idx.get(idx),
        }
    return segments_by_name

def _load_binary_mask_nrrd(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    label_name: str,
    threshold: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a binary NRRD segmentation and align to expected ``(z,y,x)``."""
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(
        np.float32, copy=False
    )
    expected_shape_tuple = tuple(expected_shape_zyx)
    if arr.ndim != NDIM_3D:
        raise ValueError(f"{label_name} NRRD must be 3D, got shape {arr.shape} from {path}")

    candidates = (
        ("zyx", arr),
        ("yxz", np.transpose(arr, (1, 2, 0))),
        ("xyz", np.transpose(arr, (2, 1, 0))),
    )
    selected: np.ndarray | None = None
    selected_layout: str | None = None
    for layout_name, candidate in candidates:
        if tuple(candidate.shape) == expected_shape_tuple:
            selected = candidate
            selected_layout = layout_name
            break

    if selected is None:
        raise ValueError(
            f"{label_name} mask shape mismatch. "
            f"Expected {expected_shape_tuple}, got {arr.shape} (and common transposes) "
            f"from {path}"
        )

    mask = (selected > float(threshold)).astype(bool, copy=False)
    if not np.any(mask):
        raise ValueError(f"{label_name} mask is empty in {path}")

    return mask, {
        "path": str(path),
        "layout": selected_layout,
        "voxels": int(np.count_nonzero(mask)),
    }

def _load_radiologist_vessel_mask_nrrd(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load radiologist vessel annotation from ``Dense_and_Vessels.seg.nrrd``."""
    import SimpleITK as sitk

    segments_by_name = _parse_seg_nrrd_segments(path)
    vessel_name: str | None = None
    vessel_label: int | None = None
    vessel_layer: int | None = None
    for name, meta in segments_by_name.items():
        if "vessel" in name.lower():
            vessel_name = name
            raw_label = meta.get("label_value")
            raw_layer = meta.get("layer")
            vessel_label = None if raw_label is None else int(raw_label)
            vessel_layer = None if raw_layer is None else int(raw_layer)
            vessel_layer_index = vessel_layer
            break

    if vessel_name is None:
        raise ValueError(
            f"Could not find a 'vessel' segment in {path}. "
            f"Found segments: {sorted(segments_by_name.keys())}"
        )

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(
        np.int16, copy=False
    )
    expected_shape_tuple = tuple(expected_shape_zyx)
    selected_arr_3d: np.ndarray
    selected_layer_layout: str | None = None
    selected_layer_index: int | None = None
    selected_layer_via_header: bool = False

    if arr.ndim == NDIM_3D:
        selected_arr_3d = arr
    elif arr.ndim == NDIM_4D:
        if arr.shape[-1] <= 32:
            layer_count = arr.shape[-1]
            selected_layer_index = (
                vessel_layer_index
                if vessel_layer_index is not None and 0 <= vessel_layer_index < layer_count
                else 0
            )
            selected_arr_3d = arr[..., selected_layer_index]
            selected_layer_layout = "layer_last"
        elif arr.shape[0] <= 32:
            layer_count = arr.shape[0]
            selected_layer_index = (
                vessel_layer_index
                if vessel_layer_index is not None and 0 <= vessel_layer_index < layer_count
                else 0
            )
            selected_arr_3d = arr[selected_layer_index, ...]
            selected_layer_layout = "layer_first"
        else:
            raise ValueError(
                "Radiologist seg NRRD 4D shape unsupported for layered decoding: "
                f"{arr.shape} from {path}"
            )
        selected_layer_via_header = (
            vessel_layer_index is not None and vessel_layer_index == selected_layer_index
        )
    else:
        raise ValueError(
            f"Radiologist seg NRRD must be 3D or layered 4D, got shape {arr.shape} from {path}"
        )

    candidates = (
        ("zyx", selected_arr_3d),
        ("yxz", np.transpose(selected_arr_3d, (1, 2, 0))),
        ("xyz", np.transpose(selected_arr_3d, (2, 1, 0))),
    )
    selected: np.ndarray | None = None
    selected_layout: str | None = None
    for layout_name, candidate in candidates:
        if candidate.shape == expected_shape_tuple:
            selected = candidate
            selected_layout = layout_name
            break

    if selected is None:
        raise ValueError(
            "Radiologist mask shape mismatch. "
            f"Expected {expected_shape_tuple}, got {selected_arr_3d.shape} "
            f"(source raw shape {arr.shape}; and common transposes) "
            f"from {path}"
        )

    if vessel_label is None:
        vessel_mask = (selected > 0).astype(bool, copy=False)
    else:
        vessel_mask = (selected == int(vessel_label)).astype(bool, copy=False)
        if not np.any(vessel_mask):
            vessel_mask = (selected > 0).astype(bool, copy=False)
    if not np.any(vessel_mask):
        raise ValueError(
            f"Radiologist vessel segment '{vessel_name}' is empty in {path}"
        )

    return vessel_mask, {
        "path": str(path),
        "segment_name": vessel_name,
        "segment_label_value": vessel_label,
        "segment_layer": selected_layer_index,
        "segment_layer_from_header": bool(selected_layer_via_header),
        "segment_layer_layout": selected_layer_layout,
        "layout": selected_layout,
        "voxels": int(np.count_nonzero(vessel_mask)),
    }

def _load_binary_mask_nrrd_cached(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    label_name: str,
    threshold: float,
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load binary NRRD mask with optional on-disk cache."""
    cache_enabled = use_cache and cache_dir is not None
    namespace = "binary_mask_nrrd"
    payload = {
        "version": IO_CACHE_VERSION,
        "loader": "binary_mask_nrrd",
        "label_name": str(label_name),
        "threshold": float(threshold),
        "expected_shape_zyx": list(expected_shape_zyx),
        "source": _file_signature(path),
    }
    key = _cache_key(payload) if cache_enabled else None

    if cache_enabled and key is not None:
        cached_arr, cached_meta = _cache_load_array(
            cache_dir,
            namespace=namespace,
            key=key,
        )
        if (
            cached_arr is not None
            and cached_meta is not None
            and cached_arr.shape == expected_shape_zyx
        ):
            mask = cached_arr.astype(bool, copy=False)
            info = dict(cached_meta.get("info", {}))
            return mask, _attach_cache_diag(
                info,
                cache_enabled=True,
                cache_dir=cache_dir,
                cache_namespace=namespace,
                cache_key_value=key,
                cache_hit=True,
            )

    mask, info = _load_binary_mask_nrrd(
        path,
        expected_shape_zyx=expected_shape_zyx,
        label_name=label_name,
        threshold=threshold,
    )
    info_with_cache = _attach_cache_diag(
        info,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir if cache_enabled else None,
        cache_namespace=namespace,
        cache_key_value=key,
        cache_hit=False,
    )
    if cache_enabled and key is not None:
        _cache_save_array(
            cache_dir,
            namespace=namespace,
            key=key,
            arr=mask.astype(np.uint8, copy=False),
            meta={"payload": payload, "info": info},
        )
    return mask, info_with_cache


def _load_radiologist_vessel_mask_nrrd_cached(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load radiologist vessel NRRD with optional on-disk cache."""
    cache_enabled = use_cache and cache_dir is not None
    namespace = "radiologist_vessel_mask_nrrd"
    payload = {
        "version": IO_CACHE_VERSION,
        "loader": "radiologist_vessel_mask_nrrd",
        "expected_shape_zyx": list(expected_shape_zyx),
        "source": _file_signature(path),
    }
    key = _cache_key(payload) if cache_enabled else None

    if cache_enabled and key is not None:
        cached_arr, cached_meta = _cache_load_array(
            cache_dir,
            namespace=namespace,
            key=key,
        )
        if (
            cached_arr is not None
            and cached_meta is not None
            and cached_arr.shape == expected_shape_zyx
        ):
            mask = cached_arr.astype(bool, copy=False)
            info = dict(cached_meta.get("info", {}))
            return mask, _attach_cache_diag(
                info,
                cache_enabled=True,
                cache_dir=cache_dir,
                cache_namespace=namespace,
                cache_key_value=key,
                cache_hit=True,
            )

    mask, info = _load_radiologist_vessel_mask_nrrd(
        path,
        expected_shape_zyx=expected_shape_zyx,
    )
    info_with_cache = _attach_cache_diag(
        info,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir if cache_enabled else None,
        cache_namespace=namespace,
        cache_key_value=key,
        cache_hit=False,
    )
    if cache_enabled and key is not None:
        _cache_save_array(
            cache_dir,
            namespace=namespace,
            key=key,
            arr=mask.astype(np.uint8, copy=False),
            meta={"payload": payload, "info": info},
        )
    return mask, info_with_cache


def _choose_flip_by_containment(
    candidate_mask_zyx: np.ndarray,
    reference_mask_zyx: np.ndarray,
) -> dict[str, Any]:
    """Select flip that maximizes fraction of candidate voxels inside reference."""
    cand = np.asarray(candidate_mask_zyx, dtype=bool)
    ref = np.asarray(reference_mask_zyx, dtype=bool)
    if cand.shape != ref.shape:
        raise ValueError(
            "Containment flip shape mismatch: "
            f"{tuple(cand.shape)} vs {tuple(ref.shape)}"
        )
    cand_voxels = int(np.count_nonzero(cand))
    if cand_voxels == 0:
        return {
            "best_flip_spec": "none",
            "best_inside_ratio": 0.0,
            "best_inside_voxels": 0,
            "all": [
                {
                    "flip_spec": "none",
                    "inside_ratio": 0.0,
                    "inside_voxels": 0,
                    "candidate_voxels": 0,
                }
            ],
        }

    rows: list[dict[str, Any]] = []
    for spec in VIZ_FLIP_SPECS:
        flipped = _apply_flip_spec(cand, spec)
        flip_key = str(spec)
        inside_voxels = int(np.count_nonzero(flipped & ref))
        inside_ratio = inside_voxels / cand_voxels
        rows.append(
            {
                "flip_spec": flip_key,
                "inside_ratio": inside_ratio,
                "inside_voxels": inside_voxels,
                "candidate_voxels": cand_voxels,
            }
        )
    # Prefer higher containment, then more inside voxels, then simpler/no flip.
    rows_sorted = sorted(
        rows,
        key=lambda d: (
            d["inside_ratio"],
            d["inside_voxels"],
            1 if d["flip_spec"] == "none" else 0,
        ),
        reverse=True,
    )
    best = rows_sorted[0]
    return {
        "best_flip_spec": best["flip_spec"],
        "best_inside_ratio": best["inside_ratio"],
        "best_inside_voxels": best["inside_voxels"],
        "all": rows_sorted,
    }


def _compute_radiologist_coverage_metrics(
    *,
    baseline_4d_exam_zyx: np.ndarray,
    baseline_4d_support_zyx: np.ndarray,
    tc4d_exam_zyx: np.ndarray,
    tc4d_support_zyx: np.ndarray,
    radiologist_mask_zyx: np.ndarray | None,
    alignment_flip_to_model: str,
    alignment_diagnostics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compute aligned radiologist hit/miss metrics for 4d/tc4d masks."""
    if radiologist_mask_zyx is None:
        return None

    def _metrics_for(pred_mask_zyx: np.ndarray) -> dict[str, float | int]:
        pred_voxels = int(np.count_nonzero(pred_mask_zyx))
        metrics = compute_hit_miss_vs_radiologist(pred_mask_zyx, radiologist_mask_zyx)
        if metrics is None:
            return {
                "pred_voxels": pred_voxels,
                "radiologist_voxels": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "miss_rate": 0.0,
                "precision_like": 0.0,
            }
        hits = int(metrics["hits"])
        return {
            "pred_voxels": pred_voxels,
            "radiologist_voxels": int(metrics["radiologist_voxels"]),
            "hits": hits,
            "misses": metrics["misses"],
            "hit_rate": metrics["hit_rate"],
            "miss_rate": metrics["miss_rate"],
            "precision_like": hits / max(pred_voxels, 1),
        }

    return {
        "alignment_flip_to_model": str(alignment_flip_to_model),
        "alignment_reference": "4d_or_tc4d_exam_union",
        "alignment_diagnostics": alignment_diagnostics,
        "4d_exam": _metrics_for(baseline_4d_exam_zyx),
        "4d_support": _metrics_for(baseline_4d_support_zyx),
        "tc4d_exam": _metrics_for(tc4d_exam_zyx),
        "tc4d_support": _metrics_for(tc4d_support_zyx),
    }


EXPECTED_DIMENSIONS_4D = 4
SPATIAL_NEIGHBOR_RADIUS_SQ = 3  # full 26-neighborhood in 3D when dt == 0
DIRECT_NEIGHBOR_COUNT = 2


def _build_offsets_4d(max_temporal_radius: int) -> np.ndarray:
    """Build 4D neighbor offsets (dt, dz, dy, dx)."""
    if max_temporal_radius < 0:
        raise ValueError("max_temporal_radius must be >= 0")

    offsets: list[tuple[int, int, int, int]] = []
    temporal_radius_sq = max_temporal_radius * max_temporal_radius
    max_spatial_dist_sq = SPATIAL_NEIGHBOR_RADIUS_SQ

    for dt in (-1, 0, 1):
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dt == 0 and dz == 0 and dy == 0 and dx == 0:
                        continue
                    spatial_dist_sq = dz * dz + dy * dy + dx * dx
                    include_dt0 = dt == 0 and 0 < spatial_dist_sq <= max_spatial_dist_sq
                    include_dt = dt != 0 and spatial_dist_sq <= temporal_radius_sq
                    if include_dt0 or include_dt:
                        offsets.append((dt, dz, dy, dx))

    return np.asarray(offsets, dtype=np.int32)


@nb.njit
def _queue_push_4d(
    q_t: np.ndarray,
    q_z: np.ndarray,
    q_y: np.ndarray,
    q_x: np.ndarray,
    ss: np.ndarray,
    t: int,
    z: int,
    y: int,
    x: int,
) -> None:
    idx = ss[1]
    q_t[idx] = t
    q_z[idx] = z
    q_y[idx] = y
    q_x[idx] = x
    idx += 1
    if idx == q_t.shape[0]:
        idx = 0
    ss[1] = idx


@nb.njit
def _queue_pop_4d(
    q_t: np.ndarray,
    q_z: np.ndarray,
    q_y: np.ndarray,
    q_x: np.ndarray,
    ss: np.ndarray,
) -> tuple[int, int, int, int]:
    idx = ss[0]
    t = q_t[idx]
    z = q_z[idx]
    y = q_y[idx]
    x = q_x[idx]
    idx += 1
    if idx == q_t.shape[0]:
        idx = 0
    ss[0] = idx
    return t, z, y, x


@nb.njit
def _queue_empty(ss: np.ndarray) -> bool:
    return ss[0] == ss[1]


@nb.njit
def _queue_clear(ss: np.ndarray) -> None:
    ss[0] = ss[1]


@nb.njit
def _collect_neighbors_4d(
    nodes: np.ndarray,
    offsets: np.ndarray,
    t: int,
    z: int,
    y: int,
    x: int,
    n_t: np.ndarray,
    n_z: np.ndarray,
    n_y: np.ndarray,
    n_x: np.ndarray,
) -> int:
    tt, zz, yy, xx = nodes.shape
    n_neighbors = 0
    for b in range(offsets.shape[0]):
        nt = t + offsets[b, 0]
        nz = z + offsets[b, 1]
        ny = y + offsets[b, 2]
        nx = x + offsets[b, 3]
        if nt < 0 or nt >= tt:
            continue
        if nz < 0 or nz >= zz:
            continue
        if ny < 0 or ny >= yy:
            continue
        if nx < 0 or nx >= xx:
            continue
        if nodes[nt, nz, ny, nx]:
            n_t[n_neighbors] = nt
            n_z[n_neighbors] = nz
            n_y[n_neighbors] = ny
            n_x[n_neighbors] = nx
            n_neighbors += 1
    return n_neighbors


@nb.njit
def _offset_exists(offsets: np.ndarray, dt: int, dz: int, dy: int, dx: int) -> bool:
    for b in range(offsets.shape[0]):
        if (
            offsets[b, 0] == dt
            and offsets[b, 1] == dz
            and offsets[b, 2] == dy
            and offsets[b, 3] == dx
        ):
            return True
    return False


@nb.njit
def _neighbors_connected_after_removal(
    nodes: np.ndarray,
    offsets: np.ndarray,
    n_neighbors: int,
    n_t: np.ndarray,
    n_z: np.ndarray,
    n_y: np.ndarray,
    n_x: np.ndarray,
    touched: np.ndarray,
    stamp: np.int32,
    q_t: np.ndarray,
    q_z: np.ndarray,
    q_y: np.ndarray,
    q_x: np.ndarray,
    qss: np.ndarray,
    target_found: np.ndarray,
) -> tuple[bool, np.int32]:
    # Fast path: with exactly two neighbors, a direct edge guarantees
    # connectivity and avoids a full BFS.
    if n_neighbors == DIRECT_NEIGHBOR_COUNT:
        dt = n_t[1] - n_t[0]
        dz = n_z[1] - n_z[0]
        dy = n_y[1] - n_y[0]
        dx = n_x[1] - n_x[0]
        if _offset_exists(offsets, dt, dz, dy, dx):
            return True, stamp

    stamp = np.int32(stamp + 1)
    _queue_clear(qss)
    if n_neighbors > 1:
        target_found[1:n_neighbors] = 0
    remaining = n_neighbors - 1

    seed_t = n_t[0]
    seed_z = n_z[0]
    seed_y = n_y[0]
    seed_x = n_x[0]
    touched[seed_t, seed_z, seed_y, seed_x] = stamp
    _queue_push_4d(q_t, q_z, q_y, q_x, qss, seed_t, seed_z, seed_y, seed_x)

    tt, zz, yy, xx = nodes.shape
    while not _queue_empty(qss):
        ct, cz, cy, cx = _queue_pop_4d(q_t, q_z, q_y, q_x, qss)
        for b in range(offsets.shape[0]):
            nt = ct + offsets[b, 0]
            nz = cz + offsets[b, 1]
            ny = cy + offsets[b, 2]
            nx = cx + offsets[b, 3]
            if nt < 0 or nt >= tt:
                continue
            if nz < 0 or nz >= zz:
                continue
            if ny < 0 or ny >= yy:
                continue
            if nx < 0 or nx >= xx:
                continue
            if not nodes[nt, nz, ny, nx]:
                continue
            if touched[nt, nz, ny, nx] == stamp:
                continue
            touched[nt, nz, ny, nx] = stamp
            for ti in range(1, n_neighbors):
                if target_found[ti] == 1:
                    continue
                if nt == n_t[ti] and nz == n_z[ti] and ny == n_y[ti] and nx == n_x[ti]:
                    target_found[ti] = 1
                    remaining -= 1
                    break
            if remaining == 0:
                return True, stamp
            _queue_push_4d(q_t, q_z, q_y, q_x, qss, nt, nz, ny, nx)

    return False, stamp


@nb.njit
def _skeletonize4d_impl(
    nodes: np.ndarray,
    anchors: np.ndarray,
    offsets: np.ndarray,
    cand_t: np.ndarray,
    cand_z: np.ndarray,
    cand_y: np.ndarray,
    cand_x: np.ndarray,
    order: np.ndarray,
    min_voxels_per_timepoint: int,
    counts_per_t: np.ndarray,
) -> tuple[int, int, int]:
    removed = 0
    restored = 0
    skipped_min = 0

    n_active = int(np.count_nonzero(nodes))
    if n_active == 0:
        return removed, restored, skipped_min

    # BFS scratch structures.
    q_t = np.empty(n_active, dtype=np.int32)
    q_z = np.empty(n_active, dtype=np.int32)
    q_y = np.empty(n_active, dtype=np.int32)
    q_x = np.empty(n_active, dtype=np.int32)
    qss = np.zeros(2, dtype=np.int64)
    touched = np.zeros(nodes.shape, dtype=np.int32)
    stamp = np.int32(1)

    max_neighbors = offsets.shape[0]
    n_t = np.empty(max_neighbors, dtype=np.int32)
    n_z = np.empty(max_neighbors, dtype=np.int32)
    n_y = np.empty(max_neighbors, dtype=np.int32)
    n_x = np.empty(max_neighbors, dtype=np.int32)
    target_found = np.zeros(max_neighbors, dtype=np.uint8)

    for k in range(order.shape[0]):
        idx = order[k]
        t = cand_t[idx]
        z = cand_z[idx]
        y = cand_y[idx]
        x = cand_x[idx]

        if not nodes[t, z, y, x]:
            continue
        if anchors[t, z, y, x]:
            continue
        if counts_per_t[t] <= min_voxels_per_timepoint:
            skipped_min += 1
            continue

        n_neighbors = _collect_neighbors_4d(
            nodes, offsets, t, z, y, x, n_t, n_z, n_y, n_x
        )
        if n_neighbors <= 1:
            nodes[t, z, y, x] = False
            counts_per_t[t] -= 1
            removed += 1
            continue

        nodes[t, z, y, x] = False
        connected, stamp = _neighbors_connected_after_removal(
            nodes,
            offsets,
            n_neighbors,
            n_t,
            n_z,
            n_y,
            n_x,
            touched,
            stamp,
            q_t,
            q_z,
            q_y,
            q_x,
            qss,
            target_found,
        )
        if connected:
            counts_per_t[t] -= 1
            removed += 1
        else:
            nodes[t, z, y, x] = True
            restored += 1

    return removed, restored, skipped_min


def _compute_anchor_mask(
    priority: np.ndarray,
    nodes: np.ndarray,
    threshold_high: float | None,
    min_anchor_fraction: float,
    min_anchor_voxels: int,
) -> np.ndarray:
    """Compute undeletable anchor voxels."""
    if min_anchor_fraction < 0:
        raise ValueError("min_anchor_fraction must be >= 0")
    if min_anchor_voxels < 0:
        raise ValueError("min_anchor_voxels must be >= 0")

    if threshold_high is None:
        anchors = np.zeros_like(nodes, dtype=bool)
    else:
        anchors = (priority >= threshold_high) & nodes
    if min_anchor_fraction <= 0.0 and min_anchor_voxels == 0:
        return anchors

    t_dim = priority.shape[0]
    for t in range(t_dim):
        active_t = nodes[t]
        n_active = int(np.count_nonzero(active_t))
        if n_active == 0:
            continue

        target = max(int(np.ceil(min_anchor_fraction * n_active)), min_anchor_voxels)
        target = min(target, n_active)
        if target <= 0:
            continue

        vals = priority[t][active_t]
        kth = n_active - target
        threshold_t = np.partition(vals, kth)[kth]
        anchors[t] |= (priority[t] >= threshold_t) & active_t

    return anchors


def skeletonize4d(
    priority: np.ndarray,
    threshold_low: float,
    *,
    threshold_high: float | None = None,
    max_temporal_radius: int = 1,
    min_voxels_per_timepoint: int = 1,
    min_anchor_fraction: float = 0.005,
    min_anchor_voxels: int = 128,
    max_candidates: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Extract a 4D center-manifold using articulation-preserving thinning."""
    if priority.ndim != EXPECTED_DIMENSIONS_4D:
        raise ValueError(
            f"`priority` must be {EXPECTED_DIMENSIONS_4D}D (t,z,y,x), got {priority.ndim}D"
        )
    if min_voxels_per_timepoint < 0:
        raise ValueError("min_voxels_per_timepoint must be >= 0")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be > 0 when provided")

    priority_f = priority.astype(np.float32, copy=False)
    nodes = priority_f >= threshold_low
    if not np.any(nodes):
        return np.zeros_like(nodes, dtype=bool)

    offsets = _build_offsets_4d(max_temporal_radius=max_temporal_radius)
    anchors = _compute_anchor_mask(
        priority_f,
        nodes,
        threshold_high,
        min_anchor_fraction=min_anchor_fraction,
        min_anchor_voxels=min_anchor_voxels,
    )

    cand_t, cand_z, cand_y, cand_x = np.nonzero(nodes & ~anchors)
    order = np.argsort(priority_f[cand_t, cand_z, cand_y, cand_x], kind="stable")
    if max_candidates is not None and len(order) > max_candidates:
        order = order[:max_candidates]

    cand_t = cand_t.astype(np.int32, copy=False)
    cand_z = cand_z.astype(np.int32, copy=False)
    cand_y = cand_y.astype(np.int32, copy=False)
    cand_x = cand_x.astype(np.int32, copy=False)
    order = order.astype(np.int64, copy=False)

    counts_per_t = np.count_nonzero(nodes, axis=(1, 2, 3)).astype(np.int64)
    removed, restored, skipped_min = _skeletonize4d_impl(
        nodes=nodes,
        anchors=anchors,
        offsets=offsets,
        cand_t=cand_t,
        cand_z=cand_z,
        cand_y=cand_y,
        cand_x=cand_x,
        order=order,
        min_voxels_per_timepoint=min_voxels_per_timepoint,
        counts_per_t=counts_per_t,
    )

    if verbose:
        total = int(np.count_nonzero(nodes))
        per_t = counts_per_t.tolist()
        print(
            "[skeleton4d] done: "
            f"removed={removed}, restored={restored}, skipped_min={skipped_min}, "
            f"retained_total={total}, retained_per_timepoint_diag={per_t}"
        )

    return nodes


def _run_4d_pipeline(
    priority_4d: np.ndarray,
) -> dict[str, Any]:
    """Run one 4D pipeline and return center-manifold and exam-level outputs."""
    active_min_temporal_support = 2 if priority_4d.shape[0] > 1 else 1
    baseline_4d_params = {
        "threshold_low": FOURD_BASELINE_PARAMS["threshold_low"],
        "threshold_high": FOURD_BASELINE_PARAMS["threshold_high"],
        "max_temporal_radius": FOURD_BASELINE_PARAMS["max_temporal_radius"],
        "min_voxels_per_timepoint": FOURD_BASELINE_PARAMS["min_voxels_per_timepoint"],
        "min_anchor_fraction": FOURD_BASELINE_PARAMS["min_anchor_fraction"],
        "min_anchor_voxels": FOURD_BASELINE_PARAMS["min_anchor_voxels"],
    }
    print(f"[4d] params={json.dumps(baseline_4d_params, sort_keys=True)}")

    t_start = time.perf_counter()
    mask_4d = skeletonize4d(
        priority_4d,
        **baseline_4d_params,
        verbose=True,
    )
    manifold_seconds = float(time.perf_counter() - t_start)

    t_collapse = time.perf_counter()
    print(f"[4d] auto_temporal_support={active_min_temporal_support}")
    selected_trial: tuple[int, np.ndarray, np.ndarray] | None = None
    for collapse_support in (
        (active_min_temporal_support,)
        if active_min_temporal_support == 1
        else (active_min_temporal_support, 1)
    ):
        try:
            exam_candidate, support_candidate, _ = collapse_4d_to_exam_skeleton(
                mask_4d,
                min_temporal_support=collapse_support,
            )
            if np.any(exam_candidate):
                selected_trial = (collapse_support, exam_candidate, support_candidate)
                break
        except ValueError as exc:
            print(
                "[4d] collapse trial error: "
                f"min_temporal_support={collapse_support}, "
                f"error={str(exc)}"
            )
            continue

    if selected_trial is None:
        support_mask = np.count_nonzero(mask_4d, axis=0) >= 1
        exam_mask = largest_component_3d(support_mask)
        if not np.any(exam_mask):
            raise ValueError("4d collapse failed for all support trials.")
        effective_support = 1
    else:
        effective_support, exam_mask, support_mask = selected_trial
    collapse_seconds = float(time.perf_counter() - t_collapse)
    retained_per_t = np.count_nonzero(mask_4d, axis=(1, 2, 3))
    print(
        "[4d] retained_voxels_per_timepoint="
        f"{retained_per_t.tolist()}"
    )

    return {
        "mask_4d": mask_4d,
        "exam_mask": exam_mask,
        "support_mask": support_mask,
        "effective_min_temporal_support": effective_support,
        "manifold_seconds": manifold_seconds,
        "collapse_seconds": collapse_seconds,
    }

def _compute_peritumoral_shell_mask(
    tumor_mask_zyx: np.ndarray | None,
    *,
    breast_mask_zyx: np.ndarray | None = None,
    inner_radius_vox: int = TUMOR_PERITUMOR_INNER_RADIUS_VOX,
    outer_radius_vox: int = TUMOR_PERITUMOR_OUTER_RADIUS_VOX,
) -> np.ndarray | None:
    """Compute a fixed-width peritumoral shell in voxel units."""
    from scipy import ndimage

    if tumor_mask_zyx is None:
        return None
    tumor = np.asarray(tumor_mask_zyx, dtype=bool)
    if tumor.ndim != NDIM_3D:
        raise ValueError(f"Tumor mask must be 3D, got {tumor.shape}")
    if not np.any(tumor):
        return None

    outer = int(max(0, outer_radius_vox))
    inner = int(max(0, min(inner_radius_vox, outer)))
    if outer <= 0:
        return np.zeros_like(tumor, dtype=bool)

    tumor_distance = ndimage.distance_transform_edt(~tumor)
    shell = (tumor_distance > float(inner)) & (tumor_distance <= float(outer)) & ~tumor

    if breast_mask_zyx is not None:
        breast = np.asarray(breast_mask_zyx, dtype=bool)
        if breast.shape != tumor.shape:
            raise ValueError(
                "Breast/tumor mask shape mismatch for peritumoral shell: "
                f"{breast.shape} vs {tumor.shape}"
            )
        shell &= breast
    return shell.astype(bool, copy=False)

def _build_case_title(
    *, case_label: str, title_prefix: str = "4D vs"
) -> str:
    """Build a consistent case-aware compare title."""
    normalized_name = "temporal consensus 4D"
    if title_prefix == "4D vs":
        return f"4D vs {normalized_name} | case: {case_label}"
    return f"{title_prefix} | case: {case_label}"


def _save_tumor_coverage_mip_png(
    baseline_4d_mask_zyx: np.ndarray,
    tc4d_mask_zyx: np.ndarray,
    tc4d_support_mask_zyx: np.ndarray,
    tumor_mask_zyx: np.ndarray,
    peritumor_mask_zyx: np.ndarray,
    output_path: Path,
    *,
    case_label: str,
    breast_mask_zyx: np.ndarray | None = None,
    radiologist_mask_zyx: np.ndarray | None = None,
    dpi: int = 180,
) -> dict[str, Any]:
    """Render orthogonal MIPs focused on tumor/peritumoral vessel coverage."""
    shape_ref = tuple(tc4d_mask_zyx.shape)
    all_volumes = [
        baseline_4d_mask_zyx,
        tc4d_support_mask_zyx,
        tumor_mask_zyx,
        peritumor_mask_zyx,
    ]
    if breast_mask_zyx is not None:
        all_volumes.append(breast_mask_zyx)
    if radiologist_mask_zyx is not None:
        all_volumes.append(radiologist_mask_zyx)
    for vol in all_volumes:
        if tuple(vol.shape) != shape_ref:
            raise ValueError(
                "Tumor coverage MIP shape mismatch: "
                f"{tuple(vol.shape)} vs {shape_ref}"
            )

    tumor = np.asarray(tumor_mask_zyx, dtype=bool)
    peritumor = np.asarray(peritumor_mask_zyx, dtype=bool)
    zone = tumor | peritumor
    lower_name = "temporal consensus 4d"

    row_masks: list[tuple[str, np.ndarray]] = [
        ("4d", np.asarray(baseline_4d_mask_zyx, dtype=bool)),
        (lower_name, np.asarray(tc4d_mask_zyx, dtype=bool)),
    ]
    if radiologist_mask_zyx is not None:
        row_masks.append(("radiologist", np.asarray(radiologist_mask_zyx, dtype=bool)))
    else:
        row_masks.append(("radiologist (missing)", np.zeros_like(zone, dtype=bool)))

    render_diag = render_vessel_coverage_mip(
        row_masks=row_masks,
        output_path=output_path,
        case_label=case_label,
        title_prefix="tumor/peritumoral vessel coverage mip",
        radiologist_mask_zyx=radiologist_mask_zyx,
        breast_mask_zyx=breast_mask_zyx,
        tumor_mask_zyx=tumor,
        peritumor_mask_zyx=peritumor,
        vessel_color="#111827",
        dpi=dpi,
    )

    tumor_voxels = int(np.count_nonzero(tumor))
    peritumor_voxels = int(np.count_nonzero(peritumor))
    zone_voxels = int(np.count_nonzero(zone))
    tc4d_support = np.asarray(tc4d_support_mask_zyx, dtype=bool)
    tc4d_exam = np.asarray(tc4d_mask_zyx, dtype=bool)
    baseline_4d = np.asarray(baseline_4d_mask_zyx, dtype=bool)
    support_tumor_voxels = int(np.count_nonzero(tc4d_support & tumor))
    support_peritumor_voxels = int(np.count_nonzero(tc4d_support & peritumor))
    exam_tumor_voxels = int(np.count_nonzero(tc4d_exam & tumor))
    exam_peritumor_voxels = int(np.count_nonzero(tc4d_exam & peritumor))
    baseline_4d_exam_tumor_voxels = int(np.count_nonzero(baseline_4d & tumor))
    baseline_4d_exam_peritumor_voxels = int(np.count_nonzero(baseline_4d & peritumor))
    zone_denom = max(zone_voxels, 1)
    diagnostics: dict[str, Any] = {
        **render_diag,
        "tumor_voxels": tumor_voxels,
        "peritumor_voxels": peritumor_voxels,
        "zone_voxels": zone_voxels,
        "tc4d_support_in_tumor_voxels": support_tumor_voxels,
        "tc4d_support_in_peritumor_voxels": support_peritumor_voxels,
        "tc4d_final_exam_in_tumor_voxels": exam_tumor_voxels,
        "tc4d_final_exam_in_peritumor_voxels": exam_peritumor_voxels,
        "baseline_4d_exam_in_tumor_voxels": baseline_4d_exam_tumor_voxels,
        "baseline_4d_exam_in_peritumor_voxels": baseline_4d_exam_peritumor_voxels,
        "tc4d_support_zone_occupancy_ratio": (
            (support_tumor_voxels + support_peritumor_voxels) / zone_denom
        ),
        "tc4d_exam_zone_occupancy_ratio": (
            (exam_tumor_voxels + exam_peritumor_voxels) / zone_denom
        ),
    }
    if radiologist_mask_zyx is not None:
        radiologist_zone = np.asarray(radiologist_mask_zyx, dtype=bool) & zone
        radiologist_tumor_voxels = int(np.count_nonzero(radiologist_zone & tumor))
        radiologist_peritumor_voxels = int(np.count_nonzero(radiologist_zone & peritumor))
        diagnostics.update(
            {
                "radiologist_in_tumor_voxels": radiologist_tumor_voxels,
                "radiologist_in_peritumor_voxels": radiologist_peritumor_voxels,
                "tc4d_support_hit_radiologist_in_zone_voxels": int(
                    np.count_nonzero((tc4d_support & radiologist_zone))
                ),
                "tc4d_exam_hit_radiologist_in_zone_voxels": int(
                    np.count_nonzero((tc4d_exam & radiologist_zone))
                ),
            }
        )
    return diagnostics

def _init_mp4_render_context(context: dict[str, Any]) -> None:
    """Pool initializer for MP4 frame rendering."""
    global _MP4_RENDER_CONTEXT
    _MP4_RENDER_CONTEXT = context

def _render_mp4_frame_to_png(frame_idx: int) -> int:
    """Render one MP4 frame to PNG using global worker context."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context = _MP4_RENDER_CONTEXT
    if context is None:
        raise RuntimeError("MP4 render context is not initialized.")

    n_panels = context["n_panels"]
    nrows = context["nrows"]
    ncols = context["ncols"]
    azim = frame_idx * 360.0 / context["n_frames"]

    fig = plt.figure(figsize=tuple(context["figsize"]))
    axes = [
        fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        for idx in range(n_panels)
    ]

    for ax in axes:
        ax.set_xlim(float(context["x_low"]), float(context["x_high"]))
        ax.set_ylim(float(context["y_low"]), float(context["y_high"]))
        ax.set_zlim(float(context["z_low"]), float(context["z_high"]))
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=float(context["elev"]), azim=azim)

    panel_specs = context["panel_specs"]
    for ax, spec in zip(axes, panel_specs):
        ax.set_title(str(spec["title"]).lower(), fontsize=12)

    bx = context.get("bx")
    by = context.get("by")
    bz = context.get("bz")
    if bx is not None and by is not None and bz is not None:
        for ax in axes:
            ax.scatter(
                bx,
                by,
                bz,
                s=0.18,
                c="#bbbbbb",
                alpha=0.06,
                marker="o",
            )

    tx = context.get("tx")
    ty = context.get("ty")
    tz = context.get("tz")
    if tx is not None and ty is not None and tz is not None:
        for ax in axes:
            ax.scatter(
                tx,
                ty,
                tz,
                s=0.25,
                c="#ff8c00",
                alpha=0.22,
                marker="o",
            )

    for ax, spec in zip(axes, panel_specs):
        coords = spec["coords"]
        if coords is not None:
            sx, sy, sz = coords
            ax.scatter(
                sx,
                sy,
                sz,
                s=float(spec["size"]),
                c=str(spec["color"]),
                alpha=float(spec["alpha"]),
                marker="o",
            )
        elif spec["empty_text"] is not None:
            ax.text2D(
                0.5,
                0.5,
                str(spec["empty_text"]).lower(),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#555555",
            )

    fig.suptitle(str(context["title"]), fontsize=13)
    frame_path = Path(str(context["frames_dir"])) / f"frame_{frame_idx:04d}.png"
    fig.savefig(frame_path, dpi=context["dpi"])
    plt.close(fig)
    return frame_idx

def _encode_png_sequence_to_mp4(
    *,
    frames_dir: Path,
    output_path: Path,
    fps: int,
    preset: str,
    crf: int,
) -> None:
    """Encode pre-rendered PNG frames into an MP4 using ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is required for MP4 encoding but was not found.")
    input_pattern = str(frames_dir / "frame_%04d.png")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)

def _save_rotating_4d_tc4d_mp4(
    baseline_4d_mask_zyx: np.ndarray,
    tc4d_mask_zyx: np.ndarray,
    output_path: Path,
    case_label: str,
    *,
    title_override: str | None = None,
    radiologist_mask_zyx: np.ndarray | None = None,
    breast_mask_zyx: np.ndarray | None = None,
    tumor_mask_zyx: np.ndarray | None = None,
    n_frames: int = 50,
    fps: int = 24,
    elev: float = 20.0,
    marker_size: float = 1.0,
    max_points_per_panel: int | None = 0,
    mp4_workers: int = 0,
    mp4_dpi: int = 140,
    mp4_encode_preset: str = "veryfast",
    mp4_crf: int = 18,
    keep_mp4_frames: bool = False,
) -> None:
    """Save rotating core MP4 (4d, tc4d, radiologist)."""
    if baseline_4d_mask_zyx.ndim != NDIM_3D or tc4d_mask_zyx.ndim != NDIM_3D:
        raise ValueError("Both masks must be 3D.")
    if tuple(baseline_4d_mask_zyx.shape) != tuple(tc4d_mask_zyx.shape):
        raise ValueError(
            "4d and tc4d masks must share shape, got "
            f"baseline_4d={baseline_4d_mask_zyx.shape}, tc4d={tc4d_mask_zyx.shape}"
        )
    if not np.any(baseline_4d_mask_zyx):
        raise ValueError("4d exam skeleton is empty.")
    if not np.any(tc4d_mask_zyx):
        raise ValueError("tc4d exam skeleton is empty.")

    lx, ly, lz = _extract_xyz(baseline_4d_mask_zyx)
    px, py, pz = _extract_xyz(tc4d_mask_zyx)
    lx, ly, lz = _subsample_xyz(lx, ly, lz, max_points=max_points_per_panel)
    px, py, pz = _subsample_xyz(px, py, pz, max_points=max_points_per_panel)

    rx: np.ndarray | None = None
    ry: np.ndarray | None = None
    rz: np.ndarray | None = None
    if radiologist_mask_zyx is not None and np.any(radiologist_mask_zyx):
        rx, ry, rz = _extract_xyz(radiologist_mask_zyx)
        rx, ry, rz = _subsample_xyz(rx, ry, rz, max_points=max_points_per_panel)

    breast_overlay_pts = _prepare_breast_overlay_points(breast_mask_zyx)
    bx: np.ndarray | None = None
    by: np.ndarray | None = None
    bz: np.ndarray | None = None
    if breast_overlay_pts is not None:
        bx, by, bz = breast_overlay_pts

    tumor_outline_pts = _prepare_tumor_outline_points(tumor_mask_zyx)
    tx: np.ndarray | None = None
    ty: np.ndarray | None = None
    tz: np.ndarray | None = None
    if tumor_outline_pts is not None:
        tx, ty, tz = tumor_outline_pts

    panel_specs: list[dict[str, Any]] = [
        {
            "title": "4d",
            "coords": (lx, ly, lz),
            "color": "#1f77b4",
            "alpha": 0.85,
            "size": marker_size,
            "empty_text": None,
        },
        {
            "title": "tc4d",
            "coords": (px, py, pz),
            "color": "#d62728",
            "alpha": 0.85,
            "size": marker_size,
            "empty_text": None,
        },
        {
            "title": "radiologist vessels",
            "coords": None if rx is None else (rx, ry, rz),
            "color": "#2ca02c",
            "alpha": 0.85,
            "size": marker_size,
            "empty_text": "not available for this case",
        },
    ]

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    for spec in panel_specs:
        coords = spec["coords"]
        if coords is None:
            continue
        cx, cy, cz = coords
        x_parts.append(cx)
        y_parts.append(cy)
        z_parts.append(cz)
    if bx is not None and by is not None and bz is not None:
        x_parts.append(bx)
        y_parts.append(by)
        z_parts.append(bz)
    if tx is not None and ty is not None and tz is not None:
        x_parts.append(tx)
        y_parts.append(ty)
        z_parts.append(tz)

    if not x_parts:
        raise ValueError("No non-empty panel content found for MP4 rendering.")

    all_x = np.concatenate(x_parts)
    all_y = np.concatenate(y_parts)
    all_z = np.concatenate(z_parts)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    z_min, z_max = float(all_z.min()), float(all_z.max())
    max_span = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
    cx_mid = 0.5 * (x_min + x_max)
    cy_mid = 0.5 * (y_min + y_max)
    cz_mid = 0.5 * (z_min + z_max)

    n_panels = len(panel_specs)
    nrows = 1
    ncols = int(math.ceil(float(n_panels) / float(nrows)))
    fig_size = (max(15.0, 5.5 * ncols), 6.5)

    if title_override is not None:
        title = str(title_override)
    else:
        title = _build_case_title(case_label=case_label)

    frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink(missing_ok=True)

    cpu_count = os.cpu_count() or 1
    if mp4_workers <= 0:
        workers = max(1, min(8, cpu_count))
    else:
        workers = max(1, mp4_workers)

    render_context: dict[str, Any] = {
        "n_panels": n_panels,
        "nrows": nrows,
        "ncols": ncols,
        "n_frames": n_frames,
        "figsize": fig_size,
        "x_low": float(cx_mid - max_span / 2.0),
        "x_high": float(cx_mid + max_span / 2.0),
        "y_low": float(cy_mid - max_span / 2.0),
        "y_high": float(cy_mid + max_span / 2.0),
        "z_low": float(cz_mid - max_span / 2.0),
        "z_high": float(cz_mid + max_span / 2.0),
        "elev": float(elev),
        "title": title,
        "panel_specs": panel_specs,
        "bx": bx,
        "by": by,
        "bz": bz,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "frames_dir": str(frames_dir),
        "dpi": mp4_dpi,
    }

    print(
        "[mp4] starting render: "
        f"panels={n_panels}, frames={n_frames}, fps={fps}, workers={workers}, "
        f"dpi={mp4_dpi}, output={output_path}"
    )
    render_start = time.perf_counter()
    if workers <= 1:
        _init_mp4_render_context(render_context)
        for frame_idx in range(n_frames):
            _render_mp4_frame_to_png(frame_idx)
            if frame_idx == 0 or frame_idx + 1 == n_frames or frame_idx % 10 == 0:
                print(f"[mp4] rendered frame {frame_idx + 1}/{n_frames}")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_mp4_render_context,
            initargs=(render_context,),
        ) as executor:
            futures = [
                executor.submit(_render_mp4_frame_to_png, frame_idx)
                for frame_idx in range(n_frames)
            ]
            completed = 0
            for future in as_completed(futures):
                _ = future.result()
                completed += 1
                if completed == 1 or completed == n_frames or completed % 10 == 0:
                    print(f"[mp4] rendered frame {completed}/{n_frames}")

    render_seconds = float(time.perf_counter() - render_start)
    encode_start = time.perf_counter()
    _encode_png_sequence_to_mp4(
        frames_dir=frames_dir,
        output_path=output_path,
        fps=fps,
        preset=str(mp4_encode_preset),
        crf=mp4_crf,
    )
    encode_seconds = float(time.perf_counter() - encode_start)
    print(
        "[mp4] render complete: "
        f"{output_path} ({render_seconds + encode_seconds:.2f} seconds, "
        f"frame_render={render_seconds:.2f}s, encode={encode_seconds:.2f}s)"
    )

    if not keep_mp4_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare 4d (hand-tuned) vs tc4d (graph-forward) 4D pipelines and "
            "render a rotating core MP4 plus tumor/peritumor vessel MIP."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="Directory containing study subdirs with per-timepoint segmentation `.npz` files.",
    )
    parser.add_argument(
        "--study-id",
        type=str,
        default=None,
        help=(
            "Study/patient ID used to auto-discover all timepoints from --input-dir "
            "(e.g., ISPY2_202539)."
        ),
    )
    parser.add_argument(
        "--input-4d",
        type=Path,
        default=None,
        help="Single .npy containing the full time series.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["tzyx", "ctzyx", "tczyx"],
        default="tzyx",
        help="Layout for --input-4d.",
    )
    parser.add_argument(
        "--npy-channel",
        type=int,
        default=1,
        help="Channel index for channelized NPY inputs.",
    )
    parser.add_argument(
        "--mp4-frames",
        type=int,
        default=50,
        help="Number of rendered frames for the rotating MP4.",
    )
    parser.add_argument(
        "--mp4-fps",
        type=int,
        default=24,
        help="Frames per second for the rotating MP4.",
    )
    parser.add_argument(
        "--render-mp4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render rotating core comparison MP4.",
    )
    parser.add_argument(
        "--render-mips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render tumor/peritumoral vessel coverage MIP when tumor mask is available.",
    )
    parser.add_argument(
        "--mip-dpi",
        type=int,
        default=180,
        help="DPI for tumor/peritumoral vessel coverage MIP.",
    )
    parser.add_argument(
        "--add-radiologist-annotations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add radiologist vessel panel + transparent breast overlay. "
            "Requires matching annotations for the case; errors if missing."
        ),
    )
    parser.add_argument(
        "--radiologist-annotations-dir",
        type=Path,
        default=DEFAULT_RADIOLOGIST_ANNOTATIONS_DIR,
        help=(
            "Base directory for Duke supplemental annotations (default: "
            "/net/projects2/vanguard/Duke-Breast-Cancer-MRI-Supplement-v3). "
            "May also point directly to `Segmentation_Masks_NRRD`."
        ),
    )
    parser.add_argument(
        "--add-tumor-outline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a lightweight tumor outline overlay to all panels.",
    )
    parser.add_argument(
        "--tumor-mask-path",
        type=Path,
        default=None,
        help="Optional explicit tumor mask path (.nii/.nii.gz/.nrrd).",
    )
    parser.add_argument(
        "--tumor-mask-dir",
        type=Path,
        default=DEFAULT_TUMOR_MASK_DIR,
        help=(
            "Directory for auto tumor mask resolution as "
            "`{study_id}.nii.gz` when --add-tumor-outline is set."
        ),
    )
    parser.add_argument(
        "--tumor-threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for tumor mask volumes.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_IO_CACHE_DIR,
        help=(
            "Directory for reusable aligned I/O cache (raw DCE + annotation masks). "
            "Defaults to .cache/compare_4d under the current working directory."
        ),
    )
    parser.add_argument(
        "--use-io-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable on-disk I/O cache for repeated debug iterations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for masks, MP4, and summary JSON.",
    )
    parser.add_argument(
        "--save-intermediate-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser

def main() -> None:
    """CLI entrypoint."""
    start_time = time.perf_counter()
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    git_metadata = _collect_git_metadata(project_root)
    print(
        "[code] git: "
        f"commit={git_metadata.get('commit_short')}, "
        f"dirty={git_metadata.get('dirty')}, "
        f"branch={git_metadata.get('branch')}"
    )

    has_study_mode = args.study_id is not None
    has_input4d_mode = args.input_4d is not None
    mode_count = has_study_mode + has_input4d_mode
    if mode_count != 1:
        raise ValueError(
            "Choose exactly one input mode: "
            "(--study-id with --input-dir) OR (--input-4d)."
        )

    io_cache_dir = Path(args.cache_dir).expanduser()
    if not io_cache_dir.is_absolute():
        io_cache_dir = (Path.cwd() / io_cache_dir).resolve()
    io_cache_enabled = args.use_io_cache
    if io_cache_enabled:
        io_cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[cache] io-cache: "
        f"enabled={io_cache_enabled}, dir={io_cache_dir}, version={IO_CACHE_VERSION}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovered_files: list[Path] | None = None
    discovered_timepoints: list[int] | None = None

    if has_study_mode:
        discovered_files, discovered_timepoints = discover_study_timepoints(
            input_dir=args.input_dir,
            study_id=args.study_id,
        )
        priority_4d = load_time_series_from_files(discovered_files)
        case_label = args.study_id
    elif has_input4d_mode:
        priority_4d = _load_time_series_from_single_npy(
            args.input_4d,
            layout=args.layout,
            npy_channel=args.npy_channel,
        )
        case_label = args.input_4d.stem

    if priority_4d.ndim != NDIM_4D:
        raise ValueError(f"Expected (t,z,y,x) array, got shape {priority_4d.shape}")

    print(f"[info] Loaded priority volume shape: {priority_4d.shape}")
    print(
        f"[info] Priority min/max: {float(priority_4d.min()):.6f}/{float(priority_4d.max()):.6f}"
    )

    baseline_4d_result = _run_4d_pipeline(priority_4d)
    tc4d_result, tc4d_params, tc4d_pipeline_diagnostics = (
        _run_graph_consensus_pipeline(priority_4d)
    )

    baseline_4d_exam = baseline_4d_result["exam_mask"].astype(bool, copy=False)
    tc4d_exam = tc4d_result["exam_mask"].astype(bool, copy=False)
    baseline_4d_4d = baseline_4d_result["mask_4d"].astype(bool, copy=False)
    tc4d_4d = tc4d_result["mask_4d"].astype(bool, copy=False)
    baseline_4d_support = baseline_4d_result["support_mask"].astype(bool, copy=False)
    tc4d_support = tc4d_result["support_mask"].astype(bool, copy=False)

    print(
        "[4d] "
        f"exam_voxels={int(np.count_nonzero(baseline_4d_exam))}, "
        f"exam_components_26={_count_components(baseline_4d_exam)}, "
        f"support_voxels={int(np.count_nonzero(baseline_4d_support))}, "
        f"manifold_voxels={int(np.count_nonzero(baseline_4d_4d))}"
    )
    print(
        "[tc4d] "
        f"exam_voxels={int(np.count_nonzero(tc4d_exam))}, "
        f"exam_components_26={_count_components(tc4d_exam)}, "
        f"support_voxels={int(np.count_nonzero(tc4d_support))}, "
        f"manifold_voxels={int(np.count_nonzero(tc4d_4d))}"
    )

    breast_mask_zyx: np.ndarray | None = None
    radiologist_mask_zyx: np.ndarray | None = None
    tumor_mask_zyx: np.ndarray | None = None
    peritumor_mask_zyx: np.ndarray | None = None
    breast_mask_processing_zyx: np.ndarray | None = None
    breast_flip_to_processing = "none"
    radiologist_alignment_diag: dict[str, Any] | None = None
    radiologist_flip_to_model = "none"
    radiologist_coverage: dict[str, Any] | None = None
    breast_overlay: dict[str, Any] = {
        "enabled": args.add_radiologist_annotations,
        "resolved_case_id": None,
        "path": None,
        "layout": None,
        "voxels": 0,
        "error": None,
    }
    tumor_overlay: dict[str, Any] = {
        "enabled": args.add_tumor_outline,
        "path": None,
        "threshold": float(args.tumor_threshold) if args.add_tumor_outline else None,
        "voxels": 0,
        "layout": None,
        "error": None,
    }
    radiologist_overlay: dict[str, Any] = {
        "enabled": args.add_radiologist_annotations,
        "requested_case_id": args.study_id if has_study_mode else None,
        "resolved_case_id": None,
        "path": None,
        "segment_name": None,
        "segment_label_value": None,
        "layout": None,
        "voxels": 0,
        "error": None,
    }
    expected_shape_zyx = tuple(baseline_4d_exam.shape)
    if args.add_radiologist_annotations:
        if not has_study_mode:
            raise ValueError(
                "--add-radiologist-annotations requires --study-id mode so annotations can be matched."
            )

        resolved_case_id = _to_breast_mri_case_id(args.study_id)
        if resolved_case_id is None:
            raise ValueError(
                "Could not map study id to a Breast_MRI case id: "
                f"{args.study_id}. Expected a DUKE_* style study id."
            )

        nrrd_root = _resolve_radiologist_nrrd_root(args.radiologist_annotations_dir)
        if not nrrd_root.exists():
            raise FileNotFoundError(
                "Radiologist annotation directory not found: "
                f"{nrrd_root} (from --radiologist-annotations-dir={args.radiologist_annotations_dir})"
            )

        breast_path = _resolve_breast_seg_path(
            nrrd_root=nrrd_root,
            case_id=resolved_case_id,
        )
        if breast_path is None:
            raise FileNotFoundError(
                "Required breast annotation missing for case "
                f"{resolved_case_id} under {nrrd_root}."
            )

        vessel_path = _resolve_radiologist_seg_path(
            nrrd_root=nrrd_root,
            case_id=resolved_case_id,
        )
        if vessel_path is None:
            raise FileNotFoundError(
                "Required Dense_and_Vessels annotation missing for case "
                f"{resolved_case_id} under {nrrd_root}."
            )

        breast_mask_zyx, breast_info = _load_binary_mask_nrrd_cached(
            breast_path,
            expected_shape_zyx=expected_shape_zyx,
            label_name="Breast",
            threshold=0.0,
            cache_dir=io_cache_dir,
            use_cache=io_cache_enabled,
        )
        radiologist_mask_zyx, radiologist_info = _load_radiologist_vessel_mask_nrrd_cached(
            vessel_path,
            expected_shape_zyx=expected_shape_zyx,
            cache_dir=io_cache_dir,
            use_cache=io_cache_enabled,
        )

        breast_overlay.update({"resolved_case_id": resolved_case_id, **breast_info})
        radiologist_overlay.update(
            {"resolved_case_id": resolved_case_id, **radiologist_info}
        )
        print(
            "[breast] loaded mask: "
            f"{breast_path} (voxels={breast_info['voxels']}, layout={breast_info['layout']}, "
            f"cache_hit={breast_info.get('cache', {}).get('hit')})"
        )
        print(
            "[radiologist] loaded vessel annotation: "
            f"{vessel_path} (segment={radiologist_info['segment_name']}, "
            f"label={radiologist_info['segment_label_value']}, "
            f"voxels={radiologist_info['voxels']}, layout={radiologist_info['layout']}, "
            f"cache_hit={radiologist_info.get('cache', {}).get('hit')})"
        )

        alignment_reference = baseline_4d_exam | tc4d_exam
        radiologist_alignment_diag = _choose_flip_by_containment(
            radiologist_mask_zyx,
            alignment_reference,
        )
        radiologist_flip_to_model = str(
            radiologist_alignment_diag.get("best_flip_spec", "none")
        )
        radiologist_mask_zyx = _apply_flip_spec(
            radiologist_mask_zyx,
            radiologist_flip_to_model,
        )
        radiologist_overlay["processing_flip_to_model"] = radiologist_flip_to_model
        radiologist_overlay["alignment_to_processing"] = radiologist_alignment_diag
        print(
            "[radiologist] aligned to model frame via 4d/tc4d containment: "
            f"flip={radiologist_flip_to_model}, "
            f"inside_ratio={float(radiologist_alignment_diag['best_inside_ratio']):.4f}, "
            f"inside_voxels={int(radiologist_alignment_diag['best_inside_voxels'])}"
        )
    else:
        print("[annotations] skipped (--no-add-radiologist-annotations)")

    if args.add_tumor_outline:
        resolved_tumor_path = args.tumor_mask_path
        if resolved_tumor_path is None:
            if not has_study_mode:
                raise ValueError(
                    "--add-tumor-outline requires --study-id mode, or provide --tumor-mask-path."
                )
            resolved_tumor_path = args.tumor_mask_dir / f"{args.study_id}.nii.gz"
        if not resolved_tumor_path.exists():
            raise FileNotFoundError(f"Tumor mask not found: {resolved_tumor_path}")

        tumor_mask_zyx, tumor_info = _load_binary_mask_nrrd_cached(
            resolved_tumor_path,
            expected_shape_zyx=expected_shape_zyx,
            label_name="Tumor",
            threshold=float(args.tumor_threshold),
            cache_dir=io_cache_dir,
            use_cache=io_cache_enabled,
        )
        tumor_overlay.update(tumor_info)
        print(
            "[tumor] loaded outline mask: "
            f"{resolved_tumor_path} (voxels={tumor_info['voxels']}, "
            f"layout={tumor_info['layout']}, threshold={args.tumor_threshold}, "
            f"cache_hit={tumor_info.get('cache', {}).get('hit')})"
        )
    else:
        print("[tumor] skipped (--no-add-tumor-outline)")

    # Priority/raw-derived tensors are in model frame. Align annotation masks into that
    # frame for any processing-time masking to avoid frame-mismatch suppression.
    if breast_mask_zyx is not None:
        if radiologist_mask_zyx is not None:
            breast_flip_to_processing = radiologist_flip_to_model
        breast_mask_processing_zyx = _apply_flip_spec(
            breast_mask_zyx,
            breast_flip_to_processing,
        )
        breast_overlay["processing_flip_to_model"] = breast_flip_to_processing
    if tumor_mask_zyx is not None:
        tumor_alignment_diag: dict[str, Any] | None = None
        if breast_mask_processing_zyx is not None:
            tumor_alignment_diag = _choose_flip_by_containment(
                tumor_mask_zyx,
                breast_mask_processing_zyx,
            )
            tumor_flip_to_processing = str(
                tumor_alignment_diag.get("best_flip_spec", "none")
            )
            tumor_mask_zyx = _apply_flip_spec(tumor_mask_zyx, tumor_flip_to_processing)
            tumor_overlay["processing_flip_to_model"] = tumor_flip_to_processing
            tumor_overlay["alignment_to_processing"] = tumor_alignment_diag
            print(
                "[tumor] aligned to model frame via breast containment: "
                f"flip={tumor_flip_to_processing}, "
                f"inside_ratio={float(tumor_alignment_diag['best_inside_ratio']):.4f}, "
                f"inside_voxels={int(tumor_alignment_diag['best_inside_voxels'])}"
            )
        else:
            tumor_overlay["processing_flip_to_model"] = "none"
            tumor_overlay["alignment_to_processing"] = None
            print(
                "[tumor] no breast mask available; using tumor mask in native frame for model processing"
            )
        peritumor_mask_zyx = _compute_peritumoral_shell_mask(
            tumor_mask_zyx,
            breast_mask_zyx=breast_mask_processing_zyx,
            inner_radius_vox=TUMOR_PERITUMOR_INNER_RADIUS_VOX,
            outer_radius_vox=TUMOR_PERITUMOR_OUTER_RADIUS_VOX,
        )
        if peritumor_mask_zyx is not None:
            print(
                "[tumor] peritumoral shell: "
                f"inner={TUMOR_PERITUMOR_INNER_RADIUS_VOX} vox, "
                f"outer={TUMOR_PERITUMOR_OUTER_RADIUS_VOX} vox, "
                f"voxels={int(np.count_nonzero(peritumor_mask_zyx))}"
            )

    radiologist_coverage = _compute_radiologist_coverage_metrics(
        baseline_4d_exam_zyx=baseline_4d_exam,
        baseline_4d_support_zyx=baseline_4d_support,
        tc4d_exam_zyx=tc4d_exam,
        tc4d_support_zyx=tc4d_support,
        radiologist_mask_zyx=radiologist_mask_zyx,
        alignment_flip_to_model=radiologist_flip_to_model,
        alignment_diagnostics=radiologist_alignment_diag,
    )
    if radiologist_coverage is not None:
        print(
            "[radiologist] aligned exam hit-rate: "
            f"4d={float(radiologist_coverage['4d_exam']['hit_rate']):.4f} "
            f"({int(radiologist_coverage['4d_exam']['hits'])}/"
            f"{int(radiologist_coverage['4d_exam']['radiologist_voxels'])}), "
            f"tc4d={float(radiologist_coverage['tc4d_exam']['hit_rate']):.4f} "
            f"({int(radiologist_coverage['tc4d_exam']['hits'])}/"
            f"{int(radiologist_coverage['tc4d_exam']['radiologist_voxels'])})"
        )

    overlap = compute_overlap_metrics(baseline_4d_exam, tc4d_exam)
    baseline_4d_exam_viz = _apply_flip_spec(baseline_4d_exam, COMPARE_4D_TC4D_VIZ_FLIP_SPEC)
    tc4d_exam_viz = _apply_flip_spec(tc4d_exam, COMPARE_4D_TC4D_VIZ_FLIP_SPEC)
    tc4d_support_viz_zyx = _apply_flip_spec(tc4d_support, COMPARE_4D_TC4D_VIZ_FLIP_SPEC)
    radiologist_mask_viz_zyx = (
        None
        if radiologist_mask_zyx is None
        else _apply_flip_spec(radiologist_mask_zyx, COMPARE_4D_TC4D_VIZ_FLIP_SPEC)
    )
    breast_mask_viz_zyx = (
        None
        if breast_mask_processing_zyx is None
        else _apply_flip_spec(
            breast_mask_processing_zyx,
            COMPARE_4D_TC4D_VIZ_FLIP_SPEC,
        )
    )
    tumor_mask_viz_zyx = (
        None
        if tumor_mask_zyx is None
        else _apply_flip_spec(tumor_mask_zyx, COMPARE_4D_TC4D_VIZ_FLIP_SPEC)
    )
    peritumor_mask_viz_zyx = (
        None
        if peritumor_mask_zyx is None
        else _apply_flip_spec(peritumor_mask_zyx, COMPARE_4D_TC4D_VIZ_FLIP_SPEC)
    )
    print(
        "[viz] applied fixed 4d/tc4d/support flip: "
        f"{COMPARE_4D_TC4D_VIZ_FLIP_SPEC}"
    )

    out_rot_core: Path | None = None
    out_tumor_coverage_mip: Path | None = None
    video_core_seconds = 0.0
    video_seconds = 0.0
    tumor_coverage_mip_seconds = 0.0
    tumor_zone_coverage_diag: dict[str, Any] | None = None

    if not args.render_mips:
        print("[mip] skipped (--no-render-mips)")
    elif tumor_mask_viz_zyx is not None and peritumor_mask_viz_zyx is not None:
        out_tumor_coverage_mip = args.output_dir / "tumor_peritumor_coverage_mip_tc4d.png"
        t_tumor_cov_start = time.perf_counter()
        try:
            tumor_zone_coverage_diag = _save_tumor_coverage_mip_png(
                baseline_4d_mask_zyx=baseline_4d_exam_viz,
                tc4d_mask_zyx=tc4d_exam_viz,
                tc4d_support_mask_zyx=tc4d_support_viz_zyx,
                tumor_mask_zyx=tumor_mask_viz_zyx,
                peritumor_mask_zyx=peritumor_mask_viz_zyx,
                breast_mask_zyx=breast_mask_viz_zyx,
                radiologist_mask_zyx=radiologist_mask_viz_zyx,
                output_path=out_tumor_coverage_mip,
                case_label=case_label,
                dpi=max(int(args.mip_dpi), 180),
            )
            tumor_coverage_mip_seconds = float(time.perf_counter() - t_tumor_cov_start)
            print(
                "[mip] wrote tumor/peritumoral coverage figure: "
                f"{out_tumor_coverage_mip} ({tumor_coverage_mip_seconds:.2f} seconds), "
                "tc4d_final_exam_in_tumor_voxels="
                f"{tumor_zone_coverage_diag['tc4d_final_exam_in_tumor_voxels']}, "
                "tc4d_final_exam_in_peritumor_voxels="
                f"{tumor_zone_coverage_diag['tc4d_final_exam_in_peritumor_voxels']}"
            )
        except Exception as exc:
            out_tumor_coverage_mip = None
            tumor_zone_coverage_diag = {
                "status": "error",
                "error": str(exc),
            }
            print(
                "[mip] tumor/peritumoral coverage figure failed; continuing. "
                f"reason={exc}"
            )
    else:
        print(
            "[mip] tumor/peritumoral coverage figure skipped "
            "(requires tumor mask and non-empty peritumoral shell)"
        )

    if args.render_mp4:
        out_rot_core = args.output_dir / "rotation_compare_4d_vs_tc4d_core.mp4"
        t_video_core_start = time.perf_counter()
        _save_rotating_4d_tc4d_mp4(
            baseline_4d_mask_zyx=baseline_4d_exam_viz,
            tc4d_mask_zyx=tc4d_exam_viz,
            title_override=_build_case_title(
                case_label=case_label,
                title_prefix="4D vs temporal consensus 4D",
            ),
            radiologist_mask_zyx=radiologist_mask_viz_zyx,
            breast_mask_zyx=breast_mask_viz_zyx,
            tumor_mask_zyx=tumor_mask_viz_zyx,
            output_path=out_rot_core,
            case_label=case_label,
            n_frames=args.mp4_frames,
            fps=args.mp4_fps,
        )
        video_core_seconds = float(time.perf_counter() - t_video_core_start)
        video_seconds = video_core_seconds
        print(
            "[timing] mp4_seconds: " f"core={video_core_seconds:.2f}, total={video_seconds:.2f}"
        )
    else:
        print("[mp4] skipped (--no-render-mp4)")

    out_4d_exam = args.output_dir / "skeleton_4d_4d_exam_mask.npy"
    out_tc4d_exam = args.output_dir / "skeleton_4d_tc4d_exam_mask.npy"

    np.save(out_4d_exam, baseline_4d_exam.astype(np.uint8))
    np.save(out_tc4d_exam, tc4d_exam.astype(np.uint8))

    summary = {
        "shape_tzyx": list(priority_4d.shape),
        "code_version": git_metadata,
        "run_config": {
            "study_id": args.study_id if has_study_mode else None,
        },
        "4d": {
            "params": FOURD_BASELINE_PARAMS,
            "effective_min_temporal_support": int(
                baseline_4d_result["effective_min_temporal_support"]
            ),
            "exam_voxels": int(np.count_nonzero(baseline_4d_exam)),
            "exam_components_26": _count_components(baseline_4d_exam),
            "support_voxels": int(np.count_nonzero(baseline_4d_support)),
            "manifold_voxels": int(np.count_nonzero(baseline_4d_4d)),
            "manifold_runtime_seconds": float(baseline_4d_result["manifold_seconds"]),
            "collapse_runtime_seconds": float(baseline_4d_result["collapse_seconds"]),
            "exam_mask_path": str(out_4d_exam),
        },
        "tc4d": {
            "params": tc4d_params,
            "effective_min_temporal_support": int(
                tc4d_result["effective_min_temporal_support"]
            ),
            "exam_voxels": int(np.count_nonzero(tc4d_exam)),
            "exam_components_26": _count_components(tc4d_exam),
            "support_voxels": int(np.count_nonzero(tc4d_support)),
            "manifold_voxels": int(np.count_nonzero(tc4d_4d)),
            "manifold_runtime_seconds": float(tc4d_result["manifold_seconds"]),
            "collapse_runtime_seconds": float(tc4d_result["collapse_seconds"]),
            "exam_mask_path": str(out_tc4d_exam),
        },
        "overlap": overlap,
        "study_mode": {
            "study_id": args.study_id,
            "input_dir": str(args.input_dir) if has_study_mode else None,
            "timepoints": discovered_timepoints
            if discovered_timepoints is not None
            else None,
            "files": (
                [str(p) for p in discovered_files]
                if discovered_files is not None
                else None
            ),
        },
        "outputs": {
            "rotation_compare_4d_vs_tc4d_core_mp4": None
            if out_rot_core is None
            else str(out_rot_core),
            "tumor_peritumor_coverage_mip_tc4d_png": None
            if out_tumor_coverage_mip is None
            else str(out_tumor_coverage_mip),
        },
        "visualization": {
            "4d_tc4d_flip_spec": COMPARE_4D_TC4D_VIZ_FLIP_SPEC,
            "processing_mask_flip_spec": (
                breast_flip_to_processing
                if breast_mask_processing_zyx is not None
                else "none"
            ),
            "tumor_zone_coverage": tumor_zone_coverage_diag,
        },
        "breast_overlay": breast_overlay,
        "radiologist_overlay": radiologist_overlay,
        "radiologist_coverage": radiologist_coverage,
        "tumor_overlay": tumor_overlay,
        "io_cache": {
            "enabled": io_cache_enabled,
            "dir": str(io_cache_dir),
            "version": IO_CACHE_VERSION,
        },
        "tc4d_pipeline_diagnostics": tc4d_pipeline_diagnostics,
        "timing_seconds": {
            "4d_total": float(baseline_4d_result["manifold_seconds"])
            + float(baseline_4d_result["collapse_seconds"]),
            "tc4d_total": float(tc4d_result["manifold_seconds"])
            + float(tc4d_result["collapse_seconds"]),
            "video_core": video_core_seconds,
            "video_total": video_seconds,
            "tumor_coverage_mip": tumor_coverage_mip_seconds,
            "total": float(time.perf_counter() - start_time),
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[done] Wrote outputs:")
    if out_rot_core is not None:
        print(f"  - {out_rot_core}")
    if out_tumor_coverage_mip is not None:
        print(f"  - {out_tumor_coverage_mip}")
    print(f"  - {out_4d_exam}")
    print(f"  - {out_tc4d_exam}")
    print(f"  - {summary_path}")
    print(f"[done] Total time: {summary['timing_seconds']['total']:.2f} seconds")


if __name__ == "__main__":
    main()
