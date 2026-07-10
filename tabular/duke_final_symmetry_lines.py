"""Generate final DUKE symmetry-line QC images from breast-mask inner walls.

Method summary
--------------
For each case, this script projects the breast mask into the same y/x view used
by the QC image and places an initial candidate line at the center of the image.
For rows where that center line sits in the gap between breasts, it searches
left and right until it reaches the two inner breast-mask walls. The final
symmetry line is the median midpoint between those row-wise wall pairs.

Outputs
-------
The script writes a side-assignment CSV and one QC PNG per selected case. Each
PNG displays the raw MRI MIP and a mask/skeleton overlay, and the top title
states which symmetry-line method was used for that case.
"""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_CSV = REPO_ROOT / "tabular" / "duke_final_symmetry_lines.csv"
DEFAULT_OUTDIR = REPO_ROOT / "qc" / "duke_final_symmetry_lines"
MAMA_MIA_IMAGE_ROOT = Path("/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images")
DEFAULT_DUKE_BREAST_ROOTS = (
    Path("/gpfs/data/karczmar-lab/workspaces/saritbose/breast_masks/DUKE"),
    Path("/gpfs/data/karczmar-lab/duke/Segmentation_Masks_NRRD"),
    Path("/gpfs/data/karczmar-lab/duke"),
)

FINAL_METHOD = "inner_walls"
PROJECTED_IMAGE_NDIMS = 2
VOLUME_NDIMS = 3
COORD_NDIMS = 3
SIDE_QC_COLUMNS = [
    "case_id",
    "dataset",
    "manifest_path",
    "bilateral_mri",
    "bilateral_breast_cancer",
    "include_for_asymmetry",
    "exclude_reason",
    "tumor_mask_path",
    "breast_mask_path",
    "centerline_path",
    "skeleton_mask_path",
    "support_mask_path",
    "morphometry_path",
    "tumor_graph_features_path",
    "image_shape_z",
    "image_shape_y",
    "image_shape_x",
    "image_midline_x",
    "left_right_axis",
    "coordinate_order_assumption",
    "tumor_centroid_x",
    "tumor_centroid_y",
    "tumor_centroid_z",
    "inferred_tumor_side",
    "inferred_contralateral_side",
    "tumor_side_source",
    "tumor_side_confidence",
    "tumor_midline_distance_vox",
    "tumor_midline_distance_fraction",
    "tumor_voxels",
    "breast_mask_available",
    "left_breast_voxels",
    "right_breast_voxels",
    "side_split_method",
    "morphometry_json_available",
    "skeleton_mask_available",
    "support_mask_available",
    "segment_count_total",
    "segment_count_assigned",
    "segment_count_midline_crossing",
    "segment_count_unassigned",
    "skeleton_voxel_count_total",
    "skeleton_voxel_count_left",
    "skeleton_voxel_count_right",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "Path to the shared DUKE/MAMA-MIA bilateral side-assignment manifest. "
            "This input is not committed with the final export."
        ),
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-id", default=None, help="Optional single-case run.")
    parser.add_argument("--tumor-threshold", type=float, default=0.5)
    parser.add_argument("--breast-threshold", type=float, default=0.5)
    parser.add_argument(
        "--x-support-fraction",
        type=float,
        default=0.05,
        help=(
            "Minimum fraction of the projected-mask rows that must provide "
            "clean inner-wall pairs for the inner-walls method to be used."
        ),
    )
    parser.add_argument(
        "--low-confidence-midline-fraction",
        type=float,
        default=0.10,
        help="Mark tumor-side inference low confidence below this x-distance fraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing this script's existing output CSV/images.",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _path_from_row(row: dict[str, str], *names: str) -> Path | None:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return Path(value)
    return None


def _load_array(path: Path) -> np.ndarray:
    if str(path).lower().endswith(".npy"):
        return np.load(path, allow_pickle=False)
    raise ValueError(f"Only .npy arrays are supported here: {path}")


def _load_medical_image(path: Path) -> np.ndarray:
    lower = str(path).lower()
    if lower.endswith(".npy"):
        return np.load(path, allow_pickle=False)

    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        try:
            import SimpleITK as sitk

            return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        except ModuleNotFoundError:
            try:
                import nibabel as nib

                return np.asarray(nib.load(str(path)).get_fdata())
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Loading NIfTI masks requires SimpleITK or nibabel. "
                    "Run this script in the vanguard environment."
                ) from exc

    if lower.endswith(".nrrd"):
        try:
            import SimpleITK as sitk

            return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Loading NRRD masks requires SimpleITK. "
                "Run this script in the vanguard environment."
            ) from exc

    raise ValueError(f"Unsupported mask format: {path}")


def _candidate_layouts(arr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    candidates = [("zyx", np.asarray(arr))]
    if arr.ndim == VOLUME_NDIMS:
        candidates.extend(
            [
                ("yxz", np.transpose(arr, (1, 2, 0))),
                ("xyz", np.transpose(arr, (2, 1, 0))),
            ]
        )
    return candidates


def _load_mask_matching_shape(
    path: Path,
    *,
    expected_shape: tuple[int, int, int],
    threshold: float,
) -> tuple[np.ndarray, str]:
    arr = _load_medical_image(path)
    if arr.ndim < VOLUME_NDIMS:
        raise ValueError(f"Mask must be at least 3D, got {arr.shape}: {path}")
    if arr.ndim > VOLUME_NDIMS:
        lead_axes = tuple(range(arr.ndim - VOLUME_NDIMS))
        arr = np.any(arr > threshold, axis=lead_axes)

    for layout, candidate in _candidate_layouts(np.asarray(arr)):
        if tuple(int(v) for v in candidate.shape) == expected_shape:
            mask = np.asarray(candidate > threshold, dtype=bool)
            if not np.any(mask):
                raise ValueError(f"Mask is empty after thresholding: {path}")
            return mask, layout

    raise ValueError(
        f"Mask shape mismatch for {path}: expected {expected_shape}, got {arr.shape}"
    )


def _load_image_matching_shape(
    path: Path,
    *,
    expected_shape: tuple[int, int, int],
) -> tuple[np.ndarray, str]:
    arr = np.asarray(_load_medical_image(path))
    if arr.ndim < VOLUME_NDIMS:
        raise ValueError(f"Image must be at least 3D, got {arr.shape}: {path}")
    if arr.ndim > VOLUME_NDIMS:
        arr = np.asarray(arr[0])

    for layout, candidate in _candidate_layouts(arr):
        if tuple(int(v) for v in candidate.shape) == expected_shape:
            return np.asarray(candidate, dtype=float), layout

    raise ValueError(
        f"Image shape mismatch for {path}: expected {expected_shape}, got {arr.shape}"
    )


def _resolve_breast_mask_path(row: dict[str, str]) -> Path | None:
    case_id = str(row.get("case_id", "")).strip()
    dataset = str(row.get("dataset", "")).strip().upper()
    if dataset != "DUKE" or not case_id.startswith("DUKE_"):
        return None

    sarit_root = DEFAULT_DUKE_BREAST_ROOTS[0]
    sarit_images_dir = sarit_root / case_id / "images"
    preferred = sarit_images_dir / f"{case_id}_0001_breast_mask.npy"
    if preferred.exists():
        return preferred
    sarit_candidates = sorted(sarit_images_dir.glob(f"{case_id}_*_breast_mask.npy"))
    if sarit_candidates:
        return sarit_candidates[0]

    suffix = case_id.split("_", maxsplit=1)[1]
    try:
        breast_case_id = f"Breast_MRI_{int(suffix):03d}"
    except ValueError:
        return None

    filename = f"Segmentation_{breast_case_id}_Breast.seg.nrrd"
    candidates = [
        DEFAULT_DUKE_BREAST_ROOTS[1] / breast_case_id / filename,
        DEFAULT_DUKE_BREAST_ROOTS[2] / breast_case_id / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _axis_values_for_mask(shape: tuple[int, int, int], x_axis: int) -> np.ndarray:
    return np.indices(shape, sparse=True)[x_axis]


def _inner_wall_midline_from_projection(
    mask_yx: np.ndarray,
    *,
    image_midline_x: float,
    min_pair_fraction: float,
) -> tuple[float | None, str]:
    """Infer a vertical split from row-wise inner wall pairs around image center."""
    if mask_yx.ndim != PROJECTED_IMAGE_NDIMS or mask_yx.shape[1] == 0:
        return None, "invalid_projection"

    start = int(round(float(image_midline_x)))
    start = min(max(start, 0), mask_yx.shape[1] - 1)
    row_midpoints: list[float] = []

    for row in np.asarray(mask_yx, dtype=bool):
        if row[start]:
            continue

        left_edge = None
        for x in range(start - 1, -1, -1):
            if row[x]:
                left_edge = x
                break

        right_edge = None
        for x in range(start + 1, row.size):
            if row[x]:
                right_edge = x
                break

        if left_edge is None or right_edge is None or right_edge <= left_edge:
            continue
        row_midpoints.append((float(left_edge) + float(right_edge)) / 2.0)

    min_pairs = max(
        1, int(math.ceil(float(mask_yx.shape[0]) * max(0.0, min_pair_fraction)))
    )
    if len(row_midpoints) < min_pairs:
        return (
            None,
            f"middle_overlap_or_too_few_inner_wall_rows:{len(row_midpoints)}<{min_pairs}",
        )

    return float(np.median(np.asarray(row_midpoints, dtype=float))), ""


def _project_to_yx(
    mask: np.ndarray,
    *,
    coord_to_axis: tuple[int, int, int],
) -> np.ndarray:
    z_axis, y_axis, x_axis = coord_to_axis
    projected = np.max(mask, axis=z_axis)
    remaining_axes = [axis for axis in (0, 1, 2) if axis != z_axis]
    if remaining_axes == [y_axis, x_axis]:
        return projected
    if remaining_axes == [x_axis, y_axis]:
        return projected.T
    return projected


def _project_image_to_yx(
    image: np.ndarray,
    *,
    coord_to_axis: tuple[int, int, int],
) -> np.ndarray:
    z_axis, y_axis, x_axis = coord_to_axis
    projected = np.max(image, axis=z_axis)
    remaining_axes = [axis for axis in (0, 1, 2) if axis != z_axis]
    if remaining_axes == [y_axis, x_axis]:
        return projected
    if remaining_axes == [x_axis, y_axis]:
        return projected.T
    return projected


def _rotate_180_about_x_axis(
    mask: np.ndarray,
    *,
    x_axis: int,
) -> np.ndarray:
    """Rotate processed skeleton masks into the raw-image display orientation."""
    flip_axes = tuple(axis for axis in (0, 1, 2) if axis != x_axis)
    return np.flip(mask, axis=flip_axes)


def _window_image(image_yx: np.ndarray) -> np.ndarray:
    values = np.asarray(image_yx, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _parse_coord_mapping(row: dict[str, str]) -> tuple[int, int, int]:
    text = str(row.get("coordinate_order_assumption", ""))
    match = re.search(r"\((\d),\s*(\d),\s*(\d)\)", text)
    if match is not None:
        return tuple(int(match.group(i)) for i in (1, 2, 3))  # type: ignore[return-value]

    x_axis_text = str(row.get("left_right_axis", "array_axis_2"))
    x_match = re.search(r"(\d+)$", x_axis_text)
    x_axis = int(x_match.group(1)) if x_match else 2
    remaining = [axis for axis in (0, 1, 2) if axis != x_axis]
    return (remaining[0], remaining[1], x_axis)


def _iter_segment_entries(morphometry: dict[str, Any]) -> list[dict[str, Any]]:
    entries_out: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for component in morphometry.values():
        if not isinstance(component, dict):
            continue
        for name, entries in component.items():
            if "bifurcation" in str(name).lower() or not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                segment = entry.get("segment")
                if not isinstance(segment, dict):
                    continue
                start = segment.get("start")
                end = segment.get("end")
                if not (
                    isinstance(start, list)
                    and isinstance(end, list)
                    and len(start) == COORD_NDIMS
                    and len(end) == COORD_NDIMS
                ):
                    continue
                try:
                    start_tuple = tuple(int(v) for v in start)
                    end_tuple = tuple(int(v) for v in end)
                except (TypeError, ValueError):
                    continue
                key = tuple(sorted((start_tuple, end_tuple)))
                if key in seen:
                    continue
                seen.add(key)
                entries_out.append({"start": start_tuple, "end": end_tuple})
    return entries_out


def _preferred_coord_to_array_axis(
    layout: str | None,
) -> tuple[int, int, int] | None:
    """Return mapping from morphometry `(z, y, x)` coord index to array axis."""
    if layout == "zyx":
        return (0, 1, 2)
    if layout == "yxz":
        return (2, 0, 1)
    if layout == "xyz":
        return (2, 1, 0)
    return None


def _detect_coord_to_array_axis(
    *,
    segments: list[dict[str, Any]],
    array_shape: tuple[int, int, int],
    preferred: tuple[int, int, int] | None,
) -> tuple[tuple[int, int, int] | None, str]:
    coords: list[tuple[int, int, int]] = []
    for segment in segments:
        coords.append(segment["start"])
        coords.append(segment["end"])
    if not coords:
        return None, "no_segment_coordinates"

    valid: list[tuple[int, int, int]] = []
    for perm in itertools.permutations(range(3)):
        if all(
            all(0 <= coord[i] < array_shape[perm[i]] for i in range(3))
            for coord in coords
        ):
            valid.append(perm)

    if preferred is not None and preferred in valid:
        suffix = "unique" if len(valid) == 1 else f"preferred_of_{len(valid)}"
        return preferred, f"morph_coords_zyx_to_array_axes_{preferred}_{suffix}"
    if len(valid) == 1:
        return valid[0], f"morph_coords_zyx_to_array_axes_{valid[0]}_unique"
    if valid:
        return valid[0], f"ambiguous_valid_mappings_{valid}_using_{valid[0]}"
    return None, "no_valid_coordinate_mapping"


def _centroid_zyx_from_mask(
    mask: np.ndarray,
    coord_to_axis: tuple[int, int, int],
) -> tuple[float, float, float]:
    coords_by_axis = np.nonzero(mask)
    axis_means = [float(np.mean(values)) for values in coords_by_axis]
    z_axis, y_axis, x_axis = coord_to_axis
    return axis_means[z_axis], axis_means[y_axis], axis_means[x_axis]


def _split_counts(
    mask: np.ndarray,
    *,
    lr_axis: int,
    midline_x: float,
    side_mask_left: np.ndarray | None = None,
    side_mask_right: np.ndarray | None = None,
) -> tuple[int, int, int]:
    total = int(np.count_nonzero(mask))
    if side_mask_left is not None and side_mask_right is not None:
        left = int(np.count_nonzero(mask & side_mask_left))
        right = int(np.count_nonzero(mask & side_mask_right))
        return total, left, right

    coords = np.nonzero(mask)[lr_axis]
    left = int(np.count_nonzero(coords < midline_x))
    right = int(np.count_nonzero(coords >= midline_x))
    return total, left, right


def _confidence(
    *,
    exclude_reasons: list[str],
    distance_fraction: float | None,
    coordinate_order_assumption: str,
    used_breast_mask: bool,
    low_confidence_midline_fraction: float,
) -> str:
    if exclude_reasons:
        return "low"
    if distance_fraction is None or not math.isfinite(distance_fraction):
        return "low"
    if not used_breast_mask:
        return "low"
    if "ambiguous" in coordinate_order_assumption:
        return "low"
    if distance_fraction < low_confidence_midline_fraction:
        return "low"
    if distance_fraction < low_confidence_midline_fraction * 1.5:
        return "medium"
    return "high"


def _empty_row(row: dict[str, str], manifest_path: Path) -> dict[str, Any]:
    centerline_path = _path_from_row(row, "centerline_path")
    support_path = _path_from_row(row, "support_path")
    return {
        "case_id": row.get("case_id", ""),
        "dataset": row.get("dataset", ""),
        "manifest_path": str(manifest_path),
        "bilateral_mri": row.get("bilateral_mri", ""),
        "bilateral_breast_cancer": row.get("bilateral_breast_cancer", ""),
        "include_for_asymmetry": row.get("include_for_asymmetry", ""),
        "exclude_reason": "",
        "tumor_mask_path": row.get("tumor_mask_path", ""),
        "breast_mask_path": "",
        "centerline_path": str(centerline_path or ""),
        "skeleton_mask_path": str(centerline_path or ""),
        "support_mask_path": str(support_path or ""),
        "morphometry_path": row.get("morphometry_path", ""),
        "tumor_graph_features_path": row.get("tumor_graph_features_path", ""),
    }


def _final_symmetry_midline(
    breast_mask: np.ndarray,
    *,
    coord_to_axis: tuple[int, int, int],
    image_center_x: float,
    x_support_fraction: float,
) -> tuple[float | None, str]:
    """Compute the final inner-wall midline.

    Returns:
        (midline_x, failure_reason)
    """
    breast_yx = _project_to_yx(
        np.asarray(breast_mask, dtype=bool), coord_to_axis=coord_to_axis
    )
    if breast_yx.size == 0 or not np.any(breast_yx):
        return None, "empty_breast_mask"

    inner_midline, failure_reason = _inner_wall_midline_from_projection(
        breast_yx,
        image_midline_x=image_center_x,
        min_pair_fraction=x_support_fraction,
    )
    if inner_midline is None:
        return None, failure_reason

    return inner_midline, ""


def _side_from_x(x_value: float, midline_x: float) -> str:
    return "left" if x_value < midline_x else "right"


def _build_case_row(
    row: dict[str, str],
    *,
    manifest_path: Path,
    tumor_threshold: float,
    breast_threshold: float,
    x_support_fraction: float,
    low_confidence_midline_fraction: float,
) -> dict[str, Any]:
    """Build one side-assignment row using the final symmetry-line method."""
    out = _empty_row(row, manifest_path)
    exclude_reasons: list[str] = []

    centerline_path = _path_from_row(row, "centerline_path")
    support_path = _path_from_row(row, "support_path")
    morphometry_path = _path_from_row(row, "morphometry_path")
    tumor_mask_path = _path_from_row(row, "tumor_mask_path")
    tumor_graph_path = _path_from_row(row, "tumor_graph_features_path")
    run_summary_path = _path_from_row(row, "run_summary_path")

    skeleton_available = bool(centerline_path and centerline_path.exists())
    support_available = bool(support_path and support_path.exists())
    morph_available = bool(morphometry_path and morphometry_path.exists())
    tumor_graph_available = bool(tumor_graph_path and tumor_graph_path.exists())

    out.update(
        {
            "morphometry_json_available": int(morph_available),
            "skeleton_mask_available": int(skeleton_available),
            "support_mask_available": int(support_available),
            "breast_mask_available": 0,
            "left_breast_voxels": "",
            "right_breast_voxels": "",
            "side_split_method": "",
            "segment_count_total": "",
            "segment_count_assigned": "",
            "segment_count_midline_crossing": "",
            "segment_count_unassigned": "",
            "skeleton_voxel_count_total": "",
            "skeleton_voxel_count_left": "",
            "skeleton_voxel_count_right": "",
        }
    )
    if not tumor_graph_available:
        exclude_reasons.append("tumor_graph_features_missing")
    if not skeleton_available:
        exclude_reasons.append("skeleton_mask_missing")
    if not morph_available:
        exclude_reasons.append("morphometry_json_missing")
    if not tumor_mask_path or not tumor_mask_path.exists():
        exclude_reasons.append("tumor_mask_missing")

    if not skeleton_available:
        out["exclude_reason"] = ";".join(exclude_reasons)
        out["tumor_side_confidence"] = "low"
        return out

    try:
        skeleton = np.asarray(_load_array(centerline_path), dtype=bool)
    except Exception as exc:
        exclude_reasons.append(f"skeleton_load_failed:{exc}")
        out["exclude_reason"] = ";".join(exclude_reasons)
        out["tumor_side_confidence"] = "low"
        return out

    if skeleton.ndim != VOLUME_NDIMS:
        exclude_reasons.append(f"skeleton_not_3d:{skeleton.shape}")
        out["exclude_reason"] = ";".join(exclude_reasons)
        out["tumor_side_confidence"] = "low"
        return out

    array_shape = tuple(int(v) for v in skeleton.shape)
    summary = _read_json(run_summary_path) if run_summary_path else {}
    tumor_context = summary.get("tumor_context", {})
    if not isinstance(tumor_context, dict):
        tumor_context = {}
    tumor_layout = str(tumor_context.get("layout", "") or "")

    morphometry = _read_json(morphometry_path) if morph_available else {}
    segments = _iter_segment_entries(morphometry)
    preferred_mapping = _preferred_coord_to_array_axis(tumor_layout)
    coord_to_axis, coord_assumption = _detect_coord_to_array_axis(
        segments=segments,
        array_shape=array_shape,
        preferred=preferred_mapping,
    )

    if coord_to_axis is None:
        exclude_reasons.append("coordinate_order_unresolved")
        coord_to_axis = preferred_mapping or (0, 1, 2)

    z_axis, y_axis, x_axis = coord_to_axis
    shape_z = array_shape[z_axis]
    shape_y = array_shape[y_axis]
    shape_x = array_shape[x_axis]
    image_center_x = float(shape_x) / 2.0
    midline_x = image_center_x
    side_split_method = "image_midline_no_breast_mask"

    out.update(
        {
            "image_shape_z": shape_z,
            "image_shape_y": shape_y,
            "image_shape_x": shape_x,
            "image_midline_x": midline_x,
            "left_right_axis": f"array_axis_{x_axis}",
            "coordinate_order_assumption": coord_assumption,
        }
    )

    left_breast = None
    right_breast = None
    used_breast_mask = False
    breast_path = _resolve_breast_mask_path(row)
    if breast_path is not None:
        out["breast_mask_path"] = str(breast_path)
        try:
            breast_mask, _ = _load_mask_matching_shape(
                breast_path,
                expected_shape=array_shape,
                threshold=breast_threshold,
            )
            inner_midline_x, inner_failure_reason = _final_symmetry_midline(
                breast_mask,
                coord_to_axis=coord_to_axis,
                image_center_x=image_center_x,
                x_support_fraction=x_support_fraction,
            )
            if inner_midline_x is None:
                exclude_reasons.append(f"inner_walls_failed:{inner_failure_reason}")
                out["side_split_method"] = ""
                raise ValueError(inner_failure_reason)
            midline_x = inner_midline_x
            side_split_method = FINAL_METHOD
            out["image_midline_x"] = midline_x
            axis_values = _axis_values_for_mask(array_shape, x_axis)
            left_breast = breast_mask & (axis_values < midline_x)
            right_breast = breast_mask & (axis_values >= midline_x)
            out.update(
                {
                    "breast_mask_available": 1,
                    "left_breast_voxels": int(np.count_nonzero(left_breast)),
                    "right_breast_voxels": int(np.count_nonzero(right_breast)),
                    "side_split_method": side_split_method,
                }
            )
            used_breast_mask = True
        except Exception as exc:
            if not any(
                str(reason).startswith("inner_walls_failed:")
                for reason in exclude_reasons
            ):
                exclude_reasons.append(f"breast_mask_load_failed:{exc}")
            out["side_split_method"] = ""
    else:
        exclude_reasons.append("breast_mask_missing")
        out["side_split_method"] = ""

    distance_fraction = None
    try:
        if tumor_mask_path is None:
            raise FileNotFoundError("no tumor mask path")
        tumor_mask, tumor_layout_loaded = _load_mask_matching_shape(
            tumor_mask_path,
            expected_shape=array_shape,
            threshold=tumor_threshold,
        )
        if tumor_layout and tumor_layout_loaded != tumor_layout:
            exclude_reasons.append(
                f"tumor_layout_changed:{tumor_layout}->{tumor_layout_loaded}"
            )
        tumor_voxels = int(np.count_nonzero(tumor_mask))
        tumor_centroid_z, tumor_centroid_y, tumor_centroid_x = _centroid_zyx_from_mask(
            tumor_mask,
            coord_to_axis,
        )
        inferred_side = _side_from_x(tumor_centroid_x, midline_x)
        contralateral_side = "right" if inferred_side == "left" else "left"
        distance_vox = abs(float(tumor_centroid_x) - midline_x)
        distance_fraction = distance_vox / float(shape_x) if shape_x else math.nan
        out.update(
            {
                "tumor_centroid_x": tumor_centroid_x,
                "tumor_centroid_y": tumor_centroid_y,
                "tumor_centroid_z": tumor_centroid_z,
                "inferred_tumor_side": inferred_side,
                "inferred_contralateral_side": contralateral_side,
                "tumor_side_source": "inferred_mask_midline",
                "tumor_midline_distance_vox": distance_vox,
                "tumor_midline_distance_fraction": distance_fraction,
                "tumor_voxels": tumor_voxels,
            }
        )
    except Exception as exc:
        exclude_reasons.append(f"tumor_mask_load_failed:{exc}")
        out.update(
            {
                "tumor_side_source": "unresolved",
                "tumor_midline_distance_fraction": "",
                "tumor_voxels": "",
            }
        )

    segment_total = len(segments)
    assigned = 0
    midline_crossing = 0
    for segment in segments:
        start_x = float(segment["start"][2])
        end_x = float(segment["end"][2])
        midpoint_x = (start_x + end_x) / 2.0
        if (start_x < midline_x <= end_x) or (end_x < midline_x <= start_x):
            midline_crossing += 1
        if math.isfinite(midpoint_x):
            assigned += 1

    out.update(
        {
            "segment_count_total": segment_total,
            "segment_count_assigned": assigned,
            "segment_count_midline_crossing": midline_crossing,
            "segment_count_unassigned": max(0, segment_total - assigned),
        }
    )

    skel_total, skel_left, skel_right = _split_counts(
        skeleton,
        lr_axis=x_axis,
        midline_x=midline_x,
        side_mask_left=left_breast,
        side_mask_right=right_breast,
    )
    out.update(
        {
            "skeleton_voxel_count_total": skel_total,
            "skeleton_voxel_count_left": skel_left,
            "skeleton_voxel_count_right": skel_right,
        }
    )

    out["include_for_asymmetry"] = int(not exclude_reasons)
    out["exclude_reason"] = ";".join(exclude_reasons)
    out["tumor_side_confidence"] = _confidence(
        exclude_reasons=exclude_reasons,
        distance_fraction=distance_fraction,
        coordinate_order_assumption=coord_assumption,
        used_breast_mask=used_breast_mask,
        low_confidence_midline_fraction=low_confidence_midline_fraction,
    )
    return out


def _make_case_figure(
    row: dict[str, str],
    *,
    outdir: Path,
    tumor_threshold: float,
    breast_threshold: float,
) -> Path | None:
    """Draw one QC image with the final symmetry-line method in the title."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_id = row["case_id"]
    skeleton = np.load(row["skeleton_mask_path"], allow_pickle=False).astype(bool)
    expected_shape = tuple(int(v) for v in skeleton.shape)
    coord_to_axis = _parse_coord_mapping(row)
    x_axis = coord_to_axis[2]
    midline_x = float(row["image_midline_x"])

    tumor, _ = _load_mask_matching_shape(
        Path(row["tumor_mask_path"]),
        expected_shape=expected_shape,
        threshold=tumor_threshold,
    )

    breast = None
    breast_path = str(row.get("breast_mask_path", "")).strip()
    if breast_path:
        try:
            breast, _ = _load_mask_matching_shape(
                Path(breast_path),
                expected_shape=expected_shape,
                threshold=breast_threshold,
            )
        except Exception:
            breast = None

    skeleton_display = _rotate_180_about_x_axis(skeleton, x_axis=x_axis)
    breast_display = (
        _rotate_180_about_x_axis(breast, x_axis=x_axis) if breast is not None else None
    )
    skeleton_yx = _project_to_yx(skeleton_display, coord_to_axis=coord_to_axis)
    tumor_yx = _project_to_yx(tumor, coord_to_axis=coord_to_axis)
    breast_yx = (
        _project_to_yx(breast_display, coord_to_axis=coord_to_axis)
        if breast_display is not None
        else np.zeros_like(skeleton_yx, dtype=bool)
    )

    raw_image_yx = None
    image_path = MAMA_MIA_IMAGE_ROOT / case_id / f"{case_id}_0001.nii.gz"
    if image_path.exists():
        try:
            raw_image, _ = _load_image_matching_shape(
                image_path,
                expected_shape=expected_shape,
            )
            raw_image_yx = _window_image(
                _project_image_to_yx(raw_image, coord_to_axis=coord_to_axis)
            )
        except Exception as exc:
            print(f"Raw image load failed for {case_id}: {exc}")

    method_text = row.get("side_split_method", "")
    side = row.get("inferred_tumor_side", "")
    contra = row.get("inferred_contralateral_side", "")
    confidence = row.get("tumor_side_confidence", "")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"{case_id} | tumor={side}, contra={contra} | "
        f"confidence={confidence} | method={method_text}",
        fontsize=12,
    )
    raw_ax, overlay_ax = axes

    if raw_image_yx is not None:
        raw_ax.imshow(raw_image_yx, cmap="gray", interpolation="nearest")
    else:
        raw_ax.imshow(np.zeros_like(skeleton_yx, dtype=float), cmap="gray")
        raw_ax.text(0.5, 0.5, "raw image unavailable", ha="center", va="center")
    raw_ax.contour(tumor_yx.astype(float), levels=[0.5], colors="red", linewidths=1.3)
    raw_ax.axvline(midline_x, color="cyan", linewidth=1.2)
    raw_ax.set_title("Raw MRI MIP + tumor outline")
    raw_ax.set_xlabel(f"x / left-right axis {x_axis}")
    raw_ax.set_ylabel("y")

    if np.any(breast_yx):
        overlay_ax.imshow(breast_yx, cmap="Greys", alpha=0.18, interpolation="nearest")
    overlay_ax.imshow(skeleton_yx, cmap="Blues", alpha=0.75, interpolation="nearest")
    overlay_ax.imshow(np.ma.masked_where(~tumor_yx, tumor_yx), cmap="Reds", alpha=0.45)
    overlay_ax.contour(
        tumor_yx.astype(float), levels=[0.5], colors="yellow", linewidths=1.3
    )
    overlay_ax.axvline(midline_x, color="cyan", linewidth=1.5)
    overlay_ax.set_title("Breast mask + skeleton + tumor")
    overlay_ax.set_xlabel(f"x / left-right axis {x_axis}")
    overlay_ax.set_ylabel("y")
    overlay_ax.text(
        0.01,
        0.99,
        f"method={method_text}\n"
        f"midline_x={midline_x:.2f}\n"
        f"dist_frac={row.get('tumor_midline_distance_fraction', '')}",
        transform=overlay_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
    )

    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = outdir / f"{case_id}_side_qc_mip.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _write_index(
    outdir: Path, rows: list[dict[str, str]], figure_paths: dict[str, Path]
) -> None:
    lines = [
        "<html><body>",
        "<h1>DUKE final symmetry-line QC</h1>",
        "<table border='1' cellspacing='0' cellpadding='4'>",
        "<tr><th>case</th><th>side</th><th>confidence</th><th>method</th>"
        "<th>midline x</th><th>distance fraction</th><th>figure</th></tr>",
    ]
    for row in rows:
        case_id = row.get("case_id", "")
        fig_path = figure_paths.get(case_id)
        fig_link = ""
        if fig_path is not None:
            fig_link = (
                f"<a href='{html.escape(fig_path.name)}'>png</a><br>"
                f"<img src='{html.escape(fig_path.name)}' width='360'>"
            )
        lines.append(
            "<tr>"
            f"<td>{html.escape(case_id)}</td>"
            f"<td>{html.escape(row.get('inferred_tumor_side', ''))}</td>"
            f"<td>{html.escape(row.get('tumor_side_confidence', ''))}</td>"
            f"<td>{html.escape(row.get('side_split_method', ''))}</td>"
            f"<td>{html.escape(row.get('image_midline_x', ''))}</td>"
            f"<td>{html.escape(row.get('tumor_midline_distance_fraction', ''))}</td>"
            f"<td>{fig_link}</td>"
            "</tr>"
        )
    lines.extend(["</table>", "</body></html>"])
    (outdir / "index.html").write_text("\n".join(lines))


def main() -> None:
    """Run the DUKE final symmetry-line export and QC figure generation."""
    args = _parse_args()
    if args.out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output CSV exists: {args.out_csv}. Use --overwrite.")
    if args.outdir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory exists: {args.outdir}. Use --overwrite."
        )

    rows = _read_manifest(args.manifest)
    if args.case_id:
        rows = [row for row in rows if row.get("case_id") == args.case_id]
    if args.max_cases is not None:
        rows = rows[: max(0, int(args.max_cases))]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIDE_QC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = _build_case_row(
                row,
                manifest_path=args.manifest.resolve(),
                tumor_threshold=float(args.tumor_threshold),
                breast_threshold=float(args.breast_threshold),
                x_support_fraction=float(args.x_support_fraction),
                low_confidence_midline_fraction=float(
                    args.low_confidence_midline_fraction
                ),
            )
            writer.writerow(out)
            output_rows.append({k: str(v) for k, v in out.items()})

    figure_paths: dict[str, Path] = {}
    for row in output_rows:
        try:
            fig_path = _make_case_figure(
                row,
                outdir=args.outdir,
                tumor_threshold=float(args.tumor_threshold),
                breast_threshold=float(args.breast_threshold),
            )
            if fig_path is not None:
                figure_paths[row["case_id"]] = fig_path
        except Exception as exc:
            print(f"QC figure failed for {row.get('case_id', '')}: {exc}")

    _write_index(args.outdir, output_rows, figure_paths)
    print(f"Wrote {len(output_rows)} rows to {args.out_csv}")
    print(f"Wrote {len(figure_paths)} QC figures and index.html to {args.outdir}")


if __name__ == "__main__":
    main()
