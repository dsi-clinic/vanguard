"""Mask loading, orientation handling, and annotation alignment helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from graph_extraction.constants import (
    NDIM_3D,
    NDIM_4D,
    SEG_NRRD_MAX_LAYERS,
    VIZ_FLIP_SPECS,
)

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


def _select_zyx_layout(
    volume: np.ndarray, expected_shape_zyx: tuple[int, int, int]
) -> tuple[np.ndarray, str]:
    """Pick the axis order whose shape matches the expected `(z, y, x)` shape."""
    candidates: tuple[tuple[str, np.ndarray], ...] = (
        ("zyx", volume),
        ("yxz", np.transpose(volume, (1, 2, 0))),
        ("xyz", np.transpose(volume, (2, 1, 0))),
    )
    for layout_name, candidate in candidates:
        if tuple(candidate.shape) == tuple(expected_shape_zyx):
            return candidate, layout_name
    raise ValueError(
        f"mask shape mismatch. Expected {expected_shape_zyx}, got {volume.shape}"
        " (and common transposes)"
    )


def _apply_flip_spec(mask_zyx: np.ndarray, flip_spec: str) -> np.ndarray:
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


def _to_breast_mri_case_id_strict(case_id: str) -> str | None:
    clean = str(case_id).strip()
    if clean == "":
        return None

    for pattern in (r"Breast_MRI_(\d+)", r"DUKE_(\d+)"):
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match is not None:
            return f"Breast_MRI_{int(match.group(1)):03d}"

    return None


def _resolve_radiologist_nrrd_root(annotations_dir: Path) -> Path:
    candidate = annotations_dir / "Segmentation_Masks_NRRD"
    if candidate.exists():
        return candidate
    return annotations_dir


def _resolve_annotation_segment_path(
    *,
    nrrd_root: Path,
    case_id: str,
    segment_glob: str,
) -> Path | None:
    case_dir = nrrd_root / case_id
    if not case_dir.exists():
        return None
    matches = sorted(case_dir.glob(segment_glob))
    if not matches:
        return None
    return matches[0]


def _resolve_radiologist_seg_path(
    *,
    nrrd_root: Path,
    case_id: str,
) -> Path | None:
    return _resolve_annotation_segment_path(
        nrrd_root=nrrd_root,
        case_id=case_id,
        segment_glob="Segmentation_*_Dense_and_Vessels.seg.nrrd",
    )


def _resolve_breast_seg_path(
    *,
    nrrd_root: Path,
    case_id: str,
) -> Path | None:
    return _resolve_annotation_segment_path(
        nrrd_root=nrrd_root,
        case_id=case_id,
        segment_glob="Segmentation_*_Breast.seg.nrrd",
    )


def _resolve_tumor_mask_path(
    *,
    tumor_mask_dir: Path,
    case_id: str,
) -> Path | None:
    tumor_extensions = (".nii.gz", ".nii", ".nrrd")
    candidates: list[Path] = [
        tumor_mask_dir / f"{case_id}{ext}" for ext in tumor_extensions
    ]
    case_id = _to_breast_mri_case_id_strict(case_id)
    if case_id is not None:
        candidates.extend(
            tumor_mask_dir / f"{case_id}{ext}" for ext in tumor_extensions
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_seg_nrrd_segments(path: Path) -> dict[str, dict[str, int | None]]:
    """Read named segment metadata from a Slicer `.seg.nrrd` header.

    Radiologist annotation files can contain several segments in one file. This
    helper extracts the segment name, stored integer label, and optional layer
    index so later code can reliably find the vessel segment.
    """
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
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a 3D NRRD mask and align it to the study volume.

    NRRD is a common medical-image file format used by tools such as 3D Slicer.
    The annotation files we use are not always saved with the same axis order.
    This helper tries the layouts we expect in this project, picks the one that
    matches `expected_shape_zyx`, thresholds it into a boolean mask, and
    records the chosen layout in the returned metadata.
    """
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(
        np.float32, copy=False
    )
    if arr.ndim != NDIM_3D:
        raise ValueError(
            f"{label_name} NRRD must be 3D, got shape {arr.shape} from {path}"
        )

    selected, selected_layout = _select_zyx_layout(arr, expected_shape_zyx)

    mask = (selected > float(threshold)).astype(bool, copy=False)
    if not np.any(mask):
        raise ValueError(f"{label_name} mask is empty in {path}")
    return mask, {
        "path": str(path),
        "layout": selected_layout,
        "voxels": int(np.count_nonzero(mask)),
    }


def _spacing_for_selected_layout(
    spacing_zyx_native: tuple[float, float, float], selected_layout: str
) -> tuple[float, float, float]:
    sz, sy, sx = (
        float(spacing_zyx_native[0]),
        float(spacing_zyx_native[1]),
        float(spacing_zyx_native[2]),
    )
    if selected_layout == "zyx":
        return (sz, sy, sx)
    if selected_layout == "yxz":
        return (sy, sx, sz)
    if selected_layout == "xyz":
        return (sx, sy, sz)
    return (sz, sy, sx)


def _load_binary_mask_with_spacing(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    label_name: str,
    threshold: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a 3D mask and return spacing in the aligned processing layout.

    This is the version used when physical distance matters. It does the same
    axis-alignment work as `_load_binary_mask_nrrd`, then also reorders the
    spacing tuple so later distance calculations are in the correct `(z, y, x)`
    coordinate system.
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if arr.ndim != NDIM_3D:
        raise ValueError(
            f"{label_name} mask must be 3D, got shape {arr.shape} from {path}"
        )

    spacing_xyz_raw = tuple(float(v) for v in image.GetSpacing())
    if len(spacing_xyz_raw) < NDIM_3D:
        raise ValueError(
            f"{label_name} mask spacing must have 3 axes, got {spacing_xyz_raw}"
        )
    spacing_zyx_native = (
        float(spacing_xyz_raw[2]),
        float(spacing_xyz_raw[1]),
        float(spacing_xyz_raw[0]),
    )

    selected, selected_layout = _select_zyx_layout(arr, expected_shape_zyx)
    spacing_zyx = _spacing_for_selected_layout(spacing_zyx_native, selected_layout)

    mask = (selected > float(threshold)).astype(bool, copy=False)
    if not np.any(mask):
        raise ValueError(f"{label_name} mask is empty in {path}")
    return mask, {
        "path": str(path),
        "layout": selected_layout,
        "voxels": int(np.count_nonzero(mask)),
        "spacing_mm_zyx": [
            float(spacing_zyx[0]),
            float(spacing_zyx[1]),
            float(spacing_zyx[2]),
        ],
    }


def _select_radiologist_vessel_layer(
    arr: np.ndarray,
    *,
    vessel_layer: int | None,
    path: Path,
) -> tuple[np.ndarray, int | None, str | None, bool]:
    """Extract the 3D vessel layer from a layered radiologist annotation.

    Some radiologist NRRDs store layers on the last axis and others on the
    first. This helper handles both cases and returns bookkeeping about which
    layer was used and whether the choice came directly from the header
    metadata.
    """
    if arr.shape[-1] <= SEG_NRRD_MAX_LAYERS:
        layer_count = int(arr.shape[-1])
        selected_layer_index = (
            int(vessel_layer)
            if vessel_layer is not None and 0 <= int(vessel_layer) < layer_count
            else 0
        )
        return (
            arr[..., selected_layer_index],
            selected_layer_index,
            "layer_last",
            vessel_layer is not None and int(vessel_layer) == selected_layer_index,
        )

    if arr.shape[0] <= SEG_NRRD_MAX_LAYERS:
        layer_count = int(arr.shape[0])
        selected_layer_index = (
            int(vessel_layer)
            if vessel_layer is not None and 0 <= int(vessel_layer) < layer_count
            else 0
        )
        return (
            arr[selected_layer_index, ...],
            selected_layer_index,
            "layer_first",
            vessel_layer is not None and int(vessel_layer) == selected_layer_index,
        )

    raise ValueError(
        "Radiologist seg NRRD 4D shape unsupported for layered decoding: "
        f"{arr.shape} from {path}"
    )


def _load_radiologist_vessel_mask_nrrd(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Load the vessel mask from the radiologist annotation file.

    The radiologist file may contain several named structures. This helper uses
    the parsed metadata to find the vessel segment, applies the stored label when
    possible, and falls back to a non-zero mask if needed.
    """
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
            break

    if vessel_name is None:
        raise ValueError(
            f"Could not find a vessel segment in {path}. "
            f"Found segments: {sorted(segments_by_name.keys())}"
        )

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int16, copy=False)
    if arr.ndim == NDIM_3D:
        selected_arr_3d = arr
        selected_layer_index: int | None = None
        selected_layer_layout: str | None = None
        selected_layer_via_header = False
    elif arr.ndim == NDIM_4D:
        (
            selected_arr_3d,
            selected_layer_index,
            selected_layer_layout,
            selected_layer_via_header,
        ) = _select_radiologist_vessel_layer(arr, vessel_layer=vessel_layer, path=path)
    else:
        raise ValueError(
            f"Radiologist seg NRRD must be 3D/4D, got shape {arr.shape} from {path}"
        )

    selected, selected_layout = _select_zyx_layout(selected_arr_3d, expected_shape_zyx)

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
        "segment_label_value": None if vessel_label is None else int(vessel_label),
        "segment_layer": None
        if selected_layer_index is None
        else int(selected_layer_index),
        "segment_layer_from_header": bool(selected_layer_via_header),
        "segment_layer_layout": selected_layer_layout,
        "layout": selected_layout,
        "voxels": int(np.count_nonzero(vessel_mask)),
    }


def _maybe_load_radiologist_context_for_mip(
    *,
    case_id: str,
    shape_zyx: tuple[int, int, int],
    radiologist_annotations_dir: Path,
) -> dict[str, object]:
    """Load optional radiologist reference masks for debug visualizations.

    These masks are used for side-by-side MIP comparisons. Missing files should
    not break the main pipeline, so the function returns a status dictionary
    instead of raising when the reference data is unavailable.
    """
    case_id = _to_breast_mri_case_id_strict(case_id)
    if case_id is None:
        return {"status": "no_case_mapping", "resolved_case_id": None}

    nrrd_root = _resolve_radiologist_nrrd_root(radiologist_annotations_dir)
    if not nrrd_root.exists():
        return {
            "status": "annotations_root_missing",
            "resolved_case_id": case_id,
            "nrrd_root": str(nrrd_root),
        }

    breast_path = _resolve_breast_seg_path(nrrd_root=nrrd_root, case_id=case_id)
    vessel_path = _resolve_radiologist_seg_path(nrrd_root=nrrd_root, case_id=case_id)
    if breast_path is None or vessel_path is None:
        return {
            "status": "annotation_files_missing",
            "resolved_case_id": case_id,
            "nrrd_root": str(nrrd_root),
            "breast_path": None if breast_path is None else str(breast_path),
            "vessel_path": None if vessel_path is None else str(vessel_path),
        }

    try:
        breast_mask, breast_info = _load_binary_mask_nrrd(
            breast_path,
            expected_shape_zyx=shape_zyx,
            label_name="Breast",
            threshold=0.0,
        )
        vessel_mask, vessel_info = _load_radiologist_vessel_mask_nrrd(
            vessel_path,
            expected_shape_zyx=shape_zyx,
        )
    except Exception as exc:
        return {
            "status": "annotation_load_failed",
            "resolved_case_id": case_id,
            "breast_path": str(breast_path),
            "vessel_path": str(vessel_path),
            "error": str(exc),
        }

    return {
        "status": "ok",
        "resolved_case_id": case_id,
        "nrrd_root": str(nrrd_root),
        "breast_path": str(breast_path),
        "vessel_path": str(vessel_path),
        "breast_info": breast_info,
        "vessel_info": vessel_info,
        "breast_mask_zyx": breast_mask,
        "radiologist_mask_zyx": vessel_mask,
    }


def _maybe_load_breast_context_for_alignment(
    *,
    case_id: str,
    shape_zyx: tuple[int, int, int],
    radiologist_annotations_dir: Path,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Load a breast mask that can be used to orient the tumor mask.

    We use the breast outline as a simple reference when deciding whether the
    tumor mask needs to be flipped to match the processed vessel outputs.
    """
    case_id = _to_breast_mri_case_id_strict(case_id)
    if case_id is None:
        return None, {"status": "no_case_mapping", "resolved_case_id": None}

    nrrd_root = _resolve_radiologist_nrrd_root(radiologist_annotations_dir)
    if not nrrd_root.exists():
        return None, {
            "status": "annotations_root_missing",
            "resolved_case_id": case_id,
            "nrrd_root": str(nrrd_root),
        }

    breast_path = _resolve_breast_seg_path(nrrd_root=nrrd_root, case_id=case_id)
    if breast_path is None:
        return None, {
            "status": "breast_mask_missing",
            "resolved_case_id": case_id,
            "nrrd_root": str(nrrd_root),
        }

    try:
        breast_mask, breast_info = _load_binary_mask_nrrd(
            breast_path,
            expected_shape_zyx=shape_zyx,
            label_name="Breast",
            threshold=0.0,
        )
    except Exception as exc:
        return None, {
            "status": "breast_mask_load_failed",
            "resolved_case_id": case_id,
            "nrrd_root": str(nrrd_root),
            "breast_path": str(breast_path),
            "error": str(exc),
        }

    return breast_mask, {
        "status": "ok",
        "resolved_case_id": case_id,
        "nrrd_root": str(nrrd_root),
        "breast_path": str(breast_path),
        "breast_info": breast_info,
    }


def _maybe_load_tumor_context_for_features(
    *,
    case_id: str,
    shape_zyx: tuple[int, int, int],
    tumor_mask_dir: Path,
    radiologist_annotations_dir: Path,
) -> tuple[np.ndarray | None, tuple[float, float, float] | None, dict[str, object]]:
    """Load the tumor mask and align it to the processing orientation.

    The saved tumor mask and the processed vessel outputs are not always stored
    with the same orientation conventions. This helper loads the tumor mask,
    checks whether a breast reference mask is available, and applies the
    best-matching flip so that later tumor-centered features are measured in the
    same coordinate system as the centerline output.

    Returns the aligned tumor mask, voxel spacing in millimeters, and a context
    dictionary that records what happened for debugging and provenance.
    """
    resolved_tumor_path = _resolve_tumor_mask_path(
        tumor_mask_dir=tumor_mask_dir,
        case_id=case_id,
    )
    if resolved_tumor_path is None:
        return (
            None,
            None,
            {
                "status": "tumor_mask_missing",
                "mask_dir": str(tumor_mask_dir),
            },
        )

    try:
        tumor_mask_model, tumor_info = _load_binary_mask_with_spacing(
            resolved_tumor_path,
            expected_shape_zyx=shape_zyx,
            label_name="Tumor",
            threshold=0.5,
        )
    except Exception as exc:
        return (
            None,
            None,
            {
                "status": "tumor_mask_load_failed",
                "path": str(resolved_tumor_path),
                "mask_dir": str(tumor_mask_dir),
                "error": str(exc),
            },
        )

    breast_mask_model, breast_context = _maybe_load_breast_context_for_alignment(
        case_id=case_id,
        shape_zyx=shape_zyx,
        radiologist_annotations_dir=radiologist_annotations_dir,
    )

    alignment_to_breast: dict[str, object] | None = None
    flip_to_model = "none"
    if breast_mask_model is not None:
        alignment_to_breast = _choose_flip_by_containment(
            tumor_mask_model,
            breast_mask_model,
        )
        flip_to_model = str(alignment_to_breast["best_flip_spec"])
        tumor_mask_model = _apply_flip_spec(tumor_mask_model, flip_to_model)

    spacing_values = tumor_info.get("spacing_mm_zyx", [1.0, 1.0, 1.0])
    spacing_mm_zyx = (
        float(spacing_values[0]),
        float(spacing_values[1]),
        float(spacing_values[2]),
    )
    context: dict[str, object] = {
        "status": "ok",
        "path": str(resolved_tumor_path),
        "mask_dir": str(tumor_mask_dir),
        "processing_flip_to_model": flip_to_model,
        "alignment_to_breast": alignment_to_breast,
        "breast_alignment_context": breast_context,
        "voxels_after_alignment": int(np.count_nonzero(tumor_mask_model)),
        **tumor_info,
    }
    return tumor_mask_model, spacing_mm_zyx, context


def _choose_flip_by_containment(
    candidate_mask_zyx: np.ndarray,
    reference_mask_zyx: np.ndarray,
) -> dict[str, object]:
    """Pick the flip where one mask best sits inside another mask.

    This is a pragmatic alignment heuristic. We try a small set of flips and
    keep the one where the candidate mask has the largest overlap inside the
    reference mask.
    """
    cand = np.asarray(candidate_mask_zyx, dtype=bool)
    ref = np.asarray(reference_mask_zyx, dtype=bool)
    if cand.shape != ref.shape:
        raise ValueError(
            "Containment flip shape mismatch: "
            f"{tuple(int(v) for v in cand.shape)} vs {tuple(int(v) for v in ref.shape)}"
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

    rows = _build_containment_rows(cand=cand, ref=ref, candidate_voxels=cand_voxels)
    rows_sorted = sorted(
        rows,
        key=lambda d: (
            float(d["inside_ratio"]),
            int(d["inside_voxels"]),
            1 if str(d["flip_spec"]) == "none" else 0,
        ),
        reverse=True,
    )
    best = rows_sorted[0]
    return {
        "best_flip_spec": str(best["flip_spec"]),
        "best_inside_ratio": float(best["inside_ratio"]),
        "best_inside_voxels": int(best["inside_voxels"]),
        "all": rows_sorted,
    }


def _build_containment_rows(
    cand: np.ndarray,
    ref: np.ndarray,
    candidate_voxels: int,
) -> list[dict[str, object]]:
    """Score each allowed flip by how much of the candidate stays inside `ref`."""
    rows: list[dict[str, object]] = []
    for spec in VIZ_FLIP_SPECS:
        flipped = _apply_flip_spec(cand, spec)
        inside_voxels = int(np.count_nonzero(flipped & ref))
        rows.append(
            {
                "flip_spec": str(spec),
                "inside_ratio": float(inside_voxels / float(candidate_voxels)),
                "inside_voxels": inside_voxels,
                "candidate_voxels": candidate_voxels,
            }
        )
    return rows
