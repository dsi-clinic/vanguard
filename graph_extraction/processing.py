"""Shared 4D skeleton processing and morphometry helpers."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from graph_extraction.core4d import (
    NDIM_3D,
    NDIM_4D,
    discover_study_timepoints,
    load_time_series_from_files,
)
from graph_extraction.skeleton_to_graph import (
    assign_component_labels,
    build_vessel_json,
    detect_bifurcations,
    edges_to_segments,
    extract_segments,
    obtain_radius_map,
    segments_to_graph,
)
from graph_extraction.tc4d import run_tc4d_from_priority
from graph_extraction.vessel_mip import render_vessel_coverage_mip

DEFAULT_RADIOLOGIST_ANNOTATIONS_DIR = Path(
    "/net/projects2/vanguard/Duke-Breast-Cancer-MRI-Supplement-v3"
)
DEFAULT_TUMOR_MASK_DIR = Path(
    "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert"
)
PROCESSING_VIZ_FLIP_SPEC = "z"
VIZ_FLIP_SPECS = ("none", "z", "y", "x", "zy", "zx", "yx", "zyx")
SEG_NRRD_MAX_LAYERS = 32

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
    """Pick a 3D orientation layout whose shape matches the expected ZYX shape."""
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

def _to_breast_mri_case_id_strict(study_id: str) -> str | None:
    """Map `DUKE_###` or `Breast_MRI_###` into `Breast_MRI_###`."""
    clean = str(study_id).strip()
    if clean == "":
        return None

    for pattern in (r"Breast_MRI_(\d+)", r"DUKE_(\d+)"):
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match is not None:
            return f"Breast_MRI_{int(match.group(1)):03d}"

    return None


def _resolve_radiologist_nrrd_root(annotations_dir: Path) -> Path:
    """Resolve `Segmentation_Masks_NRRD` from either base or direct root."""
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
    """Resolve a segment path for one Breast_MRI case using a glob pattern."""
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
    """Resolve Dense+Vessels path for one Breast_MRI case."""
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
    """Resolve breast mask path for one Breast_MRI case."""
    return _resolve_annotation_segment_path(
        nrrd_root=nrrd_root,
        case_id=case_id,
        segment_glob="Segmentation_*_Breast.seg.nrrd",
    )


def _resolve_tumor_mask_path(
    *,
    tumor_mask_dir: Path,
    study_id: str,
) -> Path | None:
    """Resolve tumor mask path from common study-id tokens."""
    tumor_extensions = (".nii.gz", ".nii", ".nrrd")
    candidates: list[Path] = [
        tumor_mask_dir / f"{study_id}{ext}" for ext in tumor_extensions
    ]
    case_id = _to_breast_mri_case_id_strict(study_id)
    if case_id is not None:
        candidates.extend(
            tumor_mask_dir / f"{case_id}{ext}" for ext in tumor_extensions
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_seg_nrrd_segments(path: Path) -> dict[str, dict[str, int | None]]:
    """Parse Slicer `.seg.nrrd` segment metadata keyed by segment name."""
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
    """Load binary NRRD and align to expected `(z,y,x)`."""
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


def _select_radiologist_vessel_layer(
    arr: np.ndarray,
    *,
    vessel_layer: int | None,
    path: Path,
) -> tuple[np.ndarray, int | None, str | None, bool]:
    """Select radiologist vessel layer and report whether selection used header index."""
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
    """Load vessel segment from Dense_and_Vessels `.seg.nrrd`."""
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
    study_id: str,
    shape_zyx: tuple[int, int, int],
    radiologist_annotations_dir: Path,
) -> dict[str, object]:
    """Attempt to load radiologist/breast masks for MIP rendering."""
    case_id = _to_breast_mri_case_id_strict(study_id)
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


def _choose_flip_by_containment(
    candidate_mask_zyx: np.ndarray,
    reference_mask_zyx: np.ndarray,
) -> dict[str, object]:
    """Select flip that maximizes candidate containment inside reference."""
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


def process_4d_study(
    *,
    input_dir: Path,
    study_id: str,
    output_dir: Path,
    force_skeleton: bool,
    force_features: bool,
    save_exam_masks: bool,
    save_center_manifold_mask: bool,
    render_mip: bool,
    mip_dpi: int,
    radiologist_annotations_dir: Path,
    tumor_mask_dir: Path,
) -> dict[str, object]:
    """Run 4D exam-level skeleton extraction + morphometry for one study."""
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton_path = output_dir / f"{study_id}_skeleton_4d_exam_mask.npy"
    support_path = output_dir / f"{study_id}_skeleton_4d_exam_support_mask.npy"
    manifold_path = output_dir / f"{study_id}_center_manifold_4d_mask.npy"
    morphometry_path = output_dir / f"{study_id}_morphometry.json"
    coverage_mip_path = output_dir / f"{study_id}_vessel_coverage_mip.png"

    skeleton_took = 0.0
    features_took = 0.0
    mip_took = 0.0

    discovered_files: list[Path] | None = None
    discovered_timepoints: list[int] | None = None
    effective_min_temporal_support: int | None = None
    tc4d_params: dict[str, object] | None = None
    tc4d_diagnostics: dict[str, object] | None = None

    if (
        save_exam_masks
        and skeleton_path.exists()
        and support_path.exists()
        and not force_skeleton
    ):
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
        )
        tc4d_result, tc4d_params, tc4d_diagnostics = run_tc4d_from_priority(
            priority_4d,
        )
        mask_4d = np.asarray(tc4d_result["mask_4d"], dtype=bool)
        skeleton_mask = np.asarray(tc4d_result["exam_mask"], dtype=bool)
        support_mask = np.asarray(tc4d_result["support_mask"], dtype=bool)
        effective_min_temporal_support = int(
            tc4d_result["effective_min_temporal_support"]
        )
        if not np.any(skeleton_mask):
            raise ValueError("TC4D produced an empty exam-level skeleton.")
        if not np.any(support_mask):
            raise ValueError("TC4D produced an empty support mask.")

        if save_exam_masks:
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

    mip_status = "skipped"
    coverage_mip_diagnostics: dict[str, object] | None = None
    radiologist_context_for_summary: dict[str, object] | None = None
    tumor_context_for_summary: dict[str, object] | None = None
    if render_mip:
        t2 = time.perf_counter()
        try:
            shape_zyx = tuple(int(v) for v in skeleton_mask.shape)
            radiologist_context = _maybe_load_radiologist_context_for_mip(
                study_id=study_id,
                shape_zyx=shape_zyx,
                radiologist_annotations_dir=radiologist_annotations_dir,
            )
            radiologist_mask_model: np.ndarray | None = None
            breast_mask_model: np.ndarray | None = None
            radiologist_mask_viz: np.ndarray | None = None
            breast_mask_viz: np.ndarray | None = None
            if radiologist_context.get("status") == "ok":
                radiologist_mask_model = np.asarray(
                    radiologist_context["radiologist_mask_zyx"], dtype=bool
                )
                breast_mask_model = np.asarray(
                    radiologist_context["breast_mask_zyx"], dtype=bool
                )
                radiologist_mask_viz = _apply_flip_spec(
                    radiologist_mask_model,
                    PROCESSING_VIZ_FLIP_SPEC,
                )
                breast_mask_viz = _apply_flip_spec(
                    breast_mask_model,
                    PROCESSING_VIZ_FLIP_SPEC,
                )

            tumor_mask_viz: np.ndarray | None = None
            resolved_tumor_path = _resolve_tumor_mask_path(
                tumor_mask_dir=tumor_mask_dir,
                study_id=study_id,
            )
            if resolved_tumor_path is None:
                tumor_context_for_summary = {
                    "status": "tumor_mask_missing",
                    "mask_dir": str(tumor_mask_dir),
                }
            else:
                try:
                    tumor_mask_model, tumor_info = _load_binary_mask_nrrd(
                        resolved_tumor_path,
                        expected_shape_zyx=shape_zyx,
                        label_name="Tumor",
                        threshold=0.5,
                    )
                    alignment_to_breast: dict[str, object] | None = None
                    flip_to_model = "none"
                    if breast_mask_model is not None:
                        alignment_to_breast = _choose_flip_by_containment(
                            tumor_mask_model,
                            breast_mask_model,
                        )
                        flip_to_model = str(alignment_to_breast["best_flip_spec"])
                        tumor_mask_model = _apply_flip_spec(
                            tumor_mask_model, flip_to_model
                        )

                    tumor_mask_viz = _apply_flip_spec(
                        tumor_mask_model,
                        PROCESSING_VIZ_FLIP_SPEC,
                    )
                    tumor_context_for_summary = {
                        "status": "ok",
                        "path": str(resolved_tumor_path),
                        "mask_dir": str(tumor_mask_dir),
                        "processing_flip_to_model": flip_to_model,
                        "alignment_to_breast": alignment_to_breast,
                        "voxels_after_alignment": int(
                            np.count_nonzero(tumor_mask_model)
                        ),
                        **tumor_info,
                    }
                except Exception as exc:
                    tumor_context_for_summary = {
                        "status": "tumor_mask_load_failed",
                        "path": str(resolved_tumor_path),
                        "mask_dir": str(tumor_mask_dir),
                        "error": str(exc),
                    }

            method_label = "tc4d"
            row_masks: list[tuple[str, np.ndarray]] = [
                (
                    method_label,
                    _apply_flip_spec(skeleton_mask, PROCESSING_VIZ_FLIP_SPEC),
                )
            ]
            if radiologist_mask_viz is not None:
                row_masks.append(("radiologist", radiologist_mask_viz))

            coverage_mip_diag = render_vessel_coverage_mip(
                row_masks=row_masks,
                output_path=coverage_mip_path,
                case_label=study_id,
                title_prefix=f"{method_label} vessel coverage mip",
                radiologist_mask_zyx=radiologist_mask_viz,
                breast_mask_zyx=breast_mask_viz,
                tumor_mask_zyx=tumor_mask_viz,
                vessel_color="#111827",
                dpi=int(mip_dpi),
            )
            radiologist_context_for_summary = {
                k: v
                for k, v in radiologist_context.items()
                if k not in {"breast_mask_zyx", "radiologist_mask_zyx"}
            }
            coverage_mip_diagnostics = {
                **coverage_mip_diag,
                "radiologist_context": radiologist_context_for_summary,
                "tumor_context": tumor_context_for_summary,
                "visualization_flip_spec": PROCESSING_VIZ_FLIP_SPEC,
            }
            mip_status = "computed"
        except Exception as exc:
            mip_status = "failed"
            coverage_mip_diagnostics = {"error": str(exc)}
        mip_took = float(time.perf_counter() - t2)

    return {
        "mode": "tc4d",
        "study_id": study_id,
        "input_dir": str(input_dir),
        "algorithm": "tc4d",
        "effective_min_temporal_support": (
            None
            if effective_min_temporal_support is None
            else int(effective_min_temporal_support)
        ),
        "skeleton_status": skeleton_status,
        "features_status": features_status,
        "skeleton_voxels": int(np.count_nonzero(skeleton_mask)),
        "support_voxels": int(np.count_nonzero(support_mask)),
        "skeleton_path": str(skeleton_path) if save_exam_masks else None,
        "support_path": str(support_path) if save_exam_masks else None,
        "manifold_path": str(manifold_path) if save_center_manifold_mask else None,
        "morphometry_path": str(morphometry_path),
        "coverage_mip_path": str(coverage_mip_path)
        if mip_status == "computed"
        else None,
        "coverage_mip_status": mip_status,
        "coverage_mip_diagnostics": coverage_mip_diagnostics,
        "radiologist_context": radiologist_context_for_summary,
        "tumor_context": tumor_context_for_summary,
        "feature_stats": feature_stats,
        "tc4d_params": tc4d_params,
        "tc4d_diagnostics": tc4d_diagnostics,
        "study_files": None
        if discovered_files is None
        else [str(p) for p in discovered_files],
        "study_timepoints": discovered_timepoints,
        "retained_per_timepoint": retained_per_t,
        "timing_seconds": {
            "skeleton": skeleton_took,
            "features": features_took,
            "coverage_mip": mip_took,
            "total": float(time.perf_counter() - start),
        },
    }
