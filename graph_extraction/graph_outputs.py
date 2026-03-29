"""Centerline-to-graph conversion, graph quality checks, and JSON writing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features.graph import compute_tumor_graph_feature_payload
from graph_extraction.constants import NDIM_3D
from graph_extraction.feature_stats import (
    _collect_morphometry_qc,
    _json_default,
    mask_to_edges_bitmask,
)
from graph_extraction.skeleton_to_graph_primitives import (
    assign_component_labels,
    build_vessel_json,
    compute_segment_metrics,
    detect_bifurcations,
    edges_to_segments,
    extract_segments,
    obtain_radius_map,
    segments_to_graph,
)


def build_graph_outputs_from_centerline(
    skeleton_mask_zyx: np.ndarray,
    vessel_reference_zyx: np.ndarray,
    output_json_path: Path,
    *,
    strict_qc: bool = False,
    tumor_mask_zyx: np.ndarray | None = None,
    tumor_spacing_mm_zyx: tuple[float, float, float] | None = None,
    tumor_graph_output_path: Path | None = None,
    kinetic_priority_4d: np.ndarray | None = None,
    kinetic_timepoints: list[int] | None = None,
    kinetic_reference_mask_zyx: np.ndarray | None = None,
) -> dict[str, object]:
    """Convert a centerline mask into graph outputs and tumor-focused features.

    This is the main bridge between a saved centerline and the JSON files used
    downstream.

    Steps:

    1. convert the skeleton mask into a graph
    2. estimate segment-level measurements such as length and radius
    3. run quality checks for invalid radii, duplicate segments, and branching issues
    4. write the lower-level morphometry JSON
    5. if tumor context is available, also write the higher-level
       `tumor_graph_features.json`

    Parameters
    ----------
    skeleton_mask_zyx:
        One-voxel-wide centerline mask.
    vessel_reference_zyx:
        Vessel support volume used to estimate local radius.
    output_json_path:
        Where to write the lower-level morphometry JSON.
    strict_qc:
        If true, fail instead of warning when quality-check counters detect invalid values.

    Returns:
    -------
    dict
        A small run summary with graph counts, quality-check status, and whether the
        tumor-centered feature JSON was written.
    """
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

    segment_metrics_by_path: dict[
        tuple[tuple[int, int, int], ...], dict[str, object]
    ] = {}
    for path in segment_paths:
        path_key = tuple(path)
        segment_metrics_by_path[path_key] = compute_segment_metrics(path, radius_map)

    qc_counters = _collect_morphometry_qc(
        segment_paths=segment_paths,
        segment_metrics_by_path=segment_metrics_by_path,
        radius_map=radius_map,
        bifurcations=bifurcations,
    )
    invalid_total = int(
        qc_counters["radius_nonfinite_node_count"]
        + qc_counters["radius_nonpositive_node_count"]
        + qc_counters["segment_radius_nonfinite_count"]
        + qc_counters["segment_radius_nonpositive_count"]
    )
    duplicate_total = int(qc_counters["duplicate_segment_path_count"])
    qc_passed = invalid_total == 0 and duplicate_total == 0
    if not qc_passed:
        qc_message = (
            "Morphometry quality checks failed: "
            f"invalid_total={invalid_total}, duplicate_paths={duplicate_total}, "
            f"counters={qc_counters}"
        )
        if strict_qc:
            raise ValueError(qc_message)
        print(f"[qc-warning] {qc_message}")

    build_vessel_json(
        graph,
        vessel_labels,
        segment_paths,
        radius_map,
        bifurcations,
        segment_metrics_by_path=segment_metrics_by_path,
        output_path=output_json_path,
    )

    tumor_graph_feature_status = "skipped_no_tumor_context"
    if tumor_graph_output_path is None:
        tumor_graph_feature_status = "skipped_no_output_path"
    elif tumor_mask_zyx is not None and tumor_spacing_mm_zyx is not None:
        tumor_payload = compute_tumor_graph_feature_payload(
            graph=graph,
            segment_paths=segment_paths,
            segment_metrics_by_path=segment_metrics_by_path,
            support_mask_zyx=vessel_reference_zyx,
            tumor_mask_zyx=tumor_mask_zyx,
            spacing_mm_zyx=tumor_spacing_mm_zyx,
            priority_4d=kinetic_priority_4d,
            study_timepoints=kinetic_timepoints,
            reference_mask_zyx=kinetic_reference_mask_zyx,
        )
        tumor_graph_output_path.write_text(
            json.dumps(tumor_payload, indent=2, default=_json_default),
            encoding="utf-8",
        )
        tumor_graph_feature_status = str(tumor_payload.get("status", "unknown"))

    return {
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "segment_count": int(len(segment_paths)),
        "component_count": int(len(vessel_labels)),
        "qc_passed": bool(qc_passed),
        "qc_invalid_count": int(invalid_total),
        "qc_duplicate_count": int(duplicate_total),
        "qc_counters": qc_counters,
        "tumor_graph_features_status": tumor_graph_feature_status,
        "tumor_graph_features_path": (
            None if tumor_graph_output_path is None else str(tumor_graph_output_path)
        ),
    }
