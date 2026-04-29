"""Kinematic vessel-feature definitions and JSON extraction helpers."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
from scipy import ndimage

from features._common import flatten_numeric_payload
from graph_extraction.constants import (
    BIFURCATION_MIN_DEGREE,
    EARLY_CYCLE_FRACTION_ALL,
    EARLY_CYCLE_FRACTION_MAJORITY,
    KINETIC_SIGNAL_EPS,
    MIN_KINETIC_TIMEPOINTS,
    MIN_PATH_POINTS,
    NDIM_4D,
    TUMOR_NEAR_MM,
    TUMOR_SHELL_SPECS,
)
from graph_extraction.feature_stats import (
    _arrival_index_from_enhancement,
    _discrete_entropy,
    _hurdle_summary,
    _quantile_summary,
    _safe_ratio,
    _segment_curvature_mean_per_mm,
    _segment_path_length_mm,
    _segment_radius_mean_mm,
    _segment_tortuosity_mm,
    _shell_name_for_signed_distance,
    _weighted_linear_fit_stats,
    _weighted_linear_slope,
)

BLOCK_NAME = "kinematic"

KINEMATIC_FEATURE_GROUPS: tuple[str, ...] = (
    "shell_segment_counts",
    "shell_kinetics",
    "normalized_shell_burdens",
    "shell_contrasts",
    "peritumoral_multi_scale_contrasts",
    "arrival_delay_vs_reference",
    "propagation_speed",
    "graph_local_propagation",
    "caliber_flow_kinetic_surrogates",
    "reference_normalized_signal",
    "temporal_heterogeneity_near_tumor",
    "early_enhancing_fraction",
    "boundary_crossing_dynamic_burden",
    "boundary_caliber_gradient",
    "component_invasion",
    "kinetic_topology_coupling",
)


def compute_tumor_kinematic_feature_payload(
    *,
    graph: nx.Graph,
    segment_paths: list[list[tuple[int, int, int]]],
    segment_metrics_by_path: dict[tuple[tuple[int, int, int], ...], dict[str, object]],
    support_mask_zyx: np.ndarray,
    tumor_mask_zyx: np.ndarray,
    spacing_mm_zyx: tuple[float, float, float],
    signed_dist_mm: np.ndarray,
    shell_voxel_counts: dict[str, int],
    priority_4d: np.ndarray,
    study_timepoints: list[int] | None,
    reference_mask_zyx: np.ndarray | None,
) -> dict[str, object]:
    """Build time-dependent vessel features around the tumor.

    `priority_4d` is the per-timepoint vessel signal volume for this study,
    aligned to the saved vessel masks. For each graph segment, this function
    samples the average signal over time and turns that curve into a set of
    simple summaries such as:

    - when enhancement first appears
    - when the segment reaches its peak
    - how quickly signal rises and falls
    - total positive enhancement over time

    Those per-segment measurements are then summarized in several ways:

    - by tumor-distance shell, so we can compare inside-tumor vessels with
      vessels a few millimeters away
    - as normalized burdens, so results are less driven by tumor size alone
    - as contrasts between shells, so we can ask whether near-tumor vessels
      behave differently from farther vessels
    - as graph-local propagation summaries, so we can capture whether signal
      appears earlier or later along connected vessel paths

    The output is designed for downstream modeling, not for direct
    visualization, so the emphasis is on stable summary statistics rather than
    raw curves.
    """
    signal_4d = np.asarray(priority_4d, dtype=float)
    if signal_4d.ndim != NDIM_4D:
        return {
            "status": "invalid_priority_shape",
            "priority_shape": [int(v) for v in signal_4d.shape],
        }
    if tuple(signal_4d.shape[1:]) != tuple(support_mask_zyx.shape):
        return {
            "status": "priority_shape_mismatch",
            "priority_shape": [int(v) for v in signal_4d.shape],
            "support_shape_zyx": [int(v) for v in support_mask_zyx.shape],
        }

    n_t = int(signal_4d.shape[0])
    if n_t < MIN_KINETIC_TIMEPOINTS:
        return {"status": "insufficient_timepoints", "timepoint_count": n_t}

    if study_timepoints is not None and len(study_timepoints) == n_t:
        time_axis = np.asarray(study_timepoints, dtype=float)
    else:
        time_axis = np.arange(n_t, dtype=float)
    if not np.all(np.isfinite(time_axis)):
        time_axis = np.arange(n_t, dtype=float)
    if np.any(np.diff(time_axis) <= 0.0):
        time_axis = np.arange(n_t, dtype=float)

    early_idx = int(max(1, np.floor(0.25 * float(n_t - 1))))
    early_time_threshold = float(time_axis[early_idx])

    support = np.asarray(support_mask_zyx, dtype=bool)
    tumor = np.asarray(tumor_mask_zyx, dtype=bool)
    radius_mm_volume = ndimage.distance_transform_edt(
        support,
        sampling=spacing_mm_zyx,
    ).astype(np.float32, copy=False)
    ref_source = "global_reference"
    ref_mask = np.ones_like(support, dtype=bool)
    if reference_mask_zyx is not None:
        reference = np.asarray(reference_mask_zyx, dtype=bool)
        if reference.shape == support.shape and np.any(reference):
            ref_mask_candidate = reference & (~tumor)
            if np.any(ref_mask_candidate):
                ref_mask = ref_mask_candidate
                ref_source = "breast_reference"
    if ref_source == "global_reference":
        ref_mask_candidate = (~support) & (~tumor)
        if np.any(ref_mask_candidate):
            ref_mask = ref_mask_candidate
            ref_source = "background_reference"

    reference_curve = np.asarray(
        [float(np.mean(signal_4d[t][ref_mask])) for t in range(n_t)],
        dtype=float,
    )
    reference_enh = reference_curve - float(reference_curve[0])
    ref_tte_idx = _arrival_index_from_enhancement(reference_enh)
    ref_tte_time = None if ref_tte_idx is None else float(time_axis[ref_tte_idx])
    reference_peak_enhancement = float(np.max(np.maximum(reference_enh, 0.0)))
    reference_auc_positive = float(
        np.trapz(np.maximum(reference_enh, 0.0), x=time_axis)
    )
    reference_peak_idx = int(np.argmax(np.maximum(reference_enh, 0.0)))
    reference_washin_slope = 0.0
    if reference_peak_idx > 0:
        ref_start_idx = 0 if ref_tte_idx is None else int(ref_tte_idx)
        ref_washin_den = float(time_axis[reference_peak_idx] - time_axis[ref_start_idx])
        if ref_washin_den > 0.0:
            reference_washin_slope = float(
                (reference_enh[reference_peak_idx] - reference_enh[ref_start_idx])
                / ref_washin_den
            )

    sz, sy, sx = (
        float(spacing_mm_zyx[0]),
        float(spacing_mm_zyx[1]),
        float(spacing_mm_zyx[2]),
    )

    edge_weight_graph = graph.copy()
    for u, v in edge_weight_graph.edges():
        dxyz = np.asarray(
            [
                (float(v[0]) - float(u[0])) * float(sx),
                (float(v[1]) - float(u[1])) * float(sy),
                (float(v[2]) - float(u[2])) * float(sz),
            ],
            dtype=float,
        )
        edge_weight_graph[u][v]["weight_mm"] = float(np.linalg.norm(dxyz))

    node_signed_map: dict[tuple[int, int, int], float] = {}
    for node in edge_weight_graph.nodes():
        x, y, z = int(node[0]), int(node[1]), int(node[2])
        node_signed_map[node] = float(signed_dist_mm[z, y, x])

    boundary_nodes: set[tuple[int, int, int]] = {
        node
        for node in edge_weight_graph.nodes()
        if abs(float(node_signed_map[node])) <= 1.0
    }
    for u, v in edge_weight_graph.edges():
        if float(node_signed_map[u]) * float(node_signed_map[v]) < 0.0:
            boundary_nodes.add(u)
            boundary_nodes.add(v)
    if not boundary_nodes:
        sorted_nodes = sorted(
            edge_weight_graph.nodes(),
            key=lambda n: abs(float(node_signed_map.get(n, 0.0))),
        )
        boundary_nodes = set(sorted_nodes[: min(16, len(sorted_nodes))])

    graph_dist_to_boundary_mm: dict[tuple[int, int, int], float] = {}
    if boundary_nodes:
        graph_dist_to_boundary_mm = nx.multi_source_dijkstra_path_length(
            edge_weight_graph,
            boundary_nodes,
            weight="weight_mm",
        )

    voxel_volume_mm3 = float(sz * sy * sx)
    tumor_voxels = int(np.count_nonzero(tumor))
    tumor_volume_mm3 = float(tumor_voxels * voxel_volume_mm3)

    tumor_indices_zyx = np.argwhere(tumor)
    centroid_zyx = np.mean(tumor_indices_zyx, axis=0)
    tumor_centroid_xyz_mm = np.asarray(
        [centroid_zyx[2] * sx, centroid_zyx[1] * sy, centroid_zyx[0] * sz], dtype=float
    )

    node_arrival_cache: dict[tuple[int, int, int], int | None] = {}

    def _arrival_idx_for_node(node_xyz: tuple[int, int, int]) -> int | None:
        if node_xyz in node_arrival_cache:
            return node_arrival_cache[node_xyz]
        x, y, z = int(node_xyz[0]), int(node_xyz[1]), int(node_xyz[2])
        curve = signal_4d[:, z, y, x]
        enh = curve - float(curve[0])
        idx = _arrival_index_from_enhancement(enh)
        node_arrival_cache[node_xyz] = idx
        return idx

    segment_rows: list[dict[str, object]] = []
    node_to_component: dict[tuple[int, int, int], int] = {}
    for comp_idx, comp_nodes in enumerate(
        nx.connected_components(edge_weight_graph), start=1
    ):
        for node in comp_nodes:
            node_to_component[node] = int(comp_idx)

    for path in segment_paths:
        if not path:
            continue
        coords_xyz = np.asarray(path, dtype=np.int64)
        coords_zyx = coords_xyz[:, [2, 1, 0]]

        curve = np.asarray(
            [
                float(
                    np.mean(
                        signal_4d[
                            t, coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]
                        ]
                    )
                )
                for t in range(n_t)
            ],
            dtype=float,
        )
        baseline = float(curve[0])
        enh = curve - baseline
        peak_idx = int(np.argmax(enh))
        peak_enh = float(enh[peak_idx])
        has_signal = bool(np.isfinite(peak_enh) and peak_enh > KINETIC_SIGNAL_EPS)
        tte_idx = _arrival_index_from_enhancement(enh)
        ttp_idx = peak_idx

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

        signed_vals = signed_dist_mm[
            coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]
        ].astype(float, copy=False)
        shell_name = _shell_name_for_signed_distance(float(np.median(signed_vals)))
        is_near_tumor = bool(np.min(np.abs(signed_vals)) <= TUMOR_NEAR_MM)
        crosses_boundary = bool(np.min(signed_vals) < 0.0 <= np.max(signed_vals))
        invasion_depth_mm = float(max(0.0, -float(np.min(signed_vals))))
        outside_reach_mm = float(max(0.0, float(np.max(signed_vals))))

        length_mm = _segment_path_length_mm(path, spacing_mm_zyx)
        radii_mm = radius_mm_volume[
            coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]
        ].astype(
            float,
            copy=False,
        )
        inside_mask = signed_vals < 0.0
        outside_mask = signed_vals >= 0.0
        radius_inside_mean_mm = (
            float(np.mean(radii_mm[inside_mask])) if np.any(inside_mask) else np.nan
        )
        radius_outside_mean_mm = (
            float(np.mean(radii_mm[outside_mask])) if np.any(outside_mask) else np.nan
        )
        radius_outside_minus_inside_mm = (
            float(radius_outside_mean_mm - radius_inside_mean_mm)
            if np.isfinite(radius_outside_mean_mm)
            and np.isfinite(radius_inside_mean_mm)
            else np.nan
        )
        radius_outside_over_inside = (
            _safe_ratio(radius_outside_mean_mm, radius_inside_mean_mm)
            if np.isfinite(radius_outside_mean_mm)
            and np.isfinite(radius_inside_mean_mm)
            else np.nan
        )
        boundary_caliber_gradient_mm_per_mm = (
            float(radius_outside_minus_inside_mm / length_mm)
            if np.isfinite(radius_outside_minus_inside_mm) and length_mm > 0.0
            else np.nan
        )
        radius_mean_mm = _segment_radius_mean_mm(path, radius_mm_volume)
        curvature_mean = _segment_curvature_mean_per_mm(path, spacing_mm_zyx)
        tortuosity = _segment_tortuosity_mm(path, spacing_mm_zyx)
        volume_burden_mm3 = float(
            math.pi * max(0.0, radius_mean_mm) ** 2 * max(0.0, length_mm)
        )
        peak_rel_reference = _safe_ratio(max(peak_enh, 0.0), reference_peak_enhancement)
        auc_rel_reference = _safe_ratio(max(auc, 0.0), reference_auc_positive)
        washin_rel_reference = _safe_ratio(washin_slope, reference_washin_slope)

        tte_time = float(time_axis[tte_idx]) if tte_idx is not None else np.nan
        arrival_delay_vs_ref = (
            float(tte_time - float(ref_tte_time))
            if (tte_idx is not None and ref_tte_time is not None)
            else np.nan
        )
        early_enhancing = bool(tte_idx is not None and tte_time <= early_time_threshold)

        endpoint_a = tuple(int(v) for v in path[0])
        endpoint_b = tuple(int(v) for v in path[-1])
        tte_a = _arrival_idx_for_node(endpoint_a)
        tte_b = _arrival_idx_for_node(endpoint_b)
        propagation_delay = np.nan
        propagation_speed_mm_per_time = np.nan
        if tte_a is not None and tte_b is not None:
            a_xyz_mm = np.asarray(
                [endpoint_a[0] * sx, endpoint_a[1] * sy, endpoint_a[2] * sz],
                dtype=float,
            )
            b_xyz_mm = np.asarray(
                [endpoint_b[0] * sx, endpoint_b[1] * sy, endpoint_b[2] * sz],
                dtype=float,
            )
            dist_a = float(np.linalg.norm(a_xyz_mm - tumor_centroid_xyz_mm))
            dist_b = float(np.linalg.norm(b_xyz_mm - tumor_centroid_xyz_mm))
            proximal_tte = (
                float(time_axis[tte_a]) if dist_a <= dist_b else float(time_axis[tte_b])
            )
            distal_tte = (
                float(time_axis[tte_b]) if dist_a <= dist_b else float(time_axis[tte_a])
            )
            propagation_delay = float(distal_tte - proximal_tte)
            if propagation_delay > 0.0 and length_mm > 0.0:
                propagation_speed_mm_per_time = float(length_mm / propagation_delay)

        graph_dist_path = np.asarray(
            [
                float(
                    graph_dist_to_boundary_mm.get(tuple(int(v) for v in node), np.nan)
                )
                for node in path
            ],
            dtype=float,
        )
        graph_dist_from_boundary_mm = (
            float(np.nanmin(graph_dist_path))
            if np.any(np.isfinite(graph_dist_path))
            else np.nan
        )
        component_id = int(node_to_component.get(endpoint_a, 0))

        segment_rows.append(
            {
                "shell": shell_name,
                "near_tumor": bool(is_near_tumor),
                "crosses_boundary": bool(crosses_boundary),
                "component_id": int(component_id),
                "length_mm": float(length_mm),
                "volume_burden_mm3": float(volume_burden_mm3),
                "radius_mean_mm": float(radius_mean_mm),
                "radius_inside_mean_mm": float(radius_inside_mean_mm),
                "radius_outside_mean_mm": float(radius_outside_mean_mm),
                "radius_outside_minus_inside_mm": float(radius_outside_minus_inside_mm),
                "radius_outside_over_inside": float(radius_outside_over_inside),
                "boundary_caliber_gradient_mm_per_mm": float(
                    boundary_caliber_gradient_mm_per_mm
                ),
                "curvature_mean": float(curvature_mean),
                "tortuosity": float(tortuosity),
                "tte_idx": -1 if tte_idx is None else int(tte_idx),
                "tte_time": float(tte_time),
                "ttp_idx": int(ttp_idx),
                "ttp_time": float(time_axis[ttp_idx]),
                "washin_slope": float(washin_slope),
                "washout_slope": float(washout_slope),
                "peak_enhancement": float(peak_enh),
                "auc": float(auc),
                "peak_rel_reference": float(peak_rel_reference),
                "auc_rel_reference": float(auc_rel_reference),
                "washin_rel_reference": float(washin_rel_reference),
                "arrival_delay_vs_reference": float(arrival_delay_vs_ref),
                "propagation_delay": float(propagation_delay),
                "propagation_speed_mm_per_time": float(propagation_speed_mm_per_time),
                "graph_distance_from_boundary_mm": float(graph_dist_from_boundary_mm),
                "invasion_depth_mm": float(invasion_depth_mm),
                "outside_reach_mm": float(outside_reach_mm),
                "has_signal": bool(has_signal),
                "early_enhancing": bool(early_enhancing),
            }
        )

    if not segment_rows:
        return {"status": "no_segments"}

    def _arr(rows: list[dict[str, object]], key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows], dtype=float)

    def _bool(rows: list[dict[str, object]], key: str) -> np.ndarray:
        return np.asarray([bool(r[key]) for r in rows], dtype=bool)

    shell_region_volume_mm3 = {
        shell_name: float(shell_voxel_counts.get(shell_name, 0))
        * float(voxel_volume_mm3)
        for shell_name, _, _ in TUMOR_SHELL_SPECS
    }
    total_segment_length_mm = float(np.sum(_arr(segment_rows, "length_mm")))
    total_segment_volume_mm3 = float(np.sum(_arr(segment_rows, "volume_burden_mm3")))

    shell_kinetics: dict[str, dict[str, object]] = {}
    shell_burdens_normalized: dict[str, dict[str, float | int]] = {}
    shell_segment_counts: dict[str, int] = {}
    for shell_name, _, _ in TUMOR_SHELL_SPECS:
        shell_rows = [r for r in segment_rows if str(r["shell"]) == shell_name]
        shell_count = int(len(shell_rows))
        shell_segment_counts[shell_name] = shell_count
        shell_lengths = _arr(shell_rows, "length_mm")
        shell_length_mm = float(np.sum(shell_lengths))
        shell_volumes = _arr(shell_rows, "volume_burden_mm3")
        shell_volume_mm3 = float(np.sum(shell_volumes))
        shell_voxels = int(shell_voxel_counts.get(shell_name, 0))
        shell_region_mm3 = float(shell_region_volume_mm3.get(shell_name, 0.0))
        shell_peak = _arr(shell_rows, "peak_enhancement")
        shell_auc = _arr(shell_rows, "auc")
        shell_has_signal = _bool(shell_rows, "has_signal")
        shell_valid_tte = np.isfinite(_arr(shell_rows, "tte_time"))

        shell_peak_weighted_volume = float(
            np.sum(shell_volumes * np.maximum(shell_peak, 0.0))
        )
        shell_auc_weighted_volume = float(
            np.sum(shell_volumes * np.maximum(shell_auc, 0.0))
        )

        shell_kinetics[shell_name] = {
            "time_to_enhancement_hurdle": _hurdle_summary(
                values=_arr(shell_rows, "tte_time"),
                valid_mask=shell_valid_tte,
                signal_mask=shell_valid_tte,
            ),
            "time_to_peak_hurdle": _hurdle_summary(
                values=_arr(shell_rows, "ttp_time"),
                valid_mask=np.isfinite(_arr(shell_rows, "ttp_time")),
                signal_mask=shell_has_signal,
            ),
            "washin_slope_hurdle": _hurdle_summary(
                values=_arr(shell_rows, "washin_slope"),
                valid_mask=np.isfinite(_arr(shell_rows, "washin_slope")),
                signal_mask=shell_has_signal,
            ),
            "washout_slope_hurdle": _hurdle_summary(
                values=_arr(shell_rows, "washout_slope"),
                valid_mask=np.isfinite(_arr(shell_rows, "washout_slope")),
                signal_mask=shell_has_signal,
            ),
            "peak_enhancement_hurdle": _hurdle_summary(
                values=shell_peak,
                valid_mask=np.isfinite(shell_peak),
                signal_mask=shell_has_signal,
            ),
            "auc_hurdle": _hurdle_summary(
                values=shell_auc,
                valid_mask=np.isfinite(shell_auc),
                signal_mask=shell_has_signal,
            ),
            "peak_rel_reference_hurdle": _hurdle_summary(
                values=_arr(shell_rows, "peak_rel_reference"),
                valid_mask=np.isfinite(_arr(shell_rows, "peak_rel_reference")),
                signal_mask=shell_has_signal,
            ),
            "auc_rel_reference_hurdle": _hurdle_summary(
                values=_arr(shell_rows, "auc_rel_reference"),
                valid_mask=np.isfinite(_arr(shell_rows, "auc_rel_reference")),
                signal_mask=shell_has_signal,
            ),
            "washin_rel_reference_hurdle": _hurdle_summary(
                values=_arr(shell_rows, "washin_rel_reference"),
                valid_mask=np.isfinite(_arr(shell_rows, "washin_rel_reference")),
                signal_mask=shell_has_signal,
            ),
        }
        shell_burdens_normalized[shell_name] = {
            "shell_region_voxels": int(shell_voxels),
            "shell_region_volume_mm3": float(shell_region_mm3),
            "shell_region_volume_per_tumor_mm3": _safe_ratio(
                shell_region_mm3, tumor_volume_mm3
            ),
            "kinetic_peak_weighted_volume_mm3": float(shell_peak_weighted_volume),
            "kinetic_auc_weighted_volume_mm3": float(shell_auc_weighted_volume),
            "peak_weighted_volume_per_tumor_mm3": _safe_ratio(
                shell_peak_weighted_volume, tumor_volume_mm3
            ),
            "auc_weighted_volume_per_tumor_mm3": _safe_ratio(
                shell_auc_weighted_volume, tumor_volume_mm3
            ),
            "peak_weighted_volume_per_shell_voxel": _safe_ratio(
                shell_peak_weighted_volume,
                float(shell_voxels),
            ),
            "auc_weighted_volume_per_shell_voxel": _safe_ratio(
                shell_auc_weighted_volume,
                float(shell_voxels),
            ),
            "peak_weighted_volume_per_shell_mm3": _safe_ratio(
                shell_peak_weighted_volume, shell_region_mm3
            ),
            "auc_weighted_volume_per_shell_mm3": _safe_ratio(
                shell_auc_weighted_volume, shell_region_mm3
            ),
            "peak_weighted_volume_per_shell_length_mm": _safe_ratio(
                shell_peak_weighted_volume, shell_length_mm
            ),
            "auc_weighted_volume_per_shell_length_mm": _safe_ratio(
                shell_auc_weighted_volume, shell_length_mm
            ),
            "peak_weighted_volume_per_shell_segment": _safe_ratio(
                shell_peak_weighted_volume, shell_count
            ),
            "auc_weighted_volume_per_shell_segment": _safe_ratio(
                shell_auc_weighted_volume, shell_count
            ),
            "total_shell_volume_burden_mm3": float(shell_volume_mm3),
            "total_shell_length_mm": float(shell_length_mm),
            "shell_length_density_mm_per_shell_mm3": _safe_ratio(
                shell_length_mm, shell_region_mm3
            ),
            "shell_volume_density_mm3_per_shell_mm3": _safe_ratio(
                shell_volume_mm3, shell_region_mm3
            ),
            "shell_length_mm_per_tumor_mm3": _safe_ratio(
                shell_length_mm, tumor_volume_mm3
            ),
            "shell_volume_burden_mm3_per_tumor_mm3": _safe_ratio(
                shell_volume_mm3, tumor_volume_mm3
            ),
            "shell_length_fraction_of_total_graph_length": _safe_ratio(
                shell_length_mm,
                total_segment_length_mm,
            ),
            "shell_volume_fraction_of_total_graph_volume": _safe_ratio(
                shell_volume_mm3,
                total_segment_volume_mm3,
            ),
        }

    near_rows = [r for r in segment_rows if bool(r["near_tumor"])]
    all_rows = segment_rows
    all_has_signal = _bool(all_rows, "has_signal")
    near_ttp_idx = np.asarray([int(r["ttp_idx"]) for r in near_rows], dtype=np.int64)
    near_peak = _arr(near_rows, "peak_enhancement")
    near_washin = _arr(near_rows, "washin_slope")
    near_washout = _arr(near_rows, "washout_slope")
    near_auc = _arr(near_rows, "auc")
    near_tte = _arr(near_rows, "tte_time")
    all_tte = _arr(all_rows, "tte_time")

    def _cv(values: np.ndarray) -> float:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0
        return _safe_ratio(float(np.std(vals)), float(np.mean(np.abs(vals))))

    temporal_heterogeneity = {
        "near_segment_count": int(len(near_rows)),
        "ttp_entropy_bits": _discrete_entropy(near_ttp_idx, n_t),
        "cv_peak_enhancement": _cv(near_peak),
        "cv_washin_slope": _cv(near_washin),
        "cv_washout_slope": _cv(near_washout),
        "cv_auc": _cv(near_auc),
        "cv_tte_time": _cv(near_tte),
    }

    all_volume = float(np.sum(_arr(all_rows, "volume_burden_mm3")))
    near_volume = float(np.sum(_arr(near_rows, "volume_burden_mm3")))
    all_length = float(np.sum(_arr(all_rows, "length_mm")))
    near_length = float(np.sum(_arr(near_rows, "length_mm")))
    early_all_volume = float(
        np.sum(
            [
                float(r["volume_burden_mm3"])
                for r in all_rows
                if bool(r["early_enhancing"])
            ]
        )
    )
    early_near_volume = float(
        np.sum(
            [
                float(r["volume_burden_mm3"])
                for r in near_rows
                if bool(r["early_enhancing"])
            ]
        )
    )
    early_shell_0_2 = [r for r in all_rows if str(r["shell"]) == "shell_0_2mm"]
    early_shell_2_5 = [r for r in all_rows if str(r["shell"]) == "shell_2_5mm"]
    early_fraction = {
        "early_phase_time_threshold": float(early_time_threshold),
        "count_fraction_all_segments": _safe_ratio(
            float(np.sum([1 for r in all_rows if bool(r["early_enhancing"])])),
            float(len(all_rows)),
        ),
        "count_fraction_near_tumor": _safe_ratio(
            float(np.sum([1 for r in near_rows if bool(r["early_enhancing"])])),
            float(len(near_rows)),
        ),
        "volume_fraction_all_segments": _safe_ratio(early_all_volume, all_volume),
        "volume_fraction_near_tumor": _safe_ratio(early_near_volume, near_volume),
        "volume_fraction_shell_0_2mm": _safe_ratio(
            float(
                np.sum(
                    [
                        float(r["volume_burden_mm3"])
                        for r in early_shell_0_2
                        if bool(r["early_enhancing"])
                    ]
                )
            ),
            float(np.sum([float(r["volume_burden_mm3"]) for r in early_shell_0_2])),
        ),
        "volume_fraction_shell_2_5mm": _safe_ratio(
            float(
                np.sum(
                    [
                        float(r["volume_burden_mm3"])
                        for r in early_shell_2_5
                        if bool(r["early_enhancing"])
                    ]
                )
            ),
            float(np.sum([float(r["volume_burden_mm3"]) for r in early_shell_2_5])),
        ),
    }

    crossing_rows = [r for r in all_rows if bool(r["crosses_boundary"])]
    crossing_early_volume = float(
        np.sum(
            [
                float(r["volume_burden_mm3"])
                for r in crossing_rows
                if bool(r["early_enhancing"])
            ]
        )
    )
    crossing_total_volume = float(
        np.sum([float(r["volume_burden_mm3"]) for r in crossing_rows])
    )
    early_fraction["count_fraction_crossing"] = _safe_ratio(
        float(sum(1 for r in crossing_rows if bool(r["early_enhancing"]))),
        float(len(crossing_rows)),
    )
    early_fraction["volume_fraction_crossing"] = _safe_ratio(
        crossing_early_volume,
        crossing_total_volume,
    )
    crossing_peak_weighted = float(
        np.sum(
            [
                float(r["volume_burden_mm3"]) * max(0.0, float(r["peak_enhancement"]))
                for r in crossing_rows
            ]
        )
    )
    crossing_auc_weighted = float(
        np.sum(
            [
                float(r["volume_burden_mm3"]) * max(0.0, float(r["auc"]))
                for r in crossing_rows
            ]
        )
    )
    boundary_dynamic = {
        "crossing_segment_count": int(len(crossing_rows)),
        "kinetic_weighted_volume_peak_mm3": float(crossing_peak_weighted),
        "kinetic_weighted_volume_auc_mm3": float(crossing_auc_weighted),
        "kinetic_weighted_volume_peak_per_tumor_mm3": _safe_ratio(
            crossing_peak_weighted, tumor_volume_mm3
        ),
        "kinetic_weighted_volume_auc_per_tumor_mm3": _safe_ratio(
            crossing_auc_weighted, tumor_volume_mm3
        ),
        "kinetic_weighted_volume_peak_per_crossing_length_mm": _safe_ratio(
            crossing_peak_weighted,
            float(np.sum(_arr(crossing_rows, "length_mm"))),
        ),
        "kinetic_weighted_volume_auc_per_crossing_length_mm": _safe_ratio(
            crossing_auc_weighted,
            float(np.sum(_arr(crossing_rows, "length_mm"))),
        ),
        "kinetic_weighted_volume_peak_per_crossing_segment": _safe_ratio(
            crossing_peak_weighted,
            float(len(crossing_rows)),
        ),
        "kinetic_weighted_volume_auc_per_crossing_segment": _safe_ratio(
            crossing_auc_weighted,
            float(len(crossing_rows)),
        ),
    }
    crossing_len = _arr(crossing_rows, "length_mm")
    crossing_r_in = _arr(crossing_rows, "radius_inside_mean_mm")
    crossing_r_out = _arr(crossing_rows, "radius_outside_mean_mm")
    crossing_r_delta = _arr(crossing_rows, "radius_outside_minus_inside_mm")
    crossing_r_ratio = _arr(crossing_rows, "radius_outside_over_inside")
    crossing_r_grad = _arr(crossing_rows, "boundary_caliber_gradient_mm_per_mm")
    valid_cross_r_in = np.isfinite(crossing_r_in)
    valid_cross_r_out = np.isfinite(crossing_r_out)
    valid_cross_delta = np.isfinite(crossing_r_delta)
    valid_cross_ratio = np.isfinite(crossing_r_ratio)
    valid_cross_grad = np.isfinite(crossing_r_grad)
    boundary_caliber_gradient = {
        "crossing_segment_count": int(len(crossing_rows)),
        "inside_radius_length_weighted_mean_mm": _safe_ratio(
            float(
                np.sum(crossing_r_in[valid_cross_r_in] * crossing_len[valid_cross_r_in])
            ),
            float(np.sum(crossing_len[valid_cross_r_in])),
        ),
        "outside_radius_length_weighted_mean_mm": _safe_ratio(
            float(
                np.sum(
                    crossing_r_out[valid_cross_r_out] * crossing_len[valid_cross_r_out]
                )
            ),
            float(np.sum(crossing_len[valid_cross_r_out])),
        ),
        "outside_minus_inside_length_weighted_mean_mm": _safe_ratio(
            float(
                np.sum(
                    crossing_r_delta[valid_cross_delta]
                    * crossing_len[valid_cross_delta]
                )
            ),
            float(np.sum(crossing_len[valid_cross_delta])),
        ),
        "outside_over_inside_hurdle": _hurdle_summary(
            values=crossing_r_ratio,
            valid_mask=valid_cross_ratio,
            signal_mask=valid_cross_ratio,
        ),
        "gradient_mm_per_mm_hurdle": _hurdle_summary(
            values=crossing_r_grad,
            valid_mask=valid_cross_grad,
            signal_mask=valid_cross_grad,
        ),
        "positive_dilation_fraction": _safe_ratio(
            float(np.count_nonzero(crossing_r_delta[valid_cross_delta] > 0.0)),
            float(np.count_nonzero(valid_cross_delta)),
        ),
    }

    component_stats: dict[int, dict[str, float]] = {}
    for row in all_rows:
        cid = int(row.get("component_id", 0))
        if cid <= 0:
            continue
        stats = component_stats.setdefault(
            cid,
            {
                "segment_count": 0.0,
                "length_mm": 0.0,
                "volume_mm3": 0.0,
                "crossing_segment_count": 0.0,
                "crossing_length_mm": 0.0,
                "crossing_volume_mm3": 0.0,
                "max_invasion_depth_mm": 0.0,
            },
        )
        seg_len = float(row["length_mm"])
        seg_vol = float(row["volume_burden_mm3"])
        stats["segment_count"] += 1.0
        stats["length_mm"] += seg_len
        stats["volume_mm3"] += seg_vol
        stats["max_invasion_depth_mm"] = max(
            stats["max_invasion_depth_mm"], float(row["invasion_depth_mm"])
        )
        if bool(row["crosses_boundary"]):
            stats["crossing_segment_count"] += 1.0
            stats["crossing_length_mm"] += seg_len
            stats["crossing_volume_mm3"] += seg_vol

    crossing_components = [
        s for s in component_stats.values() if s["crossing_segment_count"] > 0.0
    ]
    total_crossing_length = float(
        np.sum([s["crossing_length_mm"] for s in crossing_components])
    )
    largest_crossing = (
        max(crossing_components, key=lambda s: s["crossing_length_mm"])
        if crossing_components
        else None
    )
    component_invasion = {
        "component_count": int(len(component_stats)),
        "crossing_component_count": int(len(crossing_components)),
        "crossing_component_fraction": _safe_ratio(
            float(len(crossing_components)), float(len(component_stats))
        ),
        "crossing_length_concentration_max_fraction": _safe_ratio(
            float(largest_crossing["crossing_length_mm"])
            if largest_crossing is not None
            else 0.0,
            float(total_crossing_length),
        ),
        "mean_crossing_component_max_invasion_depth_mm": (
            float(np.mean([s["max_invasion_depth_mm"] for s in crossing_components]))
            if crossing_components
            else 0.0
        ),
    }
    if largest_crossing is not None:
        component_invasion.update(
            {
                "largest_crossing_component_segment_count": float(
                    largest_crossing["segment_count"]
                ),
                "largest_crossing_component_crossing_segment_count": float(
                    largest_crossing["crossing_segment_count"]
                ),
                "largest_crossing_component_crossing_length_mm": float(
                    largest_crossing["crossing_length_mm"]
                ),
                "largest_crossing_component_crossing_volume_mm3": float(
                    largest_crossing["crossing_volume_mm3"]
                ),
                "largest_crossing_component_fraction_total_length": _safe_ratio(
                    float(largest_crossing["crossing_length_mm"]),
                    float(all_length),
                ),
                "largest_crossing_component_fraction_total_volume": _safe_ratio(
                    float(largest_crossing["crossing_volume_mm3"]),
                    float(all_volume),
                ),
                "largest_crossing_component_max_invasion_depth_mm": float(
                    largest_crossing["max_invasion_depth_mm"]
                ),
            }
        )
    else:
        component_invasion.update(
            {
                "largest_crossing_component_segment_count": 0.0,
                "largest_crossing_component_crossing_segment_count": 0.0,
                "largest_crossing_component_crossing_length_mm": 0.0,
                "largest_crossing_component_crossing_volume_mm3": 0.0,
                "largest_crossing_component_fraction_total_length": 0.0,
                "largest_crossing_component_fraction_total_volume": 0.0,
                "largest_crossing_component_max_invasion_depth_mm": 0.0,
            }
        )

    node_times = np.asarray(
        [
            float(time_axis[idx]) if idx is not None else np.nan
            for idx in (
                _arrival_idx_for_node(tuple(int(v) for v in n))
                for n in edge_weight_graph.nodes()
            )
        ],
        dtype=float,
    )
    node_degrees = np.asarray(
        [int(edge_weight_graph.degree(n)) for n in edge_weight_graph.nodes()],
        dtype=np.int64,
    )
    valid_node_times = np.isfinite(node_times)
    early_node_mask = valid_node_times & (node_times <= early_time_threshold)
    bif_node_mask = node_degrees >= BIFURCATION_MIN_DEGREE
    early_bif_mask = early_node_mask & bif_node_mask

    cycles = nx.cycle_basis(edge_weight_graph)

    def _cycle_length_mm(cycle_nodes: list[tuple[int, int, int]]) -> float:
        if len(cycle_nodes) < MIN_PATH_POINTS:
            return 0.0
        total = 0.0
        for i, u in enumerate(cycle_nodes):
            v = cycle_nodes[(i + 1) % len(cycle_nodes)]
            if edge_weight_graph.has_edge(u, v):
                total += float(edge_weight_graph[u][v].get("weight_mm", 0.0))
            else:
                dxyz = np.asarray(
                    [
                        (float(v[0]) - float(u[0])) * sx,
                        (float(v[1]) - float(u[1])) * sy,
                        (float(v[2]) - float(u[2])) * sz,
                    ],
                    dtype=float,
                )
                total += float(np.linalg.norm(dxyz))
        return float(total)

    early_cycle_count_ge50 = 0
    early_cycle_count_all = 0
    early_cycle_length_ge50_mm = 0.0
    total_cycle_length_mm = 0.0
    node_time_lookup = {
        tuple(int(v) for v in node): float(node_times[i])
        for i, node in enumerate(edge_weight_graph.nodes())
    }
    for cyc in cycles:
        cyc_len_mm = _cycle_length_mm(cyc)
        total_cycle_length_mm += cyc_len_mm
        cyc_t = np.asarray(
            [node_time_lookup.get(tuple(int(v) for v in n), np.nan) for n in cyc],
            dtype=float,
        )
        valid = np.isfinite(cyc_t)
        if not np.any(valid):
            continue
        early_frac = float(np.mean(cyc_t[valid] <= early_time_threshold))
        if early_frac >= EARLY_CYCLE_FRACTION_MAJORITY:
            early_cycle_count_ge50 += 1
            early_cycle_length_ge50_mm += cyc_len_mm
        if early_frac >= EARLY_CYCLE_FRACTION_ALL:
            early_cycle_count_all += 1

    kinetic_topology_coupling = {
        "node_count": int(edge_weight_graph.number_of_nodes()),
        "valid_time_node_fraction": _safe_ratio(
            float(np.count_nonzero(valid_node_times)),
            float(edge_weight_graph.number_of_nodes()),
        ),
        "early_node_fraction": _safe_ratio(
            float(np.count_nonzero(early_node_mask)),
            float(np.count_nonzero(valid_node_times)),
        ),
        "bifurcation_node_count": int(np.count_nonzero(bif_node_mask)),
        "early_bifurcation_count": int(np.count_nonzero(early_bif_mask)),
        "early_bifurcation_fraction_of_bifurcations": _safe_ratio(
            float(np.count_nonzero(early_bif_mask)),
            float(np.count_nonzero(bif_node_mask)),
        ),
        "early_bifurcation_density_per_tumor_mm3": _safe_ratio(
            float(np.count_nonzero(early_bif_mask)),
            float(tumor_volume_mm3),
        ),
        "cycle_count": int(len(cycles)),
        "early_cycle_count_ge50": int(early_cycle_count_ge50),
        "early_cycle_count_all_nodes": int(early_cycle_count_all),
        "early_cycle_fraction_ge50": _safe_ratio(
            float(early_cycle_count_ge50), float(len(cycles))
        ),
        "early_cycle_length_burden_mm_ge50": float(early_cycle_length_ge50_mm),
        "early_cycle_length_fraction_ge50": _safe_ratio(
            float(early_cycle_length_ge50_mm),
            float(total_cycle_length_mm),
        ),
        "early_cycle_length_per_tumor_mm3_ge50": _safe_ratio(
            float(early_cycle_length_ge50_mm),
            float(tumor_volume_mm3),
        ),
    }

    near_has_signal = _bool(near_rows, "has_signal")
    reference_normalized_signal = {
        "reference_peak_enhancement": float(reference_peak_enhancement),
        "reference_auc_positive": float(reference_auc_positive),
        "reference_washin_slope": float(reference_washin_slope),
        "all_segments": {
            "peak_rel_reference_hurdle": _hurdle_summary(
                values=_arr(all_rows, "peak_rel_reference"),
                valid_mask=np.isfinite(_arr(all_rows, "peak_rel_reference")),
                signal_mask=all_has_signal,
            ),
            "auc_rel_reference_hurdle": _hurdle_summary(
                values=_arr(all_rows, "auc_rel_reference"),
                valid_mask=np.isfinite(_arr(all_rows, "auc_rel_reference")),
                signal_mask=all_has_signal,
            ),
            "washin_rel_reference_hurdle": _hurdle_summary(
                values=_arr(all_rows, "washin_rel_reference"),
                valid_mask=np.isfinite(_arr(all_rows, "washin_rel_reference")),
                signal_mask=all_has_signal,
            ),
        },
        "near_tumor_segments": {
            "peak_rel_reference_hurdle": _hurdle_summary(
                values=_arr(near_rows, "peak_rel_reference"),
                valid_mask=np.isfinite(_arr(near_rows, "peak_rel_reference")),
                signal_mask=near_has_signal,
            ),
            "auc_rel_reference_hurdle": _hurdle_summary(
                values=_arr(near_rows, "auc_rel_reference"),
                valid_mask=np.isfinite(_arr(near_rows, "auc_rel_reference")),
                signal_mask=near_has_signal,
            ),
            "washin_rel_reference_hurdle": _hurdle_summary(
                values=_arr(near_rows, "washin_rel_reference"),
                valid_mask=np.isfinite(_arr(near_rows, "washin_rel_reference")),
                signal_mask=near_has_signal,
            ),
        },
    }

    arrival_delay_values = _arr(all_rows, "arrival_delay_vs_reference")
    arrival_delay_near = _arr(near_rows, "arrival_delay_vs_reference")
    propagation_delay_values = _arr(all_rows, "propagation_delay")
    propagation_speed_values = _arr(all_rows, "propagation_speed_mm_per_time")
    propagation_speed_positive = propagation_speed_values[
        propagation_speed_values > 0.0
    ]
    propagation_delay_negative = int(np.count_nonzero(propagation_delay_values < 0.0))

    shell_hurdle_medians: dict[str, dict[str, float]] = {}
    for shell_name, shell_block in shell_kinetics.items():
        shell_hurdle_medians[shell_name] = {
            "tte": float(
                shell_block["time_to_enhancement_hurdle"]["value_given_signal"][
                    "median"
                ]
            ),
            "ttp": float(
                shell_block["time_to_peak_hurdle"]["value_given_signal"]["median"]
            ),
            "washin": float(
                shell_block["washin_slope_hurdle"]["value_given_signal"]["median"]
            ),
            "washout": float(
                shell_block["washout_slope_hurdle"]["value_given_signal"]["median"]
            ),
            "peak": float(
                shell_block["peak_enhancement_hurdle"]["value_given_signal"]["median"]
            ),
            "auc": float(shell_block["auc_hurdle"]["value_given_signal"]["median"]),
        }

    def _shell_contrast(shell_a: str, shell_b: str) -> dict[str, float]:
        a = shell_hurdle_medians.get(shell_a, {})
        b = shell_hurdle_medians.get(shell_b, {})
        out: dict[str, float] = {}
        for metric in ("tte", "ttp", "washin", "washout", "peak", "auc"):
            va = float(a.get(metric, 0.0))
            vb = float(b.get(metric, 0.0))
            out[f"{metric}_delta"] = float(va - vb)
            out[f"{metric}_ratio"] = _safe_ratio(va, vb)
            out[f"{metric}_delta_norm_l1"] = _safe_ratio(
                float(va - vb),
                float(abs(va) + abs(vb)),
            )
        return out

    def _shell_metric_slope(
        *,
        shell_names: list[str],
        shell_positions_mm: list[float],
        metric_name: str,
    ) -> float:
        vals = np.asarray(
            [
                float(shell_hurdle_medians.get(s, {}).get(metric_name, 0.0))
                for s in shell_names
            ],
            dtype=float,
        )
        pos = np.asarray(shell_positions_mm, dtype=float)
        w = np.asarray(
            [float(shell_segment_counts.get(s, 0)) for s in shell_names], dtype=float
        )
        return float(_weighted_linear_slope(pos, vals, w))

    shell_contrasts = {
        "inside_vs_0_2mm": _shell_contrast("inside_tumor", "shell_0_2mm"),
        "0_2mm_vs_2_5mm": _shell_contrast("shell_0_2mm", "shell_2_5mm"),
        "inside_vs_2_5mm": _shell_contrast("inside_tumor", "shell_2_5mm"),
        "5_10mm_vs_10_20mm": _shell_contrast("shell_5_10mm", "shell_10_20mm"),
    }
    peritumoral_multi_scale_contrasts = {
        "core_shell_gradient_per_mm": {
            "tte_slope": _shell_metric_slope(
                shell_names=["inside_tumor", "shell_0_2mm", "shell_2_5mm"],
                shell_positions_mm=[-1.0, 1.0, 3.5],
                metric_name="tte",
            ),
            "ttp_slope": _shell_metric_slope(
                shell_names=["inside_tumor", "shell_0_2mm", "shell_2_5mm"],
                shell_positions_mm=[-1.0, 1.0, 3.5],
                metric_name="ttp",
            ),
            "washin_slope": _shell_metric_slope(
                shell_names=["inside_tumor", "shell_0_2mm", "shell_2_5mm"],
                shell_positions_mm=[-1.0, 1.0, 3.5],
                metric_name="washin",
            ),
            "washout_slope": _shell_metric_slope(
                shell_names=["inside_tumor", "shell_0_2mm", "shell_2_5mm"],
                shell_positions_mm=[-1.0, 1.0, 3.5],
                metric_name="washout",
            ),
            "peak_slope": _shell_metric_slope(
                shell_names=["inside_tumor", "shell_0_2mm", "shell_2_5mm"],
                shell_positions_mm=[-1.0, 1.0, 3.5],
                metric_name="peak",
            ),
            "auc_slope": _shell_metric_slope(
                shell_names=["inside_tumor", "shell_0_2mm", "shell_2_5mm"],
                shell_positions_mm=[-1.0, 1.0, 3.5],
                metric_name="auc",
            ),
        },
    }

    graph_dist = _arr(all_rows, "graph_distance_from_boundary_mm")
    tte_valid_mask = np.isfinite(all_tte)
    graph_valid_mask = np.isfinite(graph_dist)
    fit_mask = tte_valid_mask & graph_valid_mask
    fit_stats = _weighted_linear_fit_stats(
        graph_dist[fit_mask],
        all_tte[fit_mask],
        _arr(all_rows, "length_mm")[fit_mask],
    )
    near_valid_tte_rows = [r for r in near_rows if np.isfinite(float(r["tte_time"]))]
    earliest_component_size = {
        "earliest_threshold_tte": 0.0,
        "largest_component_segments": 0,
        "largest_component_fraction_near_segments": 0.0,
    }
    if near_valid_tte_rows:
        near_tte_vals = np.asarray(
            [float(r["tte_time"]) for r in near_valid_tte_rows], dtype=float
        )
        earliest_threshold = float(np.percentile(near_tte_vals, 25.0))
        earliest_rows = [
            r for r in near_valid_tte_rows if float(r["tte_time"]) <= earliest_threshold
        ]
        comp_counts: dict[int, int] = {}
        for row in earliest_rows:
            cid = int(row.get("component_id", 0))
            comp_counts[cid] = comp_counts.get(cid, 0) + 1
        largest_comp = max(comp_counts.values()) if comp_counts else 0
        earliest_component_size = {
            "earliest_threshold_tte": float(earliest_threshold),
            "largest_component_segments": int(largest_comp),
            "largest_component_fraction_near_segments": _safe_ratio(
                float(largest_comp), float(len(near_rows))
            ),
        }

    dist_bin_edges = np.asarray([0.0, 2.0, 5.0, 10.0, 20.0, np.inf], dtype=float)
    binned_means: list[float] = []
    binned_stds: list[float] = []
    binned_counts: list[int] = []
    valid_x = graph_dist[fit_mask]
    valid_y = all_tte[fit_mask]
    for i in range(len(dist_bin_edges) - 1):
        lo = float(dist_bin_edges[i])
        hi = float(dist_bin_edges[i + 1])
        if np.isinf(hi):
            in_bin = valid_x >= lo
        else:
            in_bin = (valid_x >= lo) & (valid_x < hi)
        if not np.any(in_bin):
            binned_means.append(np.nan)
            binned_stds.append(np.nan)
            binned_counts.append(0)
            continue
        vals = valid_y[in_bin]
        binned_means.append(float(np.mean(vals)))
        binned_stds.append(float(np.std(vals)))
        binned_counts.append(int(vals.size))
    binned_means_arr = np.asarray(binned_means, dtype=float)
    binned_stds_arr = np.asarray(binned_stds, dtype=float)
    binned_counts_arr = np.asarray(binned_counts, dtype=float)
    valid_bins = binned_counts_arr > 0.0
    monotonic_pairs = 0
    monotonic_ok = 0
    valid_idx = np.where(valid_bins)[0]
    for i in range(1, len(valid_idx)):
        prev_i = int(valid_idx[i - 1])
        curr_i = int(valid_idx[i])
        monotonic_pairs += 1
        if float(binned_means_arr[curr_i]) >= float(binned_means_arr[prev_i]):
            monotonic_ok += 1

    binned_dispersion = {
        "mean_tte_std_across_distance_bins": _safe_ratio(
            float(np.sum(binned_stds_arr[valid_bins] * binned_counts_arr[valid_bins])),
            float(np.sum(binned_counts_arr[valid_bins])),
        ),
        "distance_bin_monotonic_nondecreasing_fraction": _safe_ratio(
            float(monotonic_ok),
            float(monotonic_pairs),
        ),
        "distance_bin_count": int(np.count_nonzero(valid_bins)),
    }

    graph_local_propagation = {
        "slope_tte_vs_graph_distance_from_boundary": float(fit_stats["slope"]),
        "weighted_linear_fit": fit_stats,
        "distance_binned_dispersion": binned_dispersion,
        "graph_distance_available_fraction": _safe_ratio(
            float(np.count_nonzero(graph_valid_mask)),
            float(len(all_rows)),
        ),
        "earliest_arrival_component_near_tumor": earliest_component_size,
    }

    row_length = _arr(all_rows, "length_mm")
    row_radius = _arr(all_rows, "radius_mean_mm")
    row_curve = np.abs(_arr(all_rows, "curvature_mean"))
    row_tort = np.abs(_arr(all_rows, "tortuosity"))
    row_washin = _arr(all_rows, "washin_slope")
    row_washout = _arr(all_rows, "washout_slope")
    row_peak = _arr(all_rows, "peak_enhancement")

    radius_weight = row_radius * row_length
    curvature_weight = row_curve * row_length
    tortuosity_weight = row_tort * row_length
    valid_delay = np.isfinite(arrival_delay_values)

    caliber_flow_surrogates = {
        "radius_weighted_washin_mean": _safe_ratio(
            float(np.sum(row_washin * radius_weight)),
            float(np.sum(radius_weight)),
        ),
        "radius_weighted_washout_mean": _safe_ratio(
            float(np.sum(row_washout * radius_weight)),
            float(np.sum(radius_weight)),
        ),
        "curvature_weighted_arrival_delay_mean": _safe_ratio(
            float(
                np.sum(
                    arrival_delay_values[valid_delay] * curvature_weight[valid_delay]
                )
            ),
            float(np.sum(curvature_weight[valid_delay])),
        ),
        "tortuosity_weighted_peak_enhancement_mean": _safe_ratio(
            float(np.sum(row_peak * tortuosity_weight)),
            float(np.sum(tortuosity_weight)),
        ),
    }

    return {
        "status": "ok",
        "version": "tumor_kinematic_features_v5_mm_shell_density_refnorm_topology",
        "timepoint_count": int(n_t),
        "reference_curve_source": ref_source,
        "segment_count": int(len(segment_rows)),
        "near_tumor_segment_count": int(len(near_rows)),
        "total_length_mm": float(all_length),
        "total_volume_burden_mm3": float(all_volume),
        "near_tumor_length_mm": float(near_length),
        "near_tumor_volume_burden_mm3": float(near_volume),
        "shell_segment_counts": {
            shell_name: int(shell_segment_counts.get(shell_name, 0))
            for shell_name, _, _ in TUMOR_SHELL_SPECS
        },
        "shell_kinetics": shell_kinetics,
        "normalized_shell_burdens": shell_burdens_normalized,
        "shell_contrasts": shell_contrasts,
        "peritumoral_multi_scale_contrasts": peritumoral_multi_scale_contrasts,
        "arrival_delay_vs_reference": {
            "all_segments_hurdle": _hurdle_summary(
                values=arrival_delay_values,
                valid_mask=np.isfinite(arrival_delay_values),
                signal_mask=np.isfinite(arrival_delay_values),
            ),
            "near_tumor_segments_hurdle": _hurdle_summary(
                values=arrival_delay_near,
                valid_mask=np.isfinite(arrival_delay_near),
                signal_mask=np.isfinite(arrival_delay_near),
            ),
        },
        "propagation_speed": {
            "delay_all_segments": _quantile_summary(propagation_delay_values),
            "speed_positive_delay_mm_per_time": _quantile_summary(
                propagation_speed_positive
            ),
            "negative_delay_segment_count": int(propagation_delay_negative),
        },
        "graph_local_propagation": graph_local_propagation,
        "caliber_flow_kinetic_surrogates": caliber_flow_surrogates,
        "reference_normalized_signal": reference_normalized_signal,
        "temporal_heterogeneity_near_tumor": temporal_heterogeneity,
        "early_enhancing_fraction": early_fraction,
        "boundary_crossing_dynamic_burden": boundary_dynamic,
        "boundary_caliber_gradient": boundary_caliber_gradient,
        "component_invasion": component_invasion,
        "kinetic_topology_coupling": kinetic_topology_coupling,
    }


def matches_column(column: str) -> bool:
    """Return whether a model column belongs to the kinematic block."""
    return column.startswith("kinematic_")


def extract_kinematic_json_features(payload: dict[str, Any]) -> dict[str, float]:
    """Extract the kinematic block from the saved tumor-graph JSON."""
    features: dict[str, float] = {}
    kinematic_payload = payload.get("kinematic_features")
    if not isinstance(kinematic_payload, dict):
        return features

    for group_name in KINEMATIC_FEATURE_GROUPS:
        group_value = kinematic_payload.get(group_name)
        if group_value is None:
            continue
        flatten_numeric_payload(
            group_value,
            prefix=f"kinematic_{group_name}",
            out=features,
        )
    return features
