"""Structural vessel-feature definitions and graph-block extraction helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy import ndimage

from features._common import flatten_numeric_payload, safe_float, safe_ratio
from features.kinematic import compute_tumor_kinematic_feature_payload
from graph_extraction.constants import (
    BIFURCATION_MIN_DEGREE,
    DEGREE_FOUR_PLUS,
    MIN_PATH_POINTS,
    TUMOR_BOUNDARY_NEAR_MM,
    TUMOR_NEAR_MM,
    TUMOR_SHELL_SPECS,
)
from graph_extraction.feature_stats import (
    _safe_ratio,
    _segment_curvature_mean_per_mm,
    _segment_path_length_mm,
    _segment_tortuosity_mm,
    _shell_name_for_signed_distance,
    _signed_tumor_distance_mm,
    _weighted_quantiles,
)

BLOCK_NAME = "graph"
METADATA_PREFIXES = (
    "tumor_graph_",
    "graph_",
)

STRUCTURAL_GRAPH_FEATURE_GROUPS: tuple[str, ...] = (
    "graph_totals",
    "tumor_burden",
    "boundary_crossing",
    "per_shell_topology",
    "caliber_heterogeneity_near_tumor",
    "directional_near_boundary",
    "length_weighted_shape_stats",
    "normalized_ratios",
)


def compute_tumor_graph_feature_payload(
    *,
    graph: nx.Graph,
    segment_paths: list[list[tuple[int, int, int]]],
    segment_metrics_by_path: dict[tuple[tuple[int, int, int], ...], dict[str, object]],
    support_mask_zyx: np.ndarray,
    tumor_mask_zyx: np.ndarray,
    spacing_mm_zyx: tuple[float, float, float],
    priority_4d: np.ndarray | None = None,
    study_timepoints: list[int] | None = None,
    reference_mask_zyx: np.ndarray | None = None,
) -> dict[str, object]:
    """Build tumor-centered structural features from the vessel graph.

    This is the main feature-construction function for the vessel graph. It uses
    the tumor mask to measure where each graph node and segment sits relative to
    the tumor boundary, then summarizes vessel structure inside the tumor and in
    several distance shells around it.

    The returned payload groups features into a few plain-language families:

    - overall graph summary
      how much vessel graph was found in this study
    - tumor burden
      how much vessel length or estimated vessel volume lies inside or near the tumor
    - boundary crossing
      whether segments cross from outside the tumor to inside it
    - per-shell topology
      endpoints, branch points, loops, and degree patterns in each distance shell
    - caliber and shape
      radius, curvature, and tortuosity near the tumor
    - directional features
      whether segments tend to point toward the tumor or run around it
    - normalized ratios
      versions of the same measurements divided by tumor size, shell size, or
      total vessel amount so they are easier to compare across cases

    If a 4D priority volume is available, this function also attaches the
    time-dependent vessel summaries computed by
    `compute_tumor_kinematic_feature_payload`.
    """
    if graph.number_of_nodes() <= 0:
        return {"status": "empty_graph"}
    support = np.asarray(support_mask_zyx, dtype=bool)
    tumor = np.asarray(tumor_mask_zyx, dtype=bool)
    if support.shape != tumor.shape:
        return {
            "status": "shape_mismatch",
            "support_shape_zyx": [int(v) for v in support.shape],
            "tumor_shape_zyx": [int(v) for v in tumor.shape],
        }
    if not np.any(tumor):
        return {"status": "tumor_empty"}

    sz, sy, sx = (
        float(spacing_mm_zyx[0]),
        float(spacing_mm_zyx[1]),
        float(spacing_mm_zyx[2]),
    )
    voxel_volume_mm3 = float(sz * sy * sx)
    tumor_voxels = int(np.count_nonzero(tumor))
    tumor_volume_mm3 = float(tumor_voxels * voxel_volume_mm3)
    tumor_equivalent_radius_mm = (
        float(((3.0 * tumor_volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0))
        if tumor_volume_mm3 > 0.0
        else 0.0
    )
    tumor_equivalent_surface_mm2 = float(
        4.0 * math.pi * (tumor_equivalent_radius_mm**2)
    )

    signed_dist_mm = _signed_tumor_distance_mm(
        tumor_mask_zyx=tumor, spacing_mm_zyx=spacing_mm_zyx
    )
    grad_z, grad_y, grad_x = np.gradient(signed_dist_mm, sz, sy, sx, edge_order=1)

    radius_mm_volume = ndimage.distance_transform_edt(
        support, sampling=spacing_mm_zyx
    ).astype(np.float32, copy=False)

    nodes = list(graph.nodes())
    node_xyz = np.asarray(nodes, dtype=np.int64)
    node_zyx = node_xyz[:, [2, 1, 0]]
    node_signed = signed_dist_mm[node_zyx[:, 0], node_zyx[:, 1], node_zyx[:, 2]].astype(
        float, copy=False
    )
    node_radius_mm = radius_mm_volume[
        node_zyx[:, 0], node_zyx[:, 1], node_zyx[:, 2]
    ].astype(float, copy=False)
    node_signed_map = {node: float(node_signed[i]) for i, node in enumerate(nodes)}
    node_radius_mm_map = {
        node: float(node_radius_mm[i]) for i, node in enumerate(nodes)
    }

    tumor_indices_zyx = np.argwhere(tumor)
    centroid_zyx = np.mean(tumor_indices_zyx, axis=0)
    tumor_centroid_xyz_mm = np.asarray(
        [centroid_zyx[2] * sx, centroid_zyx[1] * sy, centroid_zyx[0] * sz], dtype=float
    )

    shell_voxel_counts: dict[str, int] = {}
    for shell_name, lower, upper in TUMOR_SHELL_SPECS:
        mask = (signed_dist_mm >= float(lower)) & (signed_dist_mm < float(upper))
        shell_voxel_counts[shell_name] = int(np.count_nonzero(mask))
    shell_region_volume_mm3 = {
        shell_name: float(count) * float(voxel_volume_mm3)
        for shell_name, count in shell_voxel_counts.items()
    }

    node_shell_map = {
        node: _shell_name_for_signed_distance(node_signed_map[node]) for node in nodes
    }

    shell_length_mm = {name: 0.0 for name, _, _ in TUMOR_SHELL_SPECS}
    shell_volume_mm3 = {name: 0.0 for name, _, _ in TUMOR_SHELL_SPECS}
    total_length_mm = 0.0
    total_volume_mm3 = 0.0
    inside_length_mm = 0.0
    inside_volume_mm3 = 0.0
    near_length_mm = 0.0
    near_volume_mm3 = 0.0
    boundary_cross_length_mm = 0.0
    boundary_cross_volume_mm3 = 0.0
    boundary_cross_edge_count = 0

    radial_abs_cos: list[float] = []
    radial_weights: list[float] = []
    normal_abs_cos: list[float] = []
    normal_weights: list[float] = []

    for u, v in graph.edges():
        ux, uy, uz = (float(u[0]), float(u[1]), float(u[2]))
        vx, vy, vz = (float(v[0]), float(v[1]), float(v[2]))
        dvec_mm = np.asarray(
            [(vx - ux) * sx, (vy - uy) * sy, (vz - uz) * sz], dtype=float
        )
        length_mm = float(np.linalg.norm(dvec_mm))
        if length_mm <= 0.0:
            continue
        du = float(node_signed_map[u])
        dv = float(node_signed_map[v])
        dmid = 0.5 * (du + dv)
        shell_name = _shell_name_for_signed_distance(dmid)
        rmid_mm = max(
            0.0, 0.5 * (float(node_radius_mm_map[u]) + float(node_radius_mm_map[v]))
        )
        edge_volume_mm3 = float(math.pi * (rmid_mm**2) * length_mm)

        shell_length_mm[shell_name] += length_mm
        shell_volume_mm3[shell_name] += edge_volume_mm3
        total_length_mm += length_mm
        total_volume_mm3 += edge_volume_mm3
        if dmid < 0.0:
            inside_length_mm += length_mm
            inside_volume_mm3 += edge_volume_mm3
        if dmid <= TUMOR_NEAR_MM:
            near_length_mm += length_mm
            near_volume_mm3 += edge_volume_mm3
        if (du < 0.0 <= dv) or (dv < 0.0 <= du):
            boundary_cross_edge_count += 1
            boundary_cross_length_mm += length_mm
            boundary_cross_volume_mm3 += edge_volume_mm3

        if abs(dmid) <= TUMOR_BOUNDARY_NEAR_MM:
            dir_norm = float(np.linalg.norm(dvec_mm))
            if dir_norm <= 0.0:
                continue
            edge_dir = dvec_mm / dir_norm
            mid_xyz_mm = np.asarray(
                [(ux + vx) * 0.5 * sx, (uy + vy) * 0.5 * sy, (uz + vz) * 0.5 * sz],
                dtype=float,
            )

            radial_vec = mid_xyz_mm - tumor_centroid_xyz_mm
            radial_norm = float(np.linalg.norm(radial_vec))
            if radial_norm > 0.0:
                radial_abs_cos.append(
                    float(abs(np.dot(edge_dir, radial_vec / radial_norm)))
                )
                radial_weights.append(length_mm)

            mid_zyx = np.asarray(
                [(uz + vz) * 0.5, (uy + vy) * 0.5, (ux + vx) * 0.5], dtype=float
            )
            iz = int(np.clip(np.rint(mid_zyx[0]), 0, signed_dist_mm.shape[0] - 1))
            iy = int(np.clip(np.rint(mid_zyx[1]), 0, signed_dist_mm.shape[1] - 1))
            ix = int(np.clip(np.rint(mid_zyx[2]), 0, signed_dist_mm.shape[2] - 1))
            normal_vec = np.asarray(
                [grad_x[iz, iy, ix], grad_y[iz, iy, ix], grad_z[iz, iy, ix]],
                dtype=float,
            )
            normal_norm = float(np.linalg.norm(normal_vec))
            if normal_norm > 0.0:
                normal_abs_cos.append(
                    float(abs(np.dot(edge_dir, normal_vec / normal_norm)))
                )
                normal_weights.append(length_mm)

    per_shell_topology: dict[str, dict[str, object]] = {}
    for shell_name, _, _ in TUMOR_SHELL_SPECS:
        shell_nodes = [node for node in nodes if node_shell_map[node] == shell_name]
        subgraph = graph.subgraph(shell_nodes).copy()
        degrees = np.asarray(
            [subgraph.degree(n) for n in subgraph.nodes()], dtype=np.int64
        )
        node_count = int(subgraph.number_of_nodes())
        edge_count = int(subgraph.number_of_edges())
        endpoint_count = int(np.count_nonzero(degrees == 1))
        bifurcation_count = int(np.count_nonzero(degrees >= BIFURCATION_MIN_DEGREE))
        cycle_count = int(len(nx.cycle_basis(subgraph)))
        shell_voxels = int(shell_voxel_counts.get(shell_name, 0))
        shell_region_mm3 = float(shell_region_volume_mm3.get(shell_name, 0.0))
        shell_edge_length_mm = float(shell_length_mm.get(shell_name, 0.0))
        shell_edge_volume_mm3 = float(shell_volume_mm3.get(shell_name, 0.0))
        per_shell_topology[shell_name] = {
            "shell_region_voxels": int(shell_voxels),
            "shell_region_volume_mm3": float(shell_region_mm3),
            "shell_region_volume_per_tumor_mm3": _safe_ratio(
                shell_region_mm3, tumor_volume_mm3
            ),
            "node_count": node_count,
            "edge_count": edge_count,
            "endpoint_count": endpoint_count,
            "bifurcation_count": bifurcation_count,
            "cycle_count": cycle_count,
            "degree_distribution": {
                "deg0": int(np.count_nonzero(degrees == 0)),
                "deg1": int(np.count_nonzero(degrees == 1)),
                "deg2": int(np.count_nonzero(degrees == MIN_PATH_POINTS)),
                "deg3": int(np.count_nonzero(degrees == BIFURCATION_MIN_DEGREE)),
                "deg4plus": int(np.count_nonzero(degrees >= DEGREE_FOUR_PLUS)),
            },
            "node_density_per_shell_voxel": _safe_ratio(node_count, shell_voxels),
            "endpoint_density_per_shell_voxel": _safe_ratio(
                endpoint_count, shell_voxels
            ),
            "bifurcation_density_per_shell_voxel": _safe_ratio(
                bifurcation_count, shell_voxels
            ),
            "edge_density_per_shell_voxel": _safe_ratio(edge_count, shell_voxels),
            "cycle_density_per_shell_voxel": _safe_ratio(cycle_count, shell_voxels),
            "node_density_per_shell_mm3": _safe_ratio(node_count, shell_region_mm3),
            "endpoint_density_per_shell_mm3": _safe_ratio(
                endpoint_count, shell_region_mm3
            ),
            "bifurcation_density_per_shell_mm3": _safe_ratio(
                bifurcation_count, shell_region_mm3
            ),
            "edge_density_per_shell_mm3": _safe_ratio(edge_count, shell_region_mm3),
            "cycle_density_per_shell_mm3": _safe_ratio(cycle_count, shell_region_mm3),
            "shell_length_mm": float(shell_edge_length_mm),
            "shell_volume_burden_mm3": float(shell_edge_volume_mm3),
            "shell_length_density_mm_per_shell_mm3": _safe_ratio(
                shell_edge_length_mm, shell_region_mm3
            ),
            "shell_volume_density_mm3_per_shell_mm3": _safe_ratio(
                shell_edge_volume_mm3, shell_region_mm3
            ),
            "shell_length_mm_per_tumor_mm3": _safe_ratio(
                shell_edge_length_mm, tumor_volume_mm3
            ),
            "shell_volume_burden_mm3_per_tumor_mm3": _safe_ratio(
                shell_edge_volume_mm3, tumor_volume_mm3
            ),
            "shell_length_fraction_of_total_graph_length": _safe_ratio(
                shell_edge_length_mm,
                total_length_mm,
            ),
            "shell_volume_fraction_of_total_graph_volume": _safe_ratio(
                shell_edge_volume_mm3,
                total_volume_mm3,
            ),
        }

    near_node_mask = node_signed <= TUMOR_NEAR_MM
    near_radii_mm = node_radius_mm[near_node_mask]
    global_q90_radius_mm = (
        float(np.percentile(node_radius_mm, 90)) if node_radius_mm.size > 0 else 0.0
    )
    caliber = {
        "near_node_count": int(np.count_nonzero(near_node_mask)),
        "global_node_radius_q90_mm": global_q90_radius_mm,
        "near_radius_mean_mm": float(np.mean(near_radii_mm))
        if near_radii_mm.size > 0
        else 0.0,
        "near_radius_sd_mm": float(np.std(near_radii_mm))
        if near_radii_mm.size > 0
        else 0.0,
        "near_radius_cv": _safe_ratio(
            float(np.std(near_radii_mm)) if near_radii_mm.size > 0 else 0.0,
            float(np.mean(near_radii_mm)) if near_radii_mm.size > 0 else 0.0,
        ),
        "near_radius_q10_mm": float(np.percentile(near_radii_mm, 10))
        if near_radii_mm.size > 0
        else 0.0,
        "near_radius_q25_mm": float(np.percentile(near_radii_mm, 25))
        if near_radii_mm.size > 0
        else 0.0,
        "near_radius_q50_mm": float(np.percentile(near_radii_mm, 50))
        if near_radii_mm.size > 0
        else 0.0,
        "near_radius_q75_mm": float(np.percentile(near_radii_mm, 75))
        if near_radii_mm.size > 0
        else 0.0,
        "near_radius_q90_mm": float(np.percentile(near_radii_mm, 90))
        if near_radii_mm.size > 0
        else 0.0,
        "near_high_radius_tail_fraction_vs_global_q90": (
            float(np.mean(near_radii_mm >= global_q90_radius_mm))
            if near_radii_mm.size > 0
            else 0.0
        ),
    }

    seg_lengths_mm: list[float] = []
    seg_tortuosity: list[float] = []
    seg_curvature_mean: list[float] = []
    seg_near_lengths_mm: list[float] = []
    seg_near_tortuosity: list[float] = []
    seg_near_curvature_mean: list[float] = []
    for path in segment_paths:
        length_mm = _segment_path_length_mm(path, spacing_mm_zyx)
        if length_mm <= 0.0:
            continue
        tort = _segment_tortuosity_mm(path, spacing_mm_zyx)
        curv_mean = _segment_curvature_mean_per_mm(path, spacing_mm_zyx)
        seg_lengths_mm.append(length_mm)
        seg_tortuosity.append(tort)
        seg_curvature_mean.append(curv_mean)
        min_abs_signed = min(abs(float(node_signed_map[node])) for node in path)
        if min_abs_signed <= TUMOR_NEAR_MM:
            seg_near_lengths_mm.append(length_mm)
            seg_near_tortuosity.append(tort)
            seg_near_curvature_mean.append(curv_mean)

    seg_lengths_arr = np.asarray(seg_lengths_mm, dtype=float)
    seg_tortuosity_arr = np.asarray(seg_tortuosity, dtype=float)
    seg_curvature_arr = np.asarray(seg_curvature_mean, dtype=float)
    seg_near_lengths_arr = np.asarray(seg_near_lengths_mm, dtype=float)
    seg_near_tortuosity_arr = np.asarray(seg_near_tortuosity, dtype=float)
    seg_near_curvature_arr = np.asarray(seg_near_curvature_mean, dtype=float)

    shape_stats = {
        "global": {
            "segment_count": int(seg_lengths_arr.size),
            "length_weighted_tortuosity_quantiles": _weighted_quantiles(
                seg_tortuosity_arr, seg_lengths_arr
            ),
            "length_weighted_curvature_mean_quantiles": _weighted_quantiles(
                seg_curvature_arr, seg_lengths_arr
            ),
            "length_weighted_tortuosity_mean": _safe_ratio(
                float(np.sum(seg_tortuosity_arr * seg_lengths_arr)),
                float(np.sum(seg_lengths_arr)),
            ),
            "length_weighted_curvature_mean": _safe_ratio(
                float(np.sum(seg_curvature_arr * seg_lengths_arr)),
                float(np.sum(seg_lengths_arr)),
            ),
        },
        "near_tumor": {
            "segment_count": int(seg_near_lengths_arr.size),
            "length_weighted_tortuosity_quantiles": _weighted_quantiles(
                seg_near_tortuosity_arr, seg_near_lengths_arr
            ),
            "length_weighted_curvature_mean_quantiles": _weighted_quantiles(
                seg_near_curvature_arr, seg_near_lengths_arr
            ),
            "length_weighted_tortuosity_mean": _safe_ratio(
                float(np.sum(seg_near_tortuosity_arr * seg_near_lengths_arr)),
                float(np.sum(seg_near_lengths_arr)),
            ),
            "length_weighted_curvature_mean": _safe_ratio(
                float(np.sum(seg_near_curvature_arr * seg_near_lengths_arr)),
                float(np.sum(seg_near_lengths_arr)),
            ),
        },
    }

    radial_arr = np.asarray(radial_abs_cos, dtype=float)
    radial_w = np.asarray(radial_weights, dtype=float)
    normal_arr = np.asarray(normal_abs_cos, dtype=float)
    normal_w = np.asarray(normal_weights, dtype=float)
    directional = {
        "near_boundary_edge_count": int(len(radial_abs_cos)),
        "radial_alignment_abs_cos_mean": _safe_ratio(
            float(np.sum(radial_arr * radial_w)), float(np.sum(radial_w))
        ),
        "radial_alignment_abs_cos_quantiles": _weighted_quantiles(radial_arr, radial_w),
        "tangential_alignment_mean": 1.0
        - _safe_ratio(float(np.sum(radial_arr * radial_w)), float(np.sum(radial_w))),
        "normal_alignment_abs_cos_mean": _safe_ratio(
            float(np.sum(normal_arr * normal_w)), float(np.sum(normal_w))
        ),
        "normal_alignment_abs_cos_quantiles": _weighted_quantiles(normal_arr, normal_w),
    }

    tumor_burden = {
        "inside_tumor_volume_burden_mm3": float(inside_volume_mm3),
        "inside_tumor_length_mm": float(inside_length_mm),
        "inside_or_near_volume_burden_mm3": float(near_volume_mm3),
        "inside_or_near_length_mm": float(near_length_mm),
        "inside_tumor_volume_burden_per_tumor_mm3": _safe_ratio(
            inside_volume_mm3, tumor_volume_mm3
        ),
        "inside_or_near_volume_burden_per_tumor_mm3": _safe_ratio(
            near_volume_mm3, tumor_volume_mm3
        ),
    }

    boundary_crossing = {
        "crossing_edge_count": int(boundary_cross_edge_count),
        "crossing_length_mm": float(boundary_cross_length_mm),
        "crossing_volume_burden_mm3": float(boundary_cross_volume_mm3),
        "crossing_length_per_tumor_mm3": _safe_ratio(
            boundary_cross_length_mm, tumor_volume_mm3
        ),
        "crossing_volume_per_tumor_mm3": _safe_ratio(
            boundary_cross_volume_mm3, tumor_volume_mm3
        ),
    }

    near_shell_voxels = int(
        shell_voxel_counts.get("inside_tumor", 0)
        + shell_voxel_counts.get("shell_0_2mm", 0)
        + shell_voxel_counts.get("shell_2_5mm", 0)
    )
    near_shell_region_mm3 = float(
        shell_region_volume_mm3.get("inside_tumor", 0.0)
        + shell_region_volume_mm3.get("shell_0_2mm", 0.0)
        + shell_region_volume_mm3.get("shell_2_5mm", 0.0)
    )
    normalized = {
        "inside_length_fraction_of_total": _safe_ratio(
            inside_length_mm, total_length_mm
        ),
        "inside_volume_fraction_of_total": _safe_ratio(
            inside_volume_mm3, total_volume_mm3
        ),
        "near_length_fraction_of_total": _safe_ratio(near_length_mm, total_length_mm),
        "near_volume_fraction_of_total": _safe_ratio(near_volume_mm3, total_volume_mm3),
        "near_length_per_shell_voxel": _safe_ratio(near_length_mm, near_shell_voxels),
        "near_volume_per_shell_voxel": _safe_ratio(near_volume_mm3, near_shell_voxels),
        "near_length_per_shell_mm3": _safe_ratio(near_length_mm, near_shell_region_mm3),
        "near_volume_per_shell_mm3": _safe_ratio(
            near_volume_mm3, near_shell_region_mm3
        ),
        "inside_length_per_tumor_voxel": _safe_ratio(inside_length_mm, tumor_voxels),
        "inside_volume_per_tumor_voxel": _safe_ratio(inside_volume_mm3, tumor_voxels),
        "near_length_per_skeleton_edge": _safe_ratio(
            near_length_mm, graph.number_of_edges()
        ),
        "near_volume_per_skeleton_edge": _safe_ratio(
            near_volume_mm3, graph.number_of_edges()
        ),
        "inside_length_per_tumor_equiv_radius": _safe_ratio(
            inside_length_mm, tumor_equivalent_radius_mm
        ),
        "near_length_per_tumor_equiv_radius": _safe_ratio(
            near_length_mm, tumor_equivalent_radius_mm
        ),
        "crossing_length_per_tumor_equiv_surface": _safe_ratio(
            boundary_cross_length_mm,
            tumor_equivalent_surface_mm2,
        ),
    }

    if priority_4d is None:
        kinetic_features = {"status": "skipped_missing_time_series"}
    else:
        kinetic_features = compute_tumor_kinematic_feature_payload(
            graph=graph,
            segment_paths=segment_paths,
            segment_metrics_by_path=segment_metrics_by_path,
            support_mask_zyx=support,
            tumor_mask_zyx=tumor,
            spacing_mm_zyx=spacing_mm_zyx,
            signed_dist_mm=signed_dist_mm,
            shell_voxel_counts=shell_voxel_counts,
            priority_4d=priority_4d,
            study_timepoints=study_timepoints,
            reference_mask_zyx=reference_mask_zyx,
        )

    return {
        "status": "ok",
        "version": "tumor_graph_features_v5_mm_shell_density_refnorm_topology",
        "shell_specs_mm": [
            {"name": name, "lower": float(lower), "upper": float(upper)}
            for name, lower, upper in TUMOR_SHELL_SPECS
        ],
        "graph_totals": {
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
            "total_length_mm": float(total_length_mm),
            "total_volume_burden_mm3": float(total_volume_mm3),
        },
        "tumor_burden": tumor_burden,
        "boundary_crossing": boundary_crossing,
        "per_shell_topology": per_shell_topology,
        "caliber_heterogeneity_near_tumor": caliber,
        "directional_near_boundary": directional,
        "length_weighted_shape_stats": shape_stats,
        "normalized_ratios": normalized,
        "kinematic_features": kinetic_features,
    }


def matches_column(column: str) -> bool:
    """Return whether a model column belongs to the structural graph block."""
    return column.startswith(METADATA_PREFIXES)


def resolve_tumor_graph_features_path(
    *,
    case_id: str,
    study_dir: Path,
    summary: dict[str, Any] | None,
) -> Path | None:
    """Resolve the saved tumor-graph JSON for one study."""
    candidates: list[Path] = []
    if isinstance(summary, dict):
        path_from_summary = summary.get("tumor_graph_features_path")
        if path_from_summary:
            candidates.append(Path(str(path_from_summary)))

        feature_stats = summary.get("feature_stats", {})
        if isinstance(feature_stats, dict):
            path_from_feature_stats = feature_stats.get("tumor_graph_features_path")
            if path_from_feature_stats:
                candidates.append(Path(str(path_from_feature_stats)))

    candidates.append(study_dir / f"{case_id}_tumor_graph_features.json")

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (study_dir / candidate)
        if resolved.exists():
            return resolved
    return None


def load_tumor_graph_payload(tumor_graph_path: Path) -> dict[str, Any]:
    """Load the tumor-centered feature JSON written during graph extraction."""
    payload = json.loads(tumor_graph_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Tumor-graph payload must be a dict: {tumor_graph_path}")
    return payload


def extract_graph_json_features(payload: dict[str, Any]) -> dict[str, float]:
    """Extract the structural graph block from the saved tumor-graph JSON."""
    features: dict[str, float] = {
        "tumor_graph_features_loaded": 1.0,
    }

    status = str(payload.get("status", "")).strip().lower()
    if status:
        features["tumor_graph_status_ok"] = 1.0 if status == "ok" else 0.0

    version = str(payload.get("version", "")).strip().lower()
    if version:
        features["tumor_graph_version_is_v2"] = 1.0 if "v2" in version else 0.0
        features["tumor_graph_version_has_kinematics"] = (
            1.0 if "kinematic" in version else 0.0
        )

    for group_name in STRUCTURAL_GRAPH_FEATURE_GROUPS:
        group_value = payload.get(group_name)
        if group_value is None:
            continue
        flatten_numeric_payload(group_value, prefix=group_name, out=features)

    return features


def extract_local_graph_features(
    local_context: dict[str, Any], tumor_radii_voxels: list[int]
) -> dict[str, float]:
    """Extract graph-block features from centerline overlap with tumor-local regions."""
    if not local_context.get("available", False):
        return {}

    skeleton_voxels = float(local_context["skeleton_voxels"])
    tumor_voxels = float(local_context["tumor_voxels"])
    overlap_tumor = float(local_context["overlap_tumor"])
    skeleton_crop = np.asarray(local_context["skeleton_crop"], dtype=bool)
    dist_to_tumor = np.asarray(local_context["dist_to_tumor"], dtype=float)

    features: dict[str, float] = {
        "graph_skel_in_tumor_voxels": overlap_tumor,
        "graph_skel_in_tumor_frac_skeleton": safe_ratio(overlap_tumor, skeleton_voxels),
        "graph_skel_in_tumor_frac_tumor": safe_ratio(overlap_tumor, tumor_voxels),
        "graph_skel_per_tumor_voxel": safe_ratio(skeleton_voxels, tumor_voxels),
    }

    prev_radius = 0
    for radius in tumor_radii_voxels:
        region = dist_to_tumor <= float(radius)
        region_voxels = float(np.count_nonzero(region))
        region_overlap = float(np.count_nonzero(skeleton_crop & region))
        region_prefix = f"tumor_r{radius}"
        features[f"graph_{region_prefix}_skel_voxels"] = region_overlap
        features[f"graph_{region_prefix}_skel_frac_skeleton"] = safe_ratio(
            region_overlap, skeleton_voxels
        )
        features[f"graph_{region_prefix}_skel_frac_region"] = safe_ratio(
            region_overlap, region_voxels
        )

        if radius > prev_radius:
            shell = (dist_to_tumor > float(prev_radius)) & (
                dist_to_tumor <= float(radius)
            )
            shell_voxels = float(np.count_nonzero(shell))
            shell_overlap = float(np.count_nonzero(skeleton_crop & shell))
            shell_prefix = f"tumor_shell_r{prev_radius}_{radius}"
            features[f"graph_{shell_prefix}_skel_voxels"] = shell_overlap
            features[f"graph_{shell_prefix}_skel_frac_skeleton"] = safe_ratio(
                shell_overlap, skeleton_voxels
            )
            features[f"graph_{shell_prefix}_skel_frac_shell"] = safe_ratio(
                shell_overlap, shell_voxels
            )
        prev_radius = radius

    return features


def add_derived_graph_features(row: dict[str, Any]) -> None:
    """Add simple normalized graph descriptors derived from existing columns."""
    seg_n = row.get("morph_seg_unique_n")
    graph_nodes = row.get("graph_nodes")
    graph_edges = row.get("graph_edges")
    components = row.get("graph_component_count")
    skeleton_voxels = row.get("graph_skeleton_voxels")
    support_voxels = row.get("graph_support_voxels")
    bifurcations = row.get("morph_bifurcation_count")

    row["graph_edge_node_ratio"] = safe_ratio(graph_edges, graph_nodes)
    row["graph_component_node_ratio"] = safe_ratio(components, graph_nodes)
    row["graph_segment_node_ratio"] = safe_ratio(seg_n, graph_nodes)
    row["graph_bifurcation_node_ratio"] = safe_ratio(bifurcations, graph_nodes)
    row["graph_skeleton_support_ratio"] = safe_ratio(skeleton_voxels, support_voxels)
    row["morph_seg_unique_per_skeleton_voxel"] = safe_ratio(seg_n, skeleton_voxels)
    row["morph_bifurcation_per_segment"] = safe_ratio(bifurcations, seg_n)

    for col in (
        "graph_skeleton_voxels",
        "graph_support_voxels",
        "graph_nodes",
        "graph_edges",
        "graph_component_count",
        "morph_seg_raw_n",
        "morph_seg_unique_n",
        "morph_seg_length_sum",
        "morph_seg_volume_sum",
        "morph_bifurcation_count",
    ):
        value = safe_float(row.get(col))
        if value is None or value < 0.0:
            row[f"{col}_log1p"] = float("nan")
        else:
            row[f"{col}_log1p"] = float(np.log1p(value))
