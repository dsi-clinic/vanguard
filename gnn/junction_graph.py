"""Segment-as-edge (junction) graph for ``node_mode="junction"`` (Option A).

The topologically honest contraction of the voxel skeleton: **nodes are
junction/endpoint voxels** (degree != 2) and **edges are vessel segments**,
each carrying the segment summary as ``edge_attr``. This is the line-graph dual
of segment mode (Option B, ``gnn.segment_graph``): B puts the segment summary on
*nodes*; A puts the *same* summary vector on *edges* and keeps junctions as
first-class nodes with their own per-voxel features (the same DCE kinetics +
radius voxel mode uses) plus topological degree.

Because ``GCNConv`` ignores edge features, junction-mode graphs are consumed by
an edge-aware conv (``gnn.model.EdgeGNNClassifier``), not the plain
``GCNClassifier`` that voxel/segment mode reuse.

Unlike voxel/segment mode this builder returns a :class:`torch_geometric.data.Data`
directly (rather than going through ``from_networkx``): it must emit *both*
per-node and per-edge feature attributes, express undirected segments as
both-direction edges, and preserve parallel segments and self-loops between the
same junctions -- all of which are awkward/lossy through ``from_networkx`` but
exact when the ``edge_index``/``edge_attr`` tensors are built here. Downstream
``_finalize_data`` stacks ``data.x`` from the selected node features and
``data.edge_attr`` from the selected edge features, exactly as it stacks
``data.x`` in the other modes.

See ``gnn/DESIGN_segment_graph.md`` (§2 segment-as-edge, §5 model). Pure-cycle
components (no junction/endpoint) raise, matching segment mode's §2.1 policy.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from gnn.kinetics import DENOMINATOR_BASELINE, node_kinetic_features
from gnn.segment_graph import SEGMENT_FEATURE_ATTR, segment_summary_features
from graph_extraction.skeleton_to_graph_primitives import (
    Point3D,
    detect_bifurcations,
    extract_segments,
)

_SINGLE_TIMEPOINT = 1
_SEGMENT_DEGREE = 2

# Junction/endpoint node features: the same per-voxel DCE kinetics voxel mode
# samples (evaluated at the single junction voxel), plus local radius and the
# node's topological degree. Names are their own attribute keys (identity map),
# mirroring ``SEGMENT_FEATURE_ATTR``.
_JUNCTION_KINETIC_SCALARS: tuple[str, ...] = (
    "peak_time",
    "peak_enhancement",
    "time_to_enhancement",
    "washin_slope",
    "washout_slope",
    "auc_positive",
)
# Opening-angle summary between every pair of a bifurcation's outgoing
# segments (``graph_extraction.detect_bifurcations``, degree>=3 voxels only).
# NaN for a degree-1 endpoint node (no neighbor pair to measure an angle
# from) -- a real "not a bifurcation" case, sentinel-filled and audited via
# ``gnn.data_loader``'s ``NO_BIFURCATION_SENTINEL`` / ``no_bifurcation_count``,
# never silently defaulted. Explicitly named as the natural next junction
# feature in ``gnn/DESIGN_segment_graph.md`` §9.
_BIFURCATION_ANGLE_NAMES: tuple[str, ...] = (
    "bifurcation_angle_mean",
    "bifurcation_angle_min",
    "bifurcation_angle_max",
)
JUNCTION_NODE_FEATURE_ATTR: dict[str, str] = {
    name: name
    for name in (
        *_JUNCTION_KINETIC_SCALARS,
        "radius",
        "degree",
        *_BIFURCATION_ANGLE_NAMES,
    )
}

# Edges carry the mode-neutral segment summary -- the identical vector segment
# mode (Option B) attaches to its nodes (``gnn.segment_graph``).
JUNCTION_EDGE_FEATURE_ATTR: dict[str, str] = dict(SEGMENT_FEATURE_ATTR)


def _bifurcation_angle_summary(
    voxel_graph: nx.Graph,
) -> dict[Point3D, dict[str, float]]:
    """Per-junction opening-angle mean/min/max, keyed by voxel coordinate.

    Built once per case from ``detect_bifurcations`` (all pairwise angles, in
    degrees, between neighbor directions at each degree>=3 voxel). A degree-1
    endpoint node has no entry here -- it has no bifurcation to measure -- and
    the caller fills it with NaN for the ``_BIFURCATION_ANGLE_NAMES`` columns,
    matching how ``time_to_enhancement`` handles "no detected arrival".
    """
    summary: dict[Point3D, dict[str, float]] = {}
    for entry in detect_bifurcations(voxel_graph):
        node = tuple(entry["bifurcation"]["midpoint"])
        angles = np.asarray(list(entry["angles"].values()), dtype=float)
        summary[node] = {
            "bifurcation_angle_mean": float(np.mean(angles)),
            "bifurcation_angle_min": float(np.min(angles)),
            "bifurcation_angle_max": float(np.max(angles)),
        }
    return summary


def _junction_node_features(
    node: Point3D,
    degree: int,
    radius: float,
    dce_4d: np.ndarray,
    time_axis: np.ndarray,
    bifurcation: dict[str, float] | None,
    *,
    baseline_frame_count: int = 1,
    relative_enhancement: bool = False,
    baseline_floor_frac: float = 0.0,
    denominator: str = DENOMINATOR_BASELINE,
) -> dict[str, float]:
    """Per-voxel features for one junction/endpoint voxel, plus its degree.

    Samples the raw DCE curve at the junction voxel (``dce_4d[:, z, y, x]``,
    matching voxel mode) and normalizes ``peak_time`` / ``time_to_enhancement``
    by total acquisition duration the same way. ``time_to_enhancement`` is NaN
    when the voxel
    shows no detected arrival; that NaN is replaced with
    ``TTE_NO_ARRIVAL_SENTINEL`` when stacked into ``data.x`` in
    ``gnn.data_loader._finalize_data`` (same policy as voxel/segment modes and
    as the segment-summary ``edge_attr``), and the count is audited via
    ``graph_qc.csv``'s ``tte_no_arrival_count`` -- never silently defaulted.

    ``bifurcation`` is ``None`` for a degree-1 endpoint node (no opening angle
    to measure); its three ``_BIFURCATION_ANGLE_NAMES`` columns are then set to
    NaN, sentinel-filled/audited the same way via ``no_bifurcation_count``.
    """
    duration_seconds = float(time_axis[-1] - time_axis[0])
    if not duration_seconds > 0.0:
        duration_seconds = float(_SINGLE_TIMEPOINT)
    x, y, z = node
    kinetic = node_kinetic_features(
        dce_4d[:, z, y, x],
        time_axis,
        baseline_frame_count=baseline_frame_count,
        relative_enhancement=relative_enhancement,
        baseline_floor_frac=baseline_floor_frac,
        denominator=denominator,
    )
    tte_idx = kinetic["tte_idx"]
    features = {
        "peak_time": float(kinetic["peak_time_seconds"]) / duration_seconds,
        "peak_enhancement": float(kinetic["peak_enhancement"]),
        "time_to_enhancement": (
            float("nan")
            if tte_idx is None
            else float(kinetic["tte_seconds"]) / duration_seconds
        ),
        "washin_slope": float(kinetic["washin_slope"]),
        "washout_slope": float(kinetic["washout_slope"]),
        "auc_positive": float(kinetic["auc_positive"]),
        "radius": float(radius),
        "degree": float(degree),
    }
    if bifurcation is None:
        features.update({name: float("nan") for name in _BIFURCATION_ANGLE_NAMES})
    else:
        features.update(bifurcation)
    return features


def build_junction_graph(
    voxel_graph: nx.Graph,
    radius_map: dict[Point3D, float],
    dce_4d: np.ndarray,
    time_axis: np.ndarray,
    *,
    baseline_frame_count: int = 1,
    relative_enhancement: bool = False,
    baseline_floor_frac: float = 0.0,
    denominator: str = DENOMINATOR_BASELINE,
) -> Data:
    """Build the segment-as-edge (junction) graph from a voxel skeleton graph.

    Returns a :class:`~torch_geometric.data.Data` carrying, for each junction
    node, the ``JUNCTION_NODE_FEATURE_ATTR`` features as named ``(num_nodes,)``
    attributes and ``pos`` (the junction voxel coordinate); and for each
    directed edge, the ``JUNCTION_EDGE_FEATURE_ATTR`` (segment summary) features
    as named ``(num_edges,)`` attributes aligned with ``edge_index``. Segments
    are stored in **both directions** (matching the voxel/segment
    doubled-``num_edges`` convention), except a self-loop segment
    (``path[0] == path[-1]``), stored once. Parallel segments between the same
    two junctions are kept as separate edges (a real multigraph -- distinct
    vessels, not deduplicated).

    ``_finalize_data`` later stacks ``data.x`` from the selected node features
    and ``data.edge_attr`` from the selected edge features.

    Raises ``NotImplementedError`` for a pure-cycle connected component (all
    voxels degree 2, no junction/endpoint), matching segment mode's §2.1 policy.
    """
    for component in nx.connected_components(voxel_graph):
        if all(voxel_graph.degree(node) == _SEGMENT_DEGREE for node in component):
            raise NotImplementedError(
                "Pure-cycle skeleton component (all voxels degree 2, no "
                "junction/endpoint) has no anchor node for a segment edge. See "
                "gnn/DESIGN_segment_graph.md §2.1 for the representative-voxel "
                "policy to implement when this is first observed."
            )

    segments = extract_segments(voxel_graph)
    if not segments:
        raise ValueError(
            "extract_segments returned no segments for a non-empty voxel graph"
        )

    # Node set = union of segment endpoints, indexed in first-seen order.
    node_index: dict[Point3D, int] = {}
    for path in segments:
        for endpoint in (path[0], path[-1]):
            if endpoint not in node_index:
                node_index[endpoint] = len(node_index)
    ordered_nodes = sorted(node_index, key=node_index.__getitem__)

    bifurcation_summary = _bifurcation_angle_summary(voxel_graph)
    node_columns: dict[str, list[float]] = {
        name: [] for name in JUNCTION_NODE_FEATURE_ATTR
    }
    pos: list[list[float]] = []
    for node in ordered_nodes:
        features = _junction_node_features(
            node,
            voxel_graph.degree(node),
            radius_map[node],
            dce_4d,
            time_axis,
            bifurcation_summary.get(node),
            baseline_frame_count=baseline_frame_count,
            relative_enhancement=relative_enhancement,
            baseline_floor_frac=baseline_floor_frac,
            denominator=denominator,
        )
        for name in JUNCTION_NODE_FEATURE_ATTR:
            node_columns[name].append(features[name])
        pos.append([float(node[0]), float(node[1]), float(node[2])])

    src: list[int] = []
    dst: list[int] = []
    edge_columns: dict[str, list[float]] = {
        name: [] for name in JUNCTION_EDGE_FEATURE_ATTR
    }
    for path in segments:
        u, v = node_index[path[0]], node_index[path[-1]]
        summary = segment_summary_features(
            path,
            radius_map,
            dce_4d,
            time_axis,
            voxel_graph,
            baseline_frame_count=baseline_frame_count,
            relative_enhancement=relative_enhancement,
            baseline_floor_frac=baseline_floor_frac,
            denominator=denominator,
        )
        directed = [(u, v)] if u == v else [(u, v), (v, u)]
        for a, b in directed:
            src.append(a)
            dst.append(b)
            for name in JUNCTION_EDGE_FEATURE_ATTR:
                edge_columns[name].append(summary[name])

    data = Data()
    data.num_nodes = len(ordered_nodes)
    data.pos = torch.tensor(pos, dtype=torch.float)
    for name, values in node_columns.items():
        data[name] = torch.tensor(values, dtype=torch.float)
    data.edge_index = torch.tensor([src, dst], dtype=torch.long)
    for name, values in edge_columns.items():
        data[name] = torch.tensor(values, dtype=torch.float)

    # Connectivity of the junction graph itself (disconnected vessel trees show
    # up as > 1), counted here since edge_index is directed-both-ways.
    junction_nx = nx.Graph()
    junction_nx.add_nodes_from(range(len(ordered_nodes)))
    junction_nx.add_edges_from(zip(src, dst, strict=True))
    data.num_connected_components = nx.number_connected_components(junction_nx)
    return data
