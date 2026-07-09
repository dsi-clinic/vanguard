"""Segment-as-node (line-graph) construction for ``node_mode="segment"`` (Option B).

Contracts the voxel skeleton graph to its **line graph**: each vessel segment --
a polyline between junctions/endpoints, as traced by
``graph_extraction.extract_segments`` -- becomes one **node**, and two
segment-nodes are adjacent when they share a junction/endpoint voxel. Segment
geometry (``graph_extraction.compute_segment_metrics``: length, tortuosity,
volume, radius & curvature stats) and DCE kinetics (the same per-voxel
enhancement-curve features voxel mode uses, from ``gnn.kinetics``, summarized
along the segment) become the node's features. The result flows through the
existing ``from_networkx -> data.x -> GCNClassifier`` path with no model change,
which is the whole point of building B before A.

See ``gnn/DESIGN_segment_graph.md`` (§8, "How B is built") for the full design,
including the degenerate-topology policies this module enforces.

**Summarization / information loss (flagged per the repo's error-handling
rules):** collapsing a segment's voxels to fixed-width node features loses
within-segment heterogeneity. Kinetics are summarized as the mean and std of
the per-voxel scalars along the segment. ``time_to_enhancement`` is NaN-aware
(``nanmean``/``nanstd``): voxels with no detected arrival are excluded from the
segment's arrival mean, and a segment whose voxels *all* lack arrival gets a
NaN -- surfaced by the same feature-NaN audit voxel mode uses, never silently
defaulted.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np

from gnn.kinetics import node_kinetic_features
from graph_extraction.skeleton_to_graph_primitives import (
    Point3D,
    compute_segment_metrics,
    extract_segments,
)

_SINGLE_TIMEPOINT = 1
_SEGMENT_DEGREE = 2

# Geometry features pulled from ``compute_segment_metrics`` output. Each entry
# maps the flat segment-feature name to the (top-level-key, sub-key) path into
# that nested dict; a ``None`` sub-key means the value is the top-level scalar.
_GEOMETRY_FEATURE_PATHS: dict[str, tuple[str, str | None]] = {
    "seg_length": ("length", None),
    "seg_tortuosity": ("tortuosity", None),
    "seg_volume": ("volume", None),
    "seg_radius_mean": ("radius", "mean"),
    "seg_radius_std": ("radius", "sd"),
    "seg_radius_median": ("radius", "median"),
    "seg_radius_min": ("radius", "min"),
    "seg_radius_max": ("radius", "max"),
    "seg_curvature_mean": ("curvature", "mean"),
    "seg_curvature_std": ("curvature", "sd"),
    "seg_curvature_max": ("curvature", "max"),
}

# Per-voxel kinetic scalars (from ``gnn.kinetics.node_kinetic_features``)
# summarized along a segment. ``mean``/``std`` columns are emitted for each.
_KINETIC_SCALARS: tuple[str, ...] = (
    "peak_time",
    "peak_enhancement",
    "time_to_enhancement",
    "washin_slope",
    "auc_positive",
)

# Extra topological descriptor of the segment itself.
_TOPO_FEATURES: tuple[str, ...] = ("seg_num_voxels",)


def _segment_feature_names() -> tuple[str, ...]:
    """All segment-level feature names this module can attach to a node."""
    kinetic = tuple(
        f"seg_{scalar}_{stat}"
        for scalar in _KINETIC_SCALARS
        for stat in ("mean", "std")
    )
    return tuple(_GEOMETRY_FEATURE_PATHS) + kinetic + _TOPO_FEATURES


# Segment-mode analogue of ``data_loader._FEATURE_ATTR``: a segment feature's
# name *is* its ``nx`` node-attribute key (unlike voxel mode, where e.g.
# ``peak_time`` maps to the ``peak_time_norm`` attribute), so this is the
# identity map -- kept as an explicit dict for symmetry and so callers have one
# authoritative list of supported segment features.
SEGMENT_FEATURE_ATTR: dict[str, str] = {name: name for name in _segment_feature_names()}


def _segment_kinetic_summary(
    path: list[Point3D], dce_4d: np.ndarray, time_axis: np.ndarray
) -> dict[str, float]:
    """Mean/std of the per-voxel kinetic scalars along one segment's voxels.

    Samples the raw DCE curve at each voxel of ``path`` (``dce_4d[:, z, y, x]``,
    matching voxel mode), derives the per-voxel scalars via
    ``node_kinetic_features``, and reduces each along the segment. Arrival
    (``time_to_enhancement``) is normalized like voxel mode
    (``tte_idx / (T - 1)``) and NaN for voxels with no detected arrival, then
    summarized with ``nanmean``/``nanstd`` so no-arrival voxels don't corrupt
    the segment mean and an all-no-arrival segment is left NaN (audited, not
    defaulted).
    """
    denom = float(max(int(dce_4d.shape[0]) - 1, _SINGLE_TIMEPOINT))
    samples: dict[str, list[float]] = {scalar: [] for scalar in _KINETIC_SCALARS}
    for x, y, z in path:
        kinetic = node_kinetic_features(dce_4d[:, z, y, x], time_axis)
        samples["peak_time"].append(float(kinetic["peak_idx"]) / denom)
        samples["peak_enhancement"].append(float(kinetic["peak_enhancement"]))
        tte_idx = kinetic["tte_idx"]
        samples["time_to_enhancement"].append(
            float("nan") if tte_idx is None else float(tte_idx) / denom
        )
        samples["washin_slope"].append(float(kinetic["washin_slope"]))
        samples["auc_positive"].append(float(kinetic["auc_positive"]))

    summary: dict[str, float] = {}
    for scalar, values in samples.items():
        arr = np.asarray(values, dtype=float)
        # nan-aware only where a NaN is a valid "no signal" sentinel (arrival);
        # the other scalars are always finite, so nanmean == mean for them.
        summary[f"seg_{scalar}_mean"] = float(np.nanmean(arr))
        summary[f"seg_{scalar}_std"] = float(np.nanstd(arr))
    return summary


def _segment_node_attributes(
    path: list[Point3D],
    radius_map: dict[Point3D, float],
    dce_4d: np.ndarray,
    time_axis: np.ndarray,
) -> dict[str, float]:
    """All segment-level features for one segment polyline, as a flat dict.

    Combines geometry (``compute_segment_metrics``), kinetics
    (``_segment_kinetic_summary``), and the segment's voxel count. Keys are the
    ``SEGMENT_FEATURE_ATTR`` feature names.
    """
    metrics = compute_segment_metrics(path, radius_map)
    attrs: dict[str, float] = {}
    for name, (top_key, sub_key) in _GEOMETRY_FEATURE_PATHS.items():
        value = metrics[top_key] if sub_key is None else metrics[top_key][sub_key]
        attrs[name] = float(value)
    attrs.update(_segment_kinetic_summary(path, dce_4d, time_axis))
    attrs["seg_num_voxels"] = float(len(path))
    return attrs


def build_segment_line_graph(
    voxel_graph: nx.Graph,
    radius_map: dict[Point3D, float],
    dce_4d: np.ndarray,
    time_axis: np.ndarray,
) -> nx.Graph:
    """Build the segment line graph (Option B) from a voxel skeleton graph.

    Nodes are integer segment ids (``0..num_segments-1``) carrying the
    ``SEGMENT_FEATURE_ATTR`` features plus a ``pos`` (segment midpoint, for
    parity with voxel mode's ``data.pos``). Two segment-nodes are joined by an
    edge when their polylines share a junction/endpoint voxel. No ``edge_attr``
    is attached -- plain adjacency, so ``GCNConv`` is reused verbatim; junction
    features are an Option A concern (see ``DESIGN_segment_graph.md`` §6).

    Raises ``NotImplementedError`` for a pure-cycle connected component (all
    voxels degree 2, no junction/endpoint to anchor a segment): ``DESIGN`` §2.1
    specifies a representative-voxel policy for it, but until we actually
    observe one on real data this fails loudly rather than silently dropping
    the whole component's vessels.
    """
    for component in nx.connected_components(voxel_graph):
        if all(voxel_graph.degree(node) == _SEGMENT_DEGREE for node in component):
            raise NotImplementedError(
                "Pure-cycle skeleton component (all voxels degree 2, no "
                "junction/endpoint) has no anchor segment node. See "
                "gnn/DESIGN_segment_graph.md §2.1 for the representative-voxel "
                "policy to implement when this is first observed."
            )

    segments = extract_segments(voxel_graph)
    if not segments:
        raise ValueError(
            "extract_segments returned no segments for a non-empty voxel graph"
        )

    line = nx.Graph()
    junction_to_segments: dict[Point3D, list[int]] = defaultdict(list)
    for seg_id, path in enumerate(segments):
        attrs = _segment_node_attributes(path, radius_map, dce_4d, time_axis)
        midpoint = np.mean(np.asarray(path, dtype=float), axis=0)
        line.add_node(seg_id, pos=midpoint, **attrs)
        # A segment is anchored at both its endpoints (a self-loop segment has
        # path[0] == path[-1], so it registers once at that single junction).
        junction_to_segments[path[0]].append(seg_id)
        if path[-1] != path[0]:
            junction_to_segments[path[-1]].append(seg_id)

    for seg_ids in junction_to_segments.values():
        for i in range(len(seg_ids)):
            for j in range(i + 1, len(seg_ids)):
                if seg_ids[i] != seg_ids[j]:
                    line.add_edge(seg_ids[i], seg_ids[j])
    return line
