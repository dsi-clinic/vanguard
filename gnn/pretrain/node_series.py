"""Per-node contrast time-series for the forecasting pretext task.

The classification pipeline collapses each voxel's DCE enhancement curve to
summary kinetics (``gnn.kinetics.node_kinetic_features``) before it reaches the
model. Forecasting instead needs the *full* per-node curve on the graph, so this
module re-samples the raw 4D DCE series at each node exactly the way
``gnn.data_loader._attach_node_features`` does (``curve = dce_4d[:, z, y, x]``)
and applies the same frame-0 baseline convention (``gnn.kinetics``,
``AGENTS.md``: never per-timepoint-normalize away the kinetic meaning).

One builder per node definition (design doc §4):

- ``voxel_node_series`` -- one node per voxel; the raw curve is the signal.
- ``segment_node_series`` -- one node per vessel segment; the signal is an
  aggregate (mean) over the segment's voxels, since a segment has no single
  curve.

This is the single source of truth for "node -> forecasting signal", mirroring
how ``gnn.kinetics`` is the single source of truth for "curve -> kinetics", so
the forecasting target can never silently diverge from the measured signal.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graph_extraction.skeleton_to_graph_primitives import extract_segments

Point3D = tuple[int, int, int]

_NDIM_4D = 4  # (t, z, y, x)


def voxel_node_series(
    dce_4d: np.ndarray, nodes: list[Point3D], *, baseline_subtract: bool = True
) -> np.ndarray:
    """Stack each voxel node's DCE curve into a ``(num_nodes, T)`` matrix.

    ``dce_4d`` is the aligned raw series in ``(t, z, y, x)`` order (as loaded by
    ``gnn.raw_dce.load_raw_dce_series``); ``nodes`` are voxel coordinates in
    ``(x, y, z)`` order (the node keys used throughout ``gnn.data_loader`` and
    ``gnn.segment_graph``). Row ``i`` is node ``i``'s curve, sampled as
    ``dce_4d[:, z, y, x]`` -- identical indexing to
    ``data_loader._attach_node_features`` so voxel forecasting and voxel
    classification see the same signal.

    With ``baseline_subtract`` (the committed default, design doc §8a) each row
    is ``curve - curve[0]``: the frame-0-baselined enhancement, matching
    ``gnn.kinetics.node_kinetic_features``. The absolute intensity is scanner-
    and scale-dependent, so this is the target we forecast; per-node
    normalisation, if used, is a loss-time concern (see ``gnn.pretrain.loss``),
    not applied here.
    """
    if dce_4d.ndim != _NDIM_4D:
        raise ValueError(f"dce_4d must be 4D (t,z,y,x); got shape {dce_4d.shape}")
    if not nodes:
        raise ValueError("nodes is empty; cannot build a node-series matrix")
    num_timepoints = int(dce_4d.shape[0])
    series = np.empty((len(nodes), num_timepoints), dtype=np.float32)
    for i, node in enumerate(nodes):
        x, y, z = int(node[0]), int(node[1]), int(node[2])
        series[i] = dce_4d[:, z, y, x]
    if baseline_subtract:
        series = series - series[:, :1]
    return series


def segment_node_series(
    voxel_graph: nx.Graph, dce_4d: np.ndarray, *, baseline_subtract: bool = True
) -> np.ndarray:
    """Per-segment contrast series: mean voxel enhancement per frame, ``(num_segments, T)``.

    A segment node has no single curve, so its per-frame value (design doc §4.2)
    is an **aggregate over the segment's voxels** -- here the mean
    baseline-subtracted enhancement across the segment's voxels at each frame.

    Segments come from ``graph_extraction.extract_segments(voxel_graph)`` in the
    SAME order ``gnn.segment_graph.build_segment_line_graph`` uses to number its
    integer segment-node ids (``enumerate(extract_segments(...))``), so row ``i``
    aligns with segment-node ``i`` -- and therefore with ``data.x`` row ``i``
    after ``from_networkx`` (locked by a test). ``extract_segments`` is
    deterministic, so the two calls agree.

    Baseline subtraction matches ``voxel_node_series`` (frame-0 baseline). The
    mean is linear, so per-voxel-baseline-then-mean equals mean-then-baseline;
    either way the segment series starts at 0.

    **Modeling choice (design doc §4.2, sweepable -- see
    ``contrast_pretraining_params.md``):** using the *mean* is a decision the
    voxel case doesn't face. Median / robust summaries are alternatives, and the
    mean discards within-segment heterogeneity (already flagged in
    ``gnn.segment_graph``). Committed to mean for the first pass.
    """
    if dce_4d.ndim != _NDIM_4D:
        raise ValueError(f"dce_4d must be 4D (t,z,y,x); got shape {dce_4d.shape}")
    segments = extract_segments(voxel_graph)
    if not segments:
        raise ValueError("extract_segments returned no segments for the voxel graph")
    num_timepoints = int(dce_4d.shape[0])
    series = np.empty((len(segments), num_timepoints), dtype=np.float32)
    for i, path in enumerate(segments):
        voxel_curves = np.stack(
            [dce_4d[:, z, y, x] for x, y, z in path], axis=0
        ).astype(np.float32)  # (num_voxels_in_segment, T)
        if baseline_subtract:
            voxel_curves = voxel_curves - voxel_curves[:, :1]
        series[i] = voxel_curves.mean(axis=0)
    return series
