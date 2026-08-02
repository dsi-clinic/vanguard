"""Per-node contrast time-series for the forecasting pretext task.

The classification pipeline collapses each voxel's DCE enhancement curve to
summary kinetics (``gnn.kinetics.node_kinetic_features``) before it reaches the
model. Forecasting instead needs the *full* per-node curve on the graph, so this
module re-samples the raw 4D DCE series at each node exactly the way
``gnn.data_loader._attach_node_features`` does (``curve = dce_4d[:, z, y, x]``)
and baseline-references it through the **same** transform the kinetic path uses
(``gnn.kinetics.baseline_relative_curve`` with the case's ``baseline_frame_count``
and ``relative_enhancement``). For UChicago that is the mean-of-baseline-frames,
relative-signal-change ``(S(t) - S0) / S0`` contract; for legacy cohorts
(``baseline_frame_count=1``, absolute) it reduces to the old ``curve - curve[0]``.
Per ``AGENTS.md`` we never per-timepoint-normalize away the kinetic meaning.

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

from gnn.kinetics import baseline_relative_curve
from graph_extraction.skeleton_to_graph_primitives import extract_segments

Point3D = tuple[int, int, int]

_NDIM_4D = 4  # (t, z, y, x)
_NDIM_2D = 2
_POINT_DIM = 3  # (x, y, z)


def voxel_node_series(
    dce_4d: np.ndarray,
    nodes: list[Point3D],
    *,
    baseline_frame_count: int = 1,
    relative_enhancement: bool = False,
    baseline_override: np.ndarray | None = None,
) -> np.ndarray:
    """Stack each voxel node's baseline-referenced DCE curve into ``(num_nodes, T)``.

    ``dce_4d`` is the aligned raw series in ``(t, z, y, x)`` order (as loaded by
    ``gnn.raw_dce.load_raw_dce_series``); ``nodes`` are voxel coordinates in
    ``(x, y, z)`` order (the node keys used throughout ``gnn.data_loader`` and
    ``gnn.segment_graph``). Row ``i`` is node ``i``'s curve, sampled as
    ``dce_4d[:, z, y, x]`` -- identical indexing to
    ``data_loader._attach_node_features`` so voxel forecasting and voxel
    classification see the same signal.

    Each row is then referenced through ``gnn.kinetics.baseline_relative_curve``
    with the case's ``baseline_frame_count`` / ``relative_enhancement`` -- the
    same transform the kinetic path applies -- so the forecasting target is on
    the cohort's committed enhancement scale. Per-node normalisation, if ever
    used, is a loss-time concern (see ``gnn.pretrain.loss``), not applied here.
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
    enhancement = baseline_relative_curve(
        series,
        baseline_frame_count=baseline_frame_count,
        relative_enhancement=relative_enhancement,
        baseline_override=baseline_override,
    )
    return enhancement.astype(np.float32)


def segment_node_series(
    voxel_graph: nx.Graph,
    dce_4d: np.ndarray,
    *,
    baseline_frame_count: int = 1,
    relative_enhancement: bool = False,
) -> np.ndarray:
    """Per-segment contrast series: mean voxel enhancement per frame, ``(num_segments, T)``.

    A segment node has no single curve, so its per-frame value (design doc §4.2)
    is an **aggregate over the segment's voxels** -- here the mean per-voxel
    baseline-referenced enhancement across the segment's voxels at each frame.

    Segments come from ``graph_extraction.extract_segments(voxel_graph)`` in the
    SAME order ``gnn.segment_graph.build_segment_line_graph`` uses to number its
    integer segment-node ids (``enumerate(extract_segments(...))``), so row ``i``
    aligns with segment-node ``i`` -- and therefore with ``data.x`` row ``i``
    after ``from_networkx`` (locked by a test). ``extract_segments`` is
    deterministic, so the two calls agree.

    The transform is applied **per voxel then averaged**, matching the kinetic
    segment path (``gnn.segment_graph`` summarizes per-voxel kinetics, then means
    along the segment): each voxel is referenced against *its own* baseline. This
    matters under ``relative_enhancement`` -- ``(S - S0) / S0`` is nonlinear in
    the per-voxel ``S0``, so per-voxel-relative-then-mean differs from
    mean-then-relative; we take the former to stay consistent with the kinetic
    path (for the absolute convention the two coincide, since subtraction is
    linear).

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
        enhancement = baseline_relative_curve(
            voxel_curves,
            baseline_frame_count=baseline_frame_count,
            relative_enhancement=relative_enhancement,
        )
        series[i] = enhancement.mean(axis=0)
    return series


def junction_node_series(
    dce_4d: np.ndarray,
    pos: np.ndarray,
    *,
    baseline_frame_count: int = 1,
    relative_enhancement: bool = False,
) -> np.ndarray:
    """Per-junction contrast series: the raw curve at each junction voxel.

    **First-pass §4.3 target choice:** forecast the *raw* contrast at the
    junction voxel -- the direct voxel-analogue -- NOT a flow/derivative
    quantity. This deliberately ignores the segment dynamics that live on the
    junction graph's *edges* (``edge_attr``); the flow/derivative reformulation
    the design doc flags as possibly better-motivated (§4.3) is deferred, and is
    a one-line target swap here. Committed to the raw curve for the first pass so
    all three graph structures have a working pretext target (roadmap step 1).

    ``pos`` is the ``(num_junctions, 3)`` array of junction voxel ``(x, y, z)``
    coordinates that ``gnn.junction_graph.build_junction_graph`` puts on
    ``data.pos`` in junction-node order (row ``i`` = ``ordered_nodes[i]``), so
    the returned rows align with ``data.x``. Sampling reuses ``voxel_node_series``
    -- a junction *is* a voxel -- so the signal is identical to voxel mode's.
    """
    coords = np.asarray(pos)
    if coords.ndim != _NDIM_2D or coords.shape[1] != _POINT_DIM:
        raise ValueError(f"pos must be (num_junctions, 3); got shape {coords.shape}")
    nodes = [(int(x), int(y), int(z)) for x, y, z in coords]
    return voxel_node_series(
        dce_4d,
        nodes,
        baseline_frame_count=baseline_frame_count,
        relative_enhancement=relative_enhancement,
    )
