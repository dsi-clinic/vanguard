"""Per-node contrast time-series for the forecasting pretext task.

The classification pipeline collapses each voxel's DCE enhancement curve to
summary kinetics (``gnn.kinetics.node_kinetic_features``) before it reaches the
model. Forecasting instead needs the *full* per-node curve on the graph, so this
module re-samples the raw 4D DCE series at each node exactly the way
``gnn.data_loader._attach_node_features`` does (``curve = dce_4d[:, z, y, x]``)
and applies the same frame-0 baseline convention (``gnn.kinetics``,
``AGENTS.md``: never per-timepoint-normalize away the kinetic meaning).

This is the single source of truth for "node -> forecasting signal", mirroring
how ``gnn.kinetics`` is the single source of truth for "curve -> kinetics", so
the forecasting target can never silently diverge from the measured signal.
"""

from __future__ import annotations

import numpy as np

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
