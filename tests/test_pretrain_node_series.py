"""Tests for the forecasting node-series builders and their shared transform.

Phase 2a of the contrast-forecasting review: the pretraining path must apply the
*same* baseline/enhancement contract as the kinetic-feature path
(``gnn.kinetics.baseline_relative_curve``) rather than its old hard-coded
``curve - curve[0]``. These lock: (1) the shared transform's absolute/relative
behavior and per-row (per-voxel) baselining, (2) that the legacy convention
(``baseline_frame_count=1``, absolute) still reduces to ``curve - curve[0]``, and
(3) that the segment builder references per voxel *then* averages (nonlinear
under relative enhancement), matching the kinetic segment path.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from gnn.kinetics import baseline_relative_curve
from gnn.pretrain.node_series import (
    junction_node_series,
    segment_node_series,
    voxel_node_series,
)

# --- shared transform -------------------------------------------------------


def test_absolute_frame0_reduces_to_curve_minus_curve0() -> None:
    """Absolute convention with baseline_frame_count=1 == curve - curve[0]."""
    curve = np.array([3.0, 5.0, 8.0, 4.0])
    out = baseline_relative_curve(
        curve, baseline_frame_count=1, relative_enhancement=False
    )
    np.testing.assert_allclose(out, curve - curve[0])


def test_relative_uses_mean_of_baseline_frames() -> None:
    """Relative enhancement is (S - S0)/S0 with S0 = mean of the baseline frames."""
    curve = np.array([10.0, 30.0, 60.0])  # S0 = mean(10, 30) = 20
    out = baseline_relative_curve(
        curve, baseline_frame_count=2, relative_enhancement=True
    )
    np.testing.assert_allclose(out, (curve - 20.0) / 20.0)


def test_each_row_uses_its_own_baseline() -> None:
    """A 2D stack references each row against its own baseline, independently."""
    stack = np.array([[10.0, 10.0, 20.0], [100.0, 100.0, 110.0]])
    out = baseline_relative_curve(
        stack, baseline_frame_count=2, relative_enhancement=True
    )
    np.testing.assert_allclose(out, [[0.0, 0.0, 1.0], [0.0, 0.0, 0.1]])


def test_relative_zeroes_row_with_nonpositive_baseline() -> None:
    """A row whose baseline is <= float32 eps yields all zeros (matches kinetics)."""
    stack = np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 20.0]])
    out = baseline_relative_curve(
        stack, baseline_frame_count=2, relative_enhancement=True
    )
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1.0])


def test_baseline_frame_count_out_of_range_raises() -> None:
    """baseline_frame_count must be in [1, n_timepoints)."""
    curve = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="baseline_frame_count"):
        baseline_relative_curve(
            curve, baseline_frame_count=3, relative_enhancement=False
        )


# --- builders ---------------------------------------------------------------


def _dce_from_voxel_curves(
    curves_by_xyz: dict[tuple[int, int, int], list[float]], t: int
) -> np.ndarray:
    """Build a (t, z, y, x) DCE volume placing each curve at its (x, y, z)."""
    xs = [x for x, _, _ in curves_by_xyz]
    ys = [y for _, y, _ in curves_by_xyz]
    zs = [z for _, _, z in curves_by_xyz]
    dce = np.zeros((t, max(zs) + 1, max(ys) + 1, max(xs) + 1), dtype=np.float32)
    for (x, y, z), curve in curves_by_xyz.items():
        dce[:, z, y, x] = curve
    return dce


def test_voxel_series_matches_shared_transform_on_sampled_curves() -> None:
    """voxel_node_series == baseline_relative_curve applied to the sampled rows."""
    curves = {(0, 0, 0): [10.0, 12.0, 30.0], (1, 0, 0): [100.0, 101.0, 150.0]}
    dce = _dce_from_voxel_curves(curves, t=3)
    nodes = list(curves)
    out = voxel_node_series(
        dce, nodes, baseline_frame_count=2, relative_enhancement=True
    )
    expected = baseline_relative_curve(
        np.array([curves[n] for n in nodes], dtype=np.float32),
        baseline_frame_count=2,
        relative_enhancement=True,
    )
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_voxel_series_legacy_default_is_frame0_subtraction() -> None:
    """Legacy default (bfc=1, absolute) reproduces the old curve - curve[0]."""
    curves = {(0, 0, 0): [3.0, 5.0, 8.0], (1, 0, 0): [7.0, 7.0, 4.0]}
    dce = _dce_from_voxel_curves(curves, t=3)
    nodes = list(curves)
    out = voxel_node_series(dce, nodes)  # defaults: bfc=1, relative=False
    sampled = np.array([curves[n] for n in nodes], dtype=np.float32)
    np.testing.assert_allclose(out, sampled - sampled[:, :1])


def test_junction_series_equals_voxel_series_on_same_coords() -> None:
    """A junction is a voxel: same coords -> identical series."""
    curves = {(0, 0, 0): [10.0, 10.0, 40.0], (2, 1, 0): [50.0, 50.0, 60.0]}
    dce = _dce_from_voxel_curves(curves, t=3)
    nodes = list(curves)
    pos = np.array(nodes)
    via_junction = junction_node_series(
        dce, pos, baseline_frame_count=2, relative_enhancement=True
    )
    via_voxel = voxel_node_series(
        dce, nodes, baseline_frame_count=2, relative_enhancement=True
    )
    np.testing.assert_allclose(via_junction, via_voxel)


def test_segment_series_references_per_voxel_then_averages() -> None:
    """Segment value = mean of per-voxel relative enhancement, not relative-of-mean.

    Under relative enhancement these differ (division by a per-voxel S0 is
    nonlinear); the segment path must match the kinetic path's per-voxel-then-mean
    order.
    """
    # Two-voxel segment with very different baselines.
    curves = {(0, 0, 0): [10.0, 10.0, 20.0], (1, 0, 0): [100.0, 100.0, 110.0]}
    dce = _dce_from_voxel_curves(curves, t=3)
    graph = nx.Graph()
    graph.add_edge((0, 0, 0), (1, 0, 0))

    out = segment_node_series(
        graph, dce, baseline_frame_count=2, relative_enhancement=True
    )

    # Per-voxel relative: voxel A -> [0,0,1.0], voxel B -> [0,0,0.1]; mean -> 0.55.
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.55])
    # Relative-of-mean would give (65-55)/55 = 0.1818..., which we must NOT produce.
    assert not np.isclose(out[0, 2], (65.0 - 55.0) / 55.0)
