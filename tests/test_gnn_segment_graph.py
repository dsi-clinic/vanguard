"""Unit tests for the segment-as-node line-graph builder (``node_mode="segment"``).

These exercise ``gnn.segment_graph.build_segment_line_graph`` on tiny hand-built
skeleton graphs (a straight vessel, a Y-junction, an X-junction, a pure cycle)
so the line-graph node count, adjacency, and summarized features are checkable
by hand -- no cluster data required. See ``gnn/DESIGN_segment_graph.md`` §8.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from gnn.segment_graph import (
    SEGMENT_FEATURE_ATTR,
    build_segment_line_graph,
)

NUM_TIMEPOINTS = 3
TIME_AXIS = np.arange(NUM_TIMEPOINTS, dtype=float)
# curve = t at every voxel -> enhancement = [0, 1, 2]: peak at idx 2 (norm 1.0),
# arrival (20% of peak=2 -> 0.4) first crossed at idx 1, peak enhancement 2.0.
VOLUME_ZYX = (1, 16, 16)

# Each arm is center + 2 voxels, so a segment spans 3 voxels and 2 unit steps.
VOXELS_PER_SEGMENT = 3.0
LENGTH_PER_SEGMENT = 2.0
STRAIGHT_VESSEL_VOXELS = 5.0
STRAIGHT_VESSEL_LENGTH = 4.0

# Line-graph topology: N arms meeting at one junction -> N mutually adjacent
# segment-nodes (the complete graph K_N), each of degree N-1.
Y_ARMS, Y_EDGES, Y_DEGREE = 3, 3, 2  # K3 (triangle)
X_ARMS, X_EDGES, X_DEGREE = 4, 6, 3  # K4

# Baseline-floor cap check: a 0.01 baseline vs. a floor of 0.05*peak(2.0)=0.1
# shrinks the relative-enhancement denominator exactly 10x; the unfloored peak
# is far above physiological (~0-5), confirming the divide-by-near-zero pathology.
FLOOR_DENOM_RATIO = 10.0
PATHOLOGICAL_PEAK_MIN = 100.0


def _make_graph(edges: list[tuple[tuple, tuple]]) -> nx.Graph:
    """Build a voxel skeleton graph from (x,y,z) endpoint pairs."""
    graph = nx.Graph()
    for a, b in edges:
        graph.add_edge(tuple(a), tuple(b))
    return graph


def _line(coords: list[tuple]) -> list[tuple[tuple, tuple]]:
    """Chain a list of voxels into consecutive edges."""
    return [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]


def _uniform_inputs(graph: nx.Graph) -> tuple[dict, np.ndarray]:
    """A unit radius map plus a rising DCE volume (enhancement = t everywhere)."""
    radius_map = dict.fromkeys(graph.nodes(), 1.0)
    dce_4d = np.empty((NUM_TIMEPOINTS, *VOLUME_ZYX), dtype=np.float32)
    for t in range(NUM_TIMEPOINTS):
        dce_4d[t] = float(t)
    return radius_map, dce_4d


def test_straight_vessel_is_one_node_no_edges() -> None:
    """A single unbranched vessel collapses to one segment-node, no edges."""
    coords = [(x, 5, 0) for x in range(1, 6)]  # 5 voxels, both ends degree 1
    graph = _make_graph(_line(coords))
    radius_map, dce_4d = _uniform_inputs(graph)

    line = build_segment_line_graph(graph, radius_map, dce_4d, TIME_AXIS)

    assert line.number_of_nodes() == 1
    assert line.number_of_edges() == 0
    attrs = line.nodes[0]
    assert attrs["seg_num_voxels"] == STRAIGHT_VESSEL_VOXELS
    assert attrs["seg_length"] == pytest.approx(STRAIGHT_VESSEL_LENGTH)
    assert attrs["seg_tortuosity"] == pytest.approx(1.0)  # perfectly straight
    assert attrs["seg_radius_mean"] == pytest.approx(1.0)
    # Every voxel shares the identical rising curve -> deterministic kinetics.
    assert attrs["seg_peak_time_mean"] == pytest.approx(1.0)
    assert attrs["seg_peak_time_std"] == pytest.approx(0.0)
    assert attrs["seg_peak_enhancement_mean"] == pytest.approx(2.0)
    # Quantiles of a uniform radius map / straight-line (zero-curvature) path.
    assert attrs["seg_radius_q1"] == pytest.approx(1.0)
    assert attrs["seg_radius_q3"] == pytest.approx(1.0)
    assert attrs["seg_curvature_median"] == pytest.approx(0.0)
    assert attrs["seg_curvature_min"] == pytest.approx(0.0)
    # Both endpoints are true endpoints (degree 1) -> mean == max == 1.0.
    assert attrs["seg_degree_mean"] == pytest.approx(1.0)
    assert attrs["seg_degree_max"] == pytest.approx(1.0)
    # Enhancement peaks at the last timepoint -> no washout window (den == 0).
    assert attrs["seg_washout_slope_mean"] == pytest.approx(0.0)


def test_baseline_floor_caps_segment_relative_enhancement() -> None:
    """Baseline floor threads into segment kinetics: caps a void baseline, no-op otherwise.

    With relative enhancement and a near-void baseline the floor caps the
    otherwise-exploding per-voxel peak enhancement; with a healthy baseline
    (already above the floor) it changes nothing.
    """
    coords = [(x, 5, 0) for x in range(1, 6)]
    graph = _make_graph(_line(coords))
    radius_map = dict.fromkeys(graph.nodes(), 1.0)

    # Near-void baseline (0.01) under a peak signal of 2.0: (S - S0)/S0 explodes.
    void_baseline = np.empty((NUM_TIMEPOINTS, *VOLUME_ZYX), dtype=np.float32)
    void_baseline[0], void_baseline[1], void_baseline[2] = 0.01, 1.0, 2.0

    def seg_peak(dce_4d: np.ndarray, floor: float) -> float:
        line = build_segment_line_graph(
            graph,
            radius_map,
            dce_4d,
            TIME_AXIS,
            relative_enhancement=True,
            baseline_floor_frac=floor,
        )
        return line.nodes[0]["seg_peak_enhancement_mean"]

    unfloored = seg_peak(void_baseline, 0.0)
    floored = seg_peak(void_baseline, 0.05)
    # Denominator goes from S0=0.01 to max(0.01, 0.05*2)=0.1 -> exactly 10x
    # smaller, independent of the exact peak definition (linear in 1/denom).
    assert unfloored > PATHOLOGICAL_PEAK_MIN  # the divide-by-near-zero pathology
    assert unfloored == pytest.approx(floored * FLOOR_DENOM_RATIO, rel=1e-3)

    # Healthy baseline (1.0; floor 0.05*2=0.1 < baseline) -> floor is a no-op.
    healthy = np.stack(
        [np.full(VOLUME_ZYX, v, dtype=np.float32) for v in (1.0, 1.5, 2.0)]
    )
    assert seg_peak(healthy, 0.0) == pytest.approx(seg_peak(healthy, 0.05))


def test_y_junction_gives_triangle_line_graph() -> None:
    """Three arms meeting at one junction -> 3 segment-nodes, all pairwise adjacent."""
    center = (5, 5, 0)
    arm_x = _line([center, (6, 5, 0), (7, 5, 0)])
    arm_negx = _line([center, (4, 5, 0), (3, 5, 0)])
    arm_y = _line([center, (5, 6, 0), (5, 7, 0)])
    graph = _make_graph(arm_x + arm_negx + arm_y)
    radius_map, dce_4d = _uniform_inputs(graph)

    line = build_segment_line_graph(graph, radius_map, dce_4d, TIME_AXIS)

    # 3 segments, all sharing the degree-3 center -> a triangle (3 edges).
    assert line.number_of_nodes() == Y_ARMS
    assert line.number_of_edges() == Y_EDGES
    assert all(line.degree(n) == Y_DEGREE for n in line.nodes())
    for attrs in (line.nodes[n] for n in line.nodes()):
        assert attrs["seg_num_voxels"] == VOXELS_PER_SEGMENT
        assert attrs["seg_length"] == pytest.approx(LENGTH_PER_SEGMENT)
        # Each segment spans the degree-3 center and a degree-1 arm tip.
        assert attrs["seg_degree_mean"] == pytest.approx(2.0)
        assert attrs["seg_degree_max"] == pytest.approx(3.0)


def test_x_junction_gives_complete_graph_k4() -> None:
    """Four arms at a degree-4 junction -> 4 segment-nodes forming K4 (6 edges)."""
    center = (8, 8, 0)
    arms = (
        _line([center, (9, 8, 0), (10, 8, 0)])
        + _line([center, (7, 8, 0), (6, 8, 0)])
        + _line([center, (8, 9, 0), (8, 10, 0)])
        + _line([center, (8, 7, 0), (8, 6, 0)])
    )
    graph = _make_graph(arms)
    radius_map, dce_4d = _uniform_inputs(graph)

    line = build_segment_line_graph(graph, radius_map, dce_4d, TIME_AXIS)

    assert line.number_of_nodes() == X_ARMS
    assert line.number_of_edges() == X_EDGES  # K4
    assert all(line.degree(n) == X_DEGREE for n in line.nodes())
    for attrs in (line.nodes[n] for n in line.nodes()):
        # Each segment spans the degree-4 center and a degree-1 arm tip.
        assert attrs["seg_degree_mean"] == pytest.approx(2.5)
        assert attrs["seg_degree_max"] == pytest.approx(4.0)


def test_pure_cycle_component_raises() -> None:
    """A loop with no junction/endpoint fails loudly (DESIGN §2.1), not silently."""
    square = _line([(1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0), (1, 1, 0)])
    graph = _make_graph(square)  # every voxel degree 2
    radius_map, dce_4d = _uniform_inputs(graph)

    with pytest.raises(NotImplementedError, match="Pure-cycle"):
        build_segment_line_graph(graph, radius_map, dce_4d, TIME_AXIS)


def test_segment_feature_attr_is_identity_and_covers_node_attrs() -> None:
    """Every declared segment feature is attached to each built node."""
    coords = [(x, 5, 0) for x in range(1, 6)]
    graph = _make_graph(_line(coords))
    radius_map, dce_4d = _uniform_inputs(graph)

    line = build_segment_line_graph(graph, radius_map, dce_4d, TIME_AXIS)
    attrs = line.nodes[0]
    for name, attr_key in SEGMENT_FEATURE_ATTR.items():
        assert name == attr_key  # identity map in segment mode
        assert attr_key in attrs

    # New Tier 0/1 features are registered in the vocabulary, not just present
    # by coincidence of the loop above.
    for name in (
        "seg_radius_q1",
        "seg_radius_q3",
        "seg_curvature_median",
        "seg_curvature_min",
        "seg_curvature_q1",
        "seg_curvature_q3",
        "seg_degree_mean",
        "seg_degree_max",
        "seg_washout_slope_mean",
        "seg_washout_slope_std",
    ):
        assert name in SEGMENT_FEATURE_ATTR
