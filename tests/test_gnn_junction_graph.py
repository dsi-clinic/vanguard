"""Unit tests for the segment-as-edge junction builder (``node_mode="junction"``).

Exercises ``gnn.junction_graph.build_junction_graph`` on tiny hand-built
skeleton graphs (straight vessel, Y-junction, X-junction, pure cycle) so node
count, edge count/direction, and the split of features across nodes vs edges
are checkable by hand -- the topological dual of the segment tests. See
``gnn/DESIGN_segment_graph.md`` §2.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from gnn.junction_graph import (
    JUNCTION_EDGE_FEATURE_ATTR,
    JUNCTION_NODE_FEATURE_ATTR,
    build_junction_graph,
)

NUM_TIMEPOINTS = 3
TIME_AXIS = np.arange(NUM_TIMEPOINTS, dtype=float)
VOLUME_ZYX = (1, 16, 16)

STRAIGHT_VESSEL_NODES = 2  # two endpoints
Y_NODES, Y_SEGMENTS = 4, 3  # centre + 3 tips; 3 segments
X_NODES, X_SEGMENTS = 5, 4  # centre + 4 tips; 4 segments
CENTRE_DEGREE_Y = 3.0
CENTRE_DEGREE_X = 4.0
TIP_DEGREE = 1.0


def _make_graph(edges: list[tuple[tuple, tuple]]) -> nx.Graph:
    graph = nx.Graph()
    for a, b in edges:
        graph.add_edge(tuple(a), tuple(b))
    return graph


def _line(coords: list[tuple]) -> list[tuple[tuple, tuple]]:
    return [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]


def _uniform_inputs(graph: nx.Graph) -> tuple[dict, np.ndarray]:
    radius_map = dict.fromkeys(graph.nodes(), 1.0)
    dce_4d = np.empty((NUM_TIMEPOINTS, *VOLUME_ZYX), dtype=np.float32)
    for t in range(NUM_TIMEPOINTS):
        dce_4d[t] = float(t)
    return radius_map, dce_4d


def test_straight_vessel_two_nodes_one_segment_edge() -> None:
    """A single vessel -> 2 endpoint nodes joined by one segment (both directions)."""
    coords = [(x, 5, 0) for x in range(1, 6)]
    graph = _make_graph(_line(coords))
    radius_map, dce_4d = _uniform_inputs(graph)

    data = build_junction_graph(graph, radius_map, dce_4d, TIME_AXIS)

    assert data.num_nodes == STRAIGHT_VESSEL_NODES
    # One undirected segment -> two directed edges.
    assert data.edge_index.shape == (2, 2)
    edges = {(int(u), int(v)) for u, v in data.edge_index.t().tolist()}
    assert edges == {(0, 1), (1, 0)}
    # Both endpoints are degree 1.
    assert torch.allclose(data.degree, torch.tensor([TIP_DEGREE, TIP_DEGREE]))
    # The segment summary rides on the edges: the 5-voxel line is 4 units long.
    assert torch.allclose(data.seg_length, torch.full((2,), 4.0))
    assert torch.allclose(data.seg_num_voxels, torch.full((2,), 5.0))


def test_y_junction_star_topology() -> None:
    """Y-junction -> centre node + 3 tips, 3 segment edges (each both directions)."""
    centre = (5, 5, 0)
    arms = (
        _line([centre, (6, 5, 0), (7, 5, 0)])
        + _line([centre, (4, 5, 0), (3, 5, 0)])
        + _line([centre, (5, 6, 0), (5, 7, 0)])
    )
    graph = _make_graph(arms)
    radius_map, dce_4d = _uniform_inputs(graph)

    data = build_junction_graph(graph, radius_map, dce_4d, TIME_AXIS)

    assert data.num_nodes == Y_NODES
    assert data.edge_index.shape == (2, 2 * Y_SEGMENTS)
    # Exactly one node has degree 3 (the junction); the rest are tips (degree 1).
    degrees = sorted(data.degree.tolist())
    assert degrees == [TIP_DEGREE, TIP_DEGREE, TIP_DEGREE, CENTRE_DEGREE_Y]
    # It's a star: the junction is in every edge, so the junction graph is
    # connected with 3 undirected edges.
    assert int(data.num_connected_components) == 1


def test_x_junction_four_segments() -> None:
    """X-junction -> centre + 4 tips, 4 segment edges."""
    centre = (8, 8, 0)
    arms = (
        _line([centre, (9, 8, 0), (10, 8, 0)])
        + _line([centre, (7, 8, 0), (6, 8, 0)])
        + _line([centre, (8, 9, 0), (8, 10, 0)])
        + _line([centre, (8, 7, 0), (8, 6, 0)])
    )
    graph = _make_graph(arms)
    radius_map, dce_4d = _uniform_inputs(graph)

    data = build_junction_graph(graph, radius_map, dce_4d, TIME_AXIS)

    assert data.num_nodes == X_NODES
    assert data.edge_index.shape == (2, 2 * X_SEGMENTS)
    assert max(data.degree.tolist()) == CENTRE_DEGREE_X


def test_node_and_edge_feature_attrs_present() -> None:
    """Per-voxel features land on nodes; the segment summary lands on edges."""
    coords = [(x, 5, 0) for x in range(1, 6)]
    graph = _make_graph(_line(coords))
    radius_map, dce_4d = _uniform_inputs(graph)

    data = build_junction_graph(graph, radius_map, dce_4d, TIME_AXIS)

    for name in JUNCTION_NODE_FEATURE_ATTR:
        assert data[name].shape == (data.num_nodes,)
    for name in JUNCTION_EDGE_FEATURE_ATTR:
        assert data[name].shape == (data.edge_index.shape[1],)
    # Node and edge vocabularies are disjoint (bare per-voxel names vs seg_*).
    assert not set(JUNCTION_NODE_FEATURE_ATTR) & set(JUNCTION_EDGE_FEATURE_ATTR)


def test_pure_cycle_component_raises() -> None:
    """A loop with no junction/endpoint fails loudly (DESIGN §2.1)."""
    square = _line([(1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0), (1, 1, 0)])
    graph = _make_graph(square)
    radius_map, dce_4d = _uniform_inputs(graph)

    with pytest.raises(NotImplementedError, match="Pure-cycle"):
        build_junction_graph(graph, radius_map, dce_4d, TIME_AXIS)
