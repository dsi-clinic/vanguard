"""Unit tests for graph-derived scalar features (``gnn.graph_derived``).

Exercises ``build_graph_derived_feature_matrix`` on synthetic ``Data`` objects
carrying ``num_connected_components`` -- no cluster data required.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from gnn.graph_derived import (  # noqa: E402
    GRAPH_DERIVED_COLUMNS,
    build_graph_derived_feature_matrix,
)


def _make_data(case_id: str, num_connected_components: int) -> Data:
    data = Data(x=torch.randn(3, 2))
    data.case_id = case_id
    data.num_connected_components = num_connected_components
    return data


def test_num_connected_components_is_the_only_default_column() -> None:
    """The vocabulary is deliberately small (confound-flagged, opt-in only)."""
    assert GRAPH_DERIVED_COLUMNS == ("num_connected_components",)


def test_build_matrix_reads_values_in_data_list_order() -> None:
    """Row order matches data_list order, not any sorting of case_id."""
    data_list = [_make_data("A", 1), _make_data("B", 3), _make_data("C", 2)]
    matrix = build_graph_derived_feature_matrix(data_list, ["num_connected_components"])
    assert matrix.shape == (3, 1)
    assert matrix[:, 0].tolist() == [1.0, 3.0, 2.0]


def test_unknown_column_raises() -> None:
    """A requested column outside the supported vocabulary raises."""
    data_list = [_make_data("A", 1)]
    with pytest.raises(ValueError, match="Unknown graph-derived columns"):
        build_graph_derived_feature_matrix(data_list, ["num_nodes"])
