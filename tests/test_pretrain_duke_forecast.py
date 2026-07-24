"""Tests for the Duke forecasting-graph builder + its alignment invariant.

No real data touched: the ``from_networkx`` node-order invariant is checked on a
tiny synthetic ``networkx`` graph, task discovery on a fake (empty-file) tree, and
the ``Data -> ForecastGraph`` conversion on a synthetic ``Data``. The actual Duke
build (``_build_case`` on real skeleton/DCE files) runs only inside a Slurm job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import networkx as nx  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.utils import from_networkx  # noqa: E402

from gnn.pretrain.duke_forecast import (  # noqa: E402
    data_to_forecast_graphs,
    discover_forecast_tasks,
    resolve_smoke_horizon,
)
from gnn.pretrain.forecast import ForecastHorizon  # noqa: E402


def test_from_networkx_preserves_node_order() -> None:
    """The invariant node_series relies on: data.x rows follow list(G.nodes()).

    ``gnn.data_loader._build_case`` builds ``data.node_series`` over
    ``list(voxel_graph.nodes())`` and trusts that this matches the ``data.x`` row
    order ``from_networkx`` produces. Lock that here on a graph whose node
    insertion order is deliberately non-sorted.
    """
    graph = nx.Graph()
    for node in [(9, 0, 0), (3, 0, 0), (7, 0, 0)]:  # non-sorted insertion order
        graph.add_node(node, feat=float(node[0]))
    graph.add_edge((9, 0, 0), (3, 0, 0))
    graph.add_edge((3, 0, 0), (7, 0, 0))

    data = from_networkx(graph)
    expected = torch.tensor([float(n[0]) for n in graph.nodes()])
    torch.testing.assert_close(data.feat, expected)


def test_discover_forecast_tasks_globs_and_joins_labels(tmp_path: Path) -> None:
    """Discovery whitelists case ids, finds skeleton masks, and joins labels."""
    from gnn.data_loader import _CENTERLINE_SUFFIX

    studies = tmp_path / "studies"
    for site_case in ["DUKE_001", "DUKE_002", "ISPY2_100"]:
        d = studies / site_case.split("_")[0] / site_case
        d.mkdir(parents=True)
        (
            d / f"{site_case}{_CENTERLINE_SUFFIX}"
        ).touch()  # empty file; only name matters

    labels = tmp_path / "labels.csv"
    labels.write_text("case_id,pcr,fold\nDUKE_001,0,1\nDUKE_002,1,2\nISPY2_100,0,0\n")

    tasks = discover_forecast_tasks(studies, labels, cases=["DUKE_001", "DUKE_002"])
    got = {case_id: label for case_id, _, label in tasks}
    assert got == {"DUKE_001": 0, "DUKE_002": 1}
    for _, mask_path, _ in tasks:
        assert mask_path.name.endswith(_CENTERLINE_SUFFIX)


def test_discover_forecast_tasks_raises_on_missing_label(tmp_path: Path) -> None:
    """A whitelisted case with no label row fails loudly (no silent drop)."""
    from gnn.data_loader import _CENTERLINE_SUFFIX

    studies = tmp_path / "studies"
    d = studies / "DUKE" / "DUKE_001"
    d.mkdir(parents=True)
    (d / f"DUKE_001{_CENTERLINE_SUFFIX}").touch()
    labels = tmp_path / "labels.csv"
    labels.write_text("case_id,pcr,fold\nDUKE_999,0,1\n")

    with pytest.raises(ValueError, match="no label"):
        discover_forecast_tasks(studies, labels, cases=["DUKE_001"])


def test_discover_forecast_tasks_raises_on_missing_case(tmp_path: Path) -> None:
    """A whitelisted case whose skeleton mask is absent fails loudly."""
    from gnn.data_loader import _CENTERLINE_SUFFIX

    studies = tmp_path / "studies"
    (studies / "DUKE" / "DUKE_001").mkdir(parents=True)
    (studies / "DUKE" / "DUKE_001" / f"DUKE_001{_CENTERLINE_SUFFIX}").touch()
    labels = tmp_path / "labels.csv"
    labels.write_text("case_id,pcr\nDUKE_001,0\nDUKE_002,1\n")

    with pytest.raises(ValueError, match="no skeleton mask"):
        discover_forecast_tasks(studies, labels, cases=["DUKE_001", "DUKE_002"])


def _forecast_data(node_times: list[float], baseline_frame_count: int) -> Data:
    num_nodes, t = 3, len(node_times)
    data = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        node_series=torch.arange(num_nodes * t, dtype=torch.float).reshape(
            num_nodes, t
        ),
        node_times=torch.tensor(node_times),
    )
    data.baseline_frame_count = baseline_frame_count
    return data


def test_data_to_forecast_graphs_tiles_with_times() -> None:
    """A built Data becomes several non-overlapping timed ForecastGraphs (tiles)."""
    # T=8, baseline_frame_count=1, keep 1 -> start 0; window 4 -> tiles at 0 and 4.
    data = _forecast_data([0.0, 2.0, 4.0, 6.0, 20.0, 22.0, 24.0, 26.0], 1)
    graphs = data_to_forecast_graphs(data, ForecastHorizon(input_len=2, target_len=2))
    expected_tile_count = 2
    assert len(graphs) == expected_tile_count
    for fg in graphs:
        assert fg.x_seq.shape == (3, 2, 1)
        assert fg.target.shape == (3, 2)
        torch.testing.assert_close(fg.edge_index, data.edge_index)
        torch.testing.assert_close(fg.input_times[0], torch.tensor(0.0))  # rebased
    # Tile 1 (offset 4, t=20): rebased input times [0, 2].
    torch.testing.assert_close(graphs[1].input_times, torch.tensor([0.0, 2.0]))


def test_data_to_forecast_graphs_drops_baseline_windows() -> None:
    """Leading baseline frames are not tiled (except the kept context frame)."""
    data = _forecast_data([float(i) for i in range(8)], baseline_frame_count=3)
    graphs = data_to_forecast_graphs(
        data, ForecastHorizon(input_len=2, target_len=2), keep_baseline_context=1
    )
    # start = 3 - 1 = 2 -> single window; first input frame is series column 2.
    assert len(graphs) == 1
    torch.testing.assert_close(graphs[0].x_seq[:, :, 0], data.node_series[:, 2:4])


def test_data_to_forecast_graphs_requires_node_series() -> None:
    """Missing node_series fails loudly (the build must have attached it)."""
    data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]))
    with pytest.raises(ValueError, match="node_series"):
        data_to_forecast_graphs(data, ForecastHorizon(input_len=1, target_len=1))


def test_data_to_forecast_graphs_requires_baseline_frame_count() -> None:
    """Missing baseline_frame_count fails loudly (needed to drop baseline windows)."""
    data = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        node_series=torch.zeros(2, 8),
        node_times=torch.arange(8, dtype=torch.float),
    )
    with pytest.raises(ValueError, match="baseline_frame_count"):
        data_to_forecast_graphs(data, ForecastHorizon(input_len=2, target_len=2))


def test_resolve_smoke_horizon_unchanged_when_fits() -> None:
    """If the window fits the min T, the horizon is returned unchanged (any policy)."""
    horizon = ForecastHorizon(input_len=3, target_len=2)
    assert resolve_smoke_horizon([5, 6, 5], horizon, "strict") is horizon
    assert resolve_smoke_horizon([5, 6, 5], horizon, "truncate") is horizon


def test_resolve_smoke_horizon_strict_raises_on_short_case() -> None:
    """Strict policy raises when a case is shorter than the window (variable T)."""
    horizon = ForecastHorizon(input_len=3, target_len=2)  # window 5
    with pytest.raises(ValueError, match="variable-length"):
        resolve_smoke_horizon([5, 4], horizon, "strict")


def test_resolve_smoke_horizon_truncate_shrinks_input_len() -> None:
    """Truncate shrinks input_len so the window fits min T, keeping target_len."""
    horizon = ForecastHorizon(input_len=3, target_len=2)  # window 5
    resolved = resolve_smoke_horizon([5, 4], horizon, "truncate")  # min T = 4
    # input_len shrunk to 4 - target_len(2) = 2; target_len kept; window now 4.
    assert (resolved.input_len, resolved.target_len, resolved.window) == (2, 2, 4)


def test_resolve_smoke_horizon_truncate_raises_when_target_too_large() -> None:
    """Truncate still raises if target_len alone exceeds the min T."""
    horizon = ForecastHorizon(input_len=1, target_len=4)  # window 5
    with pytest.raises(ValueError, match="cannot fit an input horizon"):
        resolve_smoke_horizon([3], horizon, "truncate")
