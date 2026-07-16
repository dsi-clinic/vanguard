"""Unit + smoke tests for the contrast-forecasting pretraining pilot.

All synthetic -- the UChicago pipeline is not ready and no real data is touched.
Covers: node-series extraction, the input/target window split, masked MAE, the
trivial baselines, model forward shapes / edge-sensitivity, and a small learning
smoke test (the GNN forecaster reduces training loss on a structured signal).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnn.pretrain.baselines import (  # noqa: E402
    last_frame_forecast,
    temporal_mean_forecast,
)
from gnn.pretrain.forecast import ForecastHorizon, split_forecast_window  # noqa: E402
from gnn.pretrain.loss import masked_mae  # noqa: E402
from gnn.pretrain.model import ContrastForecastGNN, PerNodeForecaster  # noqa: E402
from gnn.pretrain.node_series import (  # noqa: E402
    segment_node_series,
    voxel_node_series,
)


# --------------------------------------------------------------------------- #
# node_series
# --------------------------------------------------------------------------- #
def test_voxel_node_series_matches_dce_indexing_and_baseline() -> None:
    """Row i is dce_4d[:, z, y, x] for node (x,y,z), frame-0 baseline-subtracted."""
    rng = np.random.default_rng(0)
    dce_4d = rng.random((5, 4, 4, 4)).astype(np.float32)  # (t, z, y, x)
    nodes = [(1, 2, 3), (0, 0, 0)]  # (x, y, z)
    series = voxel_node_series(dce_4d, nodes)
    assert series.shape == (2, 5)
    expected0 = dce_4d[:, 3, 2, 1] - dce_4d[0, 3, 2, 1]
    np.testing.assert_allclose(series[0], expected0, rtol=1e-6)
    # Baseline subtraction: every node's first frame is exactly 0.
    np.testing.assert_allclose(series[:, 0], np.zeros(2), atol=1e-6)


def test_voxel_node_series_can_skip_baseline() -> None:
    """Without baseline subtraction the raw curve is returned verbatim."""
    dce_4d = np.arange(2 * 1 * 1 * 1, dtype=np.float32).reshape(2, 1, 1, 1)
    series = voxel_node_series(dce_4d, [(0, 0, 0)], baseline_subtract=False)
    np.testing.assert_allclose(series[0], dce_4d[:, 0, 0, 0])


def test_voxel_node_series_rejects_non_4d() -> None:
    """A non-4D dce array fails loudly."""
    with pytest.raises(ValueError, match="4D"):
        voxel_node_series(np.zeros((3, 3)), [(0, 0, 0)])


# --------------------------------------------------------------------------- #
# segment_node_series
# --------------------------------------------------------------------------- #
def _line_voxel_graph():  # noqa: ANN202
    """A straight 3-voxel chain -> one segment (degree-1 tips, degree-2 middle)."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge((0, 0, 0), (1, 0, 0))
    graph.add_edge((1, 0, 0), (2, 0, 0))
    return graph


def _y_voxel_graph():  # noqa: ANN202
    """A Y (three arms meeting at a degree-3 junction) -> three segments."""
    import networkx as nx

    graph = nx.Graph()
    center = (2, 2, 0)
    graph.add_edge((0, 2, 0), (1, 2, 0))
    graph.add_edge((1, 2, 0), center)
    graph.add_edge((2, 0, 0), (2, 1, 0))
    graph.add_edge((2, 1, 0), center)
    graph.add_edge(center, (2, 3, 0))
    graph.add_edge((2, 3, 0), (2, 4, 0))
    return graph


def test_segment_node_series_shape_and_mean() -> None:
    """One segment -> one row = mean baseline-subtracted enhancement over its voxels."""
    rng = np.random.default_rng(1)
    dce_4d = rng.random((4, 1, 1, 3)).astype(np.float32)  # (t, z, y, x); x in 0..2
    series = segment_node_series(_line_voxel_graph(), dce_4d)
    assert series.shape == (1, 4)
    voxels = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    curves = np.stack([dce_4d[:, z, y, x] for x, y, z in voxels], axis=0)
    expected = (curves - curves[:, :1]).mean(axis=0)  # mean is order-invariant
    np.testing.assert_allclose(series[0], expected, rtol=1e-5)
    np.testing.assert_allclose(series[:, 0], np.zeros(1), atol=1e-6)


def test_segment_node_series_one_row_per_segment_and_deterministic() -> None:
    """Row count matches extract_segments (the line-graph node count), order stable."""
    from graph_extraction.skeleton_to_graph_primitives import extract_segments

    graph = _y_voxel_graph()
    dce_4d = np.random.default_rng(2).random((3, 1, 5, 3)).astype(np.float32)
    segments = extract_segments(graph)
    series = segment_node_series(graph, dce_4d)
    assert series.shape[0] == len(segments)  # one row per segment node
    # Alignment relies on extract_segments returning the same order twice.
    assert extract_segments(graph) == segments
    np.testing.assert_allclose(series[:, 0], np.zeros(series.shape[0]), atol=1e-6)


def test_segment_node_series_rejects_non_4d() -> None:
    """A non-4D dce array fails loudly."""
    with pytest.raises(ValueError, match="4D"):
        segment_node_series(_line_voxel_graph(), np.zeros((3, 3)))


# --------------------------------------------------------------------------- #
# forecast window split
# --------------------------------------------------------------------------- #
def test_split_forecast_window_shapes_and_values() -> None:
    """Input is frames [0,input_len); target is the next target_len frames."""
    series = torch.arange(3 * 6, dtype=torch.float32).reshape(3, 6)
    horizon = ForecastHorizon(input_len=4, target_len=2)
    inputs, targets = split_forecast_window(series, horizon)
    assert inputs.shape == (3, 4)
    assert targets.shape == (3, 2)
    torch.testing.assert_close(inputs, series[:, :4])
    torch.testing.assert_close(targets, series[:, 4:6])


def test_split_forecast_window_rejects_short_series() -> None:
    """Fails loudly rather than padding when T < input_len + target_len."""
    series = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="frames but the horizon needs"):
        split_forecast_window(series, ForecastHorizon(input_len=3, target_len=2))


def test_forecast_horizon_validates() -> None:
    """Both horizon lengths must be >= 1."""
    with pytest.raises(ValueError, match="input_len"):
        ForecastHorizon(input_len=0, target_len=1)
    with pytest.raises(ValueError, match="target_len"):
        ForecastHorizon(input_len=1, target_len=0)


# --------------------------------------------------------------------------- #
# masked MAE
# --------------------------------------------------------------------------- #
def test_masked_mae_unmasked_is_plain_mae() -> None:
    """With no mask, masked_mae equals the plain mean absolute error."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
    # abs errors: 0,2,3,0 -> mean 1.25
    assert masked_mae(pred, target).item() == pytest.approx(1.25)


def test_masked_mae_drops_masked_elements() -> None:
    """Masked-out elements contribute to neither numerator nor denominator."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # keep only the zero-error cells
    assert masked_mae(pred, target, mask).item() == pytest.approx(0.0)


def test_masked_mae_rejects_all_masked() -> None:
    """An all-zero mask raises rather than dividing by zero."""
    pred = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="zero elements"):
        masked_mae(pred, pred, torch.zeros(2, 2))


def test_masked_mae_rejects_shape_mismatch() -> None:
    """Mismatched pred/target shapes raise."""
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_mae(torch.zeros(2, 2), torch.zeros(2, 3))


# --------------------------------------------------------------------------- #
# trivial baselines
# --------------------------------------------------------------------------- #
def test_last_frame_forecast_repeats_last_frame() -> None:
    """Persistence baseline repeats the last observed frame across the horizon."""
    inputs = torch.tensor([[1.0, 2.0, 5.0], [0.0, 0.0, -1.0]])
    out = last_frame_forecast(inputs, target_len=3)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out[:, 0], torch.tensor([5.0, -1.0]))
    torch.testing.assert_close(out[:, 0], out[:, 2])  # constant across horizon


def test_temporal_mean_forecast_repeats_mean() -> None:
    """Temporal-mean baseline repeats each node's input-horizon mean."""
    inputs = torch.tensor([[0.0, 2.0, 4.0]])  # mean 2
    out = temporal_mean_forecast(inputs, target_len=2)
    torch.testing.assert_close(out, torch.full((1, 2), 2.0))


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _triangle_edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])


def test_gnn_forecast_forward_shape() -> None:
    """(N, input_len, C) -> (N, target_len)."""
    model = ContrastForecastGNN(in_channels=1, hidden_dim=8, target_len=3, dropout=0.0)
    x_seq = torch.randn(3, 4, 1)
    out = model(x_seq, _triangle_edge_index())
    assert out.shape == (3, 3)
    assert torch.isfinite(out).all()


def test_gnn_forecast_uses_edges() -> None:
    """Changing the graph changes the forecast -- message passing is live."""
    model = ContrastForecastGNN(in_channels=1, hidden_dim=8, target_len=2, dropout=0.0)
    model.eval()
    x_seq = torch.randn(3, 4, 1)
    full = _triangle_edge_index()
    empty = torch.empty(2, 0, dtype=torch.long)
    out_full = model(x_seq, full)
    out_empty = model(x_seq, empty)
    assert not torch.allclose(out_full, out_empty)


def test_per_node_forecaster_forward_shape() -> None:
    """Graph-free ablation: same output shape, no edge_index argument."""
    model = PerNodeForecaster(in_channels=1, hidden_dim=8, target_len=3, dropout=0.0)
    out = model(torch.randn(5, 4, 1))
    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_models_reject_bad_args() -> None:
    """Both models reject non-positive hidden_dim / target_len."""
    with pytest.raises(ValueError, match="hidden_dim"):
        ContrastForecastGNN(hidden_dim=0, target_len=2)
    with pytest.raises(ValueError, match="target_len"):
        PerNodeForecaster(target_len=0)


# --------------------------------------------------------------------------- #
# learning smoke test: propagating bolus on a path graph
# --------------------------------------------------------------------------- #
def _propagation_series(num_nodes: int, num_frames: int) -> torch.Tensor:
    """Synthetic bolus: node i's enhancement bump arrives later along the path.

    A smooth, deterministic, forecastable signal -- node i peaks at frame i, so
    the future depends on temporal position. Baseline-subtracted (frame 0 -> 0).
    """
    t = torch.arange(num_frames, dtype=torch.float32)
    arrivals = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)
    curves = torch.exp(-((t.unsqueeze(0) - arrivals) ** 2) / 2.0)
    return curves - curves[:, :1]


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_gnn_forecaster_learns_on_synthetic_propagation() -> None:
    """Training reduces forecast MAE well below its starting value (plumbing learns)."""
    torch.manual_seed(0)
    num_nodes, num_frames = 12, 8
    series = _propagation_series(num_nodes, num_frames)
    edge_index = _path_edge_index(num_nodes)
    horizon = ForecastHorizon(input_len=6, target_len=2)
    inputs, targets = split_forecast_window(series, horizon)
    x_seq = inputs.unsqueeze(-1)  # (N, input_len, 1)

    model = ContrastForecastGNN(
        in_channels=1, hidden_dim=16, target_len=horizon.target_len, dropout=0.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    first_loss = None
    last_loss = None
    for step in range(200):
        optimizer.zero_grad()
        pred = model(x_seq, edge_index)
        loss = masked_mae(pred, targets)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()

    assert first_loss is not None and last_loss is not None
    assert last_loss < 0.5 * first_loss, (first_loss, last_loss)

    # And the trained model should beat the persistence baseline on this signal.
    model.eval()
    with torch.no_grad():
        trained_mae = masked_mae(model(x_seq, edge_index), targets).item()
    baseline_mae = masked_mae(
        last_frame_forecast(inputs, horizon.target_len), targets
    ).item()
    assert trained_mae < baseline_mae, (trained_mae, baseline_mae)


# --------------------------------------------------------------------------- #
# end-to-end pretraining gates (§7.i / §7.ii) on synthetic multi-graph data
# --------------------------------------------------------------------------- #
def test_run_pretrain_gates_end_to_end() -> None:
    """The §7.i gate passes on a structured signal; the report has every key."""
    from gnn.pretrain.train import (
        build_synthetic_forecast_graphs,
        run_pretrain_gates,
    )

    horizon = ForecastHorizon(input_len=6, target_len=2)
    train_graphs = build_synthetic_forecast_graphs(
        horizon, num_graphs=6, num_nodes=12, seed=0
    )
    val_graphs = build_synthetic_forecast_graphs(
        horizon, num_graphs=3, num_nodes=12, seed=99
    )
    report = run_pretrain_gates(
        train_graphs, val_graphs, horizon, hidden_dim=16, epochs=60, seed=0
    )
    for key in ("gnn", "per_node", "last_frame", "temporal_mean"):
        assert key in report and report[key] >= 0.0
    # On a genuinely forecastable signal, the learned model must clear §7.i.
    assert report["beats_trivial"] == 1.0, report
