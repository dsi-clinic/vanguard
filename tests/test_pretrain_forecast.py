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

from gnn.kinetics import baseline_relative_curve  # noqa: E402
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


def test_baseline_relative_curve_zeroes_nonpositive_baseline_rows() -> None:
    """A row whose baseline is <= float32 eps yields all zeros (per-row, no warning)."""
    stack = np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 20.0]])
    out = baseline_relative_curve(
        stack, baseline_frame_count=2, relative_enhancement=True
    )
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1.0])


def test_baseline_relative_curve_rejects_bad_baseline_count() -> None:
    """baseline_frame_count must be in [1, n_timepoints)."""
    with pytest.raises(ValueError, match="baseline_frame_count"):
        baseline_relative_curve(
            np.array([1.0, 2.0, 3.0]),
            baseline_frame_count=3,
            relative_enhancement=False,
        )


def test_voxel_node_series_applies_relative_contract() -> None:
    """With the UChicago contract, rows are (S-S0)/S0 over the shared transform.

    Baseline S0 = mean of the first ``baseline_frame_count`` frames per voxel;
    the result must match ``gnn.kinetics.baseline_relative_curve`` applied to the
    sampled curves (the single source of truth for the enhancement transform).
    """
    dce_4d = np.arange(3 * 1 * 1 * 2, dtype=np.float32).reshape(3, 1, 1, 2)
    nodes = [(0, 0, 0), (1, 0, 0)]
    series = voxel_node_series(
        dce_4d, nodes, baseline_frame_count=2, relative_enhancement=True
    )
    sampled = np.stack([dce_4d[:, 0, 0, x] for x, _, _ in nodes], axis=0)
    expected = baseline_relative_curve(
        sampled, baseline_frame_count=2, relative_enhancement=True
    )
    np.testing.assert_allclose(series, expected, rtol=1e-6)


def test_segment_node_series_references_per_voxel_then_averages() -> None:
    """Under relative enhancement the segment references per voxel THEN means.

    Division by a per-voxel S0 is nonlinear, so per-voxel-relative-then-mean must
    differ from relative-of-the-mean; the segment path takes the former to match
    the kinetic segment path.
    """
    # Two-voxel segment, very different baselines: A -> [0,0,1.0], B -> [0,0,0.1].
    dce_4d = np.zeros((3, 1, 1, 2), dtype=np.float32)
    dce_4d[:, 0, 0, 0] = [10.0, 10.0, 20.0]
    dce_4d[:, 0, 0, 1] = [100.0, 100.0, 110.0]
    series = segment_node_series(
        _line_two_voxel_graph(),
        dce_4d,
        baseline_frame_count=2,
        relative_enhancement=True,
    )
    np.testing.assert_allclose(series[0], [0.0, 0.0, 0.55])
    assert not np.isclose(series[0, 2], (65.0 - 55.0) / 55.0)  # not relative-of-mean


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


def _line_two_voxel_graph():  # noqa: ANN202
    """A 2-voxel segment (both tips degree-1) -> one segment of two voxels."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge((0, 0, 0), (1, 0, 0))
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
# junction_node_series (raw junction-voxel curve; §4.3 first-pass target)
# --------------------------------------------------------------------------- #
def test_junction_node_series_aligns_with_pos_and_baselines() -> None:
    """Row i = raw baseline-subtracted curve at data.pos[i] (junction-node order)."""
    from gnn.junction_graph import build_junction_graph
    from gnn.pretrain.node_series import junction_node_series

    graph = _y_voxel_graph()
    num_frames = 4
    dce_4d = np.random.default_rng(3).random((num_frames, 1, 16, 16)).astype(np.float32)
    radius_map = dict.fromkeys(graph.nodes(), 1.0)
    data = build_junction_graph(
        graph, radius_map, dce_4d, np.arange(num_frames, dtype=float)
    )
    series = junction_node_series(dce_4d, data.pos.numpy())

    assert series.shape == (data.num_nodes, num_frames)
    pos = data.pos.numpy()
    for i in range(data.num_nodes):
        x, y, z = int(pos[i, 0]), int(pos[i, 1]), int(pos[i, 2])
        expected = dce_4d[:, z, y, x] - dce_4d[0, z, y, x]
        np.testing.assert_allclose(series[i], expected, rtol=1e-5)
    np.testing.assert_allclose(series[:, 0], np.zeros(data.num_nodes), atol=1e-6)


def test_junction_node_series_rejects_bad_pos_shape() -> None:
    """A pos array that isn't (num_junctions, 3) fails loudly."""
    from gnn.pretrain.node_series import junction_node_series

    dce_4d = np.zeros((3, 1, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="num_junctions, 3"):
        junction_node_series(dce_4d, np.zeros((3, 2)))


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


def test_masked_mae_broadcast_mask_matches_expanded() -> None:
    """A (N,1) mask equals its expanded (N,target_len) form (denominator regression).

    The old code took the denominator from the un-broadcast mask, so a per-node
    (N,1) mask scored target_len times high; both numerator and denominator must
    see the same broadcast mask.
    """
    pred = torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])
    target = torch.zeros_like(pred)
    node_mask = torch.tensor([[1.0], [0.0], [1.0]])  # keep nodes 0 and 2
    broadcast = masked_mae(pred, target, node_mask)
    expanded = masked_mae(pred, target, node_mask.expand_as(pred))
    # Kept errors [1,3,9,11] -> mean 6.0 (not 12.0, the pre-fix value).
    assert broadcast.item() == pytest.approx(6.0)
    assert broadcast.item() == pytest.approx(expanded.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_masked_mae_cpu_mask_applies_to_cuda_pred() -> None:
    """A CPU mask applies to a CUDA prediction instead of raising a device error."""
    pred = torch.tensor([[1.0, 3.0], [5.0, 7.0]], device="cuda")
    target = torch.zeros_like(pred)
    cpu_mask = torch.tensor([[1.0], [0.0]])  # left on CPU on purpose
    assert masked_mae(pred, target, cpu_mask).item() == pytest.approx(2.0)


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


def _times(input_len: int, target_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Equally-spaced elapsed-time axes for a horizon (frame index == seconds)."""
    frames = torch.arange(input_len + target_len, dtype=torch.float32)
    return frames[:input_len], frames[input_len:]


def test_gnn_forecast_forward_shape() -> None:
    """(N, input_len, C) + times -> (N, target_len)."""
    model = ContrastForecastGNN(in_channels=1, hidden_dim=8, dropout=0.0)
    x_seq = torch.randn(3, 4, 1)
    input_times, target_times = _times(4, 3)
    out = model(x_seq, _triangle_edge_index(), input_times, target_times)
    assert out.shape == (3, 3)
    assert torch.isfinite(out).all()


def test_gnn_forecast_uses_edges() -> None:
    """Changing the graph changes the forecast -- message passing is live."""
    model = ContrastForecastGNN(in_channels=1, hidden_dim=8, dropout=0.0)
    model.eval()
    x_seq = torch.randn(3, 4, 1)
    input_times, target_times = _times(4, 2)
    full = _triangle_edge_index()
    empty = torch.empty(2, 0, dtype=torch.long)
    out_full = model(x_seq, full, input_times, target_times)
    out_empty = model(x_seq, empty, input_times, target_times)
    assert not torch.allclose(out_full, out_empty)


def test_gnn_forecast_uses_input_and_target_times() -> None:
    """Both time inputs are consumed: perturbing either changes the forecast."""
    model = ContrastForecastGNN(in_channels=1, hidden_dim=8, dropout=0.0).eval()
    x_seq = torch.randn(3, 4, 1)
    edge = _triangle_edge_index()
    input_times, target_times = _times(4, 2)
    with torch.no_grad():
        base = model(x_seq, edge, input_times, target_times)
        diff_in = model(x_seq, edge, input_times * 5.0, target_times)
        diff_out = model(x_seq, edge, input_times, target_times + 7.0)
    assert not torch.allclose(base, diff_in)
    assert not torch.allclose(base, diff_out)


def test_query_decoder_handles_variable_target_count() -> None:
    """The time-conditioned decoder predicts one value per supplied target time."""
    model = ContrastForecastGNN(in_channels=1, hidden_dim=8, dropout=0.0).eval()
    x_seq = torch.randn(3, 4, 1)
    edge = _triangle_edge_index()
    input_times = torch.arange(4, dtype=torch.float32)
    with torch.no_grad():
        one = model(x_seq, edge, input_times, torch.tensor([4.0]))
        three = model(x_seq, edge, input_times, torch.tensor([4.0, 5.0, 6.0]))
    assert one.shape == (3, 1)
    assert three.shape == (3, 3)


def test_per_node_forecaster_forward_shape() -> None:
    """Graph-free ablation: same output shape, no edge_index argument."""
    model = PerNodeForecaster(in_channels=1, hidden_dim=8, dropout=0.0)
    input_times, target_times = _times(4, 3)
    out = model(torch.randn(5, 4, 1), input_times, target_times)
    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_models_reject_bad_args() -> None:
    """Both models reject non-positive hidden_dim / num_layers."""
    with pytest.raises(ValueError, match="hidden_dim"):
        ContrastForecastGNN(hidden_dim=0)
    with pytest.raises(ValueError, match="num_layers"):
        PerNodeForecaster(num_layers=0)


def test_augment_with_time_rejects_wrong_length() -> None:
    """input_times length must match the input horizon."""
    model = PerNodeForecaster(in_channels=1, hidden_dim=8, dropout=0.0).eval()
    with pytest.raises(ValueError, match="input_times"):
        model(torch.randn(2, 4, 1), torch.arange(3.0), torch.arange(2.0))


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
    input_times, target_times = _times(horizon.input_len, horizon.target_len)

    model = ContrastForecastGNN(in_channels=1, hidden_dim=16, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    first_loss = None
    last_loss = None
    for step in range(200):
        optimizer.zero_grad()
        pred = model(x_seq, edge_index, input_times, target_times)
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
        trained_mae = masked_mae(
            model(x_seq, edge_index, input_times, target_times), targets
        ).item()
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
