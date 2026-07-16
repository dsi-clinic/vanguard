"""Minimal pretraining loop + the design doc's §7.i / §7.ii decision gates.

This is the first-pass harness, not the production trainer. It trains
``ContrastForecastGNN`` to forecast held-out node contrast and reports the two
cheap, label-free gates the design doc requires *before* any downstream pCR
work:

- **§7.i (is the pretext task learnable at all?)** trained GNN held-out MAE vs.
  the trivial baselines (last-frame persistence, per-node temporal mean). If the
  model cannot beat these, stop -- the pretext task is learning nothing.
- **§7.ii (does the graph matter?)** trained GNN vs. the graph-free
  ``PerNodeForecaster`` on the same task. If the graph-free model matches it, the
  "graph-ey" defense (design doc §5) is not what is being learned -- surface it,
  don't hide it.

A ``ForecastGraph`` is any object exposing ``x_seq`` ``(N, input_len, C)``,
``target`` ``(N, target_len)``, and ``edge_index`` ``(2, E)`` -- e.g. a
``torch_geometric.data.Data`` with those attributes, built from real voxel
graphs (via ``gnn.pretrain.node_series``) or synthesised for a smoke test. Graphs
are processed one at a time (batch-of-one); batching is a later optimisation.

Heavy runs (real data, many graphs) belong in a Slurm job, never the login node
(AGENTS.md). The synthetic ``build_synthetic_forecast_graphs`` path is light
enough to smoke-test locally.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from gnn.pretrain.baselines import last_frame_forecast, temporal_mean_forecast
from gnn.pretrain.forecast import ForecastHorizon
from gnn.pretrain.loss import masked_mae
from gnn.pretrain.model import ContrastForecastGNN, PerNodeForecaster


@dataclass
class ForecastGraph:
    """A single graph's forecasting tensors (input horizon, target, adjacency)."""

    x_seq: torch.Tensor  # (N, input_len, C)
    target: torch.Tensor  # (N, target_len)
    edge_index: torch.Tensor  # (2, E)


def build_synthetic_forecast_graphs(
    horizon: ForecastHorizon,
    *,
    num_graphs: int,
    num_nodes: int = 12,
    noise_std: float = 0.02,
    seed: int = 0,
) -> list[ForecastGraph]:
    """Synthetic propagating-bolus graphs for smoke tests (no real data touched).

    Each graph is a path (chain) of ``num_nodes`` nodes; node ``i``'s contrast
    bump arrives at frame ``i``, so a node's future is coupled to its position
    along the vessel and (through the shared adjacency) to its neighbours' phase.
    Baseline-subtracted (frame 0 -> 0), matching ``gnn.pretrain.node_series``. A
    little Gaussian noise keeps the trivial baselines from being exactly perfect.
    This is a plumbing fixture only -- it is *not* a claim about real vasculature.
    """
    generator = torch.Generator().manual_seed(seed)
    num_frames = horizon.window
    t = torch.arange(num_frames, dtype=torch.float32)
    arrivals = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)
    base = torch.exp(-((t.unsqueeze(0) - arrivals) ** 2) / 2.0)
    base = base - base[:, :1]

    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    graphs: list[ForecastGraph] = []
    for _ in range(num_graphs):
        series = base + noise_std * torch.randn(base.shape, generator=generator)
        inputs = series[:, : horizon.input_len]
        target = series[:, horizon.input_len : horizon.window]
        graphs.append(
            ForecastGraph(
                x_seq=inputs.unsqueeze(-1), target=target, edge_index=edge_index
            )
        )
    return graphs


def _mean_graph_mae(
    per_graph_errors: list[torch.Tensor], node_counts: list[int]
) -> float:
    """Node-weighted mean MAE across graphs (a node is a node, regardless of graph)."""
    total = sum(
        err.item() * n for err, n in zip(per_graph_errors, node_counts, strict=True)
    )
    return total / sum(node_counts)


@torch.no_grad()
def evaluate_gnn(model: ContrastForecastGNN, graphs: list[ForecastGraph]) -> float:
    """Node-weighted held-out MAE of the GNN forecaster over ``graphs``."""
    model.eval()
    errs = [masked_mae(model(g.x_seq, g.edge_index), g.target) for g in graphs]
    return _mean_graph_mae(errs, [g.x_seq.shape[0] for g in graphs])


@torch.no_grad()
def evaluate_per_node(model: PerNodeForecaster, graphs: list[ForecastGraph]) -> float:
    """Node-weighted held-out MAE of the graph-free forecaster over ``graphs``."""
    model.eval()
    errs = [masked_mae(model(g.x_seq), g.target) for g in graphs]
    return _mean_graph_mae(errs, [g.x_seq.shape[0] for g in graphs])


def _baseline_mae(
    graphs: list[ForecastGraph],
    forecast_fn: Callable[[torch.Tensor, int], torch.Tensor],
    target_len: int,
) -> float:
    """Node-weighted MAE of a trivial baseline (``forecast_fn(inputs, target_len)``).

    The baseline sees only the last input channel's history (channel 0 is the
    contrast enhancement being forecast -- the target signal), consistent with
    the models forecasting that same channel.
    """
    errs = []
    for g in graphs:
        inputs = g.x_seq[:, :, 0]  # (N, input_len) -- the enhancement channel
        errs.append(masked_mae(forecast_fn(inputs, target_len), g.target))
    return _mean_graph_mae(errs, [g.x_seq.shape[0] for g in graphs])


def train_forecaster(
    model: torch.nn.Module,
    train_graphs: list[ForecastGraph],
    *,
    epochs: int,
    lr: float,
    uses_graph: bool,
) -> list[float]:
    """Train ``model`` (batch-of-one per graph); return per-epoch mean train MAE.

    ``uses_graph`` selects the forward signature: the GNN consumes ``edge_index``,
    the graph-free ablation does not. One optimiser step per graph per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_errs: list[torch.Tensor] = []
        for g in train_graphs:
            optimizer.zero_grad()
            pred = model(g.x_seq, g.edge_index) if uses_graph else model(g.x_seq)
            loss = masked_mae(pred, g.target)
            loss.backward()
            optimizer.step()
            epoch_errs.append(loss.detach())
        history.append(
            _mean_graph_mae(epoch_errs, [g.x_seq.shape[0] for g in train_graphs])
        )
    return history


def run_pretrain_gates(
    train_graphs: list[ForecastGraph],
    val_graphs: list[ForecastGraph],
    horizon: ForecastHorizon,
    *,
    in_channels: int = 1,
    hidden_dim: int = 32,
    num_layers: int = 2,
    epochs: int = 100,
    lr: float = 0.01,
    dropout: float = 0.0,
    seed: int = 0,
) -> dict[str, float]:
    """Train the GNN and the graph-free ablation; return the §7.i/§7.ii report.

    Returns held-out MAEs keyed ``gnn`` / ``per_node`` / ``last_frame`` /
    ``temporal_mean``, plus boolean gates ``beats_trivial`` (§7.i: GNN < both
    trivial baselines) and ``graph_helps`` (§7.ii: GNN meaningfully < graph-free
    ablation). These are reported honestly whether they pass or fail -- a failed
    gate is a stop signal, not something to bury.
    """
    torch.manual_seed(seed)
    gnn = ContrastForecastGNN(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        target_len=horizon.target_len,
        num_layers=num_layers,
        dropout=dropout,
    )
    train_forecaster(gnn, train_graphs, epochs=epochs, lr=lr, uses_graph=True)

    torch.manual_seed(seed)
    per_node = PerNodeForecaster(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        target_len=horizon.target_len,
        num_layers=num_layers,
        dropout=dropout,
    )
    train_forecaster(per_node, train_graphs, epochs=epochs, lr=lr, uses_graph=False)

    gnn_mae = evaluate_gnn(gnn, val_graphs)
    per_node_mae = evaluate_per_node(per_node, val_graphs)
    last_frame_mae = _baseline_mae(val_graphs, last_frame_forecast, horizon.target_len)
    temporal_mean_mae = _baseline_mae(
        val_graphs, temporal_mean_forecast, horizon.target_len
    )
    return {
        "gnn": gnn_mae,
        "per_node": per_node_mae,
        "last_frame": last_frame_mae,
        "temporal_mean": temporal_mean_mae,
        "beats_trivial": float(
            gnn_mae < last_frame_mae and gnn_mae < temporal_mean_mae
        ),
        # "meaningfully": require a >=5% relative improvement over the graph-free
        # model, so a numerically-tiny gap doesn't read as "the graph helps".
        "graph_helps": float(gnn_mae < 0.95 * per_node_mae),
    }
