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
    """A single graph's forecasting tensors (input horizon, target, adjacency, times).

    ``input_times`` / ``target_times`` are the elapsed acquisition seconds of the
    input and target frames, measured from the window's first input frame and
    shared across nodes. They carry the irregular UFAST cadence into the model
    (design review issue 3a) so adjacent frames are not treated as equally spaced.
    """

    x_seq: torch.Tensor  # (N, input_len, C)
    target: torch.Tensor  # (N, target_len)
    edge_index: torch.Tensor  # (2, E)
    input_times: torch.Tensor  # (input_len,) acquisition times
    target_times: torch.Tensor  # (target_len,) minutes from last baseline
    target_mask: torch.Tensor | None = None  # (N, target_len)
    baseline_observed: torch.Tensor | None = None  # (N,) original validity
    exam_id: str = ""

    def __post_init__(self) -> None:
        """Fill compatibility defaults for legacy all-valid smoke fixtures."""
        if self.target_mask is None:
            self.target_mask = torch.isfinite(self.target)
        if self.baseline_observed is None:
            self.baseline_observed = torch.ones(
                self.x_seq.shape[0], dtype=torch.bool, device=self.x_seq.device
            )

    @property
    def acquisition_times(self) -> torch.Tensor:
        """Explicit name used by the shared temporal encoder."""
        return self.input_times


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
    Frames are equally spaced here (unit cadence), so the elapsed-time axis is
    just the frame index -- the fixture exercises the timed model interface
    without claiming anything about real (irregular) UFAST cadence.
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

    input_times = t[: horizon.input_len]
    target_times = t[horizon.input_len : horizon.window]

    graphs: list[ForecastGraph] = []
    for _ in range(num_graphs):
        series = base + noise_std * torch.randn(base.shape, generator=generator)
        inputs = series[:, : horizon.input_len]
        target = series[:, horizon.input_len : horizon.window]
        graphs.append(
            ForecastGraph(
                x_seq=inputs.unsqueeze(-1),
                target=target,
                edge_index=edge_index,
                input_times=input_times,
                target_times=target_times,
                target_mask=torch.ones_like(target, dtype=torch.bool),
                baseline_observed=torch.ones(num_nodes, dtype=torch.bool),
            )
        )
    return graphs


def _exam_groups(
    graphs: list[ForecastGraph] | list[list[ForecastGraph]],
) -> list[list[ForecastGraph]]:
    """Normalize the legacy one-window-per-exam form to grouped exam windows."""
    if not graphs:
        raise ValueError("at least one forecast exam is required")
    if isinstance(graphs[0], ForecastGraph):
        return [[graph] for graph in graphs]  # type: ignore[list-item]
    groups = graphs  # type: ignore[assignment]
    if any(not group for group in groups):
        raise ValueError("every exam must contain at least one eligible window")
    return groups


def _mean_exam_mae(per_exam_window_errors: list[list[torch.Tensor]]) -> float:
    """Average windows within an exam, then exams equally."""
    exam_means = [
        torch.stack(errors).mean().item() for errors in per_exam_window_errors
    ]
    return float(sum(exam_means) / len(exam_means))


@torch.no_grad()
def evaluate_gnn(
    model: ContrastForecastGNN,
    graphs: list[ForecastGraph] | list[list[ForecastGraph]],
) -> float:
    """Held-out MAE averaged within exam and then across exams."""
    model.eval()
    errors = [
        [
            masked_mae(
                model(
                    g.x_seq,
                    g.edge_index,
                    g.acquisition_times,
                    g.baseline_observed,
                    g.target_times,
                ),
                g.target,
                g.target_mask,
            )
            for g in exam
        ]
        for exam in _exam_groups(graphs)
    ]
    return _mean_exam_mae(errors)


@torch.no_grad()
def evaluate_per_node(
    model: PerNodeForecaster,
    graphs: list[ForecastGraph] | list[list[ForecastGraph]],
) -> float:
    """Held-out graph-free MAE averaged within exam and then across exams."""
    model.eval()
    errors = [
        [
            masked_mae(
                model(
                    g.x_seq,
                    g.acquisition_times,
                    g.baseline_observed,
                    g.target_times,
                ),
                g.target,
                g.target_mask,
            )
            for g in exam
        ]
        for exam in _exam_groups(graphs)
    ]
    return _mean_exam_mae(errors)


def _baseline_mae(
    graphs: list[ForecastGraph] | list[list[ForecastGraph]],
    forecast_fn: Callable[[torch.Tensor, int], torch.Tensor],
    target_len: int,
) -> float:
    """Node-weighted MAE of a trivial baseline (``forecast_fn(inputs, target_len)``).

    The baseline sees only the last input channel's history (channel 0 is the
    contrast enhancement being forecast -- the target signal), consistent with
    the models forecasting that same channel.
    """
    errors = []
    for exam in _exam_groups(graphs):
        exam_errors = []
        for g in exam:
            inputs = g.x_seq[:, :, 0]
            exam_errors.append(
                masked_mae(forecast_fn(inputs, target_len), g.target, g.target_mask)
            )
        errors.append(exam_errors)
    return _mean_exam_mae(errors)


def train_forecaster(
    model: torch.nn.Module,
    train_graphs: list[ForecastGraph] | list[list[ForecastGraph]],
    *,
    epochs: int,
    lr: float,
    uses_graph: bool,
    seed: int = 0,
) -> list[float]:
    """Train ``model`` (batch-of-one per graph); return per-epoch mean train MAE.

    ``uses_graph`` selects the forward signature: the GNN consumes ``edge_index``,
    the graph-free ablation does not. One optimiser step per graph per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    exams = _exam_groups(train_graphs)
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_errs: list[list[torch.Tensor]] = []
        order = torch.randperm(len(exams), generator=generator).tolist()
        for exam_index in order:
            exam = exams[exam_index]
            window_index = int(torch.randint(len(exam), (1,), generator=generator))
            g = exam[window_index]
            optimizer.zero_grad()
            pred = (
                model(
                    g.x_seq,
                    g.edge_index,
                    g.acquisition_times,
                    g.baseline_observed,
                    g.target_times,
                )
                if uses_graph
                else model(
                    g.x_seq,
                    g.acquisition_times,
                    g.baseline_observed,
                    g.target_times,
                )
            )
            loss = masked_mae(pred, g.target, g.target_mask)
            loss.backward()
            optimizer.step()
            epoch_errs.append([loss.detach()])
        history.append(_mean_exam_mae(epoch_errs))
    return history


def run_pretrain_gates(
    train_graphs: list[ForecastGraph] | list[list[ForecastGraph]],
    val_graphs: list[ForecastGraph] | list[list[ForecastGraph]],
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
        num_layers=num_layers,
        dropout=dropout,
    )
    train_forecaster(
        gnn, train_graphs, epochs=epochs, lr=lr, uses_graph=True, seed=seed
    )

    torch.manual_seed(seed)
    per_node = PerNodeForecaster(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    train_forecaster(
        per_node,
        train_graphs,
        epochs=epochs,
        lr=lr,
        uses_graph=False,
        seed=seed,
    )

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
