"""Contrast-forecasting models: encoder (b) and the graph-free ablation.

Both predict, for every node, the contrast enhancement at a set of future query
times from an input horizon of ``(N, input_len, in_channels)`` per-frame node
features plus the acquisition times of those frames. They share a temporal head
(a per-node GRU + a time-conditioned decoder) and differ only in how each
*frame* is spatially encoded -- which is exactly the comparison the design doc's
§7.ii "does the graph matter" ablation needs:

- ``ContrastForecastGNN`` (design doc alternative (b)): each frame is encoded by
  a shared ``GCNConv`` stack, so a node's per-frame embedding depends on its
  neighbours' contemporaneous contrast -- the mechanism by which the model can
  learn bolus propagation across vascular edges. Reuses ``GCNConv`` exactly as
  ``gnn.model.GCNClassifier`` does.
- ``PerNodeForecaster``: each frame is encoded by a per-node MLP with **no
  message passing**. Same temporal head, same capacity knobs; the only thing
  removed is the graph. If this matches the GNN on held-out forecasting loss,
  the "graph-ey" defense (design doc §5) is not what is being learned.

**Physical time (design review issue 3a).** UFAST cadence is irregular and
varies several-fold between exams, so frame *index* is not frame *time*. Both
models therefore (1) append each input frame's elapsed acquisition seconds as an
extra encoder channel (the "delta-t channel", decision 1a), and (2) condition
the decoder on the elapsed seconds of each frame it is asked to predict
(decision 1b), so it forecasts "enhancement at +tau seconds" rather than "at
output slot k". Times are elapsed seconds from the window's first input frame,
shared across nodes (anatomy is static within a scan; only contrast varies).

The temporal encoder is deliberately kept separate from the spatial one
(alternative (b), not the DCGRU fusion of alternative (a)): it is the cheapest
path to a falsifying pilot and keeps the two design choices independent.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv

_NDIM_3D = 3  # (num_nodes, input_len, in_channels)
_TIME_CHANNELS = 1  # elapsed acquisition seconds, appended per frame / per query


def _augment_with_time(x_seq: torch.Tensor, input_times: torch.Tensor) -> torch.Tensor:
    """Append each input frame's elapsed time as an extra channel.

    ``x_seq`` ``(N, input_len, C)`` + ``input_times`` ``(input_len,)`` ->
    ``(N, input_len, C + 1)``. The time axis is shared across nodes, so it is
    broadcast over the node dimension.
    """
    if x_seq.ndim != _NDIM_3D:
        raise ValueError(
            f"x_seq must be 3D (N, input_len, C); got {tuple(x_seq.shape)}"
        )
    num_nodes, input_len, _ = x_seq.shape
    if input_times.ndim != 1 or input_times.shape[0] != input_len:
        raise ValueError(
            f"input_times must be 1D of length input_len={input_len}; "
            f"got shape {tuple(input_times.shape)}"
        )
    times = (
        input_times.to(x_seq.dtype)
        .view(1, input_len, _TIME_CHANNELS)
        .expand(num_nodes, input_len, _TIME_CHANNELS)
    )
    return torch.cat([x_seq, times], dim=-1)


class _TemporalForecastHead(nn.Module):
    """Per-node GRU over frame embeddings -> time-conditioned query decoder.

    Consumes a ``(N, input_len, hidden)`` sequence of per-frame node embeddings
    (nodes are the batch dimension, frames the sequence), runs a GRU, and takes
    the final hidden state as a per-node summary of the input horizon. The
    decoder is **time-conditioned** (design review decision 1b): for each target
    time ``tau`` (elapsed seconds from the window start) it predicts that node's
    enhancement from an MLP on ``[hidden_state, tau]``. It thus predicts
    "enhancement at +tau seconds", not "enhancement at output slot k", so one
    model handles the irregular / exam-varying target spacing that
    non-overlapping tiling produces -- and any number of target frames. Shared by
    both encoders so the only difference between them is spatial, never temporal.
    """

    def __init__(self, hidden_dim: int, num_gru_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + _TIME_CHANNELS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, frame_embeddings: torch.Tensor, target_times: torch.Tensor
    ) -> torch.Tensor:
        """``(N, input_len, hidden)`` + ``(target_len,)`` -> ``(N, target_len)``."""
        if target_times.ndim != 1:
            raise ValueError(
                f"target_times must be 1D (target_len,); got {tuple(target_times.shape)}"
            )
        output, _ = self.gru(frame_embeddings)
        last = output[:, -1, :]  # (N, hidden)
        num_nodes, hidden_dim = last.shape
        num_targets = target_times.shape[0]
        hidden_expanded = last.unsqueeze(1).expand(num_nodes, num_targets, hidden_dim)
        tau = (
            target_times.to(last.dtype)
            .view(1, num_targets, _TIME_CHANNELS)
            .expand(num_nodes, num_targets, _TIME_CHANNELS)
        )
        decoder_input = torch.cat([hidden_expanded, tau], dim=-1)
        return self.decoder(decoder_input).squeeze(-1)


def _check_common_args(in_channels: int, hidden_dim: int, num_layers: int) -> None:
    """Shared positivity checks (fail-fast, mirroring ``gnn.model``)."""
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")


class ContrastForecastGNN(nn.Module):
    """Per-frame ``GCNConv`` encoder + per-node temporal head (design doc (b))."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _check_common_args(in_channels, hidden_dim, num_layers)
        self.convs = nn.ModuleList()
        in_dim = in_channels + _TIME_CHANNELS  # raw enhancement + elapsed time
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.temporal = _TemporalForecastHead(hidden_dim, num_gru_layers)

    def _encode_frame(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """One frame's node features ``(N, C+1)`` -> embedding ``(N, hidden)``."""
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        return h

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        input_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> torch.Tensor:
        """Forecast ``(N, target_len)`` from a timed input horizon.

        ``x_seq`` is ``(N, input_len, in_channels)``; ``input_times`` /
        ``target_times`` are elapsed seconds ``(input_len,)`` / ``(target_len,)``
        from the window start. Each frame (with its elapsed-time channel appended)
        is passed through the shared ``GCNConv`` stack over the *same*
        ``edge_index`` (anatomy is static within a scan; only contrast varies),
        producing a per-frame node embedding; the GRU integrates them over time
        and the time-conditioned decoder reads them out at ``target_times``.
        ``edge_index`` is the (batched) vascular adjacency -- the message-passing
        path whose contribution the ``PerNodeForecaster`` ablation removes.
        """
        x_aug = _augment_with_time(x_seq, input_times)
        frame_embeddings = torch.stack(
            [
                self._encode_frame(x_aug[:, t, :], edge_index)
                for t in range(x_aug.shape[1])
            ],
            dim=1,
        )
        return self.temporal(frame_embeddings, target_times)


class PerNodeForecaster(nn.Module):
    """Graph-free counterpart of ``ContrastForecastGNN`` (design doc §7.ii ablation).

    Identical temporal head; each frame is encoded by a per-node MLP instead of a
    ``GCNConv`` stack, so no information ever crosses an edge. ``forward`` takes
    no ``edge_index`` -- the absence of the graph is structural, not a runtime
    flag, so the ablation cannot accidentally leak message passing.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _check_common_args(in_channels, hidden_dim, num_layers)
        layers: list[nn.Module] = []
        in_dim = in_channels + _TIME_CHANNELS  # raw enhancement + elapsed time
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        self.frame_encoder = nn.Sequential(*layers)
        self.temporal = _TemporalForecastHead(hidden_dim, num_gru_layers)

    def forward(
        self,
        x_seq: torch.Tensor,
        input_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> torch.Tensor:
        """Forecast ``(N, target_len)`` from a timed input horizon, no edges."""
        x_aug = _augment_with_time(x_seq, input_times)
        num_nodes, input_len, channels = x_aug.shape
        flat = x_aug.reshape(num_nodes * input_len, channels)
        embedded = self.frame_encoder(flat).reshape(num_nodes, input_len, -1)
        return self.temporal(embedded, target_times)
