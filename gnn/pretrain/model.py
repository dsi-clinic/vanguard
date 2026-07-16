"""Contrast-forecasting models: encoder (b) and the graph-free ablation.

Both predict, for every node, the next ``target_len`` contrast-enhancement
values from an input horizon of ``(N, input_len, in_channels)`` per-frame node
features. They share a temporal head (a per-node GRU + linear projection) and
differ only in how each *frame* is spatially encoded -- which is exactly the
comparison the design doc's §7.ii "does the graph matter" ablation needs:

- ``ContrastForecastGNN`` (design doc alternative (b)): each frame is encoded by
  a shared ``GCNConv`` stack, so a node's per-frame embedding depends on its
  neighbours' contemporaneous contrast -- the mechanism by which the model can
  learn bolus propagation across vascular edges. Reuses ``GCNConv`` exactly as
  ``gnn.model.GCNClassifier`` does.
- ``PerNodeForecaster``: each frame is encoded by a per-node MLP with **no
  message passing**. Same temporal head, same capacity knobs; the only thing
  removed is the graph. If this matches the GNN on held-out forecasting loss,
  the "graph-ey" defense (design doc §5) is not what is being learned.

The temporal encoder is deliberately kept separate from the spatial one
(alternative (b), not the DCGRU fusion of alternative (a)): it is the cheapest
path to a falsifying pilot and keeps the two design choices independent.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv

_NDIM_3D = 3  # (num_nodes, input_len, in_channels)


class _TemporalForecastHead(nn.Module):
    """Per-node GRU over frame embeddings -> linear projection to ``target_len``.

    Consumes a ``(N, input_len, hidden)`` sequence of per-frame node embeddings
    (nodes are the batch dimension, frames the sequence), runs a GRU, and maps
    the final hidden state to a ``(N, target_len)`` multi-step forecast (direct
    multi-horizon: all future frames predicted at once, the simplest option for
    the pilot). Shared by both encoders so the only difference between them is
    spatial, never temporal.
    """

    def __init__(self, hidden_dim: int, target_len: int, num_gru_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, target_len)

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """``(N, input_len, hidden)`` -> ``(N, target_len)``."""
        output, _ = self.gru(frame_embeddings)
        last = output[:, -1, :]
        return self.head(last)


def _check_common_args(
    in_channels: int, hidden_dim: int, input_len: int, target_len: int, num_layers: int
) -> None:
    """Shared positivity checks (fail-fast, mirroring ``gnn.model``)."""
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if input_len < 1:
        raise ValueError(f"input_len must be >= 1, got {input_len}")
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")


class ContrastForecastGNN(nn.Module):
    """Per-frame ``GCNConv`` encoder + per-node temporal head (design doc (b))."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_dim: int = 32,
        target_len: int,
        num_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _check_common_args(in_channels, hidden_dim, 1, target_len, num_layers)
        self.convs = nn.ModuleList()
        in_dim = in_channels
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.temporal = _TemporalForecastHead(hidden_dim, target_len, num_gru_layers)

    def _encode_frame(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """One frame's node features ``(N, C)`` -> embedding ``(N, hidden)``."""
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        return h

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forecast ``(N, target_len)`` from ``(N, input_len, in_channels)``.

        Each of the ``input_len`` frames is passed through the shared ``GCNConv``
        stack over the *same* ``edge_index`` (anatomy is static within a scan;
        only contrast varies), producing a per-frame node embedding; the GRU then
        integrates them over time. ``edge_index`` is the (batched) vascular
        adjacency -- the message-passing path whose contribution the
        ``PerNodeForecaster`` ablation removes.
        """
        if x_seq.ndim != _NDIM_3D:
            raise ValueError(
                f"x_seq must be 3D (N, input_len, C); got {tuple(x_seq.shape)}"
            )
        frame_embeddings = torch.stack(
            [
                self._encode_frame(x_seq[:, t, :], edge_index)
                for t in range(x_seq.shape[1])
            ],
            dim=1,
        )
        return self.temporal(frame_embeddings)


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
        target_len: int,
        num_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _check_common_args(in_channels, hidden_dim, 1, target_len, num_layers)
        layers: list[nn.Module] = []
        in_dim = in_channels
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        self.frame_encoder = nn.Sequential(*layers)
        self.temporal = _TemporalForecastHead(hidden_dim, target_len, num_gru_layers)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Forecast ``(N, target_len)`` from ``(N, input_len, in_channels)``, no edges."""
        if x_seq.ndim != _NDIM_3D:
            raise ValueError(
                f"x_seq must be 3D (N, input_len, C); got {tuple(x_seq.shape)}"
            )
        num_nodes, input_len, channels = x_seq.shape
        flat = x_seq.reshape(num_nodes * input_len, channels)
        embedded = self.frame_encoder(flat).reshape(num_nodes, input_len, -1)
        return self.temporal(embedded)
