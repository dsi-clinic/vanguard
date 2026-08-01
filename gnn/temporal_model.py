"""Shared temporal encoders for forecasting and patient-level pCR models.

The forecasting and downstream models own different heads, but they use the
same encoder object.  Encoder checkpoints can therefore be loaded strictly;
there is no manual layer copying between architectures with different inputs.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gnn.model import _graph_readout

_TIME_CHANNELS = 1
_VALIDITY_CHANNELS = 1
_NDIM_TEMPORAL = 3
BASELINE_POLICY = "finite_S0_gt_0.05_case_median_then_simultaneous_neighbor_median"


def _validate_temporal_inputs(
    node_series: torch.Tensor,
    acquisition_times: torch.Tensor,
    baseline_observed: torch.Tensor,
) -> tuple[int, int]:
    if node_series.ndim != _NDIM_TEMPORAL:
        raise ValueError(
            "node_series must be (num_nodes, num_frames, signal_channels); "
            f"got {tuple(node_series.shape)}"
        )
    num_nodes, num_frames, _ = node_series.shape
    if acquisition_times.ndim != 1 or acquisition_times.shape[0] != num_frames:
        raise ValueError(
            f"acquisition_times/input_times must be 1D with {num_frames} entries; "
            f"got {tuple(acquisition_times.shape)}"
        )
    if baseline_observed.ndim != 1 or baseline_observed.shape[0] != num_nodes:
        raise ValueError(
            f"baseline_observed must be 1D with {num_nodes} entries; "
            f"got {tuple(baseline_observed.shape)}"
        )
    return num_nodes, num_frames


def _augment_frames(
    node_series: torch.Tensor,
    acquisition_times: torch.Tensor,
    baseline_observed: torch.Tensor,
) -> torch.Tensor:
    """Append acquisition time and original-baseline validity to every frame."""
    num_nodes, num_frames = _validate_temporal_inputs(
        node_series, acquisition_times, baseline_observed
    )
    times = (
        acquisition_times.to(device=node_series.device, dtype=node_series.dtype)
        .view(1, num_frames, _TIME_CHANNELS)
        .expand(num_nodes, num_frames, _TIME_CHANNELS)
    )
    observed = (
        baseline_observed.to(device=node_series.device, dtype=node_series.dtype)
        .view(num_nodes, 1, _VALIDITY_CHANNELS)
        .expand(num_nodes, num_frames, _VALIDITY_CHANNELS)
    )
    return torch.cat([node_series, times, observed], dim=-1)


class TemporalGraphEncoder(nn.Module):
    """Per-frame GCN stack followed by a per-node GRU.

    Inputs use minutes relative to the exam's last baseline acquisition.  The
    return value is one hidden vector per node and is shared without adaptation
    by forecasting and downstream classification.
    """

    encoder_type = "graph"

    def __init__(
        self,
        *,
        signal_channels: int = 1,
        hidden_dim: int = 32,
        num_gcn_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if signal_channels < 1:
            raise ValueError("signal_channels must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if num_gcn_layers < 1:
            raise ValueError("num_gcn_layers must be >= 1")
        if num_gru_layers < 1:
            raise ValueError("num_gru_layers must be >= 1")
        self.signal_channels = signal_channels
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_gru_layers = num_gru_layers
        self.dropout_p = dropout

        self.convs = nn.ModuleList()
        in_dim = signal_channels + _TIME_CHANNELS + _VALIDITY_CHANNELS
        for _ in range(num_gcn_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

    def config(self) -> dict[str, object]:
        """Architecture/preprocessing fields required for strict reconstruction."""
        return {
            "encoder_type": self.encoder_type,
            "graph_mode": "voxel",
            "input_channels": ["relative_enhancement", "baseline_observed"],
            "signal_channels": self.signal_channels,
            "hidden_dim": self.hidden_dim,
            "num_gcn_layers": self.num_gcn_layers,
            "num_gru_layers": self.num_gru_layers,
            "dropout": self.dropout_p,
            "time_units": "minutes_since_last_baseline",
            "baseline_policy": BASELINE_POLICY,
        }

    def forward(
        self,
        node_series: torch.Tensor,
        edge_index: torch.Tensor,
        acquisition_times: torch.Tensor,
        baseline_observed: torch.Tensor,
    ) -> torch.Tensor:
        frames = _augment_frames(node_series, acquisition_times, baseline_observed)
        encoded_frames: list[torch.Tensor] = []
        for frame_idx in range(frames.shape[1]):
            h = frames[:, frame_idx, :]
            for conv in self.convs:
                h = self.dropout(torch.relu(conv(h, edge_index)))
            encoded_frames.append(h)
        sequence = torch.stack(encoded_frames, dim=1)
        output, _ = self.gru(sequence)
        return output[:, -1, :]


class PerNodeTemporalEncoder(nn.Module):
    """Graph-free matched encoder: frame MLP followed by the same GRU."""

    encoder_type = "graph_free"

    def __init__(
        self,
        *,
        signal_channels: int = 1,
        hidden_dim: int = 32,
        num_gcn_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if signal_channels < 1 or hidden_dim < 1:
            raise ValueError("signal_channels and hidden_dim must be >= 1")
        if num_gcn_layers < 1:
            raise ValueError("num_layers/num_gcn_layers must be >= 1")
        if num_gru_layers < 1:
            raise ValueError("num_gru_layers must be >= 1")
        self.signal_channels = signal_channels
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_gru_layers = num_gru_layers
        self.dropout_p = dropout
        layers: list[nn.Module] = []
        in_dim = signal_channels + _TIME_CHANNELS + _VALIDITY_CHANNELS
        for _ in range(num_gcn_layers):
            layers.extend(
                [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            in_dim = hidden_dim
        self.frame_encoder = nn.Sequential(*layers)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

    def config(self) -> dict[str, object]:
        return {
            "encoder_type": self.encoder_type,
            "graph_mode": "voxel",
            "input_channels": ["relative_enhancement", "baseline_observed"],
            "signal_channels": self.signal_channels,
            "hidden_dim": self.hidden_dim,
            "num_gcn_layers": self.num_gcn_layers,
            "num_gru_layers": self.num_gru_layers,
            "dropout": self.dropout_p,
            "time_units": "minutes_since_last_baseline",
            "baseline_policy": BASELINE_POLICY,
        }

    def forward(
        self,
        node_series: torch.Tensor,
        acquisition_times: torch.Tensor,
        baseline_observed: torch.Tensor,
    ) -> torch.Tensor:
        frames = _augment_frames(node_series, acquisition_times, baseline_observed)
        num_nodes, num_frames, channels = frames.shape
        encoded = self.frame_encoder(frames.reshape(num_nodes * num_frames, channels))
        encoded = encoded.reshape(num_nodes, num_frames, self.hidden_dim)
        output, _ = self.gru(encoded)
        return output[:, -1, :]


class TimeConditionedForecastDecoder(nn.Module):
    """Decode future enhancement at horizons relative to the last input frame."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, node_embedding: torch.Tensor, target_horizon: torch.Tensor
    ) -> torch.Tensor:
        if target_horizon.ndim != 1:
            raise ValueError("target_horizon must be one-dimensional")
        num_nodes, hidden_dim = node_embedding.shape
        num_targets = target_horizon.shape[0]
        h = node_embedding.unsqueeze(1).expand(num_nodes, num_targets, hidden_dim)
        tau = (
            target_horizon.to(node_embedding)
            .view(1, num_targets, 1)
            .expand(num_nodes, num_targets, 1)
        )
        return self.net(torch.cat([h, tau], dim=-1)).squeeze(-1)


def masked_mean_max_pool(
    node_embedding: torch.Tensor, baseline_observed: torch.Tensor
) -> torch.Tensor:
    """Return one concatenated masked mean/max vector for a single graph."""
    if (
        baseline_observed.ndim != 1
        or baseline_observed.shape[0] != node_embedding.shape[0]
    ):
        raise ValueError("baseline_observed must align with node_embedding rows")
    batch_index = torch.zeros(
        node_embedding.shape[0], dtype=torch.long, device=node_embedding.device
    )
    return _graph_readout(
        node_embedding,
        batch_index,
        "mean_max",
        baseline_observed,
    )


class TemporalGraphClassifier(nn.Module):
    """Shared graph encoder -> masked mean/max readout -> one pCR logit."""

    def __init__(self, encoder: TemporalGraphEncoder, *, graph_dim: int = 0) -> None:
        super().__init__()
        if graph_dim < 0:
            raise ValueError("graph_dim must be >= 0")
        self.encoder = encoder
        self.graph_dim = graph_dim
        self.classifier = nn.Linear(2 * encoder.hidden_dim + graph_dim, 1)

    def forward(
        self,
        node_series: torch.Tensor,
        edge_index: torch.Tensor,
        acquisition_times: torch.Tensor,
        baseline_observed: torch.Tensor,
        graph_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.encoder(
            node_series, edge_index, acquisition_times, baseline_observed
        )
        pooled = masked_mean_max_pool(embedding, baseline_observed)
        if self.graph_dim:
            if graph_features is None or graph_features.shape != (1, self.graph_dim):
                raise ValueError(
                    f"graph_features must have shape (1, {self.graph_dim})"
                )
            pooled = torch.cat([pooled, graph_features], dim=1)
        return self.classifier(pooled).view(-1)


class TemporalPerNodeClassifier(nn.Module):
    """Graph-free temporal encoder with the identical masked classifier head."""

    def __init__(self, encoder: PerNodeTemporalEncoder, *, graph_dim: int = 0) -> None:
        super().__init__()
        if graph_dim < 0:
            raise ValueError("graph_dim must be >= 0")
        self.encoder = encoder
        self.graph_dim = graph_dim
        self.classifier = nn.Linear(2 * encoder.hidden_dim + graph_dim, 1)

    def forward(
        self,
        node_series: torch.Tensor,
        acquisition_times: torch.Tensor,
        baseline_observed: torch.Tensor,
        graph_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.encoder(node_series, acquisition_times, baseline_observed)
        pooled = masked_mean_max_pool(embedding, baseline_observed)
        if self.graph_dim:
            if graph_features is None or graph_features.shape != (1, self.graph_dim):
                raise ValueError(
                    f"graph_features must have shape (1, {self.graph_dim})"
                )
            pooled = torch.cat([pooled, graph_features], dim=1)
        return self.classifier(pooled).view(-1)


def build_encoder_from_config(config: Mapping[str, object]) -> nn.Module:
    """Reconstruct an encoder from a saved checkpoint configuration."""
    kwargs = {
        "signal_channels": int(config["signal_channels"]),
        "hidden_dim": int(config["hidden_dim"]),
        "num_gcn_layers": int(config["num_gcn_layers"]),
        "num_gru_layers": int(config["num_gru_layers"]),
        "dropout": float(config["dropout"]),
    }
    encoder_type = str(config["encoder_type"])
    if encoder_type == "graph":
        encoder: nn.Module = TemporalGraphEncoder(**kwargs)
    elif encoder_type == "graph_free":
        encoder = PerNodeTemporalEncoder(**kwargs)
    else:
        raise ValueError(f"unknown encoder_type {encoder_type!r}")
    if encoder.config() != dict(config):
        raise ValueError("encoder_config does not match the supported contract")
    return encoder


def load_encoder_strict(encoder: nn.Module, checkpoint: Mapping[str, object]) -> None:
    """Validate the complete encoder contract, then load weights strictly."""
    expected = encoder.config()
    actual = checkpoint.get("encoder_config")
    if actual != expected:
        raise ValueError(
            f"encoder configuration mismatch: checkpoint={actual}, requested={expected}"
        )
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
