"""Forecasting heads around the shared temporal encoders.

Only the decoder is forecasting-specific.  The encoder objects are also used by
the downstream pCR classifiers in :mod:`gnn.temporal_model`, making checkpoint
transfer exact and strict.
"""

from __future__ import annotations

import torch
from torch import nn

from gnn.temporal_model import (
    PerNodeTemporalEncoder,
    TemporalGraphEncoder,
    TimeConditionedForecastDecoder,
)


class ContrastForecastGNN(nn.Module):
    """TemporalGraphEncoder followed by a time-conditioned forecast decoder."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
        encoder: TemporalGraphEncoder | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or TemporalGraphEncoder(
            signal_channels=in_channels,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_layers,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
        )
        self.decoder = TimeConditionedForecastDecoder(self.encoder.hidden_dim)

    def forward(
        self,
        node_series: torch.Tensor,
        edge_index: torch.Tensor,
        acquisition_times: torch.Tensor,
        baseline_observed: torch.Tensor,
        target_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compatibility for the historical Duke smoke call
        # model(x, edge, input_times, target_times). Production UChicago code
        # passes both validity and target times explicitly.
        if target_times is None:
            target_times = baseline_observed
            baseline_observed = torch.ones(
                node_series.shape[0], dtype=torch.bool, device=node_series.device
            )
        embedding = self.encoder(
            node_series, edge_index, acquisition_times, baseline_observed
        )
        target_horizon = target_times - acquisition_times[-1]
        return self.decoder(embedding, target_horizon)


class PerNodeForecaster(nn.Module):
    """Graph-free matched encoder followed by the same forecast decoder."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
        encoder: PerNodeTemporalEncoder | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or PerNodeTemporalEncoder(
            signal_channels=in_channels,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_layers,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
        )
        self.decoder = TimeConditionedForecastDecoder(self.encoder.hidden_dim)

    def forward(
        self,
        node_series: torch.Tensor,
        acquisition_times: torch.Tensor,
        baseline_observed: torch.Tensor,
        target_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_times is None:
            target_times = baseline_observed
            baseline_observed = torch.ones(
                node_series.shape[0], dtype=torch.bool, device=node_series.device
            )
        embedding = self.encoder(node_series, acquisition_times, baseline_observed)
        target_horizon = target_times - acquisition_times[-1]
        return self.decoder(embedding, target_horizon)


__all__ = [
    "ContrastForecastGNN",
    "PerNodeForecaster",
    "PerNodeTemporalEncoder",
    "TemporalGraphEncoder",
    "TimeConditionedForecastDecoder",
]
