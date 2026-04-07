"""Minimal Deep Sets classifier for tumor-local vessel point sets."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Simple ReLU MLP used for phi and rho."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to a 2D tensor."""
        return self.net(x)


class DeepSetsClassifier(nn.Module):
    """Permutation-invariant classifier for variable-length point sets."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int | None = None,
        phi_layers: int = 2,
        rho_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        latent_dim = hidden_dim if latent_dim is None else latent_dim
        self.phi = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=phi_layers,
            dropout=dropout,
        )
        self.rho = MLP(
            input_dim=(2 * latent_dim) + 1,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=rho_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Encode each point, pool by case, and classify."""
        encoded = self.phi(x)
        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        pooled_sum = torch.zeros(
            (batch_size, encoded.shape[1]),
            dtype=encoded.dtype,
            device=encoded.device,
        )
        pooled_sum.index_add_(0, batch_index, encoded)
        counts_raw = torch.bincount(batch_index, minlength=batch_size)
        counts = counts_raw.clamp_min(1)
        counts = counts.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(1)
        pooled_mean = pooled_sum / counts

        idx = batch_index.unsqueeze(1).expand_as(encoded).long()
        if getattr(encoded, "scatter_reduce_", None) is not None:
            pooled_max = torch.full(
                (batch_size, encoded.shape[1]),
                fill_value=torch.finfo(encoded.dtype).min,
                dtype=encoded.dtype,
                device=encoded.device,
            )
            pooled_max.scatter_reduce_(
                0,
                idx,
                encoded,
                reduce="amax",
                include_self=True,
            )
        else:
            pooled_max = torch.scatter_reduce(
                encoded,
                0,
                idx,
                reduce="amax",
                output_size=batch_size,
            )
            empty_rows = counts_raw == 0
            if empty_rows.any():
                pooled_max = pooled_max.clone()
                pooled_max[empty_rows] = torch.finfo(encoded.dtype).min
        log_counts = torch.log(counts)
        pooled = torch.cat([pooled_mean, pooled_max, log_counts], dim=1)
        return self.rho(pooled).view(-1)
