"""Minimal Deep Sets classifier for tumor-local vessel point sets."""

from __future__ import annotations

import torch
from torch import nn

POOLING_CHOICES = ("mean", "max", "sum", "mean_max", "mean_max_logcount")


def _pooling_width(pooling: str, latent_dim: int) -> int:
    """Return the rho input width implied by a pooling variant."""
    widths = {
        "mean": latent_dim,
        "max": latent_dim,
        "sum": latent_dim,
        "mean_max": 2 * latent_dim,
        "mean_max_logcount": (2 * latent_dim) + 1,
    }
    return widths[pooling]


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
        pooling: str = "mean_max_logcount",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if pooling not in POOLING_CHOICES:
            raise ValueError(
                f"Unknown pooling {pooling!r}. Choose from {POOLING_CHOICES}."
            )
        latent_dim = hidden_dim if latent_dim is None else latent_dim
        self.pooling = pooling
        self.phi = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=phi_layers,
            dropout=dropout,
        )
        rho_input_dim = _pooling_width(pooling, latent_dim)
        self.rho = MLP(
            input_dim=rho_input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=rho_layers,
            dropout=dropout,
        )

    def _pool(self, encoded: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Aggregate point embeddings into one vector per case."""
        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        latent_dim = encoded.shape[1]

        needs_sum = self.pooling in ("sum", "mean", "mean_max", "mean_max_logcount")
        needs_max = self.pooling in ("max", "mean_max", "mean_max_logcount")
        needs_logcount = self.pooling == "mean_max_logcount"

        chunks: list[torch.Tensor] = []

        if needs_sum:
            pooled_sum = torch.zeros(
                (batch_size, latent_dim),
                dtype=encoded.dtype,
                device=encoded.device,
            )
            pooled_sum.index_add_(0, batch_index, encoded)

            if self.pooling == "sum":
                chunks.append(pooled_sum)
            else:
                counts = torch.bincount(batch_index, minlength=batch_size).clamp_min(1)
                counts = counts.to(
                    device=encoded.device, dtype=encoded.dtype
                ).unsqueeze(1)
                chunks.append(pooled_sum / counts)

        if needs_max:
            pooled_max = torch.full(
                (batch_size, latent_dim),
                fill_value=torch.finfo(encoded.dtype).min,
                dtype=encoded.dtype,
                device=encoded.device,
            )
            for i in range(batch_size):
                mask = batch_index == i
                if mask.any():
                    pooled_max[i] = encoded[mask].max(dim=0)[0]
            chunks.append(pooled_max)

        if needs_logcount:
            counts = torch.bincount(batch_index, minlength=batch_size).clamp_min(1)
            counts = counts.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(1)
            chunks.append(torch.log(counts))

        return torch.cat(chunks, dim=1)

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Encode each point, pool by case, and classify."""
        encoded = self.phi(x)
        pooled = self._pool(encoded, batch_index)
        return self.rho(pooled).view(-1)
