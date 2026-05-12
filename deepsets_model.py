"""Minimal Deep Sets classifier for tumor-local vessel point sets."""

from __future__ import annotations

import torch
from torch import nn

POOLING_CHOICES = (
    "mean",
    "max",
    "sum",
    "mean_max",
    "mean_max_logcount",
    "attention",
    "attention_logcount",
)


def _pooling_width(pooling: str, latent_dim: int) -> int:
    """Return the rho input width implied by a pooling variant."""
    widths = {
        "mean": latent_dim,
        "max": latent_dim,
        "sum": latent_dim,
        "mean_max": 2 * latent_dim,
        "mean_max_logcount": (2 * latent_dim) + 1,
        "attention": latent_dim,
        "attention_logcount": latent_dim + 1,
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
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to a 2D tensor."""
        return self.net(x)


class AttentionPool(nn.Module):
    """Per-point attention pooling with per-case softmax normalization.

    A small MLP maps each point embedding to a scalar score; scores are
    softmaxed within each case (the case grouping is conveyed via the
    ``batch_index`` argument as in ``DeepSetsClassifier._pool``); the
    output is the score-weighted sum of point embeddings.

    The operation is permutation-invariant by construction: scoring is
    point-wise, the softmax denominator is a per-case sum which is
    permutation-invariant, and the weighted sum is permutation-invariant
    over points within a case.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        attention_hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if attention_hidden_dim <= 0:
            raise ValueError("attention_hidden_dim must be positive")
        self.latent_dim = latent_dim
        self.scorer = nn.Sequential(
            nn.Linear(latent_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(attention_hidden_dim, 1, bias=False),
        )

    @staticmethod
    def _segmented_softmax(
        scores: torch.Tensor, batch_index: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Numerically stable softmax over points grouped by batch_index.

        torch 1.11 lacks the ``include_self`` flag on ``scatter_reduce``; we use
        a per-case Python loop over the (typically small, ~8) batch dimension to
        compute per-case maxima for the softmax shift. This mirrors the existing
        per-case loop already used for ``pooled_max`` in ``DeepSetsClassifier._pool``.
        """
        if scores.numel() == 0:
            return scores
        per_case_max = torch.zeros(
            batch_size, dtype=scores.dtype, device=scores.device
        )
        for i in range(batch_size):
            mask = batch_index == i
            if mask.any():
                per_case_max[i] = scores[mask].max()
        shifted = scores - per_case_max[batch_index]
        exp_scores = torch.exp(shifted)
        denom = torch.zeros(
            batch_size, dtype=exp_scores.dtype, device=exp_scores.device
        )
        denom.index_add_(0, batch_index, exp_scores)
        eps = torch.finfo(exp_scores.dtype).tiny
        denom = denom.clamp_min(eps)
        return exp_scores / denom[batch_index]

    def forward(
        self,
        encoded: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return one (batch_size, latent_dim) weighted-sum tensor per case."""
        if encoded.shape[1] != self.latent_dim:
            raise ValueError(
                f"AttentionPool expects latent_dim={self.latent_dim}, "
                f"got {encoded.shape[1]}"
            )
        scores = self.scorer(encoded).squeeze(-1)
        weights = self._segmented_softmax(scores, batch_index, batch_size)
        weighted = encoded * weights.unsqueeze(-1)
        pooled = torch.zeros(
            (batch_size, self.latent_dim),
            dtype=encoded.dtype,
            device=encoded.device,
        )
        pooled.index_add_(0, batch_index, weighted)
        return pooled


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
        use_layer_norm: bool = False,
        attention_hidden_dim: int = 32,
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
        self.latent_dim = latent_dim
        self.phi = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=phi_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.attention_pool: AttentionPool | None
        if pooling in ("attention", "attention_logcount"):
            self.attention_pool = AttentionPool(
                latent_dim=latent_dim,
                attention_hidden_dim=attention_hidden_dim,
                dropout=dropout,
            )
        else:
            self.attention_pool = None
        rho_input_dim = _pooling_width(pooling, latent_dim)
        self.rho = MLP(
            input_dim=rho_input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=rho_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def _pool(self, encoded: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Aggregate point embeddings into one vector per case."""
        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        latent_dim = encoded.shape[1]

        needs_sum = self.pooling in ("sum", "mean", "mean_max", "mean_max_logcount")
        needs_max = self.pooling in ("max", "mean_max", "mean_max_logcount")
        needs_logcount = self.pooling in ("mean_max_logcount", "attention_logcount")
        needs_attention = self.pooling in ("attention", "attention_logcount")

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

        if needs_attention:
            assert self.attention_pool is not None
            chunks.append(
                self.attention_pool(encoded, batch_index, batch_size)
            )

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
