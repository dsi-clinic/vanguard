"""Minimal GCN classifier for whole-graph vessel-centerline classification.

MVP architecture for the GNN track: a stack of ``GCNConv`` layers, a global
mean-pool readout to one embedding per graph, and a linear classification
head. No edge features, no attention, no segment-level pooling -- those are
the deliberate next steps once this wiring is validated end-to-end (see
``gnn/README.md``), mirroring how ``deepsets/model.py`` started as a single
pooling variant before growing the others.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNClassifier(nn.Module):
    """GCNConv stack -> global mean pool -> linear head, one logit per graph."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.convs = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return one raw logit per graph in the batch (shape ``(batch_size,)``)."""
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        pooled = global_mean_pool(h, batch_index)
        return self.classifier(pooled).view(-1)
