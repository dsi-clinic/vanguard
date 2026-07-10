"""GNN classifiers for whole-graph vessel-centerline classification.

Two architectures, one per graph representation:

- ``GCNClassifier`` -- a stack of ``GCNConv`` layers, global mean-pool, linear
  head. Used by the voxel (one node per voxel) and segment (one node per
  segment, line graph) modes, whose signal lives entirely on node features.
- ``EdgeGNNClassifier`` -- the edge-aware counterpart for the junction mode
  (segment-as-edge, Option A), where each segment's summary lives on
  ``edge_attr``. ``GCNConv`` ignores edge features, so it uses
  ``EdgeConditionedConv``, a small message-passing layer that conditions each
  message on the connecting edge's features.

See ``gnn/README.md`` and ``gnn/DESIGN_segment_graph.md``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool


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


class EdgeConditionedConv(MessagePassing):
    """Message-passing layer that conditions each message on the edge features.

    For a directed edge ``j -> i`` with edge features ``e_ji``, the message is
    ``W_msg [x_j ; e_ji]`` (neighbour features concatenated with edge features),
    mean-aggregated over neighbours, plus a self term ``W_self x_i``. This is the
    minimal way to let the segment summary on ``edge_attr`` actually influence
    the junction-node embeddings -- ``GCNConv`` would discard it. Kept
    deliberately small and explicit (rather than, say, ``NNConv``) to match the
    MVP altitude of ``GCNClassifier``.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int) -> None:
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_msg = nn.Linear(in_dim + edge_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Return updated node features (shape ``(num_nodes, out_dim)``)."""
        return self.lin_self(x) + self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Message from neighbour ``j`` along its edge: ``W_msg [x_j ; e_ji]``."""
        return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))


class EdgeGNNClassifier(nn.Module):
    """EdgeConditionedConv stack -> global mean pool -> linear head.

    The edge-aware counterpart to ``GCNClassifier`` for the junction mode
    (segment-as-edge), where the per-segment summary lives on ``edge_attr``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if edge_dim <= 0:
            raise ValueError("edge_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.convs = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(EdgeConditionedConv(in_dim, hidden_dim, edge_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return one raw logit per graph in the batch (shape ``(batch_size,)``)."""
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = torch.relu(h)
            h = self.dropout(h)
        pooled = global_mean_pool(h, batch_index)
        return self.classifier(pooled).view(-1)
