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

Both classifiers optionally accept graph-level covariates (``graph_dim`` > 0,
``gnn.clinical``'s clinical/demographic feature vector, see
``gnn.data_loader``'s ``graph_features``): these aren't per-node/per-edge
signal, so they're concatenated onto the pooled graph embedding *after*
message passing, immediately before the final linear head, not fed into the
conv stack. ``graph_dim=0`` (the default) is fully backward compatible -- no
``graph_features`` argument is required or used.

See ``gnn/README.md`` and ``gnn/DESIGN_segment_graph.md``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# Readout variants for the whole-graph pooling step. A GCNConv stack followed by
# global_mean_pool collapses every node to its *average*, which discards the
# distributional tails (peak/extreme per-node signal) -- the Tier-0 sweep found
# a tabular model on node-feature *quantiles* still beats the mean-pool GNN, so
# concatenating max (and optionally sum) restores that information. See
# gnn/PLAN_advanced_modeling.md decision D2.1. Values are the number of pooled
# vectors concatenated, which scales the classifier's input width.
POOLING_WIDTHS: dict[str, int] = {"mean": 1, "mean_max": 2, "mean_max_sum": 3}


def _graph_readout(
    h: torch.Tensor,
    batch_index: torch.Tensor,
    pooling: str,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate the pooled node embeddings selected by ``pooling``.

    ``"mean"`` reproduces the original single ``global_mean_pool`` readout;
    ``"mean_max"``/``"mean_max_sum"`` append global max (and add) pools so the
    graph embedding keeps extreme/aggregate node signal, not just the average.

    ``node_mask`` (shape ``(num_nodes,)``, True = keep) restricts every pool to
    the marked nodes. Nodes it excludes still took part in message passing --
    they shaped their neighbours' embeddings in the conv stack -- but contribute
    nothing to the graph vector, so they cannot reach the graph-level loss. This
    is the whole-graph analogue of masking a per-node loss: pooling is the last
    point at which nodes are individually addressable.

    The pools are computed by weighting rather than by dropping rows, so every
    graph in the batch keeps its slot in the output even when some of its nodes
    are masked out. A graph with no kept nodes has no defined readout and raises
    rather than emitting NaN.
    """
    if node_mask is None:
        parts = [global_mean_pool(h, batch_index)]
        if pooling in ("mean_max", "mean_max_sum"):
            parts.append(global_max_pool(h, batch_index))
        if pooling == "mean_max_sum":
            parts.append(global_add_pool(h, batch_index))
        return torch.cat(parts, dim=1)

    if node_mask.shape != batch_index.shape:
        raise ValueError(
            f"node_mask shape {tuple(node_mask.shape)} does not match "
            f"batch_index {tuple(batch_index.shape)}"
        )
    keep = node_mask.to(dtype=torch.bool)
    weights = keep.to(h.dtype).unsqueeze(-1)
    kept_per_graph = global_add_pool(weights, batch_index)
    if bool((kept_per_graph == 0).any()):
        empty = int((kept_per_graph == 0).sum())
        raise ValueError(
            f"{empty} graph(s) in this batch have no unmasked nodes, so their "
            "readout is undefined. Every graph must retain at least one node "
            "with an observed baseline."
        )
    masked_sum = global_add_pool(h * weights, batch_index)
    parts = [masked_sum / kept_per_graph]
    if pooling in ("mean_max", "mean_max_sum"):
        # Excluded rows must lose the max outright, so push them to the dtype's
        # floor rather than to 0 -- a genuine all-negative embedding would
        # otherwise be beaten by a masked node.
        floored = h.masked_fill(~keep.unsqueeze(-1), torch.finfo(h.dtype).min)
        parts.append(global_max_pool(floored, batch_index))
    if pooling == "mean_max_sum":
        parts.append(masked_sum)
    return torch.cat(parts, dim=1)


class GCNClassifier(nn.Module):
    """GCNConv stack -> graph readout -> linear head, one logit per graph.

    ``pooling`` selects the readout (see ``POOLING_WIDTHS``); the default
    ``"mean"`` is the original single mean-pool, fully backward compatible.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        graph_dim: int = 0,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if graph_dim < 0:
            raise ValueError("graph_dim must be >= 0")
        if pooling not in POOLING_WIDTHS:
            raise ValueError(
                f"Unknown pooling {pooling!r}. Choose from {sorted(POOLING_WIDTHS)}."
            )
        self.graph_dim = graph_dim
        self.pooling = pooling
        self.convs = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        pooled_dim = hidden_dim * POOLING_WIDTHS[pooling]
        self.classifier = nn.Linear(pooled_dim + graph_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        graph_features: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one raw logit per graph in the batch (shape ``(batch_size,)``).

        ``graph_features`` (shape ``(batch_size, graph_dim)``) is required
        when ``graph_dim > 0`` and ignored otherwise. ``node_mask`` restricts the
        readout to the marked nodes without removing them from message passing
        (see ``_graph_readout``).
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        pooled = _graph_readout(h, batch_index, self.pooling, node_mask)
        if self.graph_dim > 0:
            if graph_features is None:
                raise ValueError("graph_features is required when graph_dim > 0")
            pooled = torch.cat([pooled, graph_features], dim=1)
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
        graph_dim: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if edge_dim <= 0:
            raise ValueError("edge_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if graph_dim < 0:
            raise ValueError("graph_dim must be >= 0")
        self.graph_dim = graph_dim
        self.convs = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(EdgeConditionedConv(in_dim, hidden_dim, edge_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim + graph_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
        graph_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one raw logit per graph in the batch (shape ``(batch_size,)``).

        ``graph_features`` (shape ``(batch_size, graph_dim)``) is required
        when ``graph_dim > 0`` and ignored otherwise.
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = torch.relu(h)
            h = self.dropout(h)
        pooled = global_mean_pool(h, batch_index)
        if self.graph_dim > 0:
            if graph_features is None:
                raise ValueError("graph_features is required when graph_dim > 0")
            pooled = torch.cat([pooled, graph_features], dim=1)
        return self.classifier(pooled).view(-1)
