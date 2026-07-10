"""Smoke test for the MVP GCN classifier."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data  # noqa: E402

from gnn.model import EdgeGNNClassifier, GCNClassifier  # noqa: E402


def _make_batch() -> Batch:
    """Two tiny graphs (a triangle and a single edge) with 2 node features."""
    graph_a = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]),
        y=torch.tensor([1.0]),
    )
    graph_b = Data(
        x=torch.randn(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0.0]),
    )
    return Batch.from_data_list([graph_a, graph_b])


def test_forward_returns_one_logit_per_graph() -> None:
    """A batch of 2 graphs yields a (2,) logit tensor."""
    batch = _make_batch()
    model = GCNClassifier(input_dim=2, hidden_dim=8, num_layers=2, dropout=0.0)
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_rejects_invalid_input_dim() -> None:
    """input_dim must be positive."""
    with pytest.raises(ValueError, match="input_dim"):
        GCNClassifier(input_dim=0)


def test_rejects_invalid_num_layers() -> None:
    """num_layers must be at least 1."""
    with pytest.raises(ValueError, match="num_layers"):
        GCNClassifier(input_dim=2, num_layers=0)


def _make_edge_batch() -> Batch:
    """Two tiny graphs with 2 node features and 3 edge features each."""
    graph_a = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]),
        edge_attr=torch.randn(6, 3),
        y=torch.tensor([1.0]),
    )
    graph_b = Data(
        x=torch.randn(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.randn(2, 3),
        y=torch.tensor([0.0]),
    )
    return Batch.from_data_list([graph_a, graph_b])


def test_edge_model_forward_returns_one_logit_per_graph() -> None:
    """EdgeGNNClassifier over a 2-graph batch yields a (2,) logit tensor."""
    batch = _make_edge_batch()
    model = EdgeGNNClassifier(
        input_dim=2, edge_dim=3, hidden_dim=8, num_layers=2, dropout=0.0
    )
    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_edge_model_uses_edge_features() -> None:
    """Changing edge_attr changes the output -- edges genuinely condition the model."""
    batch = _make_edge_batch()
    model = EdgeGNNClassifier(input_dim=2, edge_dim=3, hidden_dim=8, dropout=0.0)
    model.eval()
    baseline = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    perturbed = model(batch.x, batch.edge_index, batch.edge_attr + 5.0, batch.batch)
    assert not torch.allclose(baseline, perturbed)


def test_edge_model_rejects_invalid_edge_dim() -> None:
    """edge_dim must be positive."""
    with pytest.raises(ValueError, match="edge_dim"):
        EdgeGNNClassifier(input_dim=2, edge_dim=0)
