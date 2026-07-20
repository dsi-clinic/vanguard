"""Smoke test for the MVP GCN classifier."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data  # noqa: E402

from gnn.model import (  # noqa: E402
    POOLING_WIDTHS,
    EdgeGNNClassifier,
    GCNClassifier,
)


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


@pytest.mark.parametrize("pooling", list(POOLING_WIDTHS))
def test_pooling_variants_forward_and_classifier_width(pooling: str) -> None:
    """Each readout yields a (2,) logit and scales the head input by its width."""
    batch = _make_batch()
    model = GCNClassifier(
        input_dim=2, hidden_dim=8, num_layers=2, dropout=0.0, pooling=pooling
    )
    assert model.classifier.in_features == 8 * POOLING_WIDTHS[pooling]
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_rejects_invalid_pooling() -> None:
    """Pooling must be one of the known readout variants."""
    with pytest.raises(ValueError, match="Unknown pooling"):
        GCNClassifier(input_dim=2, pooling="bogus")


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


def _make_batch_with_graph_features(graph_dim: int) -> Batch:
    """Same two tiny graphs as ``_make_batch``, plus a per-graph feature vector."""
    graph_a = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]),
        y=torch.tensor([1.0]),
        graph_features=torch.randn(1, graph_dim),
    )
    graph_b = Data(
        x=torch.randn(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0.0]),
        graph_features=torch.randn(1, graph_dim),
    )
    return Batch.from_data_list([graph_a, graph_b])


def test_graph_dim_zero_rejects_negative() -> None:
    """graph_dim must be >= 0."""
    with pytest.raises(ValueError, match="graph_dim"):
        GCNClassifier(input_dim=2, graph_dim=-1)


def test_graph_dim_positive_requires_graph_features() -> None:
    """graph_dim > 0 without graph_features at forward time raises, not silently ignores."""
    batch = _make_batch()
    model = GCNClassifier(input_dim=2, hidden_dim=8, graph_dim=3, dropout=0.0)
    with pytest.raises(ValueError, match="graph_features"):
        model(batch.x, batch.edge_index, batch.batch)


def test_graph_dim_concatenates_and_influences_output() -> None:
    """graph_features are actually used: changing them changes the logits."""
    batch = _make_batch_with_graph_features(graph_dim=3)
    model = GCNClassifier(input_dim=2, hidden_dim=8, graph_dim=3, dropout=0.0)
    model.eval()
    baseline = model(batch.x, batch.edge_index, batch.batch, batch.graph_features)
    perturbed = model(
        batch.x, batch.edge_index, batch.batch, batch.graph_features + 5.0
    )
    assert baseline.shape == (2,)
    assert not torch.allclose(baseline, perturbed)


def test_edge_model_graph_dim_concatenates_and_influences_output() -> None:
    """Same graph_dim concat path for EdgeGNNClassifier."""
    batch = _make_batch_with_graph_features(graph_dim=2)
    batch.edge_attr = torch.randn(int(batch.edge_index.shape[1]), 3)
    model = EdgeGNNClassifier(
        input_dim=2, edge_dim=3, hidden_dim=8, graph_dim=2, dropout=0.0
    )
    model.eval()
    baseline = model(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_features
    )
    perturbed = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.graph_features + 5.0,
    )
    assert baseline.shape == (2,)
    assert not torch.allclose(baseline, perturbed)
