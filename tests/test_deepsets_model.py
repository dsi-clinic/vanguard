"""Tests for configurable pooling in DeepSetsClassifier."""

from __future__ import annotations

import pytest
import torch

from deepsets_model import POOLING_CHOICES, DeepSetsClassifier, _pooling_width

INPUT_DIM = 4
HIDDEN_DIM = 8
BATCH_SIZE = 3
POINTS_PER_CASE = (5, 3, 7)


def _make_batch(
    input_dim: int = INPUT_DIM,
    points_per_case: tuple[int, ...] = POINTS_PER_CASE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic batch of point sets with a batch_index vector."""
    x_chunks = [torch.randn(n, input_dim) for n in points_per_case]
    batch_index_chunks = [
        torch.full((n,), i, dtype=torch.long) for i, n in enumerate(points_per_case)
    ]
    return torch.cat(x_chunks, dim=0), torch.cat(batch_index_chunks, dim=0)


@pytest.mark.parametrize("pooling", POOLING_CHOICES)
def test_forward_output_shape(pooling: str) -> None:
    """Each pooling variant should produce one logit per case."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        pooling=pooling,
    )
    x, batch_index = _make_batch()
    logits = model(x, batch_index)
    assert logits.shape == (len(POINTS_PER_CASE),)


@pytest.mark.parametrize("pooling", POOLING_CHOICES)
def test_pooling_width_consistency(pooling: str) -> None:
    """The rho input dim should match _pooling_width for each variant."""
    latent_dim = HIDDEN_DIM
    expected = _pooling_width(pooling, latent_dim)
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=latent_dim,
        pooling=pooling,
    )
    rho_first_layer = model.rho.net[0]
    assert rho_first_layer.in_features == expected


def test_invalid_pooling_raises() -> None:
    """An unrecognized pooling name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown pooling"):
        DeepSetsClassifier(input_dim=INPUT_DIM, pooling="nonexistent")


@pytest.mark.parametrize("pooling", POOLING_CHOICES)
def test_gradient_flow(pooling: str) -> None:
    """Gradients should reach all parameters through every pooling path."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        pooling=pooling,
    )
    x, batch_index = _make_batch()
    logits = model(x, batch_index)
    loss = logits.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name} with pooling={pooling}"
        assert param.grad.abs().sum() > 0, (
            f"Zero gradient for {name} with pooling={pooling}"
        )


def test_default_pooling_is_mean_max_logcount() -> None:
    """Omitting the pooling argument should give the original baseline."""
    model = DeepSetsClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    assert model.pooling == "mean_max_logcount"


@pytest.mark.parametrize("pooling", POOLING_CHOICES)
def test_single_point_case(pooling: str) -> None:
    """A batch with one point per case should not crash."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        pooling=pooling,
    )
    x = torch.randn(2, INPUT_DIM)
    batch_index = torch.tensor([0, 1], dtype=torch.long)
    logits = model(x, batch_index)
    assert logits.shape == (2,)
