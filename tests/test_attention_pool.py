"""Tests for AttentionPool and the attention-pooling DeepSetsClassifier path."""

from __future__ import annotations

import pytest
import torch

from deepsets_model import (
    POOLING_CHOICES,
    AttentionPool,
    DeepSetsClassifier,
    _pooling_width,
)

ATTENTION_VARIANTS = ("attention", "attention_logcount")
INPUT_DIM = 4
HIDDEN_DIM = 8
LATENT_DIM = 8
ATTENTION_HIDDEN_DIM = 16
POINTS_PER_CASE = (5, 3, 7)


def _make_batch(
    input_dim: int = INPUT_DIM,
    points_per_case: tuple[int, ...] = POINTS_PER_CASE,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a deterministic batch of variable-length point sets."""
    generator = torch.Generator().manual_seed(seed)
    x_chunks = [
        torch.randn(n, input_dim, generator=generator) for n in points_per_case
    ]
    batch_index_chunks = [
        torch.full((n,), i, dtype=torch.long) for i, n in enumerate(points_per_case)
    ]
    return torch.cat(x_chunks, dim=0), torch.cat(batch_index_chunks, dim=0)


def test_attention_in_pooling_choices() -> None:
    """The new pooling variants must be in the public tuple."""
    for variant in ATTENTION_VARIANTS:
        assert variant in POOLING_CHOICES


@pytest.mark.parametrize("pooling", ATTENTION_VARIANTS)
def test_pooling_width_matches_rho_input(pooling: str) -> None:
    """_pooling_width must agree with the rho input layer for attention variants."""
    expected = _pooling_width(pooling, LATENT_DIM)
    if pooling == "attention":
        assert expected == LATENT_DIM
    else:
        assert expected == LATENT_DIM + 1

    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        pooling=pooling,
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
    )
    assert model.rho.net[0].in_features == expected


@pytest.mark.parametrize("pooling", ATTENTION_VARIANTS)
def test_attention_module_attached_when_used(pooling: str) -> None:
    """attention_pool submodule should only exist for attention variants."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        pooling=pooling,
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
    )
    assert isinstance(model.attention_pool, AttentionPool)


@pytest.mark.parametrize("pooling", ("mean", "mean_max_logcount"))
def test_attention_module_not_attached_otherwise(pooling: str) -> None:
    """Non-attention variants must not own an AttentionPool submodule."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, pooling=pooling
    )
    assert model.attention_pool is None


@pytest.mark.parametrize("pooling", ATTENTION_VARIANTS)
def test_forward_output_shape(pooling: str) -> None:
    """Attention pooling should yield one logit per case."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        pooling=pooling,
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
    )
    x, batch_index = _make_batch()
    logits = model(x, batch_index)
    assert logits.shape == (len(POINTS_PER_CASE),)


@pytest.mark.parametrize("pooling", ATTENTION_VARIANTS)
def test_permutation_invariance_within_case(pooling: str) -> None:
    """Shuffling points inside each case must not change per-case logits."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        pooling=pooling,
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
    )
    model.eval()
    x, batch_index = _make_batch(seed=123)

    permutation = torch.cat(
        [
            torch.randperm(int(n), generator=torch.Generator().manual_seed(i + 1))
            + int(sum(POINTS_PER_CASE[:i]))
            for i, n in enumerate(POINTS_PER_CASE)
        ]
    )
    with torch.no_grad():
        logits_orig = model(x, batch_index)
        logits_perm = model(x[permutation], batch_index[permutation])

    torch.testing.assert_close(logits_orig, logits_perm, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("pooling", ATTENTION_VARIANTS)
def test_permutation_invariance_across_case_order(pooling: str) -> None:
    """Permuting whole cases reorders logits but does not alter values."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        pooling=pooling,
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
    )
    model.eval()
    x, batch_index = _make_batch(seed=7)
    new_order = (2, 0, 1)

    new_batch_chunks: list[torch.Tensor] = []
    new_x_chunks: list[torch.Tensor] = []
    for new_idx, original_case in enumerate(new_order):
        case_mask = batch_index == original_case
        new_x_chunks.append(x[case_mask])
        new_batch_chunks.append(
            torch.full((int(case_mask.sum()),), new_idx, dtype=torch.long)
        )
    new_x = torch.cat(new_x_chunks, dim=0)
    new_batch_index = torch.cat(new_batch_chunks, dim=0)

    with torch.no_grad():
        logits_orig = model(x, batch_index)
        logits_reordered = model(new_x, new_batch_index)

    expected = logits_orig[list(new_order)]
    torch.testing.assert_close(logits_reordered, expected, atol=1e-5, rtol=1e-5)


def test_attention_weights_sum_to_one_per_case() -> None:
    """The internal softmax must produce per-case weights summing to 1."""
    torch.manual_seed(0)
    encoded = torch.randn(sum(POINTS_PER_CASE), LATENT_DIM)
    batch_index = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(POINTS_PER_CASE)]
    )
    pool = AttentionPool(
        latent_dim=LATENT_DIM, attention_hidden_dim=ATTENTION_HIDDEN_DIM
    )
    scores = pool.scorer(encoded).squeeze(-1)
    weights = AttentionPool._segmented_softmax(
        scores, batch_index, batch_size=len(POINTS_PER_CASE)
    )
    for case_idx, n in enumerate(POINTS_PER_CASE):
        mask = batch_index == case_idx
        assert mask.sum().item() == n
        case_sum = weights[mask].sum().item()
        assert case_sum == pytest.approx(1.0, abs=1e-5)


def test_attention_pool_rejects_dim_mismatch() -> None:
    """Passing the wrong latent_dim to AttentionPool must raise."""
    pool = AttentionPool(latent_dim=LATENT_DIM, attention_hidden_dim=ATTENTION_HIDDEN_DIM)
    bad = torch.randn(sum(POINTS_PER_CASE), LATENT_DIM + 3)
    batch_index = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(POINTS_PER_CASE)]
    )
    with pytest.raises(ValueError, match="latent_dim"):
        pool(bad, batch_index, batch_size=len(POINTS_PER_CASE))


def test_gradient_flows_through_attention() -> None:
    """Backprop must reach the attention scorer parameters."""
    model = DeepSetsClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        pooling="attention",
        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
    )
    x, batch_index = _make_batch(seed=42)
    logits = model(x, batch_index)
    loss = logits.sum()
    loss.backward()
    assert model.attention_pool is not None
    grads_present = [
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.attention_pool.scorer.parameters()
    ]
    assert all(grads_present)
