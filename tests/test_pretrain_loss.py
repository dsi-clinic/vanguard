"""Tests for ``gnn.pretrain.loss.masked_mae``.

Locks the two properties the masking contract promises: a mask that only needs
broadcasting (e.g. one flag per node, shape ``(N, 1)``) must give the *same* loss
as the equivalent fully-expanded ``(N, target_len)`` mask, and it must apply
regardless of which device the mask started on. Both were broken when the
denominator was computed from the un-broadcast mask (loss inflated by
``target_len``) and the mask was never moved to the prediction's device.
"""

from __future__ import annotations

import pytest
import torch

from gnn.pretrain.loss import masked_mae


def test_no_mask_is_plain_mean() -> None:
    """With no mask the loss is the plain elementwise mean absolute error."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(pred)
    assert masked_mae(pred, target).item() == pytest.approx(2.5)


def test_broadcast_mask_matches_expanded_mask() -> None:
    """A ``(N, 1)`` node mask equals its expanded ``(N, target_len)`` form.

    This is the core regression: the old denominator counted each node once while
    the numerator summed over all ``target_len`` frames, so the ``(N, 1)`` mask
    scored ``target_len`` times higher than the identical expanded mask.
    """
    pred = torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])
    target = torch.zeros_like(pred)
    node_mask = torch.tensor([[1.0], [0.0], [1.0]])  # keep nodes 0 and 2

    broadcast_loss = masked_mae(pred, target, node_mask)
    expanded_loss = masked_mae(pred, target, node_mask.expand_as(pred))

    assert broadcast_loss.item() == pytest.approx(expanded_loss.item())


def test_masked_value_is_mean_over_kept_elements() -> None:
    """The masked loss is the mean error over exactly the kept elements."""
    pred = torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])
    target = torch.zeros_like(pred)
    node_mask = torch.tensor([[1.0], [0.0], [1.0]])  # nodes 0 and 2 kept

    # Kept errors: [1, 3, 9, 11] -> mean 6.0. The dropped node contributes to
    # neither numerator nor denominator.
    assert masked_mae(pred, target, node_mask).item() == pytest.approx(6.0)


def test_all_masked_raises() -> None:
    """An all-zero mask is a caller bug, not a 0-loss no-op."""
    pred = torch.ones(3, 2)
    target = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="zero elements"):
        masked_mae(pred, target, torch.zeros(3, 1))


def test_shape_mismatch_raises() -> None:
    """pred/target shape mismatch fails loudly before any masking."""
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_mae(torch.ones(3, 2), torch.ones(3, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_mask_applies_to_cuda_prediction() -> None:
    """A CPU mask applies to a CUDA prediction instead of raising a device error."""
    pred = torch.tensor([[1.0, 3.0], [5.0, 7.0]], device="cuda")
    target = torch.zeros_like(pred)
    cpu_mask = torch.tensor([[1.0], [0.0]])  # deliberately left on CPU

    loss = masked_mae(pred, target, cpu_mask)

    # Node 0 kept: errors [1, 3] -> mean 2.0.
    assert loss.item() == pytest.approx(2.0)
