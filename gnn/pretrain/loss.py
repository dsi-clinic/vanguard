"""Masked MAE loss for contrast forecasting (design doc §6).

MAE is the committed default (matching Tang et al.), robust to the heavy-tailed
enhancement distribution. The *mechanism* for masking unreliable nodes/frames
(background / low-SNR voxels, or the pipeline's no-arrival sentinels) is
implemented here; the concrete masking *threshold* is a data-blocked decision
(design doc §8b), so the mask defaults to "use everything" and is supplied by
the caller when a policy exists. No silent NaN-swallowing -- an unmasked NaN in
the target is a bug and surfaces as a NaN loss, per the repo's fail-fast rules.
"""

from __future__ import annotations

import torch


def masked_mae(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean absolute error over ``(N, target_len)`` forecasts.

    ``mask`` (optional) is a boolean or 0/1 tensor broadcastable to ``pred``:
    ``True``/1 keeps an element, ``False``/0 drops it from both the numerator and
    the denominator, so masked-out nodes/frames contribute nothing (not a
    zero-error freebie). With no mask this is a plain mean absolute error over
    every element. Raises if a mask selects zero elements -- an all-masked batch
    is a caller bug, not a 0-loss no-op.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    abs_err = (pred - target).abs()
    if mask is None:
        return abs_err.mean()
    # Broadcast the mask to the error's full shape -- and onto its device/dtype --
    # once, then use that single tensor for both numerator and denominator. A mask
    # shaped e.g. (N, 1) otherwise inflates the loss: it broadcasts across
    # target_len inside ``abs_err * mask`` (numerator), but ``mask.sum()`` on the
    # un-broadcast (N, 1) tensor counts each node only once, so the mean comes out
    # target_len times too large. The device move also lets a CPU mask apply to a
    # CUDA prediction instead of raising a device mismatch.
    mask = torch.broadcast_to(
        mask.to(device=abs_err.device, dtype=abs_err.dtype),
        abs_err.shape,
    )
    denom = mask.sum()
    if denom.item() == 0:
        raise ValueError("mask selects zero elements; nothing to compute a loss over")
    return (abs_err * mask).sum() / denom
