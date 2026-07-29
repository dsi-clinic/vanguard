"""Trivial (non-learned) forecasting baselines for the §7.i sanity check.

The design doc's first gate: on held-out data the learned forecaster must beat
these trivial predictors on regression loss, or the pretext task is learning
nothing worth transferring and we stop before any downstream work. Both take the
``(N, input_len)`` input horizon and emit a ``(N, target_len)`` forecast, the
same shape the models produce, so they plug into the identical ``masked_mae``.

These are per-node and graph-free by construction -- they are the "is the signal
even forecastable at all" floor, distinct from the *graph* ablation
(``PerNodeForecaster`` in ``gnn.pretrain.model``), which is a learned per-node
model used to test whether message passing specifically helps (§7.ii).
"""

from __future__ import annotations

import torch

_NDIM_2D = 2  # (num_nodes, input_len)


def last_frame_forecast(inputs: torch.Tensor, target_len: int) -> torch.Tensor:
    """Persistence baseline: repeat each node's last observed frame ``target_len`` times.

    ``inputs`` is ``(N, input_len)``; returns ``(N, target_len)`` where every
    forecast frame equals the last input frame. This is the "contrast stops
    changing" null model -- a forecaster that cannot beat it has learned no
    temporal dynamics.
    """
    if inputs.ndim != _NDIM_2D:
        raise ValueError(f"inputs must be 2D (N, input_len); got {tuple(inputs.shape)}")
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    last = inputs[:, -1:]
    return last.expand(-1, target_len).contiguous()


def temporal_mean_forecast(inputs: torch.Tensor, target_len: int) -> torch.Tensor:
    """Per-node temporal-mean baseline: repeat each node's input-horizon mean.

    ``inputs`` is ``(N, input_len)``; returns ``(N, target_len)`` where every
    forecast frame equals that node's mean over its observed frames. The "no
    dynamics, just the node's typical level" null model.
    """
    if inputs.ndim != _NDIM_2D:
        raise ValueError(f"inputs must be 2D (N, input_len); got {tuple(inputs.shape)}")
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    mean = inputs.mean(dim=1, keepdim=True)
    return mean.expand(-1, target_len).contiguous()
