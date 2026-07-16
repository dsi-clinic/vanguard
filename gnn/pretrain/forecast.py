"""Forecast windowing: split a node's contrast series into input vs. target.

The pretext task (design doc §3): given a per-node contrast series of length
``T``, the model sees the first ``input_len`` frames (the **input horizon**) and
must predict the next ``target_len`` frames (the **target horizon**) at every
node. This module owns that split and the shape/validation contract; it is
deliberately model-agnostic so the GNN forecaster, the graph-free baseline, and
the trivial baselines all consume identical ``(input, target)`` tensors.

``input_len`` / ``target_len`` are *not* hard-coded here -- per the doc they are
chosen empirically by validation regression loss once real data fixes ``T``. The
``ForecastHorizon`` dataclass just carries a chosen pair and validates it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_NDIM_2D = 2  # (num_nodes, T)


@dataclass(frozen=True)
class ForecastHorizon:
    """A chosen (input_len, target_len) split of a length-``T`` series.

    ``input_len`` frames are shown to the model; ``target_len`` following frames
    are forecast. ``input_len + target_len`` must not exceed the series length
    ``T`` at split time (checked in ``split_forecast_window``). Both must be >= 1.
    """

    input_len: int
    target_len: int

    def __post_init__(self) -> None:
        """Validate that both horizon lengths are >= 1."""
        if self.input_len < 1:
            raise ValueError(f"input_len must be >= 1, got {self.input_len}")
        if self.target_len < 1:
            raise ValueError(f"target_len must be >= 1, got {self.target_len}")

    @property
    def window(self) -> int:
        """Total frames consumed (input + target)."""
        return self.input_len + self.target_len


def split_forecast_window(
    series: torch.Tensor, horizon: ForecastHorizon
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``(N, T)`` node series into ``(N, input_len)`` and ``(N, target_len)``.

    Takes the first ``horizon.window`` frames and cuts them at ``input_len``:
    the input is frames ``[0, input_len)``, the target is frames
    ``[input_len, input_len + target_len)``. Any frames beyond the window are
    ignored (a deterministic policy -- we forecast the horizon immediately after
    the input, not a random offset, to match the design doc's framing). Fails
    loudly if the series is too short rather than padding silently: variable-
    length handling is an open, data-blocked decision (design doc §8b), not
    something to paper over here.
    """
    if series.ndim != _NDIM_2D:
        raise ValueError(f"series must be 2D (N, T); got shape {tuple(series.shape)}")
    num_timepoints = series.shape[1]
    if num_timepoints < horizon.window:
        raise ValueError(
            f"series has T={num_timepoints} frames but the horizon needs "
            f"{horizon.window} (input_len={horizon.input_len} + "
            f"target_len={horizon.target_len}). Short-series handling is a "
            "data-blocked decision (design doc §8b), not defaulted here."
        )
    inputs = series[:, : horizon.input_len]
    targets = series[:, horizon.input_len : horizon.window]
    return inputs, targets
