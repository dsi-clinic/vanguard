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


def tile_forecast_windows(
    series: torch.Tensor,
    times: torch.Tensor,
    horizon: ForecastHorizon,
    *,
    baseline_frame_count: int = 1,
    keep_baseline_context: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Non-overlapping tiles over the post-baseline part of a ``(N, T)`` series.

    Instead of forecasting a single leading window (which for UChicago is all
    precontrast baseline), we tile the curve into consecutive, non-overlapping
    windows of ``horizon.window`` frames and forecast each -- the EEG-style
    windowing of Tang et al. (design review issue 3b). Every frame participates in
    at most one window and the flat baseline no longer dominates the pretext task.

    Tiling starts at frame ``baseline_frame_count - keep_baseline_context``, so:

    - the bulk of the protocol baseline is **not** tiled (the pretext task is
      blood flow, not baseline);
    - the first tile's input horizon still carries ``keep_baseline_context``
      precontrast frame(s) as wash-in onset context;
    - the full ``baseline_frame_count`` baseline frames are still used upstream to
      define S0 for the enhancement transform -- dropping them here only removes
      them as *windows*, not from the baseline estimate.

    Each returned tile is ``(inputs, target, input_times, target_times)`` with the
    times rebased to that tile's first input frame (elapsed seconds), so tiles at
    different points on an irregularly-sampled curve carry their real, differing
    cadence. Fails loudly if the post-baseline region cannot fit even one window
    (variable-length handling is a data-blocked decision, not padded here).
    """
    if series.ndim != _NDIM_2D:
        raise ValueError(f"series must be 2D (N, T); got shape {tuple(series.shape)}")
    num_timepoints = series.shape[1]
    if times.ndim != 1 or times.shape[0] != num_timepoints:
        raise ValueError(
            f"times must be 1D of length T={num_timepoints}; got {tuple(times.shape)}"
        )
    if not 1 <= baseline_frame_count < num_timepoints:
        raise ValueError(
            f"baseline_frame_count must be in [1, T={num_timepoints}); "
            f"got {baseline_frame_count}"
        )
    if not 0 <= keep_baseline_context <= baseline_frame_count:
        raise ValueError(
            f"keep_baseline_context must be in [0, baseline_frame_count="
            f"{baseline_frame_count}]; got {keep_baseline_context}"
        )
    start = baseline_frame_count - keep_baseline_context
    window = horizon.window
    if num_timepoints - start < window:
        raise ValueError(
            f"post-baseline region has {num_timepoints - start} frames "
            f"(T={num_timepoints}, start={start}) but the horizon needs {window} "
            f"(input_len={horizon.input_len} + target_len={horizon.target_len}). "
            "Short-series handling is a data-blocked decision, not padded here."
        )
    tiles: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for offset in range(start, num_timepoints - window + 1, window):
        cut = offset + horizon.input_len
        end = offset + window
        tile_times = times[offset:end] - times[offset]
        tiles.append(
            (
                series[:, offset:cut],
                series[:, cut:end],
                tile_times[: horizon.input_len],
                tile_times[horizon.input_len :],
            )
        )
    return tiles
