"""Align 3D/4D volumes to z,y,x layout expected by Deep Sets skeleton masks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from graph_extraction.constants import NDIM_4D


def align_zyx_volume_to_shape(
    volume: np.ndarray, expected_shape_zyx: tuple[int, int, int]
) -> np.ndarray:
    """Pick axis order so ``volume`` matches ``expected_shape_zyx``."""
    candidates = [
        np.asarray(volume),
        np.transpose(np.asarray(volume), (1, 2, 0)),
        np.transpose(np.asarray(volume), (2, 1, 0)),
    ]
    for candidate in candidates:
        if tuple(int(v) for v in candidate.shape) == expected_shape_zyx:
            return candidate.astype(np.float32, copy=False)
    raise ValueError(
        f"Shape mismatch after axis permutations: expected {expected_shape_zyx}, "
        f"raw {np.asarray(volume).shape}"
    )


def _spatial_permute_fn(
    volume: np.ndarray, expected_shape_zyx: tuple[int, int, int]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that maps any same-layout 3D array to ``expected_shape_zyx``."""
    raw = np.asarray(volume)
    transforms: list[Callable[[np.ndarray], np.ndarray]] = [
        lambda x: np.asarray(x),
        lambda x: np.transpose(np.asarray(x), (1, 2, 0)),
        lambda x: np.transpose(np.asarray(x), (2, 1, 0)),
    ]
    for fn in transforms:
        candidate = fn(raw)
        if tuple(int(v) for v in candidate.shape) == expected_shape_zyx:
            return fn
    raise ValueError(
        f"Shape mismatch after axis permutations: expected {expected_shape_zyx}, "
        f"raw {raw.shape}"
    )


def align_zyx_4d_to_shape(
    arr_4d: np.ndarray, expected_shape_zyx: tuple[int, int, int]
) -> np.ndarray:
    """Align ``(t, z, y, x)`` (after permutation) to match skeleton ``expected_shape_zyx``."""
    signal = np.asarray(arr_4d, dtype=np.float32)
    if signal.ndim != NDIM_4D:
        raise ValueError(f"Expected 4D array, got shape {signal.shape}")
    permute = _spatial_permute_fn(signal[0], expected_shape_zyx)
    out = np.stack(
        [
            permute(signal[t]).astype(np.float32, copy=False)
            for t in range(signal.shape[0])
        ],
        axis=0,
    )
    return out
