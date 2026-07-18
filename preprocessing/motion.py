"""QC-gated translation correction that preserves raw DCE signal."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

MINIMUM_CORRELATION_VOXELS = 32


@dataclass(frozen=True)
class MotionSettings:
    """Tracked translation-registration settings for XYZ arrays."""

    downsample_xyz: tuple[int, int, int] = (4, 4, 2)
    upsample_factor: int = 4
    max_translation_mm: float = 30.0
    minimum_correlation_delta: float = 0.0
    maximum_correlation_voxels: int = 250_000

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe settings."""
        return asdict(self)


DEFAULT_MOTION_SETTINGS = MotionSettings()


def _registration_image(volume: np.ndarray, support: np.ndarray | None) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("registration input contains nonfinite signal")
    values = (
        array[support] if support is not None and np.any(support) else array[array > 0]
    )
    if values.size < MINIMUM_CORRELATION_VOXELS:
        raise ValueError("registration support contains too few voxels")
    lower, upper = np.percentile(values, [1.0, 99.5])
    if not upper > lower:
        raise ValueError("registration support has degenerate intensity")
    clipped = np.clip(array, lower, upper)
    normalized_values = (
        clipped[support] if support is not None and np.any(support) else clipped
    )
    standard_deviation = float(normalized_values.std())
    if not standard_deviation > 0.0:
        raise ValueError("registration support has zero variance")
    normalized = (clipped - float(normalized_values.mean())) / standard_deviation
    if support is not None:
        normalized = np.where(support, normalized, 0.0)
    return np.asarray(normalized, dtype=np.float32)


def correlation_in_support(
    first: np.ndarray,
    second: np.ndarray,
    support: np.ndarray | None,
    *,
    maximum_voxels: int,
) -> float:
    """Compute Pearson correlation on a deterministic support sample."""
    if support is None or not np.any(support):
        support = (first > 0) & (second > 0)
    indices = np.flatnonzero(np.asarray(support, dtype=bool).ravel())
    if indices.size < MINIMUM_CORRELATION_VOXELS:
        return float("nan")
    if indices.size > maximum_voxels:
        indices = indices[:: int(math.ceil(indices.size / maximum_voxels))]
    first_values = np.asarray(first, dtype=np.float32).ravel()[indices]
    second_values = np.asarray(second, dtype=np.float32).ravel()[indices]
    finite = np.isfinite(first_values) & np.isfinite(second_values)
    if int(finite.sum()) < MINIMUM_CORRELATION_VOXELS:
        return float("nan")
    first_centered = first_values[finite] - float(first_values[finite].mean())
    second_centered = second_values[finite] - float(second_values[finite].mean())
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    return (
        float(np.dot(first_centered, second_centered) / denominator)
        if denominator > 0
        else float("nan")
    )


def correct_phase(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    support: np.ndarray | None,
    spacing_xyz_mm: np.ndarray,
    settings: MotionSettings = DEFAULT_MOTION_SETTINGS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Propose a translation and retain it only when in-support NCC improves."""
    fixed_array = np.asarray(fixed, dtype=np.float32)
    moving_array = np.asarray(moving, dtype=np.float32)
    if fixed_array.shape != moving_array.shape:
        raise ValueError("fixed and moving phase shapes differ")
    if np.any(fixed_array < 0) or np.any(moving_array < 0):
        raise ValueError("motion correction requires nonnegative raw signal")
    evaluation_support = (
        np.asarray(support, dtype=bool)
        if support is not None and np.any(support)
        else (fixed_array > 0) & (moving_array > 0)
    )
    fixed_registration = _registration_image(fixed_array, evaluation_support)
    moving_registration = _registration_image(moving_array, evaluation_support)
    downsample = tuple(max(1, int(value)) for value in settings.downsample_xyz)
    shift_small, error, difference_phase = phase_cross_correlation(
        fixed_registration[:: downsample[0], :: downsample[1], :: downsample[2]],
        moving_registration[:: downsample[0], :: downsample[1], :: downsample[2]],
        upsample_factor=max(1, settings.upsample_factor),
        normalization=None,
    )
    proposed_shift = np.asarray(shift_small, dtype=np.float64) * np.asarray(downsample)
    translation_mm = proposed_shift * np.asarray(spacing_xyz_mm, dtype=np.float64)
    norm_mm = float(np.linalg.norm(translation_mm))
    proposed = ndimage.shift(
        moving_array,
        shift=tuple(float(value) for value in proposed_shift),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    raw_correlation = correlation_in_support(
        fixed_array,
        moving_array,
        evaluation_support,
        maximum_voxels=settings.maximum_correlation_voxels,
    )
    proposed_correlation = correlation_in_support(
        fixed_array,
        proposed,
        evaluation_support,
        maximum_voxels=settings.maximum_correlation_voxels,
    )
    delta = proposed_correlation - raw_correlation
    within_bound = bool(np.isfinite(norm_mm) and norm_mm <= settings.max_translation_mm)
    accepted = bool(
        within_bound
        and np.isfinite(delta)
        and delta >= settings.minimum_correlation_delta
    )
    if accepted:
        corrected = proposed
        saved_shift = proposed_shift
        reason = "accepted"
    else:
        corrected = moving_array.copy()
        saved_shift = np.zeros(3, dtype=np.float64)
        reason = (
            "proposed_translation_exceeds_maximum"
            if not within_bound
            else "correlation_gain_below_minimum"
        )
    metrics: dict[str, object] = {
        "transform_accepted": accepted,
        "transform_rejection_reason": reason,
        "proposed_translation_voxels": proposed_shift.tolist(),
        "translation_voxels": saved_shift.tolist(),
        "proposed_translation_xyz_mm": translation_mm.tolist(),
        "proposed_translation_norm_mm": norm_mm,
        "raw_correlation": raw_correlation,
        "proposed_correlation": proposed_correlation,
        "corr_delta": delta,
        "phase_correlation_error": float(error),
        "phase_correlation_difference_phase": float(difference_phase),
    }
    return corrected, saved_shift, metrics
