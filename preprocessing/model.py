"""Vanguard-owned input contract for the frozen vessel models."""

from __future__ import annotations

import numpy as np

MODEL_TAIL_FRACTION = 0.001
MINIMUM_MODEL_VOXELS = 2
MODEL_SPATIAL_DIMENSIONS = 3
MAXIMUM_TAIL_FRACTION = 0.5


def model_subject_id(phase_index: int) -> str:
    """Return a stable identifier that is safe for the pinned model dataset."""
    if phase_index < 0:
        raise ValueError("phase_index must be nonnegative")
    identifier = f"hr_phase_{phase_index:04d}_id"
    if identifier.endswith(".npy"):
        raise ValueError(f"unsafe model subject identifier: {identifier}")
    return identifier


def frozen_model_intensity_preprocess(
    image: np.ndarray,
    *,
    tail_fraction: float = MODEL_TAIL_FRACTION,
) -> np.ndarray:
    """Apply the published model's tail clipping and per-volume z-score.

    This transformation is only for the frozen breast/vessel model. It must
    never be applied to the UFAST signal used for motion correction or kinetic
    features.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != MODEL_SPATIAL_DIMENSIONS:
        raise ValueError(f"model input must be 3D, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("model input contains nonfinite values")
    if not 0.0 <= tail_fraction < MAXIMUM_TAIL_FRACTION:
        raise ValueError("tail_fraction must be in [0, 0.5)")
    if array.size < MINIMUM_MODEL_VOXELS:
        raise ValueError("model input is too small")

    ordered = np.sort(array, axis=None)
    lower_index = int(ordered.size * tail_fraction)
    upper_index = -max(lower_index, 1)
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    if not upper > lower:
        raise ValueError("model input has a degenerate robust intensity range")

    normalized = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
    standard_deviation = float(normalized.std())
    if not standard_deviation > 0.0:
        raise ValueError("model input has zero variance after clipping")
    return np.asarray(
        (normalized - float(normalized.mean())) / standard_deviation,
        dtype=np.float32,
    )


def prepare_hr_phase_for_model(phase_zyx: np.ndarray) -> np.ndarray:
    """Convert native DICOM ``(z,y,x)`` signal to model ``(y,x,z)``."""
    phase = np.asarray(phase_zyx)
    if phase.ndim != MODEL_SPATIAL_DIMENSIONS:
        raise ValueError(f"HR phase must be 3D, got {phase.shape}")
    return frozen_model_intensity_preprocess(np.transpose(phase, (1, 2, 0)))
