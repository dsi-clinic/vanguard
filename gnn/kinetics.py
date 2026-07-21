"""Per-voxel DCE enhancement features shared by every GNN graph mode.

Vanguard UFAST cases use the protocol-defined precontrast-frame mean,
relative signal change, and physical acquisition seconds. Legacy cohorts keep
their established single-frame, absolute-enhancement convention. Keeping this
logic here prevents voxel, segment, and junction graphs from silently using
different definitions of the same kinetic feature.
"""

from __future__ import annotations

import numpy as np

from graph_extraction.feature_stats import _arrival_index_from_enhancement


def time_axis_from_study_timepoints(
    study_timepoints: list[int],
    physical_times_seconds: np.ndarray | None = None,
    *,
    require_physical_seconds: bool = False,
) -> np.ndarray:
    """Return a finite, strictly increasing time axis for a DCE series.

    Physical acquisition seconds take precedence when provided. Vanguard cases
    fail closed if their required physical-time sidecar is missing or invalid;
    legacy cases fall back to ``0..T-1`` when recorded indices are unusable.
    """
    if require_physical_seconds and physical_times_seconds is None:
        raise FileNotFoundError(
            "Vanguard kinetic features require ufast_times_seconds.npy"
        )
    time_axis = (
        np.asarray(physical_times_seconds, dtype=float)
        if physical_times_seconds is not None
        else np.asarray(study_timepoints, dtype=float)
    )
    if time_axis.ndim != 1 or time_axis.size != len(study_timepoints):
        raise ValueError("Time axis length must match the DCE timepoint count")
    valid = np.all(np.isfinite(time_axis)) and np.all(np.diff(time_axis) > 0.0)
    if not valid:
        if require_physical_seconds:
            raise ValueError(
                "Vanguard physical acquisition seconds must be finite and increasing"
            )
        return np.arange(len(study_timepoints), dtype=float)
    return time_axis


def node_kinetic_features(
    curve: np.ndarray,
    time_axis: np.ndarray,
    *,
    baseline_frame_count: int = 1,
    relative_enhancement: bool = False,
) -> dict[str, object]:
    """Derive kinetic features from one voxel's unscaled DCE signal curve.

    ``baseline_frame_count`` and ``relative_enhancement`` encode the acquisition
    protocol. The first baseline frames are zeroed after baseline estimation so
    precontrast noise can't be mistaken for arrival or peak enhancement.
    ``tte_idx``/``tte_seconds`` are ``None`` when no arrival is detected.
    """
    signal = np.asarray(curve, dtype=float).reshape(-1)
    times = np.asarray(time_axis, dtype=float).reshape(-1)
    if signal.size != times.size:
        raise ValueError("DCE curve and time axis lengths differ")
    if not 1 <= baseline_frame_count < signal.size:
        raise ValueError("baseline_frame_count must be in [1, n_timepoints)")

    baseline = float(np.mean(signal[:baseline_frame_count]))
    difference = signal - baseline
    if relative_enhancement:
        enhancement = (
            difference / baseline
            if np.isfinite(baseline) and baseline > np.finfo(np.float32).eps
            else np.zeros_like(difference)
        )
    else:
        enhancement = difference
    enhancement = np.asarray(enhancement, dtype=float)
    enhancement[:baseline_frame_count] = 0.0

    peak_idx = baseline_frame_count + int(np.argmax(enhancement[baseline_frame_count:]))
    peak_enhancement = float(enhancement[peak_idx])
    tte_idx = _arrival_index_from_enhancement(enhancement)
    start_idx = 0 if tte_idx is None else int(tte_idx)

    washin_den = float(times[peak_idx] - times[start_idx])
    washin_slope = (
        float((enhancement[peak_idx] - enhancement[start_idx]) / washin_den)
        if washin_den > 0.0
        else 0.0
    )
    washout_den = float(times[-1] - times[peak_idx])
    washout_slope = (
        float((enhancement[-1] - enhancement[peak_idx]) / washout_den)
        if washout_den > 0.0
        else 0.0
    )
    auc_positive = float(np.trapz(np.maximum(enhancement, 0.0), x=times))

    return {
        "baseline_signal": baseline,
        "peak_idx": peak_idx,
        "peak_time_seconds": float(times[peak_idx] - times[0]),
        "peak_enhancement": peak_enhancement,
        "tte_idx": tte_idx,
        "tte_seconds": None if tte_idx is None else float(times[tte_idx] - times[0]),
        "washin_slope": washin_slope,
        "washout_slope": washout_slope,
        "auc_positive": auc_positive,
    }
