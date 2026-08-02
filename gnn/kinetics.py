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


def baseline_relative_curve(
    curves: np.ndarray,
    *,
    baseline_frame_count: int = 1,
    relative_enhancement: bool = False,
    baseline_floor_frac: float = 0.0,
    baseline_override: np.ndarray | float | None = None,
) -> np.ndarray:
    """Baseline-reference a DCE curve (or stack of curves) along the time axis.

    Single source of truth for the "curve -> enhancement" transform, shared by
    the kinetic-feature path (``node_kinetic_features``) and the forecasting
    pretext path (``gnn.pretrain.node_series``) so the two can never silently
    diverge. Operates on the **last axis** (time), so ``curves`` may be a 1D
    ``(T,)`` curve or an ``(N, T)`` stack; each row is referenced against *its
    own* baseline (the mean of its first ``baseline_frame_count`` frames):

    - ``relative_enhancement=True``  -> ``(S(t) - S0) / S0`` (UFAST/UChicago
      contract; a row whose baseline is non-finite or <= float32 eps yields all
      zeros, matching the kinetic path).
    - ``relative_enhancement=False`` -> ``S(t) - S0`` (legacy absolute
      convention). With ``baseline_frame_count=1`` this is exactly the old
      ``curve - curve[0]`` frame-0 subtraction.

    Does **not** zero the baseline frames -- that precontrast-noise cleanup is a
    kinetic-feature concern (arrival/peak detection), applied by the caller;
    forecasting keeps the real baseline-referenced signal.
    """
    arr = np.asarray(curves, dtype=float)
    n_timepoints = arr.shape[-1]
    if not 1 <= baseline_frame_count < n_timepoints:
        raise ValueError("baseline_frame_count must be in [1, n_timepoints)")
    if baseline_floor_frac < 0.0:
        raise ValueError("baseline_floor_frac must be >= 0")
    if baseline_override is None:
        baseline = arr[..., :baseline_frame_count].mean(axis=-1, keepdims=True)
    else:
        # A repaired baseline supplied by the caller (gnn.baseline_validity):
        # this voxel's own precontrast frames were not usable, so its enhancement
        # is referenced against a locally imputed S0 instead.
        baseline = np.asarray(baseline_override, dtype=float)
        baseline = np.broadcast_to(
            baseline.reshape(baseline.shape + (1,) * (arr.ndim - baseline.ndim)),
            arr.shape[:-1] + (1,),
        )
    difference = arr - baseline
    if not relative_enhancement:
        return difference
    # Denominator for the relative transform. By default it is the raw baseline
    # (historical behavior, only the exact-zero guard below). When
    # ``baseline_floor_frac > 0`` the denominator is floored at that fraction of
    # each row's OWN signal scale (max |S| over time), so a near-void baseline
    # can't turn ``(S - S0)/S0`` into a divide-by-noise artifact: a voxel whose
    # baseline is a tiny fraction of its own peak signal would otherwise report
    # relative enhancement in the thousands-to-millions (physiological is ~0-5).
    # The floor caps peak relative enhancement at ~1/baseline_floor_frac and
    # leaves normal voxels (baseline >> floor) untouched. See AUDITING_RESULTS.md.
    denom = baseline
    if baseline_floor_frac > 0.0:
        signal_scale = np.max(np.abs(arr), axis=-1, keepdims=True)
        denom = np.maximum(baseline, baseline_floor_frac * signal_scale)
    # Divide only where the denominator is usable; rows with a non-finite or
    # <= float32-eps denominator stay zero (matches the kinetic path). Using
    # ``np.divide(where=...)`` skips the invalid division entirely rather than
    # computing it and masking afterwards, so no divide-by-zero warning fires.
    usable = np.isfinite(denom) & (denom > np.finfo(np.float32).eps)
    enhancement = np.zeros_like(difference)
    np.divide(difference, denom, out=enhancement, where=usable)
    return enhancement


def node_kinetic_features(
    curve: np.ndarray,
    time_axis: np.ndarray,
    *,
    baseline_frame_count: int = 1,
    relative_enhancement: bool = False,
    baseline_floor_frac: float = 0.0,
    baseline_override: float | None = None,
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

    baseline = (
        float(np.mean(signal[:baseline_frame_count]))
        if baseline_override is None
        else float(baseline_override)
    )
    enhancement = np.asarray(
        baseline_relative_curve(
            signal,
            baseline_frame_count=baseline_frame_count,
            relative_enhancement=relative_enhancement,
            baseline_floor_frac=baseline_floor_frac,
            baseline_override=baseline_override,
        ),
        dtype=float,
    )
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
