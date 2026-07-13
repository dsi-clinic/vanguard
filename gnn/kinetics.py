"""Per-voxel DCE enhancement-curve features, shared across node modes.

Both the voxel graph (``node_mode="voxel"``) and the segment line graph
(``node_mode="segment"``) derive their kinetic features from the *same*
per-voxel enhancement curve, using the same conventions as
``features/kinematic.py``: baseline is the timepoint-0 value (no per-timepoint
normalization, which would destroy the kinetic meaning of the curve), arrival
is estimated with
``graph_extraction.feature_stats._arrival_index_from_enhancement``, and
washin/AUC use the same formulas. Voxel mode attaches these per node; segment
mode summarizes them along a segment's voxels (see ``gnn.segment_graph``). This
module is the single source of truth so the two modes can never silently
diverge on how a curve becomes kinetics.
"""

from __future__ import annotations

import numpy as np

from graph_extraction.feature_stats import _arrival_index_from_enhancement


def time_axis_from_study_timepoints(study_timepoints: list[int]) -> np.ndarray:
    """Build a strictly increasing time axis, mirroring ``features.kinematic``.

    Falls back to plain timepoint indices (``0..T-1``) if the recorded
    timepoints are not finite and strictly increasing.
    """
    time_axis = np.asarray(study_timepoints, dtype=float)
    if not np.all(np.isfinite(time_axis)) or np.any(np.diff(time_axis) <= 0.0):
        return np.arange(len(study_timepoints), dtype=float)
    return time_axis


def node_kinetic_features(
    curve: np.ndarray, time_axis: np.ndarray
) -> dict[str, object]:
    """Derive enhancement-curve features for one voxel's raw DCE signal.

    Mirrors the per-segment convention in
    ``features.kinematic.compute_tumor_kinematic_feature_payload``: baseline is
    the timepoint-0 value, arrival is estimated with
    ``graph_extraction.feature_stats._arrival_index_from_enhancement``, and
    washin/washout/AUC use the same formulas (``washout_slope`` mirrors
    ``deepsets.build_dataset._dynamic_features_for_voxel``'s peak-to-last-
    timepoint slope).

    ``tte_idx`` is ``None`` when the voxel shows no meaningful enhancement
    (peak <= 0) -- a real "no signal" voxel, not a bug -- and the caller is
    responsible for choosing a sentinel for the tensor-facing feature.
    """
    baseline = float(curve[0])
    enh = np.asarray(curve, dtype=float) - baseline
    peak_idx = int(np.argmax(enh))
    peak_enhancement = float(enh[peak_idx])
    tte_idx = _arrival_index_from_enhancement(enh)
    start_idx = 0 if tte_idx is None else int(tte_idx)
    washin_den = float(time_axis[peak_idx] - time_axis[start_idx])
    washin_slope = (
        float((enh[peak_idx] - enh[start_idx]) / washin_den)
        if washin_den > 0.0
        else 0.0
    )
    washout_den = float(time_axis[-1] - time_axis[peak_idx])
    washout_slope = (
        float((enh[-1] - enh[peak_idx]) / washout_den) if washout_den > 0.0 else 0.0
    )
    auc_positive = float(np.trapz(np.maximum(enh, 0.0), x=time_axis))
    return {
        "peak_idx": peak_idx,
        "peak_enhancement": peak_enhancement,
        "tte_idx": tte_idx,
        "washin_slope": washin_slope,
        "washout_slope": washout_slope,
        "auc_positive": auc_positive,
    }
