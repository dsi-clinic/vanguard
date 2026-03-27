"""Shared geometry and summary-stat helpers for graph features."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from graph_extraction.constants import (
    BIFURCATION_MIN_DEGREE,
    MIN_CURVATURE_POINTS,
    MIN_LINEAR_FIT_POINTS,
    MIN_PATH_POINTS,
    NDIM_3D,
    TUMOR_SHELL_SPECS,
)
from graph_extraction.masks import _OFFSETS_3D


def mask_to_edges_bitmask(mask_zyx: np.ndarray) -> np.ndarray:
    """Convert a centerline mask into the edge encoding used by graph helpers.

    Each nonzero voxel stores which of its 26 neighboring voxels are also part
    of the centerline. `skeleton_to_graph_primitives.py` expects this
    representation when
    rebuilding graph segments from a saved mask.
    """
    if mask_zyx.ndim != NDIM_3D:
        raise ValueError(f"Expected 3D mask, got shape {mask_zyx.shape}")

    nodes = mask_zyx.astype(bool, copy=False)
    zdim, ydim, xdim = nodes.shape
    edges = np.zeros(nodes.shape, dtype=np.uint32)

    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                if not nodes[z, y, x]:
                    continue
                for b, (dz, dy, dx) in enumerate(_OFFSETS_3D):
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < zdim and 0 <= ny < ydim and 0 <= nx < xdim:
                        if nodes[nz, ny, nx]:
                            edges[z, y, x] |= np.uint32(1 << b)

    return edges


def _repair_skeleton_support_consistency(
    *,
    skeleton_mask_zyx: np.ndarray,
    support_mask_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Repair the common case where saved skeleton voxels fall outside support.

    Radius estimation is based on the support mask. If skeleton voxels sit
    outside that support, downstream radius-derived features can collapse to
    zero. This helper fixes the inconsistency and records how much repair was
    needed.
    """
    skeleton = np.asarray(skeleton_mask_zyx, dtype=bool)
    support = np.asarray(support_mask_zyx, dtype=bool)

    if skeleton.ndim != NDIM_3D or support.ndim != NDIM_3D:
        raise ValueError(
            f"skeleton/support must be 3D: skeleton={skeleton.shape}, support={support.shape}"
        )
    if tuple(skeleton.shape) != tuple(support.shape):
        raise ValueError(
            "Shape mismatch between skeleton and support: "
            f"{skeleton.shape} vs {support.shape}"
        )

    skeleton_voxels = int(np.count_nonzero(skeleton))
    support_voxels_before = int(np.count_nonzero(support))
    outside_mask = skeleton & (~support)
    outside_voxels = int(np.count_nonzero(outside_mask))

    repaired = False
    if outside_voxels > 0:
        support = support | skeleton
        repaired = True

    support_voxels_after = int(np.count_nonzero(support))
    return (
        skeleton,
        support,
        {
            "skeleton_voxels": skeleton_voxels,
            "support_voxels_before": support_voxels_before,
            "support_voxels_after": support_voxels_after,
            "skeleton_outside_support_voxels": outside_voxels,
            "repaired_support": repaired,
        },
    )


def _collect_morphometry_qc(
    *,
    segment_paths: list[list[tuple[int, int, int]]],
    segment_metrics_by_path: dict[tuple[tuple[int, int, int], ...], dict[str, object]],
    radius_map: dict[tuple[int, int, int], float],
    bifurcations: list[dict[str, object]],
) -> dict[str, int]:
    """Count common failure modes before writing morphometry outputs.

    These counters answer basic sanity-check questions such as:

    - do any node radii come out non-finite or non-positive?
    - do any segment summaries still contain invalid radii?
    - did the segment enumerator accidentally produce both A->B and B->A?
    - do we have branch points with degree greater than three?

    The returned counters are written into the run summary so later users
    can quickly tell whether a study produced questionable graph measurements.
    """
    radius_values = np.fromiter(radius_map.values(), dtype=float, count=len(radius_map))
    radius_nonfinite_node_count = int(np.count_nonzero(~np.isfinite(radius_values)))
    radius_nonpositive_node_count = int(np.count_nonzero(radius_values <= 0.0))

    segment_radius_nonpositive_count = 0
    segment_radius_nonfinite_count = 0
    duplicate_segment_paths = 0
    canonical_paths_seen: set[tuple[tuple[int, int, int], ...]] = set()

    for path in segment_paths:
        if not path:
            continue
        path_key = tuple(path)
        metric = segment_metrics_by_path[path_key]
        radius_block = metric.get("radius", {})
        radius_min = float(radius_block.get("min", 0.0))
        if not np.isfinite(radius_min):
            segment_radius_nonfinite_count += 1
        if radius_min <= 0.0:
            segment_radius_nonpositive_count += 1

        path_forward = tuple(path)
        path_reverse = tuple(reversed(path))
        path_key = path_forward if path_forward <= path_reverse else path_reverse
        if path_key in canonical_paths_seen:
            duplicate_segment_paths += 1
        else:
            canonical_paths_seen.add(path_key)

    bifurcation_degree_gt3_count = 0
    for entry in bifurcations:
        if not isinstance(entry, dict):
            continue
        bif = entry.get("bifurcation")
        if not isinstance(bif, dict):
            continue
        degree = int(bif.get("degree", 0))
        if degree > BIFURCATION_MIN_DEGREE:
            bifurcation_degree_gt3_count += 1

    return {
        "radius_nonfinite_node_count": int(radius_nonfinite_node_count),
        "radius_nonpositive_node_count": int(radius_nonpositive_node_count),
        "segment_radius_nonfinite_count": int(segment_radius_nonfinite_count),
        "segment_radius_nonpositive_count": int(segment_radius_nonpositive_count),
        "duplicate_segment_path_count": int(duplicate_segment_paths),
        "bifurcation_degree_gt3_count": int(bifurcation_degree_gt3_count),
    }


def _safe_ratio(num: float, den: float) -> float:
    denom = float(den)
    if denom <= 0.0:
        return 0.0
    return float(num) / denom


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probs: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict[str, float]:
    """Compute quantiles when some samples should count more than others.

    We use this for summaries where long or high-volume vessel segments should
    influence the percentile more than tiny segments.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(keep):
        return {f"q{int(p * 100):02d}": 0.0 for p in probs}
    v = v[keep]
    w = w[keep]
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    total = float(cdf[-1])
    if total <= 0.0:
        return {f"q{int(p * 100):02d}": 0.0 for p in probs}
    cdf = cdf / total
    out: dict[str, float] = {}
    for p in probs:
        key = f"q{int(float(p) * 100):02d}"
        out[key] = float(np.interp(float(p), cdf, v))
    return out


def _segment_path_length_mm(
    path: list[tuple[int, int, int]],
    spacing_mm_zyx: tuple[float, float, float],
) -> float:
    """Compute polyline length in millimeters for one voxel-path segment."""
    if len(path) < MIN_PATH_POINTS:
        return 0.0
    coords_xyz = np.asarray(path, dtype=float)
    diffs = np.diff(coords_xyz, axis=0)
    scale_xyz = np.asarray(
        [float(spacing_mm_zyx[2]), float(spacing_mm_zyx[1]), float(spacing_mm_zyx[0])],
        dtype=float,
    )
    diffs_mm = diffs * scale_xyz
    return float(np.sum(np.linalg.norm(diffs_mm, axis=1)))


def _segment_tortuosity_mm(
    path: list[tuple[int, int, int]],
    spacing_mm_zyx: tuple[float, float, float],
) -> float:
    """Measure how indirect a vessel segment is.

    A perfectly straight segment has tortuosity near 1.0. Larger values mean
    the segment takes a more winding path compared with the direct distance
    between its endpoints.
    """
    if len(path) < MIN_PATH_POINTS:
        return 1.0
    coords_xyz = np.asarray(path, dtype=float)
    scale_xyz = np.asarray(
        [float(spacing_mm_zyx[2]), float(spacing_mm_zyx[1]), float(spacing_mm_zyx[0])],
        dtype=float,
    )
    coords_mm = coords_xyz * scale_xyz
    diffs_mm = np.diff(coords_mm, axis=0)
    path_length_mm = float(np.sum(np.linalg.norm(diffs_mm, axis=1)))
    if path_length_mm <= 0.0:
        return 1.0
    endpoint_dist_mm = float(np.linalg.norm(coords_mm[-1] - coords_mm[0]))
    if endpoint_dist_mm <= 0.0:
        return 1.0
    return float(path_length_mm / endpoint_dist_mm)


def _segment_curvature_mean_per_mm(
    path: list[tuple[int, int, int]],
    spacing_mm_zyx: tuple[float, float, float],
) -> float:
    r"""Estimate how sharply a segment bends on average.

    Larger values mean the path changes direction more quickly as you move along
    it. The unit is \"per millimeter\", so it is comparable across voxel sizes.
    """
    if len(path) < MIN_CURVATURE_POINTS:
        return 0.0
    coords_xyz = np.asarray(path, dtype=float)
    scale_xyz = np.asarray(
        [float(spacing_mm_zyx[2]), float(spacing_mm_zyx[1]), float(spacing_mm_zyx[0])],
        dtype=float,
    )
    coords_mm = coords_xyz * scale_xyz
    v_prev = coords_mm[1:-1] - coords_mm[:-2]
    v_next = coords_mm[2:] - coords_mm[1:-1]
    n_prev = np.linalg.norm(v_prev, axis=1)
    n_next = np.linalg.norm(v_next, axis=1)
    valid = (n_prev > 0.0) & (n_next > 0.0)
    if not np.any(valid):
        return 0.0
    cosang = np.sum(v_prev[valid] * v_next[valid], axis=1) / (
        n_prev[valid] * n_next[valid]
    )
    cosang = np.clip(cosang, -1.0, 1.0)
    angles = np.arccos(cosang)
    local_arc_mm = 0.5 * (n_prev[valid] + n_next[valid])
    keep = local_arc_mm > 0.0
    if not np.any(keep):
        return 0.0
    curvature = angles[keep] / local_arc_mm[keep]
    if curvature.size == 0:
        return 0.0
    return float(np.mean(curvature))


def _segment_radius_mean_mm(
    path: list[tuple[int, int, int]],
    radius_mm_volume_zyx: np.ndarray,
) -> float:
    """Estimate the average vessel radius along one segment.

    `radius_mm_volume_zyx` is a precomputed 3D volume where each vessel voxel
    stores its estimated distance to the vessel boundary in millimeters. This
    helper samples that volume along the segment path and averages the valid
    radius values.
    """
    if len(path) == 0:
        return 0.0
    coords_xyz = np.asarray(path, dtype=np.int64)
    coords_zyx = coords_xyz[:, [2, 1, 0]]
    z = np.clip(coords_zyx[:, 0], 0, radius_mm_volume_zyx.shape[0] - 1)
    y = np.clip(coords_zyx[:, 1], 0, radius_mm_volume_zyx.shape[1] - 1)
    x = np.clip(coords_zyx[:, 2], 0, radius_mm_volume_zyx.shape[2] - 1)
    radii = np.asarray(radius_mm_volume_zyx[z, y, x], dtype=float).reshape(-1)
    radii = radii[np.isfinite(radii) & (radii > 0.0)]
    if radii.size == 0:
        return 0.0
    return float(np.mean(radii))


def _shell_name_for_signed_distance(distance_mm: float) -> str:
    """Map tumor distance in millimeters to one named distance band.

    Negative distances are inside the tumor. Positive distances are outside the
    tumor. We collapse the continuous distance into a few named shells so the
    downstream features stay interpretable, for example `inside_tumor`,
    `shell_0_2mm`, and `shell_2_5mm`.
    """
    d = float(distance_mm)
    for name, lower, upper in TUMOR_SHELL_SPECS:
        if d >= float(lower) and d < float(upper):
            return name
    return str(TUMOR_SHELL_SPECS[-1][0])


def _signed_tumor_distance_mm(
    tumor_mask_zyx: np.ndarray,
    spacing_mm_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Compute distance to the tumor boundary in millimeters.

    Values are negative inside the tumor and positive outside it. This signed
    distance map is what lets us group vessels into shells such as inside tumor,
    0 to 2 mm away, or 2 to 5 mm away.
    """
    tumor = np.asarray(tumor_mask_zyx, dtype=bool)
    outside = ndimage.distance_transform_edt(~tumor, sampling=spacing_mm_zyx).astype(
        np.float32, copy=False
    )
    inside = ndimage.distance_transform_edt(tumor, sampling=spacing_mm_zyx).astype(
        np.float32, copy=False
    )
    signed = outside
    signed[tumor] = -inside[tumor]
    return signed


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _quantile_summary(
    values: np.ndarray, *, include_count: bool = False
) -> dict[str, float | int]:
    """Summarize a numeric feature with simple robust statistics.

    Many feature groups eventually need the same kind of summary: a central
    value, spread, and tail behavior. This helper standardizes that output so
    different feature blocks use the same summary fields.

    The summary includes mean, standard deviation, median, interquartile range,
    90th percentile, minimum, and maximum. When requested, it also records how
    many valid values were summarized.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    out: dict[str, float | int]
    if arr.size == 0:
        out = {
            "mean": 0.0,
            "sd": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "q90": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
        if include_count:
            out["count"] = 0
        return out
    q25, q50, q75, q90 = np.percentile(arr, [25.0, 50.0, 75.0, 90.0])
    out = {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr)),
        "median": float(q50),
        "iqr": float(q75 - q25),
        "q90": float(q90),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if include_count:
        out["count"] = int(arr.size)
    return out


def _hurdle_summary(
    *,
    values: np.ndarray,
    valid_mask: np.ndarray,
    signal_mask: np.ndarray,
) -> dict[str, object]:
    """Summarize a sparse feature in two stages instead of one average.

    Many vessel-kinetic quantities are zero-inflated or only valid for a subset
    of segments. For example, some segments may never show meaningful
    enhancement, so it is more informative to ask two questions:

    1. how often is the quantity valid or present?
    2. among the segments where it is present, what values do we see?

    This helper returns both pieces:

    - `valid_fraction`: fraction of segments with a usable value
    - `has_signal_fraction`: fraction of segments that actually show signal
    - `value_given_signal`: robust summary of the non-zero / valid subset
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    signal = np.asarray(signal_mask, dtype=bool).reshape(-1)
    if vals.size == 0:
        return {
            "valid_fraction": 0.0,
            "has_signal_fraction": 0.0,
            "value_given_signal": _quantile_summary(np.asarray([], dtype=float)),
        }
    valid = valid & np.isfinite(vals)
    signal = signal & valid
    return {
        "valid_fraction": _safe_ratio(float(np.count_nonzero(valid)), float(vals.size)),
        "has_signal_fraction": _safe_ratio(
            float(np.count_nonzero(signal)), float(vals.size)
        ),
        "value_given_signal": _quantile_summary(vals[signal]),
    }


def _weighted_linear_slope(
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Fit a weighted straight line and return only the slope.

    Use this when a feature block only needs the direction and strength of a
    trend, for example whether a quantity tends to increase with distance from
    the tumor.
    """
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(keep) < MIN_LINEAR_FIT_POINTS:
        return 0.0
    x = x[keep]
    y = y[keep]
    w = w[keep]
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return 0.0
    x_mean = float(np.sum(w * x) / w_sum)
    y_mean = float(np.sum(w * y) / w_sum)
    cov = float(np.sum(w * (x - x_mean) * (y - y_mean)))
    var = float(np.sum(w * (x - x_mean) ** 2))
    if var <= 0.0:
        return 0.0
    return float(cov / var)


def _weighted_linear_fit_stats(
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Fit a weighted straight line and return a small diagnostics summary.

    Compared with `_weighted_linear_slope`, this returns:

    - `slope`: how fast the quantity changes
    - `intercept`: the fitted value at x = 0
    - `r2`: how well a straight line matches the data
    - `rmse` and `mae`: average error sizes

    This is used when downstream features need both the trend itself and a
    rough sense of whether that trend is stable or noisy.
    """
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(keep) < MIN_LINEAR_FIT_POINTS:
        return {
            "n": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "r2": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
        }
    x = x[keep]
    y = y[keep]
    w = w[keep]
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return {
            "n": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "r2": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
        }

    x_mean = float(np.sum(w * x) / w_sum)
    y_mean = float(np.sum(w * y) / w_sum)
    cov = float(np.sum(w * (x - x_mean) * (y - y_mean)))
    var_x = float(np.sum(w * (x - x_mean) ** 2))
    slope = float(cov / var_x) if var_x > 0.0 else 0.0
    intercept = float(y_mean - slope * x_mean)

    y_hat = intercept + slope * x
    resid = y - y_hat
    ss_res = float(np.sum(w * (resid**2)))
    ss_tot = float(np.sum(w * ((y - y_mean) ** 2)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    rmse = float(math.sqrt(ss_res / w_sum)) if w_sum > 0.0 else 0.0
    mae = float(np.sum(w * np.abs(resid)) / w_sum) if w_sum > 0.0 else 0.0
    return {
        "n": float(x.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def _arrival_index_from_enhancement(
    enhancement_curve: np.ndarray, frac_of_peak: float = 0.2
) -> int | None:
    """Estimate when enhancement first becomes meaningfully present.

    The rule is intentionally simple: find the first time point where the curve
    reaches a chosen fraction of its own peak.
    """
    enh = np.asarray(enhancement_curve, dtype=float).reshape(-1)
    enh = np.where(np.isfinite(enh), enh, 0.0)
    if enh.size == 0:
        return None
    peak = float(np.max(enh))
    if peak <= 0.0:
        return None
    threshold = max(float(frac_of_peak) * peak, 1e-6)
    hits = np.where(enh >= threshold)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def _discrete_entropy(indices: np.ndarray, n_bins: int) -> float:
    """Compute entropy for integer-valued assignments such as shell or bin IDs."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    idx = idx[(idx >= 0) & (idx < int(n_bins))]
    if idx.size == 0 or int(n_bins) <= 0:
        return 0.0
    counts = np.bincount(idx, minlength=int(n_bins)).astype(float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))
