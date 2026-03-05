"""TC4D core implementation for production extraction only."""
# ruff: noqa: E402

from __future__ import annotations

import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from graph_extraction.skeleton3d import skeletonize3d

NDIM_3D = 3
NDIM_4D = 4
MIN_OTSU_SAMPLES = 2
MIN_ROI_VOXELS = 16
MIN_ROI_SEED_VOXELS = 1024
TEMPORAL_CHAIN_LEN_SHORT = 2
TEMPORAL_CHAIN_LEN_LONG = 3

# tc4d core (graph-forward, no-learning) internal policy constants.
TC4D_THRESHOLD_LOWER_BOUND = 0.30
TC4D_THRESHOLD_UPPER_BOUND = 0.90
TC4D_EXAM_COMPONENT_KEEP_FRACTION = 0.95
GEODESIC_SEED_QUANTILE = 94.0
GEODESIC_MAX_SEEDS = 24
GEODESIC_BETA_QUANTILE = 20.0
GEODESIC_BLOB_RELAX_WITH_DYNAMICS = 0.40
GEODESIC_MAX_WORK_NODES = 400_000
GEODESIC_MIN_SUPPORT_SELECTION_FRACTION = 0.08
GEODESIC_EDGE_COST_LAMBDA_SCALE = 0.015
GEODESIC_EDGE_COST_LAMBDA_MIN = 0.001
GEODESIC_EDGE_COST_LAMBDA_MAX = 2.50
GEODESIC_UNION_MIN_SELECTED_SUPPORT_VOXELS = 48
GEODESIC_WORKSPACE_DILATION_ITERS = 1
TC4D_SPACETIME_SELECTION_PRIOR_BLEND = 0.30
TC4D_SPACETIME_SELECTION_PRIOR_TEMPORAL_NEIGHBOR_WEIGHT = 0.18
TC4D_EDGE_PERSISTENCE_NODE_WEIGHT = 0.95
TC4D_EDGE_PERSISTENCE_SEED_WEIGHT = 0.40
TC4D_EDGE_PERSISTENCE_TRAVERSAL_RELIEF = 0.35
TC4D_TEMPORAL_FLOW_SPATIAL_RADIUS = 1
TC4D_TEMPORAL_FLOW_MIN_RETAIN_FRACTION = 0.72
TC4D_TEMPORAL_FLOW_SOFT_WEIGHT = 0.24
TC4D_VED_PRECONDITION_BLEND_WEIGHT = 0.32
TC4D_VED_PRECONDITION_SMOOTH_SIGMA = 1.10
TC4D_VED_PRECONDITION_DIFFUSION_ITERS = 2
TC4D_VED_PRECONDITION_DIFFUSION_STEP = 0.45
TC4D_VED_PRECONDITION_MASK_PERCENTILE = 60.0
TC4D_VED_PRECONDITION_ROI_PAD = 8
TC4D_VED_PRECONDITION_MAX_WORK_VOXELS = 8_000_000
TC4D_FRANGI_SIGMAS = (1.0, 1.8, 2.8)
TC4D_FRANGI_NODE_WEIGHT = 0.55
TC4D_COMPONENT_REPAIR_MAX_ITERATIONS = 24
TC4D_COMPONENT_REPAIR_MAX_GAP_VOXELS = 11.0
GEODESIC_NEIGHBOR_OFFSETS_26 = tuple(
    (
        int(dz),
        int(dy),
        int(dx),
        float(math.sqrt((dz * dz) + (dy * dy) + (dx * dx))),
    )
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
)
EDGE_PERSISTENCE_NEIGHBOR_OFFSETS = tuple(
    (dz, dy, dx, edge_len)
    for dz, dy, dx, edge_len in GEODESIC_NEIGHBOR_OFFSETS_26
    if (dz > 0) or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0)
)
STRUCTURE_3X3X3_U8 = np.ones((3, 3, 3), dtype=np.uint8)


def _iter_offset_slices(
    shape_zyx: tuple[int, int, int],
    offsets: tuple[tuple[int, int, int, float], ...],
) -> Iterator[
    tuple[slice, slice, slice],
    tuple[slice, slice, slice],
    float,
]:
    """Yield valid aligned source/destination slices for spatial offsets."""
    z_dim, y_dim, x_dim = shape_zyx
    for dz, dy, dx, edge_len in offsets:
        src_z0 = max(0, -dz)
        src_z1 = z_dim - max(0, dz)
        src_y0 = max(0, -dy)
        src_y1 = y_dim - max(0, dy)
        src_x0 = max(0, -dx)
        src_x1 = x_dim - max(0, dx)

        if src_z1 <= src_z0 or src_y1 <= src_y0 or src_x1 <= src_x0:
            continue

        dst_z0 = max(0, dz)
        dst_z1 = z_dim - max(0, -dz)
        dst_y0 = max(0, dy)
        dst_y1 = y_dim - max(0, -dy)
        dst_x0 = max(0, dx)
        dst_x1 = x_dim - max(0, -dx)

        src_slice = (
            slice(src_z0, src_z1),
            slice(src_y0, src_y1),
            slice(src_x0, src_x1),
        )
        dst_slice = (
            slice(dst_z0, dst_z1),
            slice(dst_y0, dst_y1),
            slice(dst_x0, dst_x1),
        )
        yield src_slice, dst_slice, float(edge_len)


def _label_components_26(mask_zyx: np.ndarray) -> tuple[np.ndarray, int]:
    """26-connected component labeling for a 3D binary mask."""
    return ndimage.label(
        mask_zyx.astype(np.uint8),
        structure=STRUCTURE_3X3X3_U8,
    )


def _compute_otsu_threshold(
    values: np.ndarray,
    *,
    positive_only: bool = True,
    clip_bounds: tuple[float, float] | None = (0.05, 0.95),
    empty_fallback: float = 0.5,
) -> float:
    """Compute an Otsu threshold with optional positivity/clipping policy."""
    finite = values[np.isfinite(values)].astype(np.float64, copy=False)
    if positive_only:
        finite = finite[finite > 0.0]
    if finite.size < MIN_OTSU_SAMPLES:
        if clip_bounds is None and finite.size > 0:
            return float(np.min(finite))
        return empty_fallback

    vmin = np.min(finite)
    vmax = np.max(finite)
    if vmax <= vmin:
        return vmin

    hist, bins = np.histogram(finite, bins=256, range=(vmin, vmax))
    total = hist.sum()

    prob = hist.astype(np.float64) / total
    omega = np.cumsum(prob)
    centers = (bins[:-1] + bins[1:]) / 2.0
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    sigma_b2 = np.zeros_like(omega)
    valid = denom > 0
    sigma_b2[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]
    best_idx = np.argmax(sigma_b2)
    threshold = centers[best_idx]
    if clip_bounds is None:
        return threshold
    return np.clip(threshold, clip_bounds[0], clip_bounds[1])


def _score_filter_components_by_support(
    mask_zyx: np.ndarray,
    *,
    support_count_zyx: np.ndarray,
) -> np.ndarray:
    """Filter 3D components by support-weighted component score (automatic Otsu)."""
    if mask_zyx.ndim != NDIM_3D:
        raise ValueError(f"Expected 3D mask, got shape {mask_zyx.shape}")
    if support_count_zyx.ndim != NDIM_3D:
        raise ValueError(
            f"Expected 3D support_count, got shape {support_count_zyx.shape}"
        )
    if tuple(mask_zyx.shape) != tuple(support_count_zyx.shape):
        raise ValueError(
            "Mask/support_count shape mismatch: "
            f"{mask_zyx.shape} vs {support_count_zyx.shape}"
        )
    keep_fraction = np.clip(TC4D_EXAM_COMPONENT_KEEP_FRACTION, 0.0, 1.0)

    if not np.any(mask_zyx):
        return mask_zyx

    labels, n_comp = _label_components_26(mask_zyx)
    if n_comp <= 1:
        return mask_zyx

    sizes = np.bincount(labels.ravel())
    comp_sizes = sizes[1:].astype(np.float64, copy=False)
    sums = ndimage.sum(
        support_count_zyx.astype(np.float32, copy=False),
        labels,
        index=np.arange(1, n_comp + 1),
    ).astype(np.float64, copy=False)
    mean_support = sums / np.maximum(comp_sizes, 1.0)
    comp_scores = mean_support * np.log1p(comp_sizes)
    threshold = _compute_otsu_threshold(
        comp_scores,
        positive_only=False,
        clip_bounds=None,
        empty_fallback=0.0,
    )
    keep = comp_scores >= float(threshold)
    total_score = float(np.sum(comp_scores))
    if total_score > 0.0:
        ranked = np.argsort(comp_scores)[::-1]
        cumulative_fraction = np.cumsum(comp_scores[ranked]) / total_score
        n_cumulative = (
            np.searchsorted(
                cumulative_fraction,
                keep_fraction,
                side="left",
            )
            + 1
        )
        keep[ranked[:n_cumulative]] = True
    # Always retain the largest component for topology robustness.
    keep[np.argmax(comp_sizes)] = True
    keep_ids = np.flatnonzero(keep) + 1
    return np.isin(labels, keep_ids)


def _prune_micro_components(
    mask_zyx: np.ndarray,
    *,
    min_component_voxels: int,
) -> np.ndarray:
    """Drop tiny detached components while always preserving the largest one."""
    if mask_zyx.ndim != NDIM_3D:
        raise ValueError(f"Expected 3D mask, got shape {mask_zyx.shape}")
    if not np.any(mask_zyx):
        return mask_zyx

    labels, n_comp = _label_components_26(mask_zyx)
    if n_comp <= 1:
        return mask_zyx

    sizes = np.bincount(labels.ravel())
    comp_sizes = sizes[1:]
    largest_idx = np.argmax(comp_sizes)
    min_keep = max(1, min_component_voxels)

    keep = comp_sizes >= min_keep
    keep[largest_idx] = True
    keep_ids = np.flatnonzero(keep) + 1
    return np.isin(labels, keep_ids)


def _filter_components_by_support(
    mask_zyx: np.ndarray,
    *,
    support_count_zyx: np.ndarray,
) -> np.ndarray:
    """Filter components with support-aware scoring policy."""
    return _score_filter_components_by_support(
        mask_zyx,
        support_count_zyx=support_count_zyx,
    )


def _skeletonize_support_mask(
    support_mask_zyx: np.ndarray,
) -> np.ndarray:
    """Skeletonize a 3D support mask via EDT priority."""
    if not np.any(support_mask_zyx):
        return np.zeros_like(support_mask_zyx, dtype=bool)
    priority = ndimage.distance_transform_edt(support_mask_zyx).astype(
        np.float32,
        copy=False,
    )
    return skeletonize3d(priority, threshold=0.0) > 0


def _normalize_on_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Quantile-normalize values into [0, 1] only over masked voxels."""
    out = np.zeros_like(values, dtype=np.float32)
    if not np.any(mask):
        return out

    masked = values[mask]
    lo = float(np.percentile(masked, 10))
    hi = float(np.percentile(masked, 90))
    denom = max(hi - lo, 1e-6)
    out[mask] = np.clip((values[mask] - lo) / denom, 0.0, 1.0)
    return out


def _compute_frangi_tubularity_3d(
    volume_zyx: np.ndarray,
    *,
    mask_zyx: np.ndarray,
) -> np.ndarray:
    """Compute a masked, normalized 3D Frangi tubularity prior."""
    out = np.zeros_like(volume_zyx, dtype=np.float32)
    if not np.any(mask_zyx):
        return out

    coords = np.argwhere(mask_zyx)
    lo = np.min(coords, axis=0)
    hi = np.max(coords, axis=0) + 1
    pad = int(max(2, math.ceil(max(TC4D_FRANGI_SIGMAS) * 2.0)))
    shape = np.asarray(volume_zyx.shape)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, shape)
    roi = (
        slice(lo[0], hi[0]),
        slice(lo[1], hi[1]),
        slice(lo[2], hi[2]),
    )
    vol_roi = np.asarray(volume_zyx[roi], dtype=np.float32)
    mask_roi = np.asarray(mask_zyx[roi], dtype=bool)

    mask_roi_voxels = np.count_nonzero(mask_roi)
    if mask_roi_voxels < MIN_ROI_VOXELS:
        return out

    vol_roi_c = np.ascontiguousarray(vol_roi.astype(np.float32, copy=False))

    try:
        from skimage.filters import frangi

        vol_roi_norm = _normalize_on_mask(vol_roi_c, mask_roi)
        response = frangi(
            vol_roi_norm.astype(np.float32, copy=False),
            sigmas=TC4D_FRANGI_SIGMAS,
            black_ridges=False,
            mode="reflect",
        )
        response = np.nan_to_num(
            np.asarray(response, dtype=np.float32),
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        response = np.clip(response, 0.0, None)
        response_norm = _normalize_on_mask(response, mask_roi)
        response_norm *= mask_roi.astype(np.float32, copy=False)
        out[roi] = response_norm
        out *= mask_zyx.astype(np.float32, copy=False)
        return out
    except Exception:
        return out


def _compute_tc4d_ved_preconditioner_3d(
    evidence_zyx: np.ndarray,
) -> np.ndarray:
    """Build a deterministic VED-like vessel continuity prior from one 3D evidence map."""
    out = np.zeros_like(evidence_zyx, dtype=np.float32)
    mask = np.asarray(evidence_zyx > 0.0, dtype=bool)
    sigma = max(0.25, TC4D_VED_PRECONDITION_SMOOTH_SIGMA)
    n_iters = max(0, TC4D_VED_PRECONDITION_DIFFUSION_ITERS)
    step = np.clip(TC4D_VED_PRECONDITION_DIFFUSION_STEP, 0.0, 1.0)
    if not np.any(mask):
        return out

    evidence = np.nan_to_num(
        np.asarray(evidence_zyx, dtype=np.float32),
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    evidence = np.clip(evidence, 0.0, None)
    masked_vals_full = evidence[mask]
    if masked_vals_full.size >= MIN_OTSU_SAMPLES:
        q = float(
            np.percentile(
                masked_vals_full,
                np.clip(TC4D_VED_PRECONDITION_MASK_PERCENTILE, 5.0, 95.0),
            )
        )
        roi_seed = mask & (evidence >= q)
        if np.count_nonzero(roi_seed) < MIN_ROI_SEED_VOXELS:
            roi_seed = mask
    else:
        roi_seed = mask

    coords = np.argwhere(roi_seed)
    if coords.size == 0:
        coords = np.argwhere(mask)
    lo = np.min(coords, axis=0)
    hi = np.max(coords, axis=0) + 1
    pad = int(max(2, TC4D_VED_PRECONDITION_ROI_PAD))
    shape = np.asarray(evidence.shape)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, shape)
    roi = (
        slice(lo[0], hi[0]),
        slice(lo[1], hi[1]),
        slice(lo[2], hi[2]),
    )
    evidence_roi = np.asarray(evidence[roi], dtype=np.float32)
    mask_roi = np.asarray(mask[roi], dtype=bool)
    roi_shape = evidence_roi.shape
    roi_voxels = roi_shape[0] * roi_shape[1] * roi_shape[2]

    max_work_voxels = max(500_000, TC4D_VED_PRECONDITION_MAX_WORK_VOXELS)
    downsample = 1
    if roi_voxels > max_work_voxels:
        downsample = math.ceil((roi_voxels / float(max_work_voxels)) ** (1.0 / 3.0))
        downsample = max(1, min(4, downsample))
    ds = int(max(1, downsample))
    if ds > 1:
        work = (
            slice(0, None, ds),
            slice(0, None, ds),
            slice(0, None, ds),
        )
        evidence_work = np.asarray(evidence_roi[work], dtype=np.float32)
        mask_work = np.asarray(mask_roi[work], dtype=bool)
        mask_work_voxels = np.count_nonzero(mask_work)
        if mask_work_voxels < MIN_ROI_VOXELS:
            ds = 1
            evidence_work = evidence_roi
            mask_work = mask_roi
    else:
        evidence_work = evidence_roi
        mask_work = mask_roi

    evidence_norm = _normalize_on_mask(evidence_work, mask_work)
    evidence_norm *= mask_work.astype(np.float32, copy=False)

    smooth = ndimage.gaussian_filter(
        evidence_norm.astype(np.float32, copy=False),
        sigma=sigma,
        mode="reflect",
    ).astype(np.float32, copy=False)
    smooth = _normalize_on_mask(smooth, mask_work)
    smooth *= mask_work.astype(np.float32, copy=False)

    frangi_score = _compute_frangi_tubularity_3d(
        smooth,
        mask_zyx=mask_work,
    )
    frangi_score = np.clip(frangi_score, 0.0, 1.0)
    diffusion = smooth.copy()
    if n_iters > 0 and step > 0.0:
        guidance = np.clip((0.25 + (0.75 * frangi_score)), 0.25, 1.0)
        for _ in range(n_iters):
            blurred = ndimage.gaussian_filter(
                diffusion,
                sigma=sigma,
                mode="reflect",
            )
            diffusion = diffusion + (step * guidance * (blurred - diffusion))
            diffusion *= mask_work

    ved_like = np.maximum.reduce(
        [
            evidence_norm,
            diffusion,
            frangi_score,
        ]
    )
    ved_like = _normalize_on_mask(ved_like, mask_work)
    ved_like *= mask_work

    if ds > 1:
        ved_like = np.repeat(ved_like, ds, axis=0)
        ved_like = np.repeat(ved_like, ds, axis=1)
        ved_like = np.repeat(ved_like, ds, axis=2)
        ved_like = ved_like[
            : roi_shape[0],
            : roi_shape[1],
            : roi_shape[2],
        ]

    out_roi = np.zeros_like(evidence_roi, dtype=np.float32)
    out_roi[mask_roi] = ved_like[mask_roi]
    out[roi] = out_roi
    out = _normalize_on_mask(out, mask)
    out *= mask
    return out


def _build_tc4d_candidates(
    priority_4d: np.ndarray,
    *,
    ved_precondition_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an over-complete 4D candidate centerline manifold."""
    t_dim = priority_4d.shape[0]
    candidate_4d = np.zeros_like(priority_4d, dtype=bool)
    center_score_4d = np.zeros_like(priority_4d, dtype=np.float32)
    persistence_score_4d = np.zeros_like(priority_4d, dtype=np.float32)
    ved_blend = TC4D_VED_PRECONDITION_BLEND_WEIGHT
    if ved_precondition_zyx.shape != priority_4d.shape[1:]:
        raise ValueError(
            "ved precondition shape mismatch in candidate generation: "
            f"{ved_precondition_zyx.shape} vs {priority_4d.shape[1:]}"
        )
    ved_gain_map = np.clip(
        np.nan_to_num(
            ved_precondition_zyx.astype(np.float32, copy=False),
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    for t in range(t_dim):
        priority_t_raw = priority_4d[t].astype(np.float32, copy=False)
        priority_t = np.clip(
            priority_t_raw * (1.0 + (ved_blend * ved_gain_map)),
            0.0,
            1.0,
        )
        base_threshold = _compute_otsu_threshold(
            priority_t_raw,
            clip_bounds=(TC4D_THRESHOLD_LOWER_BOUND, TC4D_THRESHOLD_UPPER_BOUND),
            empty_fallback=0.5,
        )
        weak_hysteresis_threshold = base_threshold
        candidate_t = np.zeros_like(priority_t, dtype=bool)
        center_t = np.zeros_like(priority_t, dtype=np.float32)
        persistence_t = np.zeros_like(priority_t, dtype=np.float32)

        edges = skeletonize3d(priority_t, threshold=base_threshold)
        ridge = edges > 0
        vessel_t = priority_t >= base_threshold
        if np.any(ridge) and np.any(vessel_t):
            labels, n_comp = _label_components_26(vessel_t)
            if n_comp > 0:
                seed_labels = np.unique(labels[ridge])
                seed_labels = seed_labels[seed_labels > 0]
                if seed_labels.size > 0:
                    candidate_thr = np.isin(labels, seed_labels)
                else:
                    candidate_thr = ridge
            else:
                candidate_thr = ridge
        else:
            candidate_thr = ridge

        candidate_t |= candidate_thr
        if np.any(candidate_thr):
            center_t[candidate_thr] = np.maximum(
                center_t[candidate_thr],
                priority_t[candidate_thr],
            )
            persistence_t[candidate_thr] += 1.0

        weak_mask_t = priority_t >= weak_hysteresis_threshold
        lower_values = priority_t[(priority_t > 0) & ~weak_mask_t]
        if np.any(candidate_t) and lower_values.size >= MIN_OTSU_SAMPLES:
            weak_hysteresis_threshold = np.clip(
                _compute_otsu_threshold(lower_values),
                TC4D_THRESHOLD_LOWER_BOUND * 0.5,
                weak_hysteresis_threshold,
            )
            if weak_hysteresis_threshold < base_threshold:
                weak_mask_t = priority_t >= weak_hysteresis_threshold
                labels_w, n_comp_w = _label_components_26(weak_mask_t)
                if n_comp_w > 0:
                    seed_labels_w = np.unique(labels_w[candidate_t])
                    seed_labels_w = seed_labels_w[seed_labels_w > 0]
                    if seed_labels_w.size > 0:
                        candidate_weak = np.isin(labels_w, seed_labels_w) & ~candidate_t
                        if np.any(candidate_weak):
                            candidate_t |= candidate_weak
                            center_t[candidate_weak] = np.maximum(
                                center_t[candidate_weak],
                                priority_t[candidate_weak],
                            )
                            persistence_t[candidate_weak] += 0.5

        if not np.any(candidate_t):
            # Safety fallback for degenerate maps: keep top-ranked voxels.
            flat = priority_t.reshape(-1)
            keep_n = max(32, int(math.ceil(flat.size * 1e-5)))
            keep_n = min(keep_n, flat.size)
            top_idx = np.argpartition(flat, -keep_n)[-keep_n:]
            candidate_t.reshape(-1)[top_idx] = True
            center_t[candidate_t] = np.maximum(
                center_t[candidate_t],
                priority_t[candidate_t],
            )
            persistence_t[candidate_t] = 1.0

        candidate_4d[t] = candidate_t
        center_score_4d[t] = center_t
        persistence_score_4d[t] = persistence_t

        if t_dim > 1:
            for cand_t in range(t_dim):
                neigh = np.zeros_like(candidate_4d[cand_t], dtype=bool)
                for dt in (-1, 1):
                    nt = cand_t + dt
                    if nt < 0 or nt >= t_dim:
                        continue
                    neigh |= ndimage.binary_dilation(
                        candidate_4d[nt],
                        structure=STRUCTURE_3X3X3_U8,
                        iterations=2,
                    )
                if not np.any(neigh):
                    continue
                propagated = neigh & weak_mask_t & ~candidate_4d[cand_t]
                added = np.count_nonzero(propagated)
                if added <= 0:
                    continue
                candidate_4d[cand_t] |= propagated
                center_score_4d[cand_t][propagated] = np.maximum(
                    center_score_4d[cand_t][propagated],
                    priority_4d[cand_t][propagated],
                )
                persistence_score_4d[cand_t][propagated] += 0.25

    return candidate_4d, center_score_4d, persistence_score_4d


def _temporal_neighbor_support(mask_4d: np.ndarray) -> np.ndarray:
    """Count temporal neighbors using dt=+-1 with 3x3x3 spatial tolerance."""
    t_dim = mask_4d.shape[0]
    support = np.zeros_like(mask_4d, dtype=np.float32)

    for t in range(t_dim):
        acc = np.zeros(mask_4d.shape[1:], dtype=np.float32)
        for dt in (-1, 1):
            nt = t + dt
            if nt < 0 or nt >= t_dim:
                continue
            dilated = ndimage.binary_dilation(
                mask_4d[nt],
                structure=STRUCTURE_3X3X3_U8,
            )
            acc += dilated
        support[t] = acc

    return support


def _compute_spacetime_selection_prior_3d(
    *,
    mask_4d: np.ndarray,
    priority_4d: np.ndarray,
    temporal_flow_score_4d: np.ndarray,
    support_mask_zyx: np.ndarray,
    support_count_zyx: np.ndarray,
) -> np.ndarray:
    """Build a deterministic 3D prior from 4D consistency, without bridge insertion.

    This is the "selection weighting" path: we derive a spacetime-consistency prior
    and feed it into existing scoring/selection stages, rather than adding explicit
    4D bridge voxels.
    """
    support_mask = support_mask_zyx.astype(bool, copy=False)
    active_4d = mask_4d.astype(bool, copy=False)

    priority_norm_4d = _normalize_on_mask(
        priority_4d.astype(np.float32, copy=False),
        active_4d,
    )
    flow_norm_4d = _normalize_on_mask(
        temporal_flow_score_4d.astype(np.float32, copy=False),
        active_4d,
    )

    frame_score_4d = (0.58 * priority_norm_4d) + (0.42 * flow_norm_4d)
    frame_score_4d *= active_4d.astype(np.float32, copy=False)

    max_score_3d = np.max(frame_score_4d, axis=0)
    active_count_3d = np.count_nonzero(active_4d, axis=0).astype(np.float32, copy=False)
    mean_score_3d = np.divide(
        np.sum(frame_score_4d, axis=0),
        np.maximum(active_count_3d, 1.0),
    )
    support_norm_3d = np.clip(
        support_count_zyx.astype(np.float32, copy=False) / max(1, mask_4d.shape[0]),
        0.0,
        1.0,
    )

    temporal_neighbor_weight = np.clip(
        TC4D_SPACETIME_SELECTION_PRIOR_TEMPORAL_NEIGHBOR_WEIGHT,
        0.0,
        0.45,
    )
    temporal_neighbor_4d = _temporal_neighbor_support(active_4d)
    temporal_neighbor_3d = np.max(temporal_neighbor_4d, axis=0)
    temporal_neighbor_3d = np.clip(temporal_neighbor_3d / 2.0, 0.0, 1.0)

    base_weight = 1.0 - temporal_neighbor_weight
    base_mix = (0.40 * max_score_3d) + (0.30 * mean_score_3d) + (0.30 * support_norm_3d)
    prior = (base_weight * base_mix) + (temporal_neighbor_weight * temporal_neighbor_3d)
    prior = _normalize_on_mask(prior, support_mask)
    prior *= support_mask.astype(np.float32, copy=False)
    return prior


def _select_tc4d_consensus(
    candidate_4d: np.ndarray,
    priority_4d: np.ndarray,
    center_score_4d: np.ndarray,
    persistence_score_4d: np.ndarray,
) -> np.ndarray:
    """Select a temporally consistent manifold on the candidate graph."""
    evidence_weight = 0.9
    center_weight = 0.9
    persistence_weight = 1.5
    temporal_weight = 2.0
    retain_floor_fraction = 0.78
    n_terms = evidence_weight + center_weight + persistence_weight + temporal_weight

    evidence_norm_4d = np.zeros_like(priority_4d, dtype=np.float32)
    center_norm_4d = np.zeros_like(priority_4d, dtype=np.float32)
    for t in range(priority_4d.shape[0]):
        cand_t = candidate_4d[t]
        evidence_norm_4d[t] = _normalize_on_mask(priority_4d[t], cand_t)
        center_norm_4d[t] = _normalize_on_mask(center_score_4d[t], cand_t)

    temporal_support = _temporal_neighbor_support(candidate_4d)
    selected_next = np.zeros_like(candidate_4d, dtype=bool)

    for t in range(candidate_4d.shape[0]):
        cand_t = candidate_4d[t]
        if not np.any(cand_t):
            continue

        temporal_score = np.clip(temporal_support[t] / 2.0, 0.0, 1.0)
        total_score = (
            (evidence_weight * evidence_norm_4d[t])
            + (center_weight * center_norm_4d[t])
            + (persistence_weight * persistence_score_4d[t])
            + (temporal_weight * temporal_score)
        )
        total_score = total_score / (n_terms + 1e-6)

        cand_total = total_score[cand_t]
        score_cutoff = _compute_otsu_threshold(cand_total)
        keep_t = cand_t & (total_score >= score_cutoff)

        n_cand = cand_total.size
        min_keep = max(1, math.ceil(retain_floor_fraction * n_cand))
        if np.count_nonzero(keep_t) < min_keep:
            kth = max(0, n_cand - min_keep)
            score_cutoff = np.partition(cand_total, kth)[kth]
            keep_t = cand_t & (total_score >= score_cutoff)

        if not np.any(keep_t):
            # Last-resort fallback for this timepoint.
            fallback_idx = np.argmax(cand_total)
            keep_positions = np.flatnonzero(cand_t)
            keep_flat = keep_positions[fallback_idx]
            keep_t.reshape(-1)[keep_flat] = True

        selected_next[t] = keep_t

    return selected_next


def _line_voxels_3d(
    p0: tuple[int, int, int],
    p1: tuple[int, int, int],
) -> np.ndarray:
    """Return integer 3D voxels on a straight line between two points."""
    z0, y0, x0 = p0
    z1, y1, x1 = p1
    steps = max(abs(z1 - z0), abs(y1 - y0), abs(x1 - x0), 1)
    zz = np.rint(np.linspace(z0, z1, num=steps + 1)).astype(np.int32)
    yy = np.rint(np.linspace(y0, y1, num=steps + 1)).astype(np.int32)
    xx = np.rint(np.linspace(x0, x1, num=steps + 1)).astype(np.int32)
    coords = np.stack([zz, yy, xx], axis=1)
    return np.unique(coords, axis=0)


def _repair_exam_skeleton_components(
    exam_mask: np.ndarray,
    *,
    support_count_zyx: np.ndarray,
) -> np.ndarray:
    """Repair disconnected exam components with support-guided straight bridges."""
    if not np.any(exam_mask):
        return exam_mask

    repaired = exam_mask.copy()
    support_mask = support_count_zyx >= 1
    max_gap_voxels = float(TC4D_COMPONENT_REPAIR_MAX_GAP_VOXELS)

    # Greedy component reconnect: repeatedly link nearest non-anchor components
    # back to the current anchor set when the gap is reasonably small.
    for _ in range(TC4D_COMPONENT_REPAIR_MAX_ITERATIONS):
        labels_cur, n_cur = _label_components_26(repaired)
        if n_cur <= 1:
            break
        sizes_cur = np.bincount(labels_cur.ravel())
        sizes_cur[0] = 0
        anchor_label = np.argmax(sizes_cur)
        anchor_mask = labels_cur == anchor_label
        anchor_dt, anchor_idx = ndimage.distance_transform_edt(
            ~anchor_mask,
            return_indices=True,
        )

        best_dist = np.inf
        best_line: np.ndarray | None = None
        for lab in range(1, n_cur + 1):
            if lab == anchor_label:
                continue
            comp_mask = labels_cur == lab
            dist_vals = anchor_dt[comp_mask]
            min_idx_local = np.argmin(dist_vals)
            min_dist = float(dist_vals[min_idx_local])
            if min_dist > max_gap_voxels:
                continue
            comp_coords = np.argwhere(comp_mask)
            src = tuple(comp_coords[min_idx_local])
            dst = tuple(anchor_idx[:, src[0], src[1], src[2]])
            line = _line_voxels_3d(src, dst)
            line_z = np.clip(line[:, 0], 0, repaired.shape[0] - 1)
            line_y = np.clip(line[:, 1], 0, repaired.shape[1] - 1)
            line_x = np.clip(line[:, 2], 0, repaired.shape[2] - 1)
            line_support = support_mask[line_z, line_y, line_x]
            if not np.any(line_support):
                continue
            supported_line = np.stack(
                [line_z[line_support], line_y[line_support], line_x[line_support]],
                axis=1,
            )
            if min_dist < best_dist:
                best_dist = min_dist
                best_line = supported_line

        if best_line is None:
            break

        line_arr = np.asarray(best_line, dtype=np.int32)
        before = np.count_nonzero(repaired)
        repaired[line_arr[:, 0], line_arr[:, 1], line_arr[:, 2]] = True
        added = np.count_nonzero(repaired) - before
        if added <= 0:
            break
    return repaired


def _compute_coarse_dynamic_score_3d(
    source_4d: np.ndarray,
    *,
    mask_zyx: np.ndarray,
) -> np.ndarray:
    """Build a robust coarse dynamic score map from a 4D time series."""
    t_dim = source_4d.shape[0]
    pre = source_4d[0].astype(np.float32, copy=False)
    peak = pre.copy()
    rise = np.zeros_like(pre, dtype=np.float32)
    prev = pre
    for t in range(1, t_dim):
        cur = source_4d[t].astype(np.float32, copy=False)
        np.maximum(peak, cur, out=peak)
        delta = cur - prev
        np.maximum(rise, np.maximum(delta, 0.0), out=rise)
        prev = cur
    peak_minus_pre = np.maximum(peak - pre, 0.0)

    peak_norm = _normalize_on_mask(peak_minus_pre, mask_zyx)
    rise_norm = _normalize_on_mask(rise, mask_zyx)
    return (0.65 * peak_norm) + (0.35 * rise_norm)


def _compute_temporal_edge_persistence_score_3d(
    selected_4d: np.ndarray,
    *,
    support_mask_zyx: np.ndarray,
) -> np.ndarray:
    """Estimate temporal persistence of local edges from repeated 3D adjacencies."""
    out = np.zeros(support_mask_zyx.shape, dtype=np.float32)

    t_dim = selected_4d.shape[0]
    if t_dim < TEMPORAL_CHAIN_LEN_SHORT:
        return out

    shape = support_mask_zyx.shape
    sel = selected_4d.astype(bool, copy=False)
    accum = np.zeros(shape, dtype=np.float32)
    accum_weight = np.zeros(shape, dtype=np.float32)

    for src_slice, dst_slice, _edge_len in _iter_offset_slices(
        shape,
        EDGE_PERSISTENCE_NEIGHBOR_OFFSETS,
    ):
        subshape = tuple(sl.stop - sl.start for sl in src_slice)
        edge_count = np.zeros(subshape, dtype=np.uint8)
        edge_consecutive = np.zeros(subshape, dtype=np.uint8)
        prev_edge: np.ndarray | None = None
        for t in range(t_dim):
            edge_t = sel[t][src_slice] & sel[t][dst_slice]
            edge_count += edge_t
            if prev_edge is not None:
                edge_consecutive += edge_t & prev_edge
            prev_edge = edge_t

        active_edge = edge_count > 0
        if not np.any(active_edge):
            continue

        edge_frac = edge_count.astype(np.float32, copy=False) / float(max(1, t_dim))
        consec_frac = edge_consecutive.astype(np.float32, copy=False) / float(
            max(1, t_dim - 1)
        )
        edge_score = (0.60 * edge_frac) + (0.40 * consec_frac)
        edge_score *= active_edge

        accum[src_slice] += edge_score
        accum[dst_slice] += edge_score
        accum_weight[src_slice] += active_edge
        accum_weight[dst_slice] += active_edge

    valid = accum_weight > 0
    out[valid] = accum[valid] / np.maximum(accum_weight[valid], 1e-6)
    out = _normalize_on_mask(out, support_mask_zyx)
    out *= support_mask_zyx
    return out


def _compute_tubularity_and_blobness_from_mask(
    mask_zyx: np.ndarray,
    *,
    dynamics_score_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute low-cost tubularity and blobness proxies from local 3D occupancy."""
    mask_u8 = mask_zyx.astype(np.uint8, copy=False)
    neighbor_count = ndimage.convolve(
        mask_u8,
        STRUCTURE_3X3X3_U8,
        mode="constant",
        cval=0,
    )
    neighbor_count = neighbor_count - mask_u8
    n = neighbor_count.astype(np.float32, copy=False)

    # Tube-like centerlines have moderate local degree, while large blobs show
    # very high isotropic neighborhood occupancy.
    mask_f32 = mask_zyx.astype(np.float32, copy=False)
    tubularity = np.exp(-((n - 4.0) ** 2) / (2.0 * (3.5**2))).astype(
        np.float32,
        copy=False,
    )
    tubularity *= mask_f32
    blobness = np.clip((n - 10.0) / 12.0, 0.0, 1.0)
    blobness *= mask_f32

    relax = float(np.clip(GEODESIC_BLOB_RELAX_WITH_DYNAMICS, 0.0, 0.9))
    blobness *= 1.0 - (relax * np.clip(dynamics_score_zyx, 0.0, 1.0))

    return np.clip(tubularity, 0.0, 1.0), np.clip(blobness, 0.0, 1.0)


def _build_tc4d_node_and_traversal_fields(
    *,
    selected_4d: np.ndarray,
    priority_4d: np.ndarray,
    support_mask_zyx: np.ndarray,
    support_count_zyx: np.ndarray,
    edge_persistence_score_zyx: np.ndarray,
    temporal_flow_score_zyx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build deterministic node/edge scalar fields for geodesic connected selection."""
    mask = support_mask_zyx.astype(bool, copy=False)

    priority_4d_f32 = priority_4d.astype(np.float32, copy=False)
    evidence_3d = np.zeros(priority_4d_f32.shape[1:], dtype=np.float32)
    for t in range(priority_4d_f32.shape[0]):
        layer = priority_4d_f32[t] * selected_4d[t]
        np.maximum(evidence_3d, layer, out=evidence_3d)
    if not np.any(evidence_3d[mask]):
        evidence_3d = np.max(priority_4d_f32, axis=0)
    evidence_norm = _normalize_on_mask(evidence_3d, mask)

    t_dim = selected_4d.shape[0]
    temporal_norm = np.clip(
        support_count_zyx.astype(np.float32, copy=False) / float(max(1, t_dim)),
        0.0,
        1.0,
    )

    dynamics_score = _compute_coarse_dynamic_score_3d(
        priority_4d.astype(np.float32, copy=False),
        mask_zyx=mask,
    )
    temporal_flow_norm = _normalize_on_mask(
        np.clip(
            temporal_flow_score_zyx.astype(np.float32, copy=False),
            0.0,
            1.0,
        ),
        mask,
    )

    edge_persistence_norm = _normalize_on_mask(
        np.clip(
            edge_persistence_score_zyx.astype(np.float32, copy=False),
            0.0,
            1.0,
        ),
        mask,
    )
    temporal_flow_weight = np.clip(TC4D_TEMPORAL_FLOW_SOFT_WEIGHT, 0.0, 0.8)

    tubularity_score, blobness_penalty = _compute_tubularity_and_blobness_from_mask(
        mask,
        dynamics_score_zyx=dynamics_score,
    )
    frangi_source = (0.60 * evidence_norm) + (0.40 * dynamics_score)
    frangi_score = _compute_frangi_tubularity_3d(
        frangi_source,
        mask_zyx=mask,
    )
    tubularity_score = np.clip(
        np.maximum(tubularity_score, frangi_score),
        0.0,
        1.0,
    )
    blobness_penalty = np.clip(
        blobness_penalty * (1.0 - (0.25 * frangi_score)),
        0.0,
        1.0,
    )

    combined_raw = (
        (0.95 * evidence_norm)
        + (0.85 * temporal_norm)
        + (0.70 * dynamics_score)
        + (1.10 * tubularity_score)
        + (TC4D_EDGE_PERSISTENCE_NODE_WEIGHT * edge_persistence_norm)
        + (TC4D_FRANGI_NODE_WEIGHT * frangi_score)
        + (1.10 * temporal_flow_weight * temporal_flow_norm)
        - (0.75 * blobness_penalty)
    )
    node_prize = _normalize_on_mask(combined_raw, mask)

    traversal_base = -np.log(np.clip(evidence_norm, 1e-3, 1.0)).astype(
        np.float32,
        copy=False,
    )
    traversal_temporal = 1.0 + (0.45 * (1.0 - temporal_norm))
    traversal_blob = 1.0 + (1.35 * blobness_penalty)
    traversal_tube_relief = np.clip(
        0.30
        + (0.50 * tubularity_score)
        + (0.45 * frangi_score)
        + (0.55 * temporal_flow_weight * temporal_flow_norm)
        + (TC4D_EDGE_PERSISTENCE_TRAVERSAL_RELIEF * edge_persistence_norm),
        0.20,
        1.10,
    )
    traversal_dyn_relief = np.clip(0.75 + (0.25 * dynamics_score), 0.75, 1.0)
    traversal_cost = (traversal_base * traversal_temporal * traversal_blob) / (
        traversal_tube_relief * traversal_dyn_relief
    )
    traversal_cost[~mask] = np.inf

    seed_score = (
        node_prize
        + (0.35 * temporal_norm)
        + (0.10 * dynamics_score)
        + (0.20 * tubularity_score)
        + (TC4D_EDGE_PERSISTENCE_SEED_WEIGHT * edge_persistence_norm)
        + (0.25 * frangi_score)
        + (0.80 * temporal_flow_weight * temporal_flow_norm)
    )
    seed_score[~mask] = -np.inf

    return {
        "node_prize": node_prize,
        "traversal_cost": traversal_cost,
        "seed_score": seed_score,
    }


def _select_geodesic_connected_subtree(
    *,
    support_mask_zyx: np.ndarray,
    traversal_mask_zyx: np.ndarray,
    node_prize_zyx: np.ndarray,
    traversal_cost_zyx: np.ndarray,
    seed_score_zyx: np.ndarray,
) -> np.ndarray:
    """Select a connected subtree via single-root geodesic tree + prize-minus-cost DP."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    support_mask = support_mask_zyx.astype(bool, copy=False)
    traversal_or_support = traversal_mask_zyx.astype(bool, copy=False) | support_mask
    finite_traversal = np.isfinite(traversal_cost_zyx)
    mask = traversal_or_support & finite_traversal
    if not np.any(mask):
        mask = support_mask.copy()

    if not np.any(support_mask):
        return np.zeros_like(support_mask, dtype=bool)

    work_nodes = np.count_nonzero(mask)
    if work_nodes > GEODESIC_MAX_WORK_NODES:
        keep_quantile = np.clip(
            100.0 * (1.0 - (GEODESIC_MAX_WORK_NODES / work_nodes)),
            0.0,
            99.9,
        )
        keep_threshold = np.percentile(seed_score_zyx[mask], keep_quantile)
        mask = mask & (seed_score_zyx >= keep_threshold)
        mask |= support_mask
        if np.count_nonzero(mask) < max(1024, GEODESIC_MAX_WORK_NODES // 4):
            mask = traversal_or_support & finite_traversal

    node_ids = -np.ones(mask.shape, dtype=np.int32)
    support_coords = np.argwhere(mask)
    n_nodes = support_coords.shape[0]
    node_ids[mask] = np.arange(n_nodes, dtype=np.int32)
    support_on_work = support_mask[mask]

    finite_seed_vals = seed_score_zyx[mask]
    seed_mask = mask
    if finite_seed_vals.size > 0:
        seed_threshold = np.percentile(finite_seed_vals, GEODESIC_SEED_QUANTILE)
        seed_mask = mask & (seed_score_zyx >= seed_threshold)
    seed_voxel_flat = np.flatnonzero(seed_mask)
    if seed_voxel_flat.size == 0:
        seed_voxel_flat = np.flatnonzero(mask)
    seed_order = np.argsort(seed_score_zyx.ravel()[seed_voxel_flat])[::-1][
        :GEODESIC_MAX_SEEDS
    ]
    seed_node_ids_arr = node_ids.ravel()[seed_voxel_flat[seed_order]]

    node_cost = traversal_cost_zyx[mask].astype(np.float32, copy=False)
    node_prize = np.clip(node_prize_zyx[mask].astype(np.float32, copy=False), 0.0, 1.0)
    dist = np.full(n_nodes, np.inf, dtype=np.float32)
    parent = np.full(n_nodes, -1, dtype=np.int32)
    ROOT_SENTINEL = np.int32(-2)
    virtual_root_id = n_nodes
    try:
        rows_chunks: list[np.ndarray] = []
        cols_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        for src_slice, dst_slice, edge_len in _iter_offset_slices(
            mask.shape,
            GEODESIC_NEIGHBOR_OFFSETS_26,
        ):
            valid = mask[src_slice] & mask[dst_slice]
            if not np.any(valid):
                continue
            src_ids = node_ids[src_slice][valid]
            dst_ids = node_ids[dst_slice][valid]
            step_cost = (
                0.5 * (node_cost[src_ids] + node_cost[dst_ids]) * edge_len
            ).astype(np.float32, copy=False)
            rows_chunks.append(src_ids)
            cols_chunks.append(dst_ids)
            weight_chunks.append(step_cost)

        if not rows_chunks:
            raise ValueError("No edges were found in support mask geodesic graph.")

        rows = np.concatenate(rows_chunks, axis=0)
        cols = np.concatenate(cols_chunks, axis=0)
        weights = np.concatenate(weight_chunks, axis=0)
        vr_rows = np.full(seed_node_ids_arr.size, virtual_root_id, dtype=np.int32)
        vr_cols = seed_node_ids_arr
        vr_weights = np.zeros(seed_node_ids_arr.size, dtype=np.float32)
        rows = np.concatenate([rows, vr_rows], axis=0)
        cols = np.concatenate([cols, vr_cols], axis=0)
        weights = np.concatenate([weights, vr_weights], axis=0)
        graph = csr_matrix(
            (weights, (rows, cols)),
            shape=(n_nodes + 1, n_nodes + 1),
            dtype=np.float32,
        )
        dist_all, predecessors_all = dijkstra(
            csgraph=graph,
            directed=True,
            indices=virtual_root_id,
            return_predecessors=True,
            min_only=False,
        )
        dist = dist_all[:n_nodes].astype(np.float32, copy=False)
        predecessors = predecessors_all[:n_nodes].astype(np.int32, copy=False)
        parent.fill(-1)
        valid_pred = (predecessors >= 0) & (predecessors < n_nodes)
        parent[valid_pred] = predecessors[valid_pred]
        parent[predecessors == virtual_root_id] = ROOT_SENTINEL
        reachable_tmp = np.isfinite(dist)
        parent[reachable_tmp & (parent < 0)] = ROOT_SENTINEL
    except Exception as exc:
        raise ValueError("Scipy geodesic shortest-path solve failed.") from exc
    reachable = np.isfinite(dist)
    reachable_ids = np.flatnonzero(reachable)
    if reachable_ids.size == 0:
        return np.zeros_like(mask, dtype=bool)

    parent_edge_cost = np.zeros(n_nodes, dtype=np.float32)
    child_ids = np.flatnonzero((parent > -1) & reachable)
    parent_ids = parent[child_ids]
    child_coords = support_coords[child_ids]
    parent_coords = support_coords[parent_ids]
    deltas = (child_coords - parent_coords).astype(np.float32, copy=False)
    edge_lens = np.sqrt(np.sum(deltas * deltas, axis=1)).astype(np.float32, copy=False)
    edge_lens = np.clip(edge_lens, 1.0, 1.75)
    parent_edge_cost[child_ids] = (
        0.5 * (node_cost[child_ids] + node_cost[parent_ids]) * edge_lens
    )

    reachable_support = reachable & support_on_work
    reachable_support_count = np.count_nonzero(reachable_support)
    if reachable_support_count:
        prize_floor = np.percentile(
            node_prize[reachable_support], GEODESIC_BETA_QUANTILE
        )
    else:
        prize_floor = np.percentile(node_prize[reachable], GEODESIC_BETA_QUANTILE)
    node_gain = node_prize - prize_floor

    median_edge_cost = (
        np.median(parent_edge_cost[child_ids]) if child_ids.size > 0 else 1.0
    )
    median_node_prize = (
        np.median(node_prize[reachable_support])
        if reachable_support_count
        else np.median(node_prize[reachable])
    )
    positive_reachable = reachable_support & (node_gain > 0.0)
    positive_gain = node_gain[positive_reachable]
    median_positive_gain = np.median(positive_gain) if positive_gain.size > 0 else 0.0
    gain_for_lambda = max(median_positive_gain, 0.05 * median_node_prize, 1e-3)
    lambda_edge = np.clip(
        (gain_for_lambda / max(median_edge_cost, 1e-6))
        * GEODESIC_EDGE_COST_LAMBDA_SCALE,
        GEODESIC_EDGE_COST_LAMBDA_MIN,
        GEODESIC_EDGE_COST_LAMBDA_MAX,
    )

    order_asc = reachable_ids[np.argsort(dist[reachable_ids])]
    order_desc = order_asc[::-1]
    root_nodes = np.flatnonzero((parent == ROOT_SENTINEL) & reachable)
    root_nodes = root_nodes if root_nodes.size else seed_node_ids_arr

    dp = np.full(n_nodes, -np.inf, dtype=np.float32)
    dp[reachable] = node_gain[reachable]
    for u in order_desc:
        pid = parent[u]
        if pid >= 0 and np.isfinite(dp[u]) and np.isfinite(dp[pid]):
            contribution = dp[u] - (lambda_edge * parent_edge_cost[u])
            if contribution > 0.0:
                dp[pid] = dp[pid] + contribution

    include = np.zeros(n_nodes, dtype=bool)
    include[root_nodes] = dp[root_nodes] > 0.0
    if not np.any(include[root_nodes]):
        include[root_nodes[node_prize[root_nodes].argmax()]] = True
    for u in order_asc:
        pu = parent[u]
        if pu >= 0 and include[pu] and np.isfinite(dp[u]):
            contribution = dp[u] - (lambda_edge * parent_edge_cost[u])
            if contribution > 0.0:
                include[u] = True

    if not np.any(include & support_on_work):
        include[root_nodes[node_prize[root_nodes].argmax()]] = True

    selected_coords = support_coords[include]

    selected_mask = np.zeros_like(support_mask, dtype=bool)
    selected_mask[
        selected_coords[:, 0],
        selected_coords[:, 1],
        selected_coords[:, 2],
    ] = True

    return selected_mask


def _apply_temporal_directed_consistency_filter(
    mask_4d: np.ndarray,
) -> np.ndarray:
    """Compute temporal directed-consistency soft score map."""
    if mask_4d.ndim != NDIM_4D:
        raise ValueError(f"Expected 4D mask (t,z,y,x), got shape {mask_4d.shape}")

    active = mask_4d.astype(bool, copy=False)
    t_dim = active.shape[0]
    active_total = np.count_nonzero(active)
    if active_total == 0 or t_dim <= 1:
        return active.astype(np.float32, copy=False)

    radius = max(0, min(2, TC4D_TEMPORAL_FLOW_SPATIAL_RADIUS))
    footprint = np.ones(
        (2 * radius + 1, 2 * radius + 1, 2 * radius + 1),
        dtype=np.uint8,
    )
    binary_structure = footprint.astype(bool, copy=False)

    forward_len = np.zeros_like(active, dtype=np.uint8)
    backward_len = np.zeros_like(active, dtype=np.uint8)
    forward_len[0] = active[0].astype(np.uint8, copy=False)
    backward_len[-1] = active[-1].astype(np.uint8, copy=False)

    for t in range(1, t_dim):
        prev_best = ndimage.maximum_filter(
            forward_len[t - 1],
            footprint=footprint,
            mode="constant",
            cval=0,
        )
        step = np.minimum(
            prev_best.astype(np.uint16, copy=False) + 1,
            255,
        ).astype(np.uint8, copy=False)
        forward_len[t] = np.where(active[t], step, 0)

    for t in range(t_dim - 2, -1, -1):
        next_best = ndimage.maximum_filter(
            backward_len[t + 1],
            footprint=footprint,
            mode="constant",
            cval=0,
        )
        step = np.minimum(
            next_best.astype(np.uint16, copy=False) + 1,
            255,
        ).astype(np.uint8, copy=False)
        backward_len[t] = np.where(active[t], step, 0)

    chain_len = (
        forward_len.astype(np.int16, copy=False)
        + backward_len.astype(np.int16, copy=False)
        - 1
    ).astype(np.int16, copy=False)
    chain_len[~active] = 0
    inv_t_dim = 1.0 / t_dim
    consistency_score = np.clip(
        chain_len * inv_t_dim,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    consistency_score[~active] = 0.0

    required_chain_len = (
        TEMPORAL_CHAIN_LEN_SHORT
        if t_dim <= TEMPORAL_CHAIN_LEN_SHORT
        else TEMPORAL_CHAIN_LEN_LONG
    )
    filtered = active & (chain_len >= required_chain_len)
    retain_fraction = np.count_nonzero(filtered) / active_total

    target_retain_fraction = np.clip(TC4D_TEMPORAL_FLOW_MIN_RETAIN_FRACTION, 0.50, 0.98)
    if (
        retain_fraction < target_retain_fraction
        and required_chain_len > TEMPORAL_CHAIN_LEN_SHORT
    ):
        required_chain_len = TEMPORAL_CHAIN_LEN_SHORT
        filtered = active & (chain_len >= required_chain_len)
    bridge_score = max(required_chain_len, TEMPORAL_CHAIN_LEN_SHORT) * inv_t_dim

    if t_dim >= TEMPORAL_CHAIN_LEN_LONG:
        for t in range(1, t_dim - 1):
            if not np.any(active[t]):
                continue
            prev_context = ndimage.binary_dilation(
                filtered[t - 1],
                structure=binary_structure,
            )
            next_context = ndimage.binary_dilation(
                filtered[t + 1],
                structure=binary_structure,
            )
            bridge = active[t] & (~filtered[t]) & prev_context & next_context
            if np.any(bridge):
                filtered[t] |= bridge
                consistency_score[t][bridge] = np.maximum(
                    consistency_score[t][bridge],
                    bridge_score,
                )

    return consistency_score


def _finalize_exam_mask_topology(
    exam_mask_zyx: np.ndarray,
    *,
    support_count_zyx: np.ndarray,
) -> np.ndarray:
    """Apply final exam-mask cleanup steps after connectivity stages."""
    exam_mask_zyx = _filter_components_by_support(
        exam_mask_zyx,
        support_count_zyx=support_count_zyx,
    )

    if np.any(exam_mask_zyx):
        thin_mask = _skeletonize_support_mask(exam_mask_zyx)
        if np.any(thin_mask):
            exam_mask_zyx = thin_mask

    for cleanup_fn in (
        _filter_components_by_support,
        _repair_exam_skeleton_components,
        _filter_components_by_support,
    ):
        exam_mask_zyx = cleanup_fn(
            exam_mask_zyx,
            support_count_zyx=support_count_zyx,
        )

    exam_voxels = np.count_nonzero(exam_mask_zyx)
    micro_comp_min_voxels = max(8, round(float(exam_voxels) * 0.0015))
    exam_mask_zyx = _prune_micro_components(
        exam_mask_zyx,
        min_component_voxels=micro_comp_min_voxels,
    )

    return exam_mask_zyx


def _collapse_tc4d_with_temporal_hysteresis(
    mask_4d: np.ndarray,
    *,
    priority_4d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse 4D manifold with temporal hysteresis + component-scored pruning."""
    if mask_4d.ndim != NDIM_4D:
        raise ValueError(f"Expected 4D mask (t,z,y,x), got shape {mask_4d.shape}")

    min_temporal_support = min(2, mask_4d.shape[0])

    mask_4d = mask_4d.astype(bool, copy=False)
    temporal_flow_score_4d = _apply_temporal_directed_consistency_filter(mask_4d)

    temporal_flow_score_3d = np.max(
        np.where(mask_4d, temporal_flow_score_4d, 0.0),
        axis=0,
    ).astype(np.float32, copy=False)

    support_count = np.count_nonzero(mask_4d, axis=0)

    support_mask = support_count >= min_temporal_support
    if not np.any(support_mask):
        raise ValueError(
            "4D collapse produced an empty core support mask. "
            "Lower min-temporal-support or adjust graph consensus."
        )
    if min_temporal_support > 1:
        weak_mask = support_count >= (min_temporal_support - 1)
        if np.any(weak_mask & ~support_mask):
            labels, _ = _label_components_26(weak_mask)
            support_mask = np.isin(labels, np.unique(labels[support_mask]))

    prior_blend_weight = np.clip(
        TC4D_SPACETIME_SELECTION_PRIOR_BLEND,
        0.0,
        0.8,
    )
    spacetime_prior_3d = _compute_spacetime_selection_prior_3d(
        mask_4d=mask_4d,
        priority_4d=priority_4d,
        temporal_flow_score_4d=temporal_flow_score_4d,
        support_mask_zyx=support_mask,
        support_count_zyx=support_count,
    )
    if prior_blend_weight > 0.0 and np.any(spacetime_prior_3d[support_mask] > 0.0):
        flow_norm = _normalize_on_mask(
            temporal_flow_score_3d.astype(np.float32, copy=False),
            support_mask,
        )
        temporal_flow_score_3d = np.clip(
            ((1.0 - prior_blend_weight) * flow_norm)
            + (prior_blend_weight * spacetime_prior_3d),
            0.0,
            1.0,
        )
        temporal_flow_score_3d *= support_mask.astype(np.float32, copy=False)

    edge_persistence_score = _compute_temporal_edge_persistence_score_3d(
        mask_4d,
        support_mask_zyx=support_mask,
    )

    exam_mask_raw = _skeletonize_support_mask(support_mask)
    if not np.any(exam_mask_raw):
        raise ValueError(
            "Exam-level skeleton is empty after temporal hysteresis collapse."
        )
    exam_mask = _filter_components_by_support(
        exam_mask_raw,
        support_count_zyx=support_count,
    )

    exam_mask_pre_geodesic = exam_mask.copy()
    geodesic_fields = _build_tc4d_node_and_traversal_fields(
        selected_4d=mask_4d,
        priority_4d=priority_4d,
        support_mask_zyx=support_mask,
        support_count_zyx=support_count,
        edge_persistence_score_zyx=edge_persistence_score,
        temporal_flow_score_zyx=temporal_flow_score_3d,
    )

    workspace_mask = ndimage.binary_dilation(
        np.any(mask_4d, axis=0),
        structure=STRUCTURE_3X3X3_U8,
        iterations=GEODESIC_WORKSPACE_DILATION_ITERS,
    )
    workspace_mask |= support_mask
    geodesic_support_mask = _select_geodesic_connected_subtree(
        support_mask_zyx=support_mask,
        traversal_mask_zyx=workspace_mask,
        node_prize_zyx=geodesic_fields["node_prize"],
        traversal_cost_zyx=geodesic_fields["traversal_cost"],
        seed_score_zyx=geodesic_fields["seed_score"],
    )
    geodesic_exam_mask = np.zeros_like(geodesic_support_mask, dtype=bool)
    if np.any(geodesic_support_mask):
        exam_raw = _skeletonize_support_mask(geodesic_support_mask)
        if np.any(exam_raw):
            geodesic_exam_mask = _filter_components_by_support(
                exam_raw,
                support_count_zyx=support_count,
            )

    selected_support_voxels = np.count_nonzero(geodesic_support_mask)
    support_selection_fraction = np.count_nonzero(
        geodesic_support_mask & support_mask
    ) / max(1, np.count_nonzero(support_mask))
    if (
        selected_support_voxels >= GEODESIC_UNION_MIN_SELECTED_SUPPORT_VOXELS
        and support_selection_fraction >= GEODESIC_MIN_SUPPORT_SELECTION_FRACTION
    ):
        merged_exam = geodesic_exam_mask | exam_mask_pre_geodesic
        merged_exam = _filter_components_by_support(
            merged_exam,
            support_count_zyx=support_count,
        )
        exam_mask = merged_exam

    exam_mask = _finalize_exam_mask_topology(
        exam_mask,
        support_count_zyx=support_count,
    )

    return exam_mask, support_mask


def _run_tc4d_graph_consensus_pipeline(
    priority_4d: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run tc4d graph-forward no-learning pipeline."""
    t_total_start = time.perf_counter()
    ved_prior_src = np.max(
        priority_4d.astype(np.float32, copy=False),
        axis=0,
    ).astype(np.float32, copy=False)
    ved_precond_zyx = _compute_tc4d_ved_preconditioner_3d(ved_prior_src)

    candidate_t0 = time.perf_counter()
    candidate_4d, center_score_4d, persistence_score_4d = _build_tc4d_candidates(
        priority_4d,
        ved_precondition_zyx=ved_precond_zyx,
    )
    candidate_seconds = float(time.perf_counter() - candidate_t0)
    consensus_t0 = time.perf_counter()
    selected_4d = _select_tc4d_consensus(
        candidate_4d=candidate_4d,
        priority_4d=priority_4d,
        center_score_4d=center_score_4d,
        persistence_score_4d=persistence_score_4d,
    )
    consensus_seconds = float(time.perf_counter() - consensus_t0)
    manifold_seconds = float(time.perf_counter() - t_total_start)
    active_min_temporal_support = min(2, selected_4d.shape[0])
    collapse_t0 = time.perf_counter()
    exam_mask, support_mask = _collapse_tc4d_with_temporal_hysteresis(
        selected_4d,
        priority_4d=priority_4d,
    )
    if not np.any(exam_mask):
        raise ValueError("collapse produced empty exam skeleton")
    collapse_seconds = float(time.perf_counter() - collapse_t0)

    total_tc4d_seconds = float(time.perf_counter() - t_total_start)

    result = {
        "mask_4d": selected_4d,
        "exam_mask": exam_mask,
        "support_mask": support_mask,
        "effective_min_temporal_support": active_min_temporal_support,
        "manifold_seconds": manifold_seconds,
        "collapse_seconds": collapse_seconds,
    }
    params: dict[str, Any] = {
        "algorithm": "tc4d_graph_forward_no_learning",
        "policy": "tc4d_auto_ved_spt4d_subtree",
        "min_temporal_support": "auto",
    }
    diagnostics: dict[str, Any] = {
        "timing_seconds": {
            "candidate_generation": candidate_seconds,
            "graph_consensus": consensus_seconds,
            "collapse": collapse_seconds,
            "total": total_tc4d_seconds,
        },
    }
    return result, params, diagnostics


def run_tc4d_from_priority(
    priority_4d: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Public entrypoint for production tc4d extraction from `(t,z,y,x)` logits."""
    return _run_tc4d_graph_consensus_pipeline(priority_4d)
