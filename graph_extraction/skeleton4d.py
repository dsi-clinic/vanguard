"""Topology-preserving center-manifold extraction for 4D volumes."""

from __future__ import annotations

import numba as nb
import numpy as np

EXPECTED_DIMENSIONS_4D = 4
SPATIAL_NEIGHBOR_RADIUS_SQ = 3  # full 26-neighborhood in 3D when dt == 0
DIRECT_NEIGHBOR_COUNT = 2


def _build_offsets_4d(max_temporal_radius: int) -> np.ndarray:
    """Build 4D neighbor offsets (dt, dz, dy, dx)."""
    if max_temporal_radius < 0:
        raise ValueError("max_temporal_radius must be >= 0")

    offsets: list[tuple[int, int, int, int]] = []
    temporal_radius_sq = max_temporal_radius * max_temporal_radius

    for dt in (-1, 0, 1):
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dt == 0 and dz == 0 and dy == 0 and dx == 0:
                        continue
                    spatial_dist_sq = dz * dz + dy * dy + dx * dx
                    if dt == 0:
                        if 0 < spatial_dist_sq <= SPATIAL_NEIGHBOR_RADIUS_SQ:
                            offsets.append((dt, dz, dy, dx))
                    elif spatial_dist_sq <= temporal_radius_sq:
                        offsets.append((dt, dz, dy, dx))

    return np.asarray(offsets, dtype=np.int32)


@nb.njit
def _queue_push_4d(
    q_t: np.ndarray,
    q_z: np.ndarray,
    q_y: np.ndarray,
    q_x: np.ndarray,
    ss: np.ndarray,
    t: int,
    z: int,
    y: int,
    x: int,
) -> None:
    idx = ss[1]
    q_t[idx] = t
    q_z[idx] = z
    q_y[idx] = y
    q_x[idx] = x
    idx += 1
    if idx == q_t.shape[0]:
        idx = 0
    ss[1] = idx


@nb.njit
def _queue_pop_4d(
    q_t: np.ndarray,
    q_z: np.ndarray,
    q_y: np.ndarray,
    q_x: np.ndarray,
    ss: np.ndarray,
) -> tuple[int, int, int, int]:
    idx = ss[0]
    t = q_t[idx]
    z = q_z[idx]
    y = q_y[idx]
    x = q_x[idx]
    idx += 1
    if idx == q_t.shape[0]:
        idx = 0
    ss[0] = idx
    return t, z, y, x


@nb.njit
def _queue_empty(ss: np.ndarray) -> bool:
    return ss[0] == ss[1]


@nb.njit
def _queue_clear(ss: np.ndarray) -> None:
    ss[0] = ss[1]


@nb.njit
def _collect_neighbors_4d(
    nodes: np.ndarray,
    offsets: np.ndarray,
    t: int,
    z: int,
    y: int,
    x: int,
    n_t: np.ndarray,
    n_z: np.ndarray,
    n_y: np.ndarray,
    n_x: np.ndarray,
) -> int:
    tt, zz, yy, xx = nodes.shape
    n_neighbors = 0
    for b in range(offsets.shape[0]):
        nt = t + offsets[b, 0]
        nz = z + offsets[b, 1]
        ny = y + offsets[b, 2]
        nx = x + offsets[b, 3]
        if nt < 0 or nt >= tt:
            continue
        if nz < 0 or nz >= zz:
            continue
        if ny < 0 or ny >= yy:
            continue
        if nx < 0 or nx >= xx:
            continue
        if nodes[nt, nz, ny, nx]:
            n_t[n_neighbors] = nt
            n_z[n_neighbors] = nz
            n_y[n_neighbors] = ny
            n_x[n_neighbors] = nx
            n_neighbors += 1
    return n_neighbors


@nb.njit
def _offset_exists(offsets: np.ndarray, dt: int, dz: int, dy: int, dx: int) -> bool:
    for b in range(offsets.shape[0]):
        if (
            offsets[b, 0] == dt
            and offsets[b, 1] == dz
            and offsets[b, 2] == dy
            and offsets[b, 3] == dx
        ):
            return True
    return False


@nb.njit
def _neighbors_connected_after_removal(
    nodes: np.ndarray,
    offsets: np.ndarray,
    n_neighbors: int,
    n_t: np.ndarray,
    n_z: np.ndarray,
    n_y: np.ndarray,
    n_x: np.ndarray,
    touched: np.ndarray,
    stamp: np.int32,
    q_t: np.ndarray,
    q_z: np.ndarray,
    q_y: np.ndarray,
    q_x: np.ndarray,
    qss: np.ndarray,
    target_found: np.ndarray,
) -> tuple[bool, np.int32]:
    # Fast path: with exactly two neighbors, a direct edge guarantees
    # connectivity and avoids a full BFS.
    if n_neighbors == DIRECT_NEIGHBOR_COUNT:
        dt = n_t[1] - n_t[0]
        dz = n_z[1] - n_z[0]
        dy = n_y[1] - n_y[0]
        dx = n_x[1] - n_x[0]
        if _offset_exists(offsets, dt, dz, dy, dx):
            return True, stamp

    stamp = np.int32(stamp + 1)
    _queue_clear(qss)
    for i in range(1, n_neighbors):
        target_found[i] = 0
    remaining = n_neighbors - 1

    seed_t = n_t[0]
    seed_z = n_z[0]
    seed_y = n_y[0]
    seed_x = n_x[0]
    touched[seed_t, seed_z, seed_y, seed_x] = stamp
    _queue_push_4d(q_t, q_z, q_y, q_x, qss, seed_t, seed_z, seed_y, seed_x)

    tt, zz, yy, xx = nodes.shape
    while not _queue_empty(qss):
        ct, cz, cy, cx = _queue_pop_4d(q_t, q_z, q_y, q_x, qss)
        for b in range(offsets.shape[0]):
            nt = ct + offsets[b, 0]
            nz = cz + offsets[b, 1]
            ny = cy + offsets[b, 2]
            nx = cx + offsets[b, 3]
            if nt < 0 or nt >= tt:
                continue
            if nz < 0 or nz >= zz:
                continue
            if ny < 0 or ny >= yy:
                continue
            if nx < 0 or nx >= xx:
                continue
            if not nodes[nt, nz, ny, nx]:
                continue
            if touched[nt, nz, ny, nx] == stamp:
                continue
            touched[nt, nz, ny, nx] = stamp
            for ti in range(1, n_neighbors):
                if target_found[ti] == 1:
                    continue
                if nt == n_t[ti] and nz == n_z[ti] and ny == n_y[ti] and nx == n_x[ti]:
                    target_found[ti] = 1
                    remaining -= 1
                    break
            if remaining == 0:
                return True, stamp
            _queue_push_4d(q_t, q_z, q_y, q_x, qss, nt, nz, ny, nx)

    return False, stamp


@nb.njit
def _skeletonize4d_impl(
    nodes: np.ndarray,
    anchors: np.ndarray,
    offsets: np.ndarray,
    cand_t: np.ndarray,
    cand_z: np.ndarray,
    cand_y: np.ndarray,
    cand_x: np.ndarray,
    order: np.ndarray,
    min_voxels_per_timepoint: int,
    counts_per_t: np.ndarray,
) -> tuple[int, int, int]:
    removed = 0
    restored = 0
    skipped_min = 0

    n_active = int(np.count_nonzero(nodes))
    if n_active == 0:
        return removed, restored, skipped_min

    # BFS scratch structures.
    q_t = np.empty(n_active, dtype=np.int32)
    q_z = np.empty(n_active, dtype=np.int32)
    q_y = np.empty(n_active, dtype=np.int32)
    q_x = np.empty(n_active, dtype=np.int32)
    qss = np.zeros(2, dtype=np.int64)
    touched = np.zeros(nodes.shape, dtype=np.int32)
    stamp = np.int32(1)

    max_neighbors = offsets.shape[0]
    n_t = np.empty(max_neighbors, dtype=np.int32)
    n_z = np.empty(max_neighbors, dtype=np.int32)
    n_y = np.empty(max_neighbors, dtype=np.int32)
    n_x = np.empty(max_neighbors, dtype=np.int32)
    target_found = np.zeros(max_neighbors, dtype=np.uint8)

    for k in range(order.shape[0]):
        idx = order[k]
        t = cand_t[idx]
        z = cand_z[idx]
        y = cand_y[idx]
        x = cand_x[idx]

        if not nodes[t, z, y, x]:
            continue
        if anchors[t, z, y, x]:
            continue
        if counts_per_t[t] <= min_voxels_per_timepoint:
            skipped_min += 1
            continue

        n_neighbors = _collect_neighbors_4d(
            nodes, offsets, t, z, y, x, n_t, n_z, n_y, n_x
        )
        if n_neighbors <= 1:
            nodes[t, z, y, x] = False
            counts_per_t[t] -= 1
            removed += 1
            continue

        nodes[t, z, y, x] = False
        connected, stamp = _neighbors_connected_after_removal(
            nodes,
            offsets,
            n_neighbors,
            n_t,
            n_z,
            n_y,
            n_x,
            touched,
            stamp,
            q_t,
            q_z,
            q_y,
            q_x,
            qss,
            target_found,
        )
        if connected:
            counts_per_t[t] -= 1
            removed += 1
        else:
            nodes[t, z, y, x] = True
            restored += 1

    return removed, restored, skipped_min


def _compute_anchor_mask(
    priority: np.ndarray,
    nodes: np.ndarray,
    threshold_high: float | None,
    min_anchor_fraction: float,
    min_anchor_voxels: int,
) -> np.ndarray:
    """Compute undeletable anchor voxels."""
    if min_anchor_fraction < 0:
        raise ValueError("min_anchor_fraction must be >= 0")
    if min_anchor_voxels < 0:
        raise ValueError("min_anchor_voxels must be >= 0")

    if threshold_high is None:
        anchors = np.zeros_like(nodes, dtype=bool)
    else:
        anchors = (priority >= threshold_high) & nodes
    if min_anchor_fraction <= 0.0 and min_anchor_voxels == 0:
        return anchors

    t_dim = priority.shape[0]
    for t in range(t_dim):
        active_t = nodes[t]
        n_active = int(np.count_nonzero(active_t))
        if n_active == 0:
            continue

        target = max(int(np.ceil(min_anchor_fraction * n_active)), min_anchor_voxels)
        target = min(target, n_active)
        if target <= 0:
            continue

        vals = priority[t][active_t]
        kth = n_active - target
        threshold_t = np.partition(vals, kth)[kth]
        anchors[t] |= (priority[t] >= threshold_t) & active_t

    return anchors


def skeletonize4d(
    priority: np.ndarray,
    threshold_low: float,
    *,
    threshold_high: float | None = None,
    max_temporal_radius: int = 1,
    min_voxels_per_timepoint: int = 1,
    min_anchor_fraction: float = 0.005,
    min_anchor_voxels: int = 128,
    max_candidates: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Extract a 4D center-manifold using articulation-preserving thinning."""
    if priority.ndim != EXPECTED_DIMENSIONS_4D:
        raise ValueError(
            f"`priority` must be {EXPECTED_DIMENSIONS_4D}D (t,z,y,x), got {priority.ndim}D"
        )
    if min_voxels_per_timepoint < 0:
        raise ValueError("min_voxels_per_timepoint must be >= 0")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be > 0 when provided")

    priority_f = priority.astype(np.float32, copy=False)
    nodes = priority_f >= threshold_low
    if not np.any(nodes):
        return np.zeros_like(nodes, dtype=bool)

    offsets = _build_offsets_4d(max_temporal_radius=max_temporal_radius)
    anchors = _compute_anchor_mask(
        priority_f,
        nodes,
        threshold_high,
        min_anchor_fraction=min_anchor_fraction,
        min_anchor_voxels=min_anchor_voxels,
    )

    cand_t, cand_z, cand_y, cand_x = np.nonzero(nodes & ~anchors)
    order = np.argsort(priority_f[cand_t, cand_z, cand_y, cand_x], kind="stable")
    if max_candidates is not None and len(order) > max_candidates:
        order = order[:max_candidates]

    cand_t = cand_t.astype(np.int32, copy=False)
    cand_z = cand_z.astype(np.int32, copy=False)
    cand_y = cand_y.astype(np.int32, copy=False)
    cand_x = cand_x.astype(np.int32, copy=False)
    order = order.astype(np.int64, copy=False)

    counts_per_t = np.count_nonzero(nodes, axis=(1, 2, 3)).astype(np.int64)
    removed, restored, skipped_min = _skeletonize4d_impl(
        nodes=nodes,
        anchors=anchors,
        offsets=offsets,
        cand_t=cand_t,
        cand_z=cand_z,
        cand_y=cand_y,
        cand_x=cand_x,
        order=order,
        min_voxels_per_timepoint=min_voxels_per_timepoint,
        counts_per_t=counts_per_t,
    )

    if verbose:
        total = int(np.count_nonzero(nodes))
        per_t = counts_per_t.tolist()
        print(
            "[skeleton4d] done: "
            f"removed={removed}, restored={restored}, skipped_min={skipped_min}, "
            f"retained_total={total}, retained_per_timepoint_diag={per_t}"
        )

    return nodes
