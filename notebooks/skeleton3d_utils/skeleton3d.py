# skeleton3d.py
"""Topology-preserving morphological skeletonization for 3D volumes.

Each voxel is treated as a node connected to its potential 26 neighbors (in a
3×3×3 cube). Voxels are iteratively removed in order of increasing priority,
but only if their removal does not disconnect any of their neighbors (checked
via Breadth First Search).

Public:
    skeletonize3d(priority, threshold) -> np.ndarray of uint32 bitmasks
"""

import numba as nb
import numpy as np

# Neighborhood definitions (3D only)

# Definition of relative coordinates of all 26 possible neighbors around
# each voxel. Generate all combinations of shifts by -1, 0, +1 in each
# dimension and then remove 0, 0, 0 (itself).
_OFFSETS_3D = np.array(
    [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ],
    dtype=np.int64,
)

# Reverse index for opposite directions
_REV_3D = np.empty(26, dtype=np.int64)
for i, (dz, dy, dx) in enumerate(_OFFSETS_3D):
    opp = (-dz, -dy, -dx)
    _REV_3D[i] = np.where((_OFFSETS_3D == opp).all(axis=1))[0][0]

EXPECTED_DIMENSIONS = 3


# Ring-buffer queue utilities


@nb.njit
def _queue_push_3d(Q: np.ndarray, ss: np.ndarray, k: int, i: int, j: int) -> None:
    """Add a voxel’s coordinates to the BFS queue.

    It uses a circular (ring) buffer so it can efficiently reuse the same
    fixed-size array instead of growing a list.”

    Inputs:
        -Q: a 2D NumPy array of shape (N, 3) — it stores voxel coordinates.
        Each row is [k, i, j] (z, y, x).
        -ss: a small array of size 2, [start, stop], that keeps track of which
        part of the queue is active.
            - ss[0] = start index (where to pop from)
            - ss[1] = stop index (where to push next)
        - k, i, j: the coordinates of the voxel you want to enqueue.
    """
    Q[ss[1], 0] = k
    Q[ss[1], 1] = i
    Q[ss[1], 2] = j
    ss[1] = 0 if (ss[1] + 1 == len(Q)) else ss[1] + 1


@nb.njit
def _queue_pop_3d(Q: np.ndarray, ss: np.ndarray) -> tuple[int, int, int]:
    """Pop (k, i, j) coordinates from the BFS ring-buffer queue.

    Removes and returns the voxel coordinates located at the current
    queue "start" position (ss[0]) and advances the start pointer by one.
    If the start pointer reaches the end of the buffer, it wraps around
    to 0 (circular behavior).

    Args:
        Q (np.ndarray): Array of shape (N, 3) storing voxel coordinates
            currently in the BFS queue.
        ss (np.ndarray): Two-element array [start, stop] indicating
            the active range within the circular queue.

    Returns:
        tuple[int, int, int]: The next voxel coordinates (k, i, j)
            to be processed in the BFS traversal.
    """
    k = Q[ss[0], 0]
    i = Q[ss[0], 1]
    j = Q[ss[0], 2]
    ss[0] = 0 if (ss[0] + 1 == len(Q)) else ss[0] + 1
    return k, i, j


@nb.njit
def _queue_empty(ss: np.ndarray) -> bool:
    """Check whether the BFS ring-buffer queue is empty.

    The queue is considered empty when the start and stop
    indices are equal, meaning there are no elements left
    to pop between them.

    Args:
        ss (np.ndarray): Two-element array [start, stop] that
            tracks the active range within the circular queue.

    Returns:
        bool: True if the queue is empty, False otherwise.
    """
    return ss[0] == ss[1]


@nb.njit
def _queue_clear(ss: np.ndarray) -> None:
    """Clear the BFS ring-buffer queue.

    Resets the queue by setting the start index equal to the stop index,
    effectively marking it as empty without modifying any stored data.

    Args:
        ss (np.ndarray): Two-element array [start, stop] that tracks
            the active range within the circular queue.
    """
    ss[0] = ss[1]


# 3D skeletonization core


@nb.njit
def _add_edges_3d(nodes: np.ndarray, edges: np.ndarray, k: int, i: int, j: int) -> None:
    """Update the edge bitmask for voxel (k, i, j) based on its 26-connected neighbors.

    For the voxel at position (k, i, j), this function checks all 26 possible
    neighbor directions defined in _OFFSETS_3D. For each valid neighbor that
    exists within the volume bounds and is active in `nodes`, the corresponding
    bit in `edges[k, i, j]` is set to 1.

    In other words, it encodes which neighboring voxels are directly connected
    to this voxel as a 26-bit integer mask.

    Args:
        nodes (np.ndarray): 3D boolean array (Z, H, W), where True indicates
            that a voxel is currently part of the skeleton.
        edges (np.ndarray): 3D uint32 array (Z, H, W) storing per-voxel
            26-bit connectivity masks. Modified in place.
        k (int): Z-index (depth) of the voxel.
        i (int): Y-index (row) of the voxel.
        j (int): X-index (column) of the voxel.
    """
    # Identify nodes and shape
    Z, H, W = nodes.shape
    # Iterate through all the possible combinations,
    # b is the index of the neighbor
    for b in range(26):
        # Lookup the offset vector for neighbor b
        dz, dy, dx = _OFFSETS_3D[b]
        # Compute the neighbor’s coordinates by adding the offset to the current
        #  voxel’s
        nk, ni, nj = k + dz, i + dy, j + dx
        # Ensure the neighbor indices are inside the volume (Z, H, W are the
        # shape).
        # Ensure the neighbor voxel is active (True) in nodes (i.e., part of the
        # current object/graph).
        if 0 <= nk < Z and 0 <= ni < H and 0 <= nj < W and nodes[nk, ni, nj]:
            # Set the b-th bit in this voxel’s edge mask to 1, meaning “we have
            # a connection to neighbor b
            edges[k, i, j] |= np.uint32(1 << b)


@nb.njit
def _add_node_3d(nodes: np.ndarray, edges: np.ndarray, k: int, i: int, j: int) -> None:
    """Activate a voxel (k, i, j) as a node and update all relevant edge connections.

    Sets the voxel at (k, i, j) to True in the `nodes` array and updates its
    26-connected neighbor links in the `edges` bitmask array. After adding the
    new node, the function also refreshes the edge masks of all neighboring
    active voxels so that bidirectional connections are correctly established.

    Args:
        nodes (np.ndarray): 3D boolean array (Z, H, W), where True indicates
            active voxels (nodes) in the graph.
        edges (np.ndarray): 3D uint32 array (Z, H, W) storing 26-bit connectivity
            masks for each voxel. Modified in place.
        k (int): Z-index (depth) of the voxel being added.
        i (int): Y-index (row) of the voxel being added.
        j (int): X-index (column) of the voxel being added.
    """
    # Activate voxel k, i, j as node
    nodes[k, i, j] = True
    # Build this voxel’s connectivity mask. Looks at all 26 neighbors and sets
    # the appropriate bits in edges[k, i, j] for any active neighbors.
    _add_edges_3d(nodes, edges, k, i, j)
    Z, H, W = nodes.shape
    for b in range(26):
        # Get the neighbor offset (how far to move in z, y, x for neighbor b).
        dz, dy, dx = _OFFSETS_3D[b]
        # Compute the neighbor’s coordinates by applying the offset to (k, i, j).
        nk, ni, nj = k + dz, i + dy, j + dx
        # Check bounds and activity. Only proceed if the neighbor is inside the
        # volume and currently active (True in nodes).
        if 0 <= nk < Z and 0 <= ni < H and 0 <= nj < W and nodes[nk, ni, nj]:
            # Refresh the neighbor’s connectivity mask.
            _add_edges_3d(nodes, edges, nk, ni, nj)


@nb.njit
def _remove_node_3d(
    nodes: np.ndarray, edges: np.ndarray, k: int, i: int, j: int
) -> None:
    """Remove a voxel from the active node set and update neighbor connectivity.

    Marks the voxel as inactive in `nodes` and clears its edge bitmask in `edges`.
    For all 26 possible neighbor directions, if the neighbor voxel is active,
    the function removes the reverse connection bit in the neighbor's edge mask.
    This ensures that no active voxel maintains an edge pointing to a node
    that has just been deleted.

    Args:
        nodes (np.ndarray): 3D boolean array (Z, H, W), where True marks active voxels.
        edges (np.ndarray): 3D uint32 array (Z, H, W) containing 26-bit neighbor masks.
                            Modified in place.
        k (int): Z-index (depth) of the voxel to remove.
        i (int): Y-index (row) of the voxel to remove.
        j (int): X-index (column) of the voxel to remove.
    """
    # Deactivate this voxel — mark it as no longer part of the structure being kept.
    nodes[k, i, j] = False
    # Clear its edge connections.
    edges[k, i, j] = np.uint32(0)
    Z, H, W = nodes.shape
    for b in range(26):
        # Get the 3D offset
        dz, dy, dx = _OFFSETS_3D[b]
        # Compute the neighbor’s coordinates using the offset from the current voxel.
        nk, ni, nj = k + dz, i + dy, j + dx
        # Check two things:
        #   1. The neighbor is within the bounds of the volume.
        #   2. The neighbor voxel is active (True in nodes).
        if 0 <= nk < Z and 0 <= ni < H and 0 <= nj < W and nodes[nk, ni, nj]:
            # if b corresponds to “+x” (right), then rb corresponds to “−x” (left).
            rb = _REV_3D[b]
            # Remove the reverse edge in the neighbor’s mask:
            edges[nk, ni, nj] &= np.uint32(~(1 << rb))


@nb.njit
def _set_seeds_3d(edges: np.ndarray, seeds: np.ndarray, k: int, i: int, j: int) -> int:
    """Collect all 3D neighbors connected to voxel (k, i, j).

    Reads the 26-bit connectivity mask from `edges[k, i, j]` and enumerates
    all neighbors that are directly connected to this voxel. The 3D coordinates
    of these connected neighbors are written sequentially into the `seeds`
    array, which is then used to initialize the BFS region-growing step.

    Args:
        edges (np.ndarray): 3D uint32 array (Z, H, W) where each voxel’s
            26-bit mask encodes which neighbors are connected.
        seeds (np.ndarray): Preallocated array of shape (26, 3) to store
            the coordinates of connected neighbors. Modified in place.
        k (int): Z-index (depth) of the center voxel.
        i (int): Y-index (row) of the center voxel.
        j (int): X-index (column) of the center voxel.

    Returns:
        int: Number of connected neighbors found and written to `seeds`.
    """
    n = 0
    # Get the 26-bit edge mask of this voxel
    mask = edges[k, i, j]
    for b in range(26):
        # Bit test: shift the mask b bits to the right, then check the least
        # significant bit.
        #   If it’s 1 → this voxel is connected to neighbor b.
        #   If it’s 0 → no connection in that direction.
        if (mask >> b) & 1:
            dz, dy, dx = _OFFSETS_3D[b]
            # Compute the neighbor’s coordinates and store them in the next
            # available row in seeds
            seeds[n, 0] = k + dz
            seeds[n, 1] = i + dy
            seeds[n, 2] = j + dx
            n += 1  # Increment the neighbor counter
    # Return how many connected neighbors were found.
    return n


@nb.njit
def _remove_if_not_articulation_3d(
    nodes: np.ndarray,
    edges: np.ndarray,
    seeds: np.ndarray,
    Q: np.ndarray,
    Qss: np.ndarray,
    touched: np.ndarray,
    tss: np.ndarray,
    k: int,
    i: int,
    j: int,
) -> None:
    """Test and remove voxel (k, i, j) if it is not an articulation point.

    This function attempts to remove a voxel while preserving the local
    topology of the graph. It first identifies all directly connected
    neighbors of the voxel using `_set_seeds_3d()`, then temporarily
    removes the voxel. Using a breadth-first search (BFS), it checks
    whether all remaining neighbors are still connected to each other.
    If any neighbor becomes disconnected, the voxel is restored.

    Args:
        nodes (np.ndarray): 3D boolean array (Z, H, W) of active voxels.
        edges (np.ndarray): 3D uint32 array (Z, H, W) with 26-bit masks of connectivity.
        seeds (np.ndarray): (26, 3) array for storing coordinates of neighbor voxels.
        Q (np.ndarray): (N, 3) queue array for BFS traversal.
        Qss (np.ndarray): Two-element array [start, stop] controlling the BFS queue.
        touched (np.ndarray): 3D int array tracking visited voxels with timestamps.
        tss (np.ndarray): Two-element array [start_stamp, stop_stamp] used to manage
            unique BFS visitation ranges.
        k (int): Z-index of the voxel being tested.
        i (int): Y-index of the voxel being tested.
        j (int): X-index of the voxel being tested.

    Returns:
        None. The voxel is removed in-place if safe; otherwise restored.
    """
    # Collect all connected neighbor voxels of (k, i, j) into seeds.
    num = _set_seeds_3d(edges, seeds, k, i, j)
    # Temporarily remove the voxel and update its neighbors’ edge masks.
    # After this, (k, i, j) is gone from the graph, we’ll check if its removal
    # disconnects neighbors.
    _remove_node_3d(nodes, edges, k, i, j)

    # If the voxel had no connected neighbors, removing it cannot break
    # connectivity — we’re done.
    if num == 0:
        return  # delete isolated voxel
    # Select the first neighbor as the BFS “target” or reference voxel.
    tk, ti, tj = seeds[0, 0], seeds[0, 1], seeds[0, 2]
    # Mark it as “visited” using the current timestamp tss[0].
    touched[tk, ti, tj] = tss[0]
    tss[1] += 1  # Increment tss[1]

    # Loop over the remaining neighbors
    for s in range(1, num):
        # Get coordinates of the next neighbor we’ll test for connectivity.
        sk, si, sj = seeds[s, 0], seeds[s, 1], seeds[s, 2]
        # Reset the BFS queue and enqueue the current neighbor as the starting point.
        _queue_clear(Qss)
        _queue_push_3d(Q, Qss, sk, si, sj)

        connected = False
        # Start a BFS traversal: while the queue has elements,
        # pop the next voxel to explore.
        while not _queue_empty(Qss):
            ck, ci, cj = _queue_pop_3d(Q, Qss)
            stamp = touched[ck, ci, cj]  # prevent re-visiting voxels
            # confirming connectivity. Mark as connected and stop this BFS.
            if tss[0] <= stamp < tss[1]:
                connected = True
                break
            # If this voxel was already visited in the current search iteration,
            #  skip it.
            if stamp == tss[1]:
                continue
            # Mark the voxel as “visited” in this BFS iteration.
            touched[ck, ci, cj] = tss[1]

            # Look at all connected neighbors of the current voxel
            # For each connected neighbor, compute its coordinates and enqueue it.
            # This continues the BFS to explore all reachable voxels.
            mask = edges[ck, ci, cj]
            for b in range(26):
                if (mask >> b) & 1:
                    nk = ck + _OFFSETS_3D[b, 0]
                    ni = ci + _OFFSETS_3D[b, 1]
                    nj = cj + _OFFSETS_3D[b, 2]
                    _queue_push_3d(Q, Qss, nk, ni, nj)

        tss[1] += 1
        # If BFS failed to reach the target region (neighbors disconnected):
        #   Update timestamp window.
        #   Restore the voxel using _add_node_3d()
        #   Stop the function — we can’t remove this voxel without breaking
        #   topology.
        if not connected:
            tss[0] = tss[1]
            _add_node_3d(nodes, edges, k, i, j)
            return
    # update timestamps to mark the end of this test, so future BFS runs
    # have a clean time window
    tss[0] = tss[1]


@nb.njit
def _skeletonize_impl_3d(
    nodes: np.ndarray,
    edges: np.ndarray,
    seeds: np.ndarray,
    Q: np.ndarray,
    Qss: np.ndarray,
    touched: np.ndarray,
    tss: np.ndarray,
    kk: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
) -> None:
    """Core implementation of the 3D topology-preserving thinning algorithm.

    Iterates through all active voxels (given by kk, ii, jj) in order of
    increasing priority and attempts to remove each one if it is not an
    articulation point. The actual removal and connectivity testing are
    performed by `_remove_if_not_articulation_3d()`.

    Args:
        nodes (np.ndarray): 3D boolean array (Z, H, W) marking active voxels.
        edges (np.ndarray): 3D uint32 array (Z, H, W) of 26-bit connectivity
        masks.
        seeds (np.ndarray): (26, 3) array for temporarily storing neighbor
        coordinates.
        Q (np.ndarray): (N, 3) BFS queue used during connectivity testing.
        Qss (np.ndarray): Two-element array [start, stop] controlling the BFS
        queue state.
        touched (np.ndarray): 3D int array tracking BFS visitation stamps.
        tss (np.ndarray): Two-element array [start_stamp, stop_stamp] for
        managing visit timestamps.
        kk (np.ndarray): Z indices of active voxels sorted by priority.
        ii (np.ndarray): Y indices of active voxels sorted by priority.
        jj (np.ndarray): X indices of active voxels sorted by priority.

    Returns:
        None. Modifies `nodes` and `edges` in place to represent the final skeleton.
    """
    for idx in range(len(kk)):
        # kk, ii, jj are the z, y, x coordinates of voxels that were initially
        # active
        # They have already been sorted so that low-priority voxels are
        # processed (and potentially removed) first.
        _remove_if_not_articulation_3d(
            nodes, edges, seeds, Q, Qss, touched, tss, kk[idx], ii[idx], jj[idx]
        )


# Public function


def skeletonize3d(priority: np.ndarray, threshold: float) -> np.ndarray:
    """Perform topology-preserving skeletonization on a 3D volume.

    This is the main entry point for the 3D thinning algorithm. Voxels are
    iteratively removed in order of increasing priority (lowest values first),
    but only if their removal does not disconnect neighboring voxels. The
    output is a volume of the same shape where each voxel’s 26-bit integer
    encodes its connectivity to adjacent voxels in the final skeleton graph.

    Args:
        priority (np.ndarray): 3D array (Z, H, W) containing voxel values that
            define the thinning order. Lower values are removed earlier.
            For example, this may represent intensity, distance, or probability.
        threshold (float): Minimum value to include a voxel in the graph.
            Voxels below this threshold are excluded from the start.

    Returns:
        np.ndarray: 3D uint32 array (Z, H, W) representing the final skeleton.
            Each voxel stores a 26-bit bitmask where bit b = 1 if connected
            to neighbor direction b as defined in `_OFFSETS_3D`.
    """
    # Ensure the input array really is 3D
    if priority.ndim != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"`priority` must be {EXPECTED_DIMENSIONS}D, received {priority.ndim}D"
        )
    # Create a binary mask of active voxels.
    nodes = priority > threshold
    # Initialize an empty connectivity array, same shape as the input.
    # Each voxel will later store a 26-bit integer (inside the 32-bit container).
    edges = np.zeros_like(nodes, dtype=np.uint32)

    Z, H, W = nodes.shape
    # For every active voxel, call _add_edges_3d() to compute its connections
    # to neighbors.
    # After this loop:
    #   Every voxel knows which of its 26 neighbors are part of the same structure.
    #   edges is now a full 3D connectivity map of the original object.
    for k in range(Z):
        for i in range(H):
            for j in range(W):
                if nodes[k, i, j]:
                    _add_edges_3d(nodes, edges, k, i, j)

    # Extract the indices of all active voxels
    kk, ii, jj = np.nonzero(nodes)
    # Sort all active voxels by their priority value (ascending).
    order = np.argsort(priority[kk, ii, jj])
    kk, ii, jj = kk[order], ii[order], jj[order]

    # Preallocate space for the neighbor list used during articulation testing.
    # At most, 26 neighbors per voxel.
    seeds = np.empty((26, 3), dtype=np.int64)
    Q = np.empty((len(kk), 3), dtype=np.int64)
    Qss = np.zeros(2, dtype=np.int64)
    touched = np.zeros(nodes.shape, dtype=np.int64)
    tss = np.array([1, 1], dtype=np.int64)
    # Call the core loop of the algorithm.
    _skeletonize_impl_3d(nodes, edges, seeds, Q, Qss, touched, tss, kk, ii, jj)
    return edges
