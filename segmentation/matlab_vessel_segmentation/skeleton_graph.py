"""Skeleton graph analysis: prune short branches + bridge gaps.

Ports (behaviourally) these MATLAB files:
  KineticAnalysisFunctions/skel2graph3d/Skel2Graph3D.m, Graph2Skel3D.m, pk_follow_link.m
  KineticAnalysisFunctions/VesselAnalysis/Conn_nearest_points_v2.m
  KineticAnalysisFunctions/VesselAnalysis/Vessel_morph.m

WHAT THIS STEP DOES, IN PLAIN LANGUAGE
--------------------------------------
`Skeleton3D` gives a raw 1-voxel-wide skeleton that is noisy: lots of tiny hair-like
"spurs" and small broken gaps. `Vessel_morph` cleans it into a vessel centerline:
  1. delete short leaf branches (spurs) below a length threshold,
  2. bridge small gaps between branch pieces that are close in space,
  3. delete leaf branches again below a (mm-based) minimum length.
The cleaned centerline is then used as the SEED for growing the final vessel mask.

IMPLEMENTATION NOTE (faithfulness)
----------------------------------
The MATLAB code builds an explicit node/link graph via linear-index bookkeeping. We
reproduce the *behaviour* (voxel-degree classification, branch = chain of degree-2
voxels between junctions/endpoints, leaf-pruning by branch length, and the exact
gap-bridging loop of Conn_nearest_points_v2) using scipy connected components, which is
equivalent for the quantity that flows downstream (a cleaned binary centerline, then
binarised as `skel==1`). Per-branch tortuosity (an *analysis* output, not used to build
the mask) is provided as a simple extra, not a line-by-line port.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

_K3 = np.ones((3, 3, 3), dtype=int)  # 26-connectivity kernel (incl. centre)
_S26 = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity structure


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    """Number of 26-neighbours (excluding self) for each foreground voxel."""
    s = skel.astype(int)
    nc = ndi.convolve(s, _K3, mode="constant", cval=0) - s
    nc[~skel] = 0
    return nc


def _branch_decompose(skel: np.ndarray):
    """Split a skeleton into junctions and chains (branch interiors).

    Returns (chain_labels, n_chains, junction_mask, endpoint_mask, degree).
    A "chain" is a 26-connected run of non-junction voxels; each chain is one branch.
    """
    skel = skel.astype(bool)
    deg = _neighbor_count(skel)
    junction = skel & (deg >= 3)  # branch points
    endpoint = skel & (deg == 1)  # free ends (leaf tips)
    chains = skel & ~junction
    lab, n = ndi.label(chains, structure=_S26)
    return lab, n, junction, endpoint, deg


def _prune_short_leaves(skel: np.ndarray, thr: int) -> np.ndarray:
    """Iteratively delete leaf branches shorter than `thr` voxels.

    Mirrors Skel2Graph3D's rule: an endpoint-terminated (leaf) link is kept only if its
    length exceeds THR; node-to-node links are always kept. We repeat until stable,
    which reproduces Vessel_morph's "iterate until network length changes < 0.5%" loop.
    """
    if thr <= 0:
        return skel.astype(bool)
    skel = skel.astype(bool)
    while True:
        lab, n, junction, endpoint, _ = _branch_decompose(skel)
        if n == 0:
            break
        to_remove = np.zeros_like(skel)
        # dilate junctions once so we can test which chains touch a junction
        junc_dil = ndi.binary_dilation(junction, structure=_S26)
        changed = False
        # count how many distinct junction clusters each chain touches is expensive;
        # a chain is a "leaf" if it contains a free endpoint (deg==1 tip). That is the
        # operational definition of a prunable spur here.
        for lbl in range(1, n + 1):
            comp = lab == lbl
            length = int(comp.sum())
            if length >= thr:
                continue
            has_endpoint = bool((comp & endpoint).any())
            touches_junction = bool((comp & junc_dil).any())
            # leaf spur: has a free end AND (dangles from <=1 junction OR is isolated)
            if has_endpoint and (length < thr):
                # keep genuine node-to-node bridges: those touch a junction on BOTH
                # ends and have no free endpoint -> already excluded by has_endpoint.
                to_remove |= comp
                changed = True
            elif not touches_junction and length < thr:
                # isolated tiny fragment
                to_remove |= comp
                changed = True
        if not changed:
            break
        skel = skel & ~to_remove
        # junctions that lost all but <=1 branch decay to path/endpoint automatically
        # on the next iteration's re-decomposition.
    return skel


def _endpoints(skel: np.ndarray) -> np.ndarray:
    deg = _neighbor_count(skel)
    return skel & (deg == 1)


def conn_nearest_points(skel: np.ndarray, maxd_vox: float, ratio) -> np.ndarray:
    """Port of Conn_nearest_points_v2.m — bridge nearby branch pieces.

    Repeatedly: take the shortest connected component, find its endpoints, and connect
    the endpoint whose nearest voxel on ANY other component is closest (measured in
    anisotropic voxel units via `ratio`) with a straight line, if that distance <= MaxD.
    Stop when one component remains or the nearest gap exceeds MaxD.
    """
    skel = skel.astype(bool)
    ratio = np.asarray(ratio, dtype=np.float64)
    cc_lab, n = ndi.label(skel, structure=_S26)
    guard = 0
    min_d = 0.0
    while n > 1 and min_d < maxd_vox and guard < 10000:
        guard += 1
        sizes = np.bincount(cc_lab.ravel())
        sizes[0] = 0
        order = np.argsort(sizes)  # ascending, shortest first
        order = order[sizes[order] > 1]  # skip single-voxel / background
        connected = False
        min_d = np.inf
        for lbl in order[:-1]:  # not the longest
            tskel = cc_lab == lbl
            other = skel & ~tskel
            if not other.any():
                continue
            ox, oy, oz = np.where(other)
            other_pts = np.stack([ox, oy, oz], axis=1).astype(np.float64)
            tree = cKDTree(other_pts * ratio)
            eps = np.where(_endpoints(tskel))
            eps = np.stack(eps, axis=1).astype(np.float64)
            if eps.size == 0:
                continue
            dists, idxs = tree.query(eps * ratio)
            j = int(np.argmin(dists))
            d = float(dists[j])
            if d < min_d:
                min_d = d
            if d <= maxd_vox:
                p1 = eps[j].astype(int)
                p2 = other_pts[idxs[j]].astype(int)
                N = int(round(max(np.abs(p1 - p2)) * np.sqrt(2)))
                N = max(N, 2)
                line = np.stack(
                    [
                        np.round(np.linspace(p1[0], p2[0], N)),
                        np.round(np.linspace(p1[1], p2[1], N)),
                        np.round(np.linspace(p1[2], p2[2], N)),
                    ],
                    axis=0,
                ).astype(int)
                skel[line[0], line[1], line[2]] = True
                cc_lab, n = ndi.label(skel, structure=_S26)
                connected = True
                break
        if not connected:
            break
    return skel


def vessel_morph(skel: np.ndarray, spacing) -> dict:
    """Port of Vessel_morph.m -> returns dict with 'skel' (cleaned) and 'skel_label'.

    spacing = [PixelSpacing_row, PixelSpacing_col, SpacingBetweenSlices] (mm).
    minlen = 10 mm and MaxD = 15 mm, converted to voxels exactly as MATLAB.
    """
    skel = skel.astype(bool)
    scale = float(spacing[0])  # in-plane mm/voxel
    ratio = np.array([1.0, 1.0, round(spacing[2] / spacing[0], 2)])
    minlen = int(round(10.0 / scale))  # 10 mm in voxels
    maxd = int(round(15.0 / scale))  # 15 mm in voxels

    # Step 1: initial spur prune (Skel2Graph3D(skel,10)) + stabilise (THR=0 loop).
    s2 = _prune_short_leaves(skel, 10)
    # Step 2: bridge gaps up to MaxD (anisotropic).
    s2 = conn_nearest_points(s2, maxd, ratio)
    # Step 3: final leaf prune at the mm-based minimum length.
    s2 = _prune_short_leaves(s2, minlen)

    # Build a per-branch label image (chains between junctions get distinct labels;
    # junction voxels are added back so skel_label>0 == the full cleaned centerline).
    lab, n, junction, _endp, _deg = _branch_decompose(s2)
    skel_label = lab.copy()
    if junction.any():
        skel_label[junction] = (
            n + 1
        )  # junctions share one extra label; >0 is what matters

    return {
        "skel": s2,
        "skel_label": skel_label,
        "n_branches": int(n),
        "spacing": np.asarray(spacing, dtype=float),
        "ratio": ratio,
        "minlen_vox": minlen,
        "maxd_vox": maxd,
    }
