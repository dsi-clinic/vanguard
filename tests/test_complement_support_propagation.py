"""Tests for restricting SegVessel's vessel-width mask to accepted branches.

The complement stage updates the centerline and must also propagate support so
radius/caliber features do not treat newly added centerline voxels as
one-voxel-wide. These tests check the restriction helper in isolation (no
SegVessel / Hessian run).
"""

from __future__ import annotations

import numpy as np

from preprocessing.complement import restrict_vmask_to_accepted_branches


def test_restrict_vmask_keeps_only_accepted_branch_width() -> None:
    """Vessel voxels nearest an accepted branch stay; the other branch is dropped."""
    labeled = np.zeros((20, 20, 10), dtype=np.int32)
    labeled[10, 2:8, 5] = 1  # branch 1 (accepted)
    labeled[10, 12:18, 5] = 2  # branch 2 (rejected)

    vmask = np.zeros_like(labeled, dtype=bool)
    # Width around each branch: a 3-voxel-thick tube in the row direction.
    vmask[9:12, 2:8, 5] = True
    vmask[9:12, 12:18, 5] = True

    restricted = restrict_vmask_to_accepted_branches(vmask, labeled, keep_labels=[1])

    assert restricted[9:12, 2:8, 5].all()
    assert not restricted[9:12, 12:18, 5].any()
    assert int(restricted.sum()) == int(vmask[9:12, 2:8, 5].sum())


def test_restrict_vmask_empty_keep_labels_returns_empty() -> None:
    """No accepted branches means no support voxels are propagated."""
    labeled = np.zeros((8, 8, 4), dtype=np.int32)
    labeled[4, 2:6, 2] = 1
    vmask = labeled > 0
    restricted = restrict_vmask_to_accepted_branches(vmask, labeled, keep_labels=[])
    assert not restricted.any()


def test_restrict_vmask_includes_centerline_neighborhood_not_just_skeleton() -> None:
    """Accepted-branch support is wider than the 1-voxel centerline."""
    labeled = np.zeros((15, 15, 8), dtype=np.int32)
    labeled[7, 3:12, 4] = 3
    vmask = np.zeros_like(labeled, dtype=bool)
    vmask[6:9, 3:12, 3:6] = True  # thick tube around the branch

    restricted = restrict_vmask_to_accepted_branches(vmask, labeled, keep_labels={3})
    assert restricted[7, 3:12, 4].all()  # centerline covered
    assert restricted[6, 5, 4] and restricted[8, 5, 4]  # width kept
    assert int(restricted.sum()) == int(vmask.sum())
