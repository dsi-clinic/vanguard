"""Unit tests for the bilateral -> single-breast splitter (``gnn.breast_split``).

Synthetic 3D arrays only -- no cluster data required. Shape convention
throughout: ``(z, y, x)`` = ``(4, 10, 20)``, two rectangular "breasts"
separated by an empty gap on the x-axis (left breast x in [2,8), right
breast x in [12,18)), matching the DUKE inner-wall-detectable layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnn.breast_split import (
    resolve_tumor_mask_orientation,
    save_split_skeleton,
    save_split_support_mask,
    split_case,
)

_SHAPE = (4, 10, 20)


def _two_breast_mask() -> np.ndarray:
    mask = np.zeros(_SHAPE, dtype=bool)
    mask[:, :, 2:8] = True
    mask[:, :, 12:18] = True
    return mask


def _skeleton_in_both_breasts() -> np.ndarray:
    skeleton = np.zeros(_SHAPE, dtype=bool)
    skeleton[:, 4:6, 3:7] = True
    skeleton[:, 4:6, 13:17] = True
    return skeleton


def _tumor_mask(*, x_slice: slice) -> np.ndarray:
    tumor = np.zeros(_SHAPE, dtype=bool)
    tumor[:, 4:6, x_slice] = True
    return tumor


def test_tumor_on_left_retains_left_breast_only() -> None:
    """A left-side tumor keeps only the left breast's skeleton voxels."""
    result = split_case(
        "CASE_L",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
    )
    assert result.included
    assert result.retained_side == "left"
    assert result.tumor_side_voxel_fraction == 1.0
    # Only the left-breast skeleton voxels survive; the right breast's
    # skeleton voxels are dropped entirely, left ones pass through unchanged.
    expected = _skeleton_in_both_breasts().copy()
    expected[:, :, 12:18] = False
    assert np.array_equal(result.skeleton_single_breast, expected)


def test_tumor_on_right_retains_right_breast_only() -> None:
    """A right-side tumor keeps only the right breast's skeleton voxels."""
    result = split_case(
        "CASE_R",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=_tumor_mask(x_slice=slice(13, 17)),
    )
    assert result.included
    assert result.retained_side == "right"
    assert not np.any(result.skeleton_single_breast[:, :, 2:8])
    assert np.any(result.skeleton_single_breast[:, :, 12:18])


def test_tumor_split_across_midline_excluded_as_bilateral() -> None:
    """A tumor with comparable volume on both sides is excluded, not guessed."""
    tumor = np.zeros(_SHAPE, dtype=bool)
    tumor[:, 4:6, 4:7] = True  # left side, near midline
    tumor[:, 4:6, 13:16] = True  # right side, roughly equal volume
    result = split_case(
        "CASE_BOTH",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=tumor,
    )
    assert not result.included
    assert result.exclude_reason == "tumor_overlap_both_sides"
    assert result.skeleton_single_breast is None


def test_empty_tumor_mask_excluded_as_unknown_side() -> None:
    """No tumor mask means no way to infer the tumor-bearing side."""
    result = split_case(
        "CASE_NO_TUMOR",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=np.zeros(_SHAPE, dtype=bool),
    )
    assert not result.included
    assert result.exclude_reason == "tumor_mask_empty"


def test_empty_breast_mask_excluded() -> None:
    """No breast mask means no midline can be computed at all."""
    result = split_case(
        "CASE_NO_BREAST",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=np.zeros(_SHAPE, dtype=bool),
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
    )
    assert not result.included
    assert result.exclude_reason == "breast_mask_empty"


def test_tumor_outside_breast_mask_excluded_as_misregistered() -> None:
    """A tumor mask that doesn't overlap either breast signals bad registration."""
    tumor = np.zeros(_SHAPE, dtype=bool)
    tumor[:, 4:6, 9:11] = True  # entirely in the inter-breast gap
    result = split_case(
        "CASE_MISALIGNED",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=tumor,
    )
    assert not result.included
    assert result.exclude_reason == "tumor_mostly_outside_breast_mask"


def test_native_unilateral_input_cannot_find_a_midline() -> None:
    """Single-breast input has no second wall, so midline detection fails on it.

    This is not a case the module is meant to handle: per the
    harmonized-cohort plan, native unilateral cases must be routed around
    ``split_case`` entirely and used unchanged -- this test documents *why*
    that routing decision is required, not optional.
    """
    breast_mask = np.zeros(_SHAPE, dtype=bool)
    breast_mask[:, :, 2:8] = True
    skeleton = np.zeros(_SHAPE, dtype=bool)
    skeleton[:, 4:6, 3:7] = True
    result = split_case(
        "CASE_UNILATERAL",
        skeleton=skeleton,
        breast_mask=breast_mask,
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
    )
    assert not result.included
    assert result.exclude_reason.startswith("midline_detection_failed")


def test_mismatched_shapes_raise() -> None:
    """Shape mismatches between inputs are a caller bug, not silently coerced."""
    with pytest.raises(ValueError, match="shape mismatch"):
        split_case(
            "CASE_BAD_SHAPE",
            skeleton=np.zeros((2, 2, 2), dtype=bool),
            breast_mask=_two_breast_mask(),
            tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
        )


def test_support_mask_is_split_alongside_skeleton() -> None:
    """A support mask passed to split_case is restricted to the same retained side."""
    support_mask = np.zeros(_SHAPE, dtype=bool)
    support_mask[:, 3:7, 2:8] = True  # wider than the skeleton, left breast
    support_mask[:, 3:7, 12:18] = True  # wider than the skeleton, right breast

    result = split_case(
        "CASE_SUPPORT",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
        support_mask=support_mask,
    )
    assert result.included
    assert result.retained_side == "left"
    assert result.support_single_breast is not None
    assert not np.any(result.support_single_breast[:, :, 12:18])
    assert np.any(result.support_single_breast[:, :, 2:8])


def test_support_mask_omitted_leaves_support_single_breast_none() -> None:
    """Without a support_mask argument, support_single_breast stays None."""
    result = split_case(
        "CASE_NO_SUPPORT",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
    )
    assert result.included
    assert result.support_single_breast is None


def test_support_mask_shape_mismatch_raises() -> None:
    """A support_mask with the wrong shape is a caller bug, not silently coerced."""
    with pytest.raises(ValueError, match="shape mismatch support_mask"):
        split_case(
            "CASE_BAD_SUPPORT_SHAPE",
            skeleton=_skeleton_in_both_breasts(),
            breast_mask=_two_breast_mask(),
            tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
            support_mask=np.zeros((2, 2, 2), dtype=bool),
        )


def test_save_split_support_mask_writes_original_filename_convention(
    tmp_path: Path,
) -> None:
    """The saved support mask keeps the *original* exam-support-mask filename.

    Not a "single_breast"-suffixed variant -- gnn/data_loader.py's
    _build_case looks it up via a fixed pattern that doesn't know about
    breast-split substitution.
    """
    support_mask = np.zeros(_SHAPE, dtype=bool)
    support_mask[:, 3:7, 2:8] = True
    result = split_case(
        "CASE_SAVE_SUPPORT",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
        support_mask=support_mask,
    )
    out_path = save_split_support_mask(result, tmp_path)
    assert out_path.name == "CASE_SAVE_SUPPORT_skeleton_4d_exam_support_mask.npy"
    loaded = np.load(out_path)
    assert np.array_equal(loaded, result.support_single_breast)


def test_save_split_support_mask_raises_without_support_mask(tmp_path: Path) -> None:
    """Saving a support mask that was never given to split_case is a caller bug."""
    result = split_case(
        "CASE_NO_SUPPORT_SAVE",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
    )
    with pytest.raises(ValueError, match="no support_mask was passed"):
        save_split_support_mask(result, tmp_path)


def test_save_split_skeleton_writes_expected_filename(tmp_path: Path) -> None:
    """The saved array's filename follows the exam-mask naming convention."""
    result = split_case(
        "CASE_SAVE",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=_two_breast_mask(),
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
    )
    out_path = save_split_skeleton(result, tmp_path)
    assert out_path.name == "CASE_SAVE_skeleton_4d_single_breast_mask.npy"
    loaded = np.load(out_path)
    assert np.array_equal(loaded, result.skeleton_single_breast)


def test_save_split_skeleton_raises_for_excluded_case(tmp_path: Path) -> None:
    """Saving an excluded case's (nonexistent) skeleton is a caller bug."""
    result = split_case(
        "CASE_EXCLUDED",
        skeleton=_skeleton_in_both_breasts(),
        breast_mask=np.zeros(_SHAPE, dtype=bool),
        tumor_mask=_tumor_mask(x_slice=slice(3, 7)),
    )
    with pytest.raises(ValueError, match="excluded"):
        save_split_skeleton(result, tmp_path)


def test_resolve_tumor_mask_orientation_flips_when_it_recovers_overlap() -> None:
    """A tumor mask entirely outside the breast mask that would be entirely inside it if flipped.

    In that case, axis 0 gets flipped.
    """
    breast_mask = np.zeros(_SHAPE, dtype=bool)
    breast_mask[0:2, 4:6, 3:7] = True  # only in the first half of axis 0
    tumor_mask = np.zeros(_SHAPE, dtype=bool)
    tumor_mask[2:4, 4:6, 3:7] = True  # only in the second half of axis 0 (mirror)

    resolved, was_flipped = resolve_tumor_mask_orientation(
        tumor_mask, breast_mask=breast_mask
    )
    assert was_flipped
    assert np.array_equal(resolved, np.flip(tumor_mask, axis=0))
    assert np.array_equal(resolved & breast_mask, resolved)  # fully contained now


def test_resolve_tumor_mask_orientation_keeps_already_good_overlap() -> None:
    """A tumor mask that already overlaps the breast mask is left unflipped.

    Ties/near-ties favor no-op over flipping.
    """
    breast_mask = np.zeros(_SHAPE, dtype=bool)
    breast_mask[:, 4:6, 3:7] = True
    tumor_mask = np.zeros(_SHAPE, dtype=bool)
    tumor_mask[:, 4:6, 3:7] = True  # identical to breast region on this axis

    resolved, was_flipped = resolve_tumor_mask_orientation(
        tumor_mask, breast_mask=breast_mask
    )
    assert not was_flipped
    assert np.array_equal(resolved, tumor_mask)


def test_resolve_tumor_mask_orientation_prefers_higher_overlap_not_just_nonzero() -> (
    None
):
    """When both orientations overlap somewhat, the strictly better one wins."""
    breast_mask = np.zeros(_SHAPE, dtype=bool)
    breast_mask[:, 4:6, 3:7] = True
    tumor_mask = np.zeros(_SHAPE, dtype=bool)
    tumor_mask[0:3, 4:6, 3:7] = True  # 3/4 of axis-0 extent overlaps already

    resolved, was_flipped = resolve_tumor_mask_orientation(
        tumor_mask, breast_mask=breast_mask
    )
    assert not was_flipped
    assert np.array_equal(resolved, tumor_mask)
