"""Tests for the MATLAB-route complement quality gate.

The gate exists to answer one question: does a candidate segment that the merged
skeleton does not have look like a real vessel, or like noise? These tests put both
kinds of thing in front of it on the same synthetic exam and check it separates them,
plus the two structural properties the gate's design turns on:

  * a candidate that partly overlaps the merged skeleton must stay whole, because
    classifying voxels before finding components used to cut exactly that case in two
    and leave its novel part too short to pass;
  * the calibration reference must be built from whole components too, so both
    populations are the same kind of object.

Skeletons here are hand-built binary volumes rather than SegVessel output -- the gate
takes a skeleton, a vesselness map and an enhancement map as plain arrays, so it can be
tested without running the (slow) Hessian filter.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.complement import filter_complement

SHAPE = (60, 60, 30)
SPACING = (1.0, 1.0, 1.0)  # the UFAST grid is 1 mm isotropic


def _line(
    volume: np.ndarray, row: int, col_start: int, col_stop: int, slice_index: int
) -> None:
    volume[row, col_start:col_stop, slice_index] = True


@pytest.fixture
def exam() -> dict:
    """A synthetic exam: several known vessels, one new vessel, and a field of noise.

    ``merged`` holds seven vessels the production merge already found -- comfortably
    above MIN_CONFIRMED_COMPONENTS_FOR_CALIBRATION, so a test that consumes one of them
    still leaves a usable calibration set. ``matlab`` holds the same seven, plus one
    genuinely new vessel and twelve scattered specks. Vesselness and enhancement are
    high on every real vessel and low on the specks, which is the signal the gate is
    meant to use.
    """
    merged = np.zeros(SHAPE, dtype=bool)
    matlab = np.zeros(SHAPE, dtype=bool)

    known = [(row, 5, 45, 10) for row in (10, 17, 24, 31, 38, 45, 52)]
    for row, start, stop, slice_index in known:
        _line(merged, row, start, stop, slice_index)
        _line(matlab, row, start, stop, slice_index)

    new_vessel = (25, 5, 45, 20)
    _line(matlab, *new_vessel)

    rng = np.random.default_rng(0)
    speck_positions = rng.integers(
        [0, 0, 0], [SHAPE[0], SHAPE[1] - 4, SHAPE[2]], size=(12, 3)
    )
    for position in speck_positions:
        matlab[position[0], position[1] : position[1] + 3, position[2]] = True

    vesselness = np.full(SHAPE, 0.01, dtype=np.float32)
    enhancement = np.full(SHAPE, 1.0, dtype=np.float32)
    real_vessels = merged.copy()
    _line(real_vessels, *new_vessel)
    vesselness[real_vessels] = 0.5
    enhancement[real_vessels] = 100.0

    breast = np.zeros(SHAPE, dtype=bool)
    breast[2:58, 2:58, 2:28] = True

    return {
        "matlab_skeleton_yxz": matlab,
        "vesselness_yxz": vesselness,
        "enhancement_yxz": enhancement,
        "merged_skeleton_yxz": merged,
        "breast_mask_yxz": breast,
        "spacing_yxz": SPACING,
        "new_vessel": new_vessel,
        "speck_positions": speck_positions,
    }


def test_keeps_a_genuinely_new_vessel(exam: dict) -> None:
    """A separate, well-enhancing, tubular segment is the thing this stage is for."""
    kept, stats = filter_complement(
        **{
            k: exam[k]
            for k in (
                "matlab_skeleton_yxz",
                "vesselness_yxz",
                "enhancement_yxz",
                "merged_skeleton_yxz",
                "breast_mask_yxz",
                "spacing_yxz",
            )
        }
    )

    row, start, stop, slice_index = exam["new_vessel"]
    new_vessel_voxels = kept[row, start:stop, slice_index]
    assert new_vessel_voxels.all(), "the new vessel should be kept end to end"
    assert stats["n_kept_components"] >= 1


def test_rejects_scattered_specks(exam: dict) -> None:
    """Low-vesselness, low-enhancement specks must not survive the gate."""
    kept, _stats = filter_complement(
        **{
            k: exam[k]
            for k in (
                "matlab_skeleton_yxz",
                "vesselness_yxz",
                "enhancement_yxz",
                "merged_skeleton_yxz",
                "breast_mask_yxz",
                "spacing_yxz",
            )
        }
    )

    speck_mask = np.zeros(SHAPE, dtype=bool)
    for position in exam["speck_positions"]:
        speck_mask[position[0], position[1] : position[1] + 3, position[2]] = True
    speck_mask &= ~exam["merged_skeleton_yxz"]

    assert not (kept & speck_mask).any(), "no speck should reach the output"


def test_never_returns_voxels_the_merge_already_has(exam: dict) -> None:
    """`n_kept` has to mean added coverage, not re-reporting known vessels."""
    kept, stats = filter_complement(
        **{
            k: exam[k]
            for k in (
                "matlab_skeleton_yxz",
                "vesselness_yxz",
                "enhancement_yxz",
                "merged_skeleton_yxz",
                "breast_mask_yxz",
                "spacing_yxz",
            )
        }
    )

    assert not (kept & exam["merged_skeleton_yxz"]).any()
    assert stats["n_kept"] == int(kept.sum())


def test_partly_overlapping_segment_is_judged_whole(exam: dict) -> None:
    """A segment that runs along a known vessel and then continues past it.

    Classifying voxels before finding components split this into a "confirmed" part
    and a short "candidate" stub; the stub then failed the size and elongation gates,
    so the novel extension was lost. Judged whole, the extension survives.
    """
    matlab = exam["matlab_skeleton_yxz"].copy()
    vesselness = exam["vesselness_yxz"].copy()
    enhancement = exam["enhancement_yxz"].copy()

    # The known vessel at row 10 occupies columns 5..44. This segment overlaps its
    # last 20 columns and then carries on for another 13.
    _line(matlab, 10, 25, 58, 10)
    extension = np.zeros(SHAPE, dtype=bool)
    _line(extension, 10, 45, 58, 10)
    vesselness[extension] = 0.5
    enhancement[extension] = 100.0

    kept, _stats = filter_complement(
        matlab_skeleton_yxz=matlab,
        vesselness_yxz=vesselness,
        enhancement_yxz=enhancement,
        merged_skeleton_yxz=exam["merged_skeleton_yxz"],
        breast_mask_yxz=exam["breast_mask_yxz"],
        spacing_yxz=SPACING,
    )
    # The two columns closest to the known vessel fall inside the merge tolerance and
    # are already covered, so they are not "added"; everything beyond them should be.
    novel = extension & ~exam["merged_skeleton_yxz"]
    assert (
        (kept & novel).sum() >= extension.sum() - 3
    ), "the part extending past the known vessel should be added"


def test_component_statistics_match_a_direct_readout(exam: dict) -> None:
    """The grouped-by-label statistics must equal the obvious per-component readout."""
    from scipy import ndimage as ndi

    from preprocessing.complement import _component_elongation, _per_component_stats

    labeled, n = ndi.label(exam["matlab_skeleton_yxz"], structure=np.ones((3, 3, 3)))
    stats = _per_component_stats(
        labeled, exam["vesselness_yxz"], exam["enhancement_yxz"], SPACING
    )
    assert len(stats) == n

    for label, entry in stats.items():
        component = labeled == label
        coords_mm = np.argwhere(component).astype(np.float64) * np.asarray(SPACING)
        assert entry["size"] == int(component.sum())
        assert entry["mean_vesselness"] == pytest.approx(
            float(exam["vesselness_yxz"][component].mean())
        )
        assert entry["mean_enhancement"] == pytest.approx(
            float(exam["enhancement_yxz"][component].mean())
        )
        assert entry["elongation"] == pytest.approx(_component_elongation(coords_mm))
