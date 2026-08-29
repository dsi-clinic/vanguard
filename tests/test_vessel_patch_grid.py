"""The vessel patch grid must cover the whole volume, at every acquisition size.

`Dataset3DDivided` takes a fixed *count* of patches per axis and converts it to a
stride of ``(length - input_dim) // (divisions - 1)``, so the stride grows with the
volume. The pipeline used to hardcode 8x8x3, which silently stopped covering large
matrices: past ``stride > input_dim`` consecutive patches no longer touch and
`predict_vessel_batched` asserts on voxels no patch analyzed. That failed 15 of 283
exams in the v6 pCR preprocessing run (864^2-896^2 acquisitions, plus one
768x768x326 failing on z alone).

These tests pin both halves of the fix: large shapes are now covered, and -- just as
important -- shapes the old grid already handled keep their exact patch layout, so
the change cannot perturb existing outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from segmentation.vessel_patch_grid import (
    VESSEL_INPUT_DIM,
    VESSEL_MIN_XY_DIVISIONS,
    VESSEL_MIN_Z_DIVISIONS,
    divisions_for_axis,
    vessel_divisions_for_inputs,
)

# Real acquisition shapes from the v6 pCR cohort that the fixed 8x8x3 grid failed on.
FAILED_UNDER_LEGACY_GRID = [
    (784, 784, 200),
    (784, 784, 250),
    (864, 864, 210),
    (864, 864, 260),
    (880, 880, 220),
    (896, 896, 250),
    (768, 768, 326),  # in-plane is fine; z alone is uncovered
]

# Real shapes the legacy grid handled. 768 in-plane gives a stride of exactly
# input_dim and is the largest covered size, so it is the boundary case.
COVERED_UNDER_LEGACY_GRID = [
    (400, 400, 215),
    (432, 432, 250),
    (512, 512, 250),
    (640, 640, 230),
    (768, 768, 240),
    (528, 528, 280),
]


def _starts(length: int, divisions: int) -> list[int]:
    """Reproduce the submodule's per-axis patch starts for one axis."""
    step = (length - VESSEL_INPUT_DIM) // (divisions - 1)
    return [
        index * step if index != divisions - 1 else length - VESSEL_INPUT_DIM
        for index in range(divisions)
    ]


def _axis_covered(length: int, divisions: int) -> bool:
    """True when the patches along one axis leave no gap.

    Coverage of the 3-D volume is separable: the boxes are the full cartesian
    product of the per-axis starts, so a voxel is analyzed exactly when each of its
    three coordinates is covered on its own axis.
    """
    hit = np.zeros(length, dtype=bool)
    for start in _starts(length, divisions):
        hit[start : start + VESSEL_INPUT_DIM] = True
    return bool(hit.all())


def _grid_for(shape: tuple[int, int, int]) -> tuple[int, int]:
    x_y = max(
        divisions_for_axis(shape[0], VESSEL_MIN_XY_DIVISIONS),
        divisions_for_axis(shape[1], VESSEL_MIN_XY_DIVISIONS),
    )
    return x_y, divisions_for_axis(shape[2], VESSEL_MIN_Z_DIVISIONS)


@pytest.mark.parametrize("shape", FAILED_UNDER_LEGACY_GRID)
def test_large_shapes_were_broken_and_are_now_covered(
    shape: tuple[int, int, int],
) -> None:
    """Guards against the test itself going vacuous if the legacy grid ever changes."""
    legacy_gap = not (
        _axis_covered(shape[0], VESSEL_MIN_XY_DIVISIONS)
        and _axis_covered(shape[1], VESSEL_MIN_XY_DIVISIONS)
        and _axis_covered(shape[2], VESSEL_MIN_Z_DIVISIONS)
    )
    assert legacy_gap, f"{shape} no longer reproduces the defect; pick another shape"

    x_y_divisions, z_division = _grid_for(shape)
    assert _axis_covered(shape[0], x_y_divisions)
    assert _axis_covered(shape[1], x_y_divisions)
    assert _axis_covered(shape[2], z_division)


@pytest.mark.parametrize("shape", COVERED_UNDER_LEGACY_GRID)
def test_previously_working_shapes_keep_the_legacy_grid(
    shape: tuple[int, int, int],
) -> None:
    """The floor at 8/3 makes the fix a no-op wherever the old grid already worked."""
    assert _grid_for(shape) == (VESSEL_MIN_XY_DIVISIONS, VESSEL_MIN_Z_DIVISIONS)


def test_every_axis_length_is_covered_and_never_shrinks_the_grid() -> None:
    """Sweep every plausible axis length rather than trusting the sampled shapes."""
    for length in range(VESSEL_INPUT_DIM, 1301):
        for minimum in (VESSEL_MIN_XY_DIVISIONS, VESSEL_MIN_Z_DIVISIONS):
            divisions = divisions_for_axis(length, minimum)
            assert divisions >= minimum, (length, minimum, divisions)
            assert _axis_covered(length, divisions), (length, minimum, divisions)
            if _axis_covered(length, minimum):
                assert divisions == minimum, (
                    f"length {length} was already covered by {minimum} divisions; "
                    f"the fix must not change its patch layout"
                )


def test_three_dimensional_denominator_has_no_unanalyzed_voxels() -> None:
    """End-to-end on a real volume: the accumulator denominator is nowhere zero.

    775 in-plane is the smallest failing width, which keeps this to ~60 MB.
    """
    shape = (775, 775, 100)
    x_y_divisions, z_division = _grid_for(shape)

    def denominator(nxy: int, nz: int) -> np.ndarray:
        denom = np.zeros(shape, dtype=np.uint8)
        for x in _starts(shape[0], nxy):
            for y in _starts(shape[1], nxy):
                for z in _starts(shape[2], nz):
                    denom[
                        x : x + VESSEL_INPUT_DIM,
                        y : y + VESSEL_INPUT_DIM,
                        z : z + VESSEL_INPUT_DIM,
                    ] += 1
        return denom

    legacy = denominator(VESSEL_MIN_XY_DIVISIONS, VESSEL_MIN_Z_DIVISIONS)
    assert not (legacy != 0).all(), "expected the legacy grid to leave gaps here"

    fixed = denominator(x_y_divisions, z_division)
    assert (fixed != 0).all()


def test_divisions_are_read_from_the_input_headers(tmp_path: Path) -> None:
    """`vessel_divisions_for_inputs` drives off the STEP-1 shapes on disk."""
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    np.save(
        ordinary / "hr_phase_0000_id.npy", np.zeros((200, 200, 120), dtype=np.uint8)
    )
    assert vessel_divisions_for_inputs(ordinary) == (
        VESSEL_MIN_XY_DIVISIONS,
        VESSEL_MIN_Z_DIVISIONS,
    )

    large = tmp_path / "large"
    large.mkdir()
    # Header-only read, so a sparse memmap avoids materialising the volume.
    memmap = np.lib.format.open_memmap(
        large / "hr_phase_0000_id.npy", mode="w+", dtype=np.uint8, shape=(800, 800, 100)
    )
    del memmap
    x_y_divisions, z_division = vessel_divisions_for_inputs(large)
    assert x_y_divisions > VESSEL_MIN_XY_DIVISIONS
    assert _axis_covered(800, x_y_divisions)
    assert z_division == VESSEL_MIN_Z_DIVISIONS

    # The grid must cover the *largest* volume present, not merely the first one.
    np.save(large / "hr_phase_0001_id.npy", np.zeros((200, 200, 120), dtype=np.uint8))
    assert vessel_divisions_for_inputs(large) == (x_y_divisions, z_division)

    with pytest.raises(FileNotFoundError):
        vessel_divisions_for_inputs(tmp_path / "missing")
