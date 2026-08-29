#!/usr/bin/env python3
"""Choose the vessel-inference patch grid from the volume being segmented.

``Dataset3DDivided`` samples a *fixed number* of patches along each axis and turns
that count into a stride:

    stride = (length - input_dim) // (divisions - 1)

so the stride grows with the volume rather than staying put. The pipeline used to
pass a hardcoded 8x8x3 grid, which quietly stopped covering large acquisitions:
once ``stride > input_dim`` consecutive patches no longer touch, the volume keeps
voxels that no patch ever analyzed, and ``predict_vessel_batched`` refuses to
divide its accumulator by a zero denominator. With ``input_dim=96`` that made
every exam with an in-plane matrix >= 775, or >= 290 slices, fail outright.

The helpers here derive the counts from the actual STEP-1 input shapes, floored at
the historical 8 and 3. Because of that floor they return the legacy grid for
every shape the legacy grid already covered, so switching to them cannot change
any output that previously succeeded.

Note that ``stride <= input_dim`` is necessary but *not* sufficient for coverage.
The final patch is pinned to ``length - input_dim`` rather than placed on the
stride, and the stride is floored, so the last jump is ``stride + span % (n - 1)``.
Length 289 with 3 divisions is the smallest counterexample: the stride is exactly
96, yet patches end at 192 and the last one starts at 193. Coverage is therefore
tested directly instead of being predicted by a closed form.

Kept free of torch and of the segmentation submodule so it stays importable (and
testable) on its own.
"""

from __future__ import annotations

from pathlib import Path

VESSEL_INPUT_DIM = 96
VESSEL_MIN_XY_DIVISIONS = 8
VESSEL_MIN_Z_DIVISIONS = 3

# Two offsets are the fewest that define a stride at all, and STEP-1 inputs are
# single-channel volumes.
_MIN_DIVISIONS = 2
_SPATIAL_DIMENSIONS = 3


def axis_patch_starts(length: int, divisions: int) -> list[int]:
    """Patch offsets ``generate_divided_boxes_dict`` produces along one axis."""
    span = max(int(length) - VESSEL_INPUT_DIM, 0)
    if divisions < _MIN_DIVISIONS or span == 0:
        return [0]
    step = span // (divisions - 1)
    return [
        index * step if index != divisions - 1 else span for index in range(divisions)
    ]


def axis_is_covered(length: int, divisions: int) -> bool:
    """True when patches along one axis leave no unanalyzed voxel.

    Coverage of the volume is separable: the boxes are the full cartesian product
    of the per-axis offsets, so a voxel is analyzed exactly when each of its three
    coordinates is covered on its own axis.
    """
    reach = 0
    for start in sorted(set(axis_patch_starts(length, divisions))):
        if start > reach:
            return False
        reach = max(reach, start + VESSEL_INPUT_DIM)
    return reach >= max(int(length), VESSEL_INPUT_DIM)


def divisions_for_axis(length: int, minimum: int) -> int:
    """Smallest division count >= ``minimum`` that covers the axis completely.

    Searched rather than solved: see the module docstring for why the obvious
    closed form is wrong at the boundary.
    """
    divisions = max(int(minimum), _MIN_DIVISIONS)
    limit = divisions + max(int(length), VESSEL_INPUT_DIM)
    while divisions <= limit:
        if axis_is_covered(length, divisions):
            return divisions
        divisions += 1
    raise ValueError(f"no patch grid covers an axis of length {length}")


def vessel_divisions_for_inputs(step1_dir: Path | str) -> tuple[int, int]:
    """Pick ``(x_y_divisions, z_division)`` covering every volume in ``step1_dir``.

    Only the ``.npy`` headers are read, so this stays cheap no matter how large
    the volumes are.
    """
    from numpy.lib import format as npy_format

    shapes = []
    for path in sorted(Path(step1_dir).glob("*.npy")):
        with path.open("rb") as handle:
            npy_format.read_magic(handle)
            shapes.append(npy_format.read_array_header_1_0(handle)[0])
    if not shapes:
        raise FileNotFoundError(f"no STEP-1 model inputs in {step1_dir}")
    if any(len(shape) != _SPATIAL_DIMENSIONS for shape in shapes):
        raise ValueError(f"expected 3-D STEP-1 inputs in {step1_dir}, got {shapes}")

    x_y_divisions = max(
        max(
            divisions_for_axis(shape[0], VESSEL_MIN_XY_DIVISIONS),
            divisions_for_axis(shape[1], VESSEL_MIN_XY_DIVISIONS),
        )
        for shape in shapes
    )
    z_division = max(
        divisions_for_axis(shape[2], VESSEL_MIN_Z_DIVISIONS) for shape in shapes
    )
    return x_y_divisions, z_division
