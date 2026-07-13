"""Tests for the DUKE final symmetry-line inner-wall method."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TABULAR_DIR = REPO_ROOT / "tabular"
if str(TABULAR_DIR) not in sys.path:
    sys.path.insert(0, str(TABULAR_DIR))

from duke_final_symmetry_lines import (  # noqa: E402
    FINAL_METHOD,
    _inner_wall_midline_from_projection,
)

EXPECTED_MEDIAN_MIDLINE = 5.5
EXPECTED_CENTERLINE_MIDLINE = 5.0
EXPECTED_DUKE_ROWS = 284


def test_inner_wall_midline_uses_median_row_pair_midpoints() -> None:
    """The exported midline is the median of clean row-wise wall midpoints."""
    mask_yx = np.zeros((5, 12), dtype=bool)
    mask_yx[0, [1, 9]] = True
    mask_yx[1, [2, 9]] = True
    mask_yx[2, [3, 8]] = True
    mask_yx[3, [2, 10]] = True
    mask_yx[4, [3, 9]] = True

    midline, reason = _inner_wall_midline_from_projection(
        mask_yx,
        image_midline_x=5.0,
        min_pair_fraction=0.6,
    )

    assert reason == ""
    assert midline == EXPECTED_MEDIAN_MIDLINE


def test_inner_wall_midline_skips_rows_where_center_is_inside_mask() -> None:
    """Rows with mask at image center do not contribute wall-pair midpoints."""
    mask_yx = np.zeros((4, 11), dtype=bool)
    mask_yx[0, [2, 8]] = True
    mask_yx[1, [3, 7]] = True
    mask_yx[2, 5] = True
    mask_yx[3, 5] = True

    midline, reason = _inner_wall_midline_from_projection(
        mask_yx,
        image_midline_x=5.0,
        min_pair_fraction=0.5,
    )

    assert reason == ""
    assert midline == EXPECTED_CENTERLINE_MIDLINE


def test_inner_wall_midline_requires_enough_clean_wall_pairs() -> None:
    """The helper fails when too few rows expose clean left/right inner walls."""
    mask_yx = np.zeros((6, 11), dtype=bool)
    mask_yx[0, [2, 8]] = True
    mask_yx[1:, 5] = True

    midline, reason = _inner_wall_midline_from_projection(
        mask_yx,
        image_midline_x=5.0,
        min_pair_fraction=0.5,
    )

    assert midline is None
    assert reason == "middle_overlap_or_too_few_inner_wall_rows:1<3"


def test_final_duke_export_uses_only_inner_walls() -> None:
    """The committed DUKE export contains only the final inner-wall method."""
    csv_path = REPO_ROOT / "tabular" / "duke_final_symmetry_lines.csv"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == EXPECTED_DUKE_ROWS
    assert {row["side_split_method"] for row in rows} == {FINAL_METHOD}
