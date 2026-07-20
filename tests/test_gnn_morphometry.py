"""Unit tests for morphometry graph-level features (``gnn.morphometry``).

Exercises ``build_morphometry_feature_matrix`` on synthetic
``<case_id>_morphometry.json`` fixtures matching the real nested schema
(component -> segment/bifurcation entries) -- no cluster data required.
Includes a zero-segment component to exercise the NaN-impute path
documented in ``AUDITING_RESULTS.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.morphometry import MORPHOMETRY_COLUMNS, build_morphometry_feature_matrix


def _write_morphometry(path: Path, *, num_segments: int) -> None:
    """Write a synthetic morphometry.json with ``num_segments`` segment entries."""
    segments = [
        {
            "segment": {"start": [0, 0, 0], "end": [i + 1, 0, 0]},
            "radius": {
                "mean": 1.5,
                "sd": 0.1,
                "median": 1.5,
                "min": 1.4,
                "q1": 1.4,
                "q3": 1.6,
                "max": 1.7,
            },
            "length": float(i + 1),
            "tortuosity": 1.1,
            "volume": 10.0,
            "curvature": {
                "mean": 0.2,
                "sd": 0.05,
                "median": 0.2,
                "min": 0.1,
                "q1": 0.15,
                "q3": 0.25,
                "max": 0.3,
            },
        }
        for i in range(num_segments)
    ]
    bifurcations = (
        [
            {
                "bifurcation": {
                    "midpoint": [1, 0, 0],
                    "points_angle": [[0, 0, 0], [2, 0, 0], [1, 1, 0]],
                    "degree": 3,
                },
                "angles": {"pair1": 90.0, "pair2": 120.0, "pair3": 150.0},
            }
        ]
        if num_segments > 0
        else []
    )
    morph = {"1": {"component_1": segments, "component_1 bifurcation": bifurcations}}
    path.write_text(json.dumps(morph))


def test_build_matrix_extracts_real_values(tmp_path: Path) -> None:
    """Requested columns are extracted correctly for each case."""
    path_a = tmp_path / "A_morphometry.json"
    path_b = tmp_path / "B_morphometry.json"
    _write_morphometry(path_a, num_segments=3)
    _write_morphometry(path_b, num_segments=5)

    matrix, names = build_morphometry_feature_matrix(
        {"A": path_a, "B": path_b},
        ["A", "B"],
        ["morph_seg_length_n", "morph_bifurcation_count"],
    )
    assert names == ["morph_seg_length_n", "morph_bifurcation_count"]
    assert matrix[:, 0].tolist() == [3.0, 5.0]
    assert matrix[:, 1].tolist() == [1.0, 1.0]


def test_empty_component_produces_nan_that_gets_imputed(tmp_path: Path) -> None:
    """A case with zero segments yields NaN in array_stats, mean-imputed away."""
    path_empty = tmp_path / "EMPTY_morphometry.json"
    path_normal = tmp_path / "NORMAL_morphometry.json"
    _write_morphometry(path_empty, num_segments=0)
    _write_morphometry(path_normal, num_segments=4)

    matrix, names = build_morphometry_feature_matrix(
        {"EMPTY": path_empty, "NORMAL": path_normal},
        ["EMPTY", "NORMAL"],
        ["morph_seg_length_mean"],
    )
    assert not (matrix != matrix).any()  # no NaN survives (x != x is the NaN check)
    # The empty case's NaN was imputed to the mean of the (single) real value.
    assert matrix[0, 0] == pytest.approx(matrix[1, 0])
    assert names == ["morph_seg_length_mean"]


def test_unknown_column_raises(tmp_path: Path) -> None:
    """A requested column outside the supported vocabulary raises."""
    path = tmp_path / "A_morphometry.json"
    _write_morphometry(path, num_segments=1)
    with pytest.raises(ValueError, match="Unknown morphometry columns"):
        build_morphometry_feature_matrix({"A": path}, ["A"], ["not_a_real_column"])


def test_missing_case_raises(tmp_path: Path) -> None:
    """A case_id absent from morphometry_paths_by_case is a caller bug, not an expected case."""
    path = tmp_path / "A_morphometry.json"
    _write_morphometry(path, num_segments=1)
    with pytest.raises(KeyError, match="No morphometry_path"):
        build_morphometry_feature_matrix(
            {"A": path}, ["A", "ZZZ"], ["morph_seg_length_n"]
        )


def test_output_row_order_matches_case_ids_order(tmp_path: Path) -> None:
    """Rows are aligned to the requested case_ids order, not dict insertion order."""
    path_a = tmp_path / "A_morphometry.json"
    path_b = tmp_path / "B_morphometry.json"
    _write_morphometry(path_a, num_segments=2)
    _write_morphometry(path_b, num_segments=7)

    matrix, _names = build_morphometry_feature_matrix(
        {"A": path_a, "B": path_b}, ["B", "A"], ["morph_seg_length_n"]
    )
    assert matrix[:, 0].tolist() == [7.0, 2.0]


def test_all_columns_pinned_and_extractable(tmp_path: Path) -> None:
    """MORPHOMETRY_COLUMNS matches what extract_morphometry_features actually emits."""
    path = tmp_path / "A_morphometry.json"
    _write_morphometry(path, num_segments=2)
    matrix, names = build_morphometry_feature_matrix(
        {"A": path}, ["A"], list(MORPHOMETRY_COLUMNS)
    )
    assert matrix.shape == (1, len(MORPHOMETRY_COLUMNS))
    assert names == list(MORPHOMETRY_COLUMNS)
