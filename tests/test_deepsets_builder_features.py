"""Tests for Deep Sets point-feature construction."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from config import DEFAULT_CONFIG

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    from build_deepsets_dataset import (
        SUPPORT_MASK_PATTERN,
        _build_case_set,
        _load_support_radius_mm,
        build_deepsets_manifest_frame,
        resolve_deepsets_point_features,
    )
else:
    SUPPORT_MASK_PATTERN = "{case_id}_skeleton_4d_exam_support_mask.npy"

EXPANDED_FEATURES = [
    "curvature_rad",
    "signed_distance_tumor_mm",
    "abs_distance_tumor_mm",
    "inside_tumor",
    "skeleton_node_degree",
    "is_endpoint",
    "is_chain",
    "is_junction",
    "offset_from_tumor_centroid_norm_x",
    "offset_from_tumor_centroid_norm_y",
    "offset_from_tumor_centroid_norm_z",
    "direction_to_tumor_centroid_x",
    "direction_to_tumor_centroid_y",
    "direction_to_tumor_centroid_z",
    "local_vessel_radius_mm",
]


def _line_case_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a three-point skeleton crossing a one-voxel tumor."""
    skeleton = np.zeros((3, 3, 3), dtype=bool)
    skeleton[1, 1, 0:3] = True
    tumor = np.zeros_like(skeleton)
    tumor[1, 1, 1] = True
    support = np.zeros_like(skeleton)
    support[1, 1, 1] = True
    support_radius = ndimage.distance_transform_edt(support, sampling=(1.0, 1.0, 1.0))
    return skeleton, tumor, support_radius


def test_default_config_resolves_to_curvature_only() -> None:
    """The default config should preserve the legacy feature set."""
    raw_features = DEFAULT_CONFIG["feature_toggles"]["deepsets_point_features"]
    assert resolve_deepsets_point_features(raw_features) == ["curvature_rad"]
    assert resolve_deepsets_point_features(None) == ["curvature_rad"]


def test_invalid_point_feature_raises() -> None:
    """Unsupported feature names should fail before any dataset build starts."""
    with pytest.raises(ValueError, match="Unknown Deep Sets point features"):
        resolve_deepsets_point_features(["curvature_rad", "not_a_feature"])


def test_expanded_point_features_are_serialized_in_configured_order() -> None:
    """Synthetic masks should produce deterministic expanded point features."""
    skeleton, tumor, support_radius = _line_case_masks()
    case_set = _build_case_set(
        case_id="CASE_001",
        label=1,
        skeleton_mask_zyx=skeleton,
        tumor_mask_zyx=tumor,
        spacing_mm_zyx=(1.0, 1.0, 1.0),
        local_radius_mm=10.0,
        tumor_equiv_radius_mm=1.0,
        point_feature_names=EXPANDED_FEATURES,
        support_radius_mm_zyx=support_radius,
    )

    assert case_set is not None
    assert case_set["feature_names"] == EXPANDED_FEATURES
    x = case_set["x"].numpy()
    assert x.shape == (3, len(EXPANDED_FEATURES))

    col = {name: idx for idx, name in enumerate(EXPANDED_FEATURES)}
    left, center, right = x

    assert left[col["signed_distance_tumor_mm"]] == pytest.approx(1.0)
    assert center[col["signed_distance_tumor_mm"]] == pytest.approx(-1.0)
    assert center[col["abs_distance_tumor_mm"]] == pytest.approx(1.0)
    assert center[col["inside_tumor"]] == pytest.approx(1.0)
    assert left[col["inside_tumor"]] == pytest.approx(0.0)

    assert left[col["skeleton_node_degree"]] == pytest.approx(1.0)
    assert center[col["skeleton_node_degree"]] == pytest.approx(2.0)
    assert left[col["is_endpoint"]] == pytest.approx(1.0)
    assert center[col["is_chain"]] == pytest.approx(1.0)
    assert center[col["is_junction"]] == pytest.approx(0.0)
    assert center[col["curvature_rad"]] == pytest.approx(math.pi)

    assert left[col["offset_from_tumor_centroid_norm_x"]] == pytest.approx(-0.1)
    assert center[col["offset_from_tumor_centroid_norm_x"]] == pytest.approx(0.0)
    assert right[col["offset_from_tumor_centroid_norm_x"]] == pytest.approx(0.1)
    assert left[col["direction_to_tumor_centroid_x"]] == pytest.approx(1.0)
    assert center[col["direction_to_tumor_centroid_x"]] == pytest.approx(0.0)
    assert right[col["direction_to_tumor_centroid_x"]] == pytest.approx(-1.0)

    assert left[col["local_vessel_radius_mm"]] == pytest.approx(0.0)
    assert center[col["local_vessel_radius_mm"]] == pytest.approx(1.0)


def test_support_radius_loader_reports_missing_and_bad_masks(tmp_path: Path) -> None:
    """Support-radius loading should return skip reasons instead of fake values."""
    radius, reason = _load_support_radius_mm(
        study_dir=tmp_path,
        case_id="CASE_002",
        expected_shape_zyx=(2, 2, 2),
        spacing_mm_zyx=(1.0, 1.0, 1.0),
    )
    assert radius is None
    assert reason == "missing"

    np.save(
        tmp_path / SUPPORT_MASK_PATTERN.format(case_id="CASE_002"),
        np.zeros((1, 1, 1), dtype=bool),
    )
    radius, reason = _load_support_radius_mm(
        study_dir=tmp_path,
        case_id="CASE_002",
        expected_shape_zyx=(2, 2, 2),
        spacing_mm_zyx=(1.0, 1.0, 1.0),
    )
    assert radius is None
    assert reason == "bad"


def test_empty_manifest_frame_preserves_columns() -> None:
    """An all-skipped build should still be able to write an empty manifest."""
    manifest_df = build_deepsets_manifest_frame([], ["site", "tumor_subtype"])

    assert manifest_df.empty
    assert list(manifest_df.columns) == [
        "case_id",
        "set_path",
        "label",
        "dataset",
        "num_points",
        "local_radius_mm",
        "tumor_equiv_radius_mm",
        "used_fallback_nearest_points",
        "site",
        "tumor_subtype",
    ]


def test_toy_modes_remain_compatible_with_selected_features() -> None:
    """Toy modes should preserve their existing label-feature behavior."""
    skeleton, tumor, _ = _line_case_masks()
    toy_only_set = _build_case_set(
        case_id="CASE_003",
        label=1,
        skeleton_mask_zyx=skeleton,
        tumor_mask_zyx=tumor,
        spacing_mm_zyx=(1.0, 1.0, 1.0),
        local_radius_mm=10.0,
        tumor_equiv_radius_mm=1.0,
        point_feature_names=EXPANDED_FEATURES,
        toy_only=True,
    )

    assert toy_only_set is not None
    assert toy_only_set["feature_names"] == ["toy_perfect_label"]
    assert toy_only_set["x"].shape == (3, 1)
    assert np.allclose(toy_only_set["x"].numpy(), 1.0)

    toy_appended_set = _build_case_set(
        case_id="CASE_004",
        label=1,
        skeleton_mask_zyx=skeleton,
        tumor_mask_zyx=tumor,
        spacing_mm_zyx=(1.0, 1.0, 1.0),
        local_radius_mm=10.0,
        tumor_equiv_radius_mm=1.0,
        point_feature_names=["curvature_rad", "inside_tumor"],
        toy_perfect_feature=True,
    )

    assert toy_appended_set is not None
    assert toy_appended_set["feature_names"] == [
        "curvature_rad",
        "inside_tumor",
        "toy_perfect_label",
    ]
    assert np.allclose(toy_appended_set["x"].numpy()[:, -1], 1.0)
