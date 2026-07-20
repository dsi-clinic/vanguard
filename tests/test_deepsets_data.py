"""Tests for deepsets.data.SavedSetLookup, in particular its keep_features param.

keep_features is the mechanism LOCO uses to select a column subset from a
case built once under the full loco_superset regime, without a rebuild per
covariate -- see evaluation/loco_deepsets.py and
docs/loco_feature_importance.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from deepsets.data import SavedSetLookup  # noqa: E402

FEATURE_NAMES = ["signed_distance_mm", "degree", "curvature_rad", "peak_enhancement"]


def _write_case(tmp_path: Path, case_id: str) -> Path:
    """Write one synthetic case .pt payload with a small, known feature matrix."""
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=torch.float32,
    )
    payload = {
        "x": x,
        "y": torch.tensor([1.0], dtype=torch.float32),
        "case_id": case_id,
        "feature_names": list(FEATURE_NAMES),
        "num_points": 2,
    }
    path = tmp_path / f"{case_id}.pt"
    torch.save(payload, path)
    return path


@pytest.fixture
def manifest_df(tmp_path: Path) -> pd.DataFrame:
    """Two synthetic cases with a known, fixed 4-column feature matrix each."""
    path_a = _write_case(tmp_path, "case_a")
    path_b = _write_case(tmp_path, "case_b")
    return pd.DataFrame(
        {"case_id": ["case_a", "case_b"], "set_path": [str(path_a), str(path_b)]}
    )


def test_get_without_keep_features_returns_every_column(
    manifest_df: pd.DataFrame,
) -> None:
    """With no keep_features, .get() returns every cached column unchanged."""
    lookup = SavedSetLookup(manifest_df)
    payload = lookup.get("case_a")
    assert payload["feature_names"] == FEATURE_NAMES
    assert payload["x"].shape == (2, 4)


def test_keep_features_subsets_shape_and_feature_names(
    manifest_df: pd.DataFrame,
) -> None:
    """keep_features narrows both x's column count and feature_names."""
    lookup = SavedSetLookup(manifest_df, keep_features=["curvature_rad", "degree"])
    payload = lookup.get("case_a")
    assert payload["feature_names"] == ["curvature_rad", "degree"]
    assert payload["x"].shape == (2, 2)


def test_keep_features_values_are_order_faithful_not_storage_order(
    manifest_df: pd.DataFrame,
) -> None:
    """keep_features=[curvature_rad, degree] selects cols 2,1 in that order.

    Not whatever order they happen to sit in in the stored tensor.
    """
    lookup = SavedSetLookup(manifest_df, keep_features=["curvature_rad", "degree"])
    payload = lookup.get("case_a")
    # Row 0 storage order is [signed_distance_mm=1, degree=2, curvature_rad=3, peak_enhancement=4]
    # so curvature_rad=3 then degree=2, not the storage order 2 then 3.
    assert payload["x"][0].tolist() == [3.0, 2.0]
    assert payload["x"][1].tolist() == [30.0, 20.0]


def test_keep_features_unknown_name_raises(manifest_df: pd.DataFrame) -> None:
    """Requesting a feature name absent from the case's own feature_names fails loud."""
    lookup = SavedSetLookup(
        manifest_df, keep_features=["curvature_rad", "not_a_feature"]
    )
    with pytest.raises(ValueError, match="not_a_feature"):
        lookup.get("case_a")


def test_subset_applies_keep_features_to_every_case(manifest_df: pd.DataFrame) -> None:
    """.subset() applies the same keep_features narrowing to every case ID."""
    lookup = SavedSetLookup(manifest_df, keep_features=["degree"])
    payloads = lookup.subset(["case_a", "case_b"])
    assert all(p["x"].shape == (2, 1) for p in payloads)
    assert all(p["feature_names"] == ["degree"] for p in payloads)
