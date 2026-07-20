"""Identity-seam tests for ``build_centerline_features``'s adapter usage.

The dataset adapter is required (multi-dataset migration Step 5), so there is
no ``adapter=None`` path left to compare against. What's still worth pinning:
a case's ``dataset`` column always comes from ``adapter.case_dataset_name()``
-- the case-id prefix -- even when the parent directory name disagrees, so a
misfiled study directory can't silently report the wrong cohort.
"""

from __future__ import annotations

from pathlib import Path

from config import DEFAULT_CONFIG, ConfigNode, _deep_merge


def _config(centerline_root: Path) -> ConfigNode:
    """A minimal run config that only walks the centerline tree for identity.

    Every feature block that would need real metadata (clinical, morphometry,
    tumor masks, tumor-graph JSON) is disabled so the test depends on nothing
    but the directory layout.
    """
    overrides = {
        "data_paths": {
            "centerline_root": str(centerline_root),
            "tumor_mask_root": str(centerline_root / "no_masks"),
        },
        "feature_toggles": {
            "use_vascular": True,
            "use_clinical": False,
            "require_centerline_file": False,
            "include_missing_centerline_rows": True,
            "use_morphometry": False,
            "use_tumor_local_features": False,
            "use_tumor_graph_features_json": False,
            "bilateral_filter": None,
            "dataset_include": None,
        },
    }
    return ConfigNode._wrap(_deep_merge(DEFAULT_CONFIG, overrides))


def _make_study(root: Path, dataset_dir: str, case_id: str) -> None:
    """Create one empty study directory ``root/<dataset_dir>/<case_id>/``."""
    (root / dataset_dir / case_id).mkdir(parents=True, exist_ok=True)


def test_dataset_identity_follows_adapter_for_mamamia_shaped_tree(
    tmp_path: Path,
) -> None:
    """On a MAMA-MIA-shaped tree, identity matches both the adapter and the dir."""
    from cohorts.mamamia import MamaMiaDataset
    from tabular.cohort import build_centerline_features

    root = tmp_path / "studies"
    _make_study(root, "DUKE", "DUKE_001")
    _make_study(root, "ISPY2", "ISPY2_045")
    _make_study(root, "NACT", "NACT_007")
    config = _config(root)

    result = build_centerline_features(
        config, adapter=MamaMiaDataset(cohort=None, root=tmp_path)
    )

    assert sorted(result["dataset"]) == ["DUKE", "ISPY2", "NACT"]


def test_dataset_identity_follows_case_id_prefix_not_directory_name(
    tmp_path: Path,
) -> None:
    """When the directory name lies, identity follows the case-id prefix.

    This is the seam Step 2 introduced: cohort identity is the adapter's
    authoritative answer, not whatever directory a study happens to sit in.
    """
    from cohorts.mamamia import MamaMiaDataset
    from tabular.cohort import build_centerline_features

    root = tmp_path / "studies"
    # Directory says "MISLABELED" but the case id prefix says "DUKE".
    _make_study(root, "MISLABELED", "DUKE_777")
    config = _config(root)

    result = build_centerline_features(
        config, adapter=MamaMiaDataset(cohort=None, root=tmp_path)
    )

    assert result["dataset"].tolist() == ["DUKE"]
