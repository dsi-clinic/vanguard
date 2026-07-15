"""Step 2 parity tests: the dataset adapter is a no-op for MAMA-MIA identity.

These are the CI-safe counterpart to the full-data gate in
``scripts/validate_adapter_feature_parity.py``. They build a tiny synthetic
centerline tree (no real data needed) and check two things:

1. On a MAMA-MIA-shaped tree (each study's parent directory equals its case-id
   prefix), ``build_centerline_features`` produces the *identical* table with
   and without a ``MamaMiaDataset`` adapter — the invariant Step 2 must hold
   (see cohorts/README.md).
2. On a deliberately mismatched tree (directory name != id prefix), the adapter
   result follows ``case_dataset_name(case_id)`` rather than the directory name,
   proving the identity seam is actually live and not dead code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def test_adapter_is_identity_noop_for_mamamia_shaped_tree(tmp_path: Path) -> None:
    """Adapter on vs. off yields byte-identical tables when dir == id prefix."""
    from cohorts.mamamia import MamaMiaDataset
    from tabular.cohort import build_centerline_features

    root = tmp_path / "studies"
    _make_study(root, "DUKE", "DUKE_001")
    _make_study(root, "ISPY2", "ISPY2_045")
    _make_study(root, "NACT", "NACT_007")
    config = _config(root)

    without = build_centerline_features(config, adapter=None)
    with_adapter = build_centerline_features(
        config, adapter=MamaMiaDataset(cohort="duke", root=tmp_path)
    )

    # Same rows in the same order, and identical everywhere (identity is the
    # only thing the adapter touches, and it agrees here).
    pd.testing.assert_frame_equal(without, with_adapter)
    assert sorted(with_adapter["dataset"]) == ["DUKE", "ISPY2", "NACT"]


def test_adapter_seam_is_live_when_dir_disagrees_with_prefix(tmp_path: Path) -> None:
    """When the directory name lies, the adapter follows the case-id prefix."""
    from cohorts.mamamia import MamaMiaDataset
    from tabular.cohort import build_centerline_features

    root = tmp_path / "studies"
    # Directory says "MISLABELED" but the case id prefix says "DUKE".
    _make_study(root, "MISLABELED", "DUKE_777")
    config = _config(root)

    without = build_centerline_features(config, adapter=None)
    with_adapter = build_centerline_features(
        config, adapter=MamaMiaDataset(cohort="duke", root=tmp_path)
    )

    # Fallback trusts the directory; the adapter trusts the id prefix. This is
    # exactly the three-way disagreement Step 2 exists to collapse (see cohorts/README.md).
    assert without["dataset"].tolist() == ["MISLABELED"]
    assert with_adapter["dataset"].tolist() == ["DUKE"]
