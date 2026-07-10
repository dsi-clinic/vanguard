"""Tests for consuming adapter-provided CV folds in the eval path (Step 3 follow-up).

``cohorts.resolve_folds`` parses and validates a dataset's shipped folds, but
until this wiring, nothing in the pipeline actually consumed them -- setting
``split_policy: provided`` had no effect on a real run. These tests cover the
seam that closes that gap: ``tabular.train._apply_provided_folds`` merges the
resolved folds onto the feature table, and ``prepare_evaluation_context``
excludes the merged column from the model's input features so it can't leak in
as a predictor when a run opts into ``model_params.split_mode: predefined``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cohorts import MamaMiaDataset, UChicagoDataset
from config import DEFAULT_CONFIG, ConfigNode, _deep_merge
from tabular.train import _apply_provided_folds, prepare_evaluation_context


def _write_manifest(tmp_path: Path) -> Path:
    """Write a tiny UChicago-shaped manifest CSV and return its root dir."""
    root = tmp_path / "uc"
    root.mkdir()
    csv_path = root / "dce2d_internal_ultrafast_manifest.csv"
    csv_path.write_text(
        "exam_id,dataset,patient_key,fold,pcr,phase_files\n"
        'e1,simbiosys,p1,0,1.0,"[""/a/p0.nii.gz""]"\n'
        'e2,uch_nac,p2,1,0.0,"[""/b/p0.nii.gz""]"\n'
        'e3,her2_naclike,p3,0,1.0,"[""/c/p0.nii.gz""]"\n'
    )
    return root


def _config(dataset_overrides: dict) -> ConfigNode:
    return ConfigNode._wrap(_deep_merge(DEFAULT_CONFIG, {"dataset": dataset_overrides}))


def test_apply_provided_folds_merges_manifest_folds(tmp_path: Path) -> None:
    """When split_policy resolves to 'provided', the fold column is merged in."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = _config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "auto"}
    )
    feats_df = pd.DataFrame({"case_id": ["e1", "e2", "e3"], "pcr": [1, 0, 1]})

    merged = _apply_provided_folds(feats_df, config, adapter)

    assert "fold" in merged.columns
    assert dict(zip(merged["case_id"], merged["fold"], strict=True)) == {
        "e1": 0,
        "e2": 1,
        "e3": 0,
    }


def test_apply_provided_folds_noop_when_policy_is_compute(tmp_path: Path) -> None:
    """With split_policy forced to 'compute', the table is returned unchanged."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = _config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "compute"}
    )
    feats_df = pd.DataFrame({"case_id": ["e1", "e2", "e3"], "pcr": [1, 0, 1]})

    merged = _apply_provided_folds(feats_df, config, adapter)

    pd.testing.assert_frame_equal(merged, feats_df)


def test_apply_provided_folds_uses_configured_split_col_name(tmp_path: Path) -> None:
    """The merged column is named after model_params.split_col, not hardcoded."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = ConfigNode._wrap(
        _deep_merge(
            DEFAULT_CONFIG,
            {
                "dataset": {
                    "name": "uchicago",
                    "cohort": None,
                    "root": "/x",
                    "split_policy": "auto",
                },
                "model_params": {"split_col": "provided_fold"},
            },
        )
    )
    feats_df = pd.DataFrame({"case_id": ["e1", "e2", "e3"], "pcr": [1, 0, 1]})

    merged = _apply_provided_folds(feats_df, config, adapter)

    assert "provided_fold" in merged.columns
    assert "fold" not in merged.columns


def test_apply_provided_folds_noop_for_mamamia_default_compute() -> None:
    """MAMA-MIA's default policy is 'compute', so nothing is merged by default."""
    adapter = MamaMiaDataset(cohort="duke", root=Path("/x"))
    config = _config(
        {"name": "mamamia", "cohort": "duke", "root": "/x", "split_policy": "auto"}
    )
    feats_df = pd.DataFrame({"case_id": ["DUKE_001", "DUKE_002"], "pcr": [1, 0]})

    merged = _apply_provided_folds(feats_df, config, adapter)

    pd.testing.assert_frame_equal(merged, feats_df)


def test_predefined_split_col_excluded_from_model_features() -> None:
    """The merged fold column must never leak into X as a predictor.

    Regression check for the gap the review flagged: prepare_evaluation_context
    didn't drop model_params.split_col, so a predefined-mode run would have fed
    the fold assignment itself into the model.
    """
    feats_df = pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "pcr": [1, 0, 1, 0],
            "dataset": ["uchicago"] * 4,
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "feature_b": [1.0, 2.0, 3.0, 4.0],
            "fold": [0, 0, 1, 1],
        }
    )
    config = ConfigNode._wrap(
        _deep_merge(
            DEFAULT_CONFIG,
            {
                "model_params": {"split_mode": "predefined", "split_col": "fold"},
                "feature_toggles": {"use_clinical": False},
            },
        )
    )

    context = prepare_evaluation_context(feats_df, config)

    expected_n_folds = 2
    assert "fold" not in context["X"].columns
    assert set(context["X"].columns) == {"feature_a", "feature_b"}
    assert len(context["splits"]) == expected_n_folds
