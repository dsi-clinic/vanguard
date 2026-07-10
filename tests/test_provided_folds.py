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
import pytest

from cohorts import MamaMiaDataset, UChicagoDataset
from config import DEFAULT_CONFIG, ConfigNode, _deep_merge
from tabular.train import (
    _apply_group_keys,
    _apply_provided_folds,
    prepare_evaluation_context,
)


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


def test_apply_provided_folds_raises_on_missing_fold(tmp_path: Path) -> None:
    """A modeled case with no provided fold fails closed, unconditionally.

    'e4' is in the feature table but absent from the manifest, so the left merge
    leaves its fold null. Attaching provided folds must cover every modeled case;
    otherwise create_predefined_splits would build a spurious empty-validation
    fold and leak those rows into every fold's training set.
    """
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
                "model_params": {"split_mode": "predefined", "split_col": "fold"},
            },
        )
    )
    feats_df = pd.DataFrame({"case_id": ["e1", "e2", "e3", "e4"], "pcr": [1, 0, 1, 0]})

    with pytest.raises(ValueError, match="no provided fold"):
        _apply_provided_folds(feats_df, config, adapter)


def test_apply_provided_folds_fails_closed_even_in_random_mode(tmp_path: Path) -> None:
    """Fail-closed does not depend on split_mode: a missing fold always raises.

    Even a run that would compute random splits must not silently attach an
    incomplete provided-fold column; the dataset's policy is "provided", so the
    folds are expected to cover every case.
    """
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = _config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "auto"}
    )  # model_params.split_mode defaults to "random"
    feats_df = pd.DataFrame({"case_id": ["e1", "e2", "e3", "e4"], "pcr": [1, 0, 1, 0]})

    with pytest.raises(ValueError, match="no provided fold"):
        _apply_provided_folds(feats_df, config, adapter)


def test_apply_provided_folds_raises_on_duplicate_fold_mapping(tmp_path: Path) -> None:
    """A manifest that maps one exam to two folds is rejected before the merge."""
    root = tmp_path / "uc"
    root.mkdir()
    (root / "dce2d_internal_ultrafast_manifest.csv").write_text(
        "exam_id,dataset,patient_key,fold,pcr,phase_files\n"
        'e1,simbiosys,p1,0,1.0,"[""/a/p0.nii.gz""]"\n'
        'e1,simbiosys,p1,1,1.0,"[""/a/p0.nii.gz""]"\n'  # e1 mapped twice
        'e2,uch_nac,p2,1,0.0,"[""/b/p0.nii.gz""]"\n'
    )
    adapter = UChicagoDataset(root=root)
    config = _config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "auto"}
    )
    feats_df = pd.DataFrame({"case_id": ["e1", "e2"], "pcr": [1, 0]})

    with pytest.raises(ValueError, match="more than once"):
        _apply_provided_folds(feats_df, config, adapter)


def test_apply_provided_folds_rejects_split_col_collision(tmp_path: Path) -> None:
    """split_col must not clobber case_id or the label column."""
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
                "model_params": {"split_col": "pcr"},  # collides with the label column
            },
        )
    )
    feats_df = pd.DataFrame({"case_id": ["e1", "e2", "e3"], "pcr": [1, 0, 1]})

    with pytest.raises(ValueError, match="collides"):
        _apply_provided_folds(feats_df, config, adapter)


def test_apply_provided_folds_rejects_duplicate_case_in_features(
    tmp_path: Path,
) -> None:
    """A duplicate case_id in the feature table cannot fan across folds."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = _config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "auto"}
    )
    feats_df = pd.DataFrame({"case_id": ["e1", "e1", "e2", "e3"], "pcr": [1, 1, 0, 1]})

    with pytest.raises((ValueError, pd.errors.MergeError)):
        _apply_provided_folds(feats_df, config, adapter)


def test_apply_group_keys_populates_patient_key(tmp_path: Path) -> None:
    """_apply_group_keys fills group_col from adapter.group_key (patient grouping)."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = ConfigNode._wrap(
        _deep_merge(
            DEFAULT_CONFIG,
            {
                "dataset": {"name": "uchicago", "cohort": None, "root": "/x"},
                "model_params": {"group_col": "patient_key"},
            },
        )
    )
    feats_df = pd.DataFrame({"case_id": ["e1", "e2", "e3"], "pcr": [1, 0, 1]})

    grouped = _apply_group_keys(feats_df, config, adapter)

    assert dict(zip(grouped["case_id"], grouped["patient_key"], strict=True)) == {
        "e1": "p1",
        "e2": "p2",
        "e3": "p3",
    }


def test_apply_group_keys_does_not_override_existing_column(tmp_path: Path) -> None:
    """An already-populated group column (e.g. clinical site) is left untouched."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = ConfigNode._wrap(
        _deep_merge(
            DEFAULT_CONFIG,
            {
                "dataset": {"name": "uchicago", "cohort": None, "root": "/x"},
                "model_params": {"group_col": "patient_key"},
            },
        )
    )
    feats_df = pd.DataFrame(
        {"case_id": ["e1", "e2"], "pcr": [1, 0], "patient_key": ["preset1", "preset2"]}
    )

    grouped = _apply_group_keys(feats_df, config, adapter)

    assert grouped["patient_key"].tolist() == ["preset1", "preset2"]


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


def test_group_col_excluded_from_model_features_when_adapter_configured() -> None:
    """The adapter grouping key (patient_key) must not leak into X as a feature.

    With a dataset configured and use_group_split on, group_col holds an identity
    (patient_key); prepare_evaluation_context must drop it from the model inputs
    even though use_group_split would otherwise keep the group column as a
    feature for legacy (non-adapter) runs.
    """
    feats_df = pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "pcr": [1, 0, 1, 0],
            "dataset": ["uchicago"] * 4,
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "patient_key": ["p1", "p1", "p2", "p2"],
        }
    )
    config = ConfigNode._wrap(
        _deep_merge(
            DEFAULT_CONFIG,
            {
                "dataset": {"name": "uchicago", "cohort": None, "root": "/x"},
                "model_params": {
                    "split_mode": "random",
                    "use_group_split": True,
                    "group_col": "patient_key",
                    "n_splits": 2,
                },
                "feature_toggles": {"use_clinical": False},
            },
        )
    )

    context = prepare_evaluation_context(feats_df, config)

    assert "patient_key" not in context["X"].columns
    assert set(context["X"].columns) == {"feature_a"}
