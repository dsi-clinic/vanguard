"""Tests for modeling/ablation.py's adapter wiring (Step 5).

Before Step 5, ``_prepare_full_dataset`` built an adapter but never applied
provided folds / patient grouping / ``report_by`` the way ``tabular/train.py``
does, so an ablation matrix run over a dataset with provided folds silently
ignored them. ``run_ablation_matrix`` now mirrors that wiring, with one
sequencing difference driven by what varies across the family/mode loop:

- Provided folds are attached **once per arm**, before the loop, because
  ``config.dataset.split_policy`` doesn't vary by family/mode -- attaching
  twice would hit ``_apply_provided_folds``' split_col-collision guard.
- Group keys are resolved **fresh per (arm, family, mode)**, because
  ``model_params.use_group_split`` does vary by split mode.

These tests pin both halves of that sequencing directly, without running a
full ablation matrix (real CV fitting needs more synthetic data than is
useful here; the per-piece helpers are already unit-tested in
tests/test_provided_folds.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cohorts import UChicagoDataset
from config import DEFAULT_CONFIG, ConfigNode, _deep_merge
from tabular.train import (
    _adapter_populates_group_col,
    _apply_group_keys,
    _apply_provided_folds,
)


def _write_manifest(tmp_path: Path) -> Path:
    """Write a tiny UChicago-shaped manifest CSV and return its root dir."""
    root = tmp_path / "uc"
    root.mkdir()
    (root / "dce2d_internal_ultrafast_manifest.csv").write_text(
        "exam_id,dataset,patient_key,fold,pcr,phase_files\n"
        'e1,simbiosys,p1,0,1.0,"[""/a/p0.nii.gz""]"\n'
        'e2,uch_nac,p2,1,0.0,"[""/b/p0.nii.gz""]"\n'
    )
    return root


def _uchicago_config() -> ConfigNode:
    return ConfigNode._wrap(
        _deep_merge(
            DEFAULT_CONFIG,
            {"dataset": {"name": "uchicago", "cohort": None, "root": "/x"}},
        )
    )


def test_applying_provided_folds_twice_raises_on_the_second_call(
    tmp_path: Path,
) -> None:
    """Regression guard for why ablation.py applies folds once per arm.

    If a caller instead called _apply_provided_folds once per (family, mode)
    cell on the same arm_df -- the naive per-cell mirror of the per-cell group
    logic -- the second call would find model_params.split_col already
    present and raise, rather than silently corrupting the fold assignment.
    """
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = _uchicago_config()
    feats_df = pd.DataFrame({"case_id": ["e1", "e2"], "pcr": [1, 0]})

    once = _apply_provided_folds(feats_df, config, adapter)
    assert "fold" in once.columns

    with pytest.raises(ValueError, match="collides"):
        _apply_provided_folds(once, config, adapter)


def test_group_keys_can_be_recomputed_per_mode_without_collision(
    tmp_path: Path,
) -> None:
    """Unlike folds, group-key population is safe to call fresh every mode.

    _apply_group_keys copies rather than mutates in place and is a no-op when
    the column is already present, so ablation.py calling it once per
    (arm, family, mode) -- because use_group_split varies by mode -- doesn't
    need the same once-per-arm guard folds do. Uses UChicagoDataset (manifest-
    driven group_key, no real clinical files needed) with split_policy
    overridden to "compute" -- the scenario _needs_group_keys actually gates
    on, regardless of which adapter is in play.
    """
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    compute_config = ConfigNode._wrap(
        _deep_merge(
            DEFAULT_CONFIG,
            {
                "dataset": {
                    "name": "uchicago",
                    "cohort": None,
                    "root": "/x",
                    "split_policy": "compute",
                },
                "model_params": {"group_col": "patient_key", "use_group_split": False},
            },
        )
    )
    arm_df = pd.DataFrame(
        {
            "case_id": ["e1", "e2"],
            "pcr": [1, 0],
            "patient_key": ["preset1", "preset2"],
        }
    )

    # Mode with use_group_split=False: patient_key is (hypothetically) already
    # a genuine feature here, not populated by the adapter, so it must not be
    # flagged for dropping.
    compute_config.model_params.use_group_split = False
    assert _adapter_populates_group_col(arm_df, compute_config, adapter) is False

    # Mode with use_group_split=True, but patient_key already present: still
    # not adapter-populated (mirrors _apply_group_keys' own no-op-when-present
    # rule), so a genuine feature is never mistaken for an identity key.
    compute_config.model_params.use_group_split = True
    assert _adapter_populates_group_col(arm_df, compute_config, adapter) is False
    unchanged = _apply_group_keys(arm_df, compute_config, adapter)
    assert unchanged["patient_key"].tolist() == ["preset1", "preset2"]

    # A column absent from the frame *is* the case this whole seam exists
    # for -- and it must be recomputable across repeated per-mode calls
    # without raising the way folds would on a second application.
    no_group_col_df = arm_df.drop(columns=["patient_key"])
    assert _adapter_populates_group_col(no_group_col_df, compute_config, adapter)
    first = _apply_group_keys(no_group_col_df, compute_config, adapter)
    second = _apply_group_keys(no_group_col_df, compute_config, adapter)
    assert (
        first["patient_key"].tolist()
        == second["patient_key"].tolist()
        == [
            "p1",
            "p2",
        ]
    )
