"""Regression tests for per-fold graph-feature preprocessing (CV-leakage fix).

The clinical imputer/one-hot encoder and the morphometry mean-imputer must be
fit on each cross-validation fold's *training* cases only -- never once over
the whole cohort at dataset-build time. Otherwise validation cases contribute
to the imputer means/modes and to the one-hot category vocabulary, so the
training representation carries information about the validation distribution
(optimistic CV, inconsistent with deployment on a genuinely unseen cohort).

These exercise ``gnn.train._fit_transform_graph_features`` -- the function that
does the per-fold fit -- on a tiny hand-built raw-input frame whose expected
imputed/encoded values differ depending on whether the fit saw the validation
cases, so a regression to whole-cohort fitting fails loudly here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from gnn.train import _fit_transform_graph_features  # noqa: E402

# Raw (un-imputed, un-encoded) inputs. Column order is the fixed source order
# the sidecar uses: clinical (age, tumor_subtype), then morphometry.
# Validation case V1 carries extreme values that would swing every cohort-wide
# statistic; a leakage-free per-fold fit must ignore them.
_INPUTS = pd.DataFrame(
    {
        "age": [40.0, np.nan, 1000.0],
        "tumor_subtype": ["luminal", "luminal", "her2"],
        "morph_seg_length_mean": [10.0, np.nan, 1000.0],
    },
    index=pd.Index(["T1", "T2", "V1"], name="case_id"),
)
_GROUPS = (["age", "tumor_subtype"], [], ["morph_seg_length_mean"])
# Column layout of the transformed vector: age, then the single train-fit
# one-hot column (tumor_subtype=luminal), then morphometry.
_AGE_COL, _LUMINAL_COL, _MORPH_COL = 0, 1, 2

# T1's age; the only non-missing age in the training split, so it is also the
# training-split mean used to impute T2. The whole-cohort mean would be
# (40 + 1000) / 2 = 520 -- what a leaky fit would wrongly fill.
_TRAIN_AGE_MEAN = 40.0
# Likewise the training-split morphometry mean (only T1 is non-missing).
_TRAIN_MORPH_MEAN = 10.0


def _fit() -> dict[str, np.ndarray]:
    return _fit_transform_graph_features(
        _INPUTS, train_case_ids=["T1", "T2"], val_case_ids=["V1"], groups=_GROUPS
    )


def test_numeric_imputer_uses_training_mean_only() -> None:
    """A missing training value is filled with the train mean, not the cohort mean."""
    feats = _fit()
    assert feats["T2"][_AGE_COL] == pytest.approx(_TRAIN_AGE_MEAN)


def test_morphometry_imputer_uses_training_mean_only() -> None:
    """The morphometry mean-imputer never sees the validation case's value."""
    feats = _fit()
    assert feats["T2"][_MORPH_COL] == pytest.approx(_TRAIN_MORPH_MEAN)


def test_category_unseen_in_training_is_encoded_as_unknown() -> None:
    """A category present only in validation gets an all-zero (unknown) encoding."""
    feats = _fit()
    # One-hot vocabulary is fit on the train split (only "luminal"), so V1's
    # "her2" is unknown -> the luminal indicator is 0 and no new column appears.
    assert feats["T1"][_LUMINAL_COL] == pytest.approx(1.0)
    assert feats["V1"][_LUMINAL_COL] == pytest.approx(0.0)
    widths = {name: vec.shape[0] for name, vec in feats.items()}
    assert set(widths.values()) == {len(_GROUPS[0]) + 1}  # age + 1 one-hot + morph == 3


def test_validation_own_present_values_pass_through() -> None:
    """Non-missing validation values are transformed, not overwritten by imputation."""
    feats = _fit()
    assert feats["V1"][_AGE_COL] == pytest.approx(1000.0)
    assert feats["V1"][_MORPH_COL] == pytest.approx(1000.0)
