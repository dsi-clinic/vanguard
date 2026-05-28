"""Tests for analysis/feature_ranking.py (Issue #151).

These are deliberately small synthetic tests that lock in the public contract
of the ranking functions: tidy schema, deterministic ordering, robustness to
NaNs / constant columns, and consistency across methods on cases where the
ground-truth ranking is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.feature_ranking import (
    rank_features,
    rank_features_by_lr_coef,
    rank_features_by_univariate_auc,
    rank_features_by_xgb_gain,
    top_k_features,
)

N_SAMPLES = 200
N_GOOD_FEATURES = 3
N_NOISE_FEATURES = 5
RANDOM_SEED = 0
TOP_K_FOR_BASIC_TEST = 3


def _make_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    n_good: int = N_GOOD_FEATURES,
    n_noise: int = N_NOISE_FEATURES,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a synthetic dataset where the first ``n_good`` columns predict y.

    The good columns are constructed as ``label + small_noise`` (so they are
    almost perfectly informative) and the remaining columns are independent
    Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)

    good = np.stack(
        [y + 0.05 * rng.standard_normal(n_samples) for _ in range(n_good)],
        axis=1,
    )
    noise = rng.standard_normal((n_samples, n_noise))

    columns = [f"good_{i}" for i in range(n_good)] + [
        f"noise_{i}" for i in range(n_noise)
    ]
    X = pd.DataFrame(np.concatenate([good, noise], axis=1), columns=columns)
    return X, y


def _good_feature_names(n_good: int = N_GOOD_FEATURES) -> set[str]:
    return {f"good_{i}" for i in range(n_good)}


@pytest.mark.parametrize(
    "ranker",
    [rank_features_by_univariate_auc, rank_features_by_xgb_gain],
)
def test_ranker_schema_and_completeness(ranker) -> None:  # noqa: ANN001
    """The output is a tidy DataFrame with one row per feature."""
    X, y = _make_synthetic_dataset()
    result = ranker(X, y)

    assert list(result.columns) == ["feature", "score", "rank", "method"]
    assert len(result) == X.shape[1]
    assert set(result["feature"]) == set(X.columns)
    assert set(result["rank"]) == set(range(1, X.shape[1] + 1))
    assert (result["method"] == ranker.__name__.removeprefix("rank_features_by_")).all()


def test_univariate_auc_picks_good_features() -> None:
    """All good features should rank above all noise features."""
    X, y = _make_synthetic_dataset()
    result = rank_features_by_univariate_auc(X, y)
    top = result.sort_values("rank").head(N_GOOD_FEATURES)
    assert set(top["feature"]) == _good_feature_names()


def test_xgb_gain_picks_good_features() -> None:
    """XGBoost should also rank the good features at the top."""
    X, y = _make_synthetic_dataset()
    result = rank_features_by_xgb_gain(X, y, n_estimators=100, max_depth=3)
    top = result.sort_values("rank").head(N_GOOD_FEATURES)
    assert set(top["feature"]) == _good_feature_names()


def test_lr_abs_coef_picks_good_features() -> None:
    """L2 logistic regression should also rank the good features at the top."""
    X, y = _make_synthetic_dataset()
    result = rank_features_by_lr_coef(X, y)
    top = result.sort_values("rank").head(N_GOOD_FEATURES)
    assert set(top["feature"]) == _good_feature_names()


def test_ranker_handles_constant_column() -> None:
    """A constant column should sink to the bottom, not raise."""
    X, y = _make_synthetic_dataset()
    X = X.copy()
    X["constant"] = 1.0
    result = rank_features_by_univariate_auc(X, y)
    constant_row = result.set_index("feature").loc["constant"]
    assert constant_row["score"] == -np.inf
    assert int(constant_row["rank"]) == X.shape[1]


def test_ranker_handles_all_nan_column() -> None:
    """A column that is all NaN should sink to the bottom."""
    X, y = _make_synthetic_dataset()
    X = X.copy()
    X["all_nan"] = np.nan
    result = rank_features_by_univariate_auc(X, y)
    nan_row = result.set_index("feature").loc["all_nan"]
    assert nan_row["score"] == -np.inf
    assert int(nan_row["rank"]) == X.shape[1]


def test_ranker_handles_partial_nan_column() -> None:
    """A column with NaNs but informative finite values should still rank well."""
    X, y = _make_synthetic_dataset()
    X = X.copy()
    perturbed = X["good_0"].copy()
    perturbed.iloc[:50] = np.nan
    X["good_0"] = perturbed

    result = rank_features_by_univariate_auc(X, y)
    assert (
        int(result.loc[result["feature"] == "good_0", "rank"].iloc[0])
        <= N_GOOD_FEATURES
    )


def test_ranker_validates_y_shape() -> None:
    """Mismatched y length should raise a ValueError."""
    X, y = _make_synthetic_dataset()
    with pytest.raises(ValueError):
        rank_features_by_univariate_auc(X, y[:10])


def test_ranker_validates_y_class_count() -> None:
    """Single-class y should raise."""
    X, _ = _make_synthetic_dataset()
    y_const = np.zeros(X.shape[0], dtype=int)
    with pytest.raises(ValueError):
        rank_features_by_univariate_auc(X, y_const)


def test_dispatcher_method_routing() -> None:
    """``rank_features`` should dispatch to the same result as the named function."""
    X, y = _make_synthetic_dataset()
    direct = rank_features_by_univariate_auc(X, y)
    via_dispatch = rank_features(X, y, method="univariate_auc")
    pd.testing.assert_frame_equal(direct, via_dispatch)


def test_dispatcher_rejects_unknown_method() -> None:
    """Unknown ranking method names should raise ValueError."""
    X, y = _make_synthetic_dataset()
    with pytest.raises(ValueError):
        rank_features(X, y, method="not_a_method")  # type: ignore[arg-type]


def test_top_k_features_basic() -> None:
    """top_k_features returns names in rank order, length min(k, n)."""
    X, y = _make_synthetic_dataset()
    result = rank_features_by_univariate_auc(X, y)
    top_3 = top_k_features(result, k=TOP_K_FOR_BASIC_TEST)
    assert top_3 == sorted(
        top_3,
        key=lambda name: int(result.loc[result["feature"] == name, "rank"].iloc[0]),
    )
    assert len(top_3) == TOP_K_FOR_BASIC_TEST
    assert set(top_3) == _good_feature_names()


def test_top_k_features_k_zero() -> None:
    """``top_k_features`` with ``k=0`` returns an empty list."""
    X, y = _make_synthetic_dataset()
    result = rank_features_by_univariate_auc(X, y)
    assert top_k_features(result, k=0) == []
