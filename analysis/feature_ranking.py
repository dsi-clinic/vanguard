"""Per-feature ranking utilities for top-K feature evaluation (Issue #151).

Each public ranking function takes a numeric feature matrix ``X`` plus a binary
target ``y`` and returns a tidy DataFrame with one row per feature:

    case_id,  feature,  score,  rank,  method

Higher ``score`` is always interpreted as "more predictive"; ``rank`` is 1 for
the most predictive feature. The ``method`` column is identical across rows
within a single ranking call and identifies which ranker produced the table —
this makes it cheap to ``pd.concat`` rankings from different methods and group
on ``method`` downstream.

Design notes:

- All rankers ignore non-finite values (NaN / Inf) on a per-feature basis when
  computing the score. Features that have fewer than two finite samples or
  fewer than two unique values are demoted to the bottom of the ranking with
  ``score = -np.inf`` rather than dropped, so the caller still gets a complete
  per-feature record.
- The rankers are intentionally pure: they do not do their own NaN imputation
  or scaling and do not fit a model that needs an outer ``ColumnTransformer``.
  This keeps the per-fold orchestration code (which lives elsewhere) free to
  decide on the preprocessing contract.
- The tree-based ranker (``rank_features_by_xgb_gain``) and the linear ranker
  (``rank_features_by_lr_coef``) both rely on the caller to pass in a numeric
  ``X``; categorical columns must be one-hot encoded upstream.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

MIN_BINARY_CLASSES = 2
MIN_FEATURE_SAMPLES = 2
MIN_FEATURE_UNIQUE = 2

RankingMethod = Literal["univariate_auc", "xgb_gain", "lr_abs_coef"]


def _validate_inputs(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Coerce inputs to numpy and align feature names with column count."""
    if isinstance(X, pd.DataFrame):
        inferred_names = [str(c) for c in X.columns]
        X_arr = X.to_numpy(dtype=float, copy=False)
    else:
        X_arr = np.asarray(X, dtype=float)
        inferred_names = None
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n_samples, n_features = X_arr.shape
    if feature_names is None:
        feature_names = inferred_names or [f"f{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length {len(feature_names)} does not match "
            f"X column count {n_features}."
        )

    y_arr = np.asarray(y).astype(int, copy=False)
    if y_arr.ndim != 1 or y_arr.shape[0] != n_samples:
        raise ValueError(
            f"y must be a 1D array of length {n_samples}; got shape {y_arr.shape}."
        )
    if np.unique(y_arr).size < MIN_BINARY_CLASSES:
        raise ValueError("y must contain at least two distinct class labels.")

    return X_arr, y_arr, list(feature_names)


def _ranked_frame(
    feature_names: list[str],
    scores: np.ndarray,
    method: RankingMethod,
) -> pd.DataFrame:
    """Build the tidy ranking frame, breaking score ties on feature index."""
    if len(feature_names) != scores.shape[0]:
        raise ValueError("feature_names and scores must have the same length.")

    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)

    return (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "score": scores.astype(float, copy=False),
                "rank": ranks.astype(int, copy=False),
                "method": method,
            }
        )
        .sort_values("rank", kind="stable")
        .reset_index(drop=True)
    )


def rank_features_by_univariate_auc(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Rank features by their per-feature absolute AUC against ``y``.

    Score is ``max(auc, 1 - auc)`` so that a feature that is monotonically
    *anti-correlated* with the label is treated as equally informative to one
    that is monotonically *positively* correlated.
    """
    X_arr, y_arr, names = _validate_inputs(X, y, feature_names)
    n_features = X_arr.shape[1]
    scores = np.full(n_features, -np.inf, dtype=float)

    for j in range(n_features):
        valid = np.isfinite(X_arr[:, j])
        if int(valid.sum()) < MIN_FEATURE_SAMPLES:
            continue
        xv = X_arr[valid, j]
        yv = y_arr[valid]
        if (
            np.unique(xv).size < MIN_FEATURE_UNIQUE
            or np.unique(yv).size < MIN_BINARY_CLASSES
        ):
            continue
        try:
            auc = float(roc_auc_score(yv, xv))
        except ValueError:
            continue
        scores[j] = max(auc, 1.0 - auc)

    return _ranked_frame(names, scores, method="univariate_auc")


def rank_features_by_xgb_gain(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_names: list[str] | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 0,
) -> pd.DataFrame:
    """Rank features by mean XGBoost gain-importance after a quick fit.

    The classifier is fit with ``tree_method='hist'`` and conservative defaults
    so the call is cheap inside a per-fold ranking loop. Missing values are
    handled natively by XGBoost; no imputation is applied.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - defensive import guard
        raise ImportError(
            "rank_features_by_xgb_gain requires xgboost to be installed."
        ) from exc

    X_arr, y_arr, names = _validate_inputs(X, y, feature_names)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        tree_method="hist",
        importance_type="gain",
        n_jobs=1,
        eval_metric="logloss",
    )
    clf.fit(X_arr, y_arr)

    importances = np.asarray(clf.feature_importances_, dtype=float)
    if importances.shape[0] != len(names):
        raise RuntimeError(
            f"XGBoost reported {importances.shape[0]} importances for "
            f"{len(names)} features."
        )

    return _ranked_frame(names, importances, method="xgb_gain")


def rank_features_by_lr_coef(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_names: list[str] | None = None,
    *,
    penalty: str = "l2",
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Rank features by ``|coef|`` of an L2-regularized logistic regression.

    Features are standardized before fitting so coefficients are comparable
    across features. Non-finite cells in ``X`` are replaced with the column
    mean of finite values (or 0 if a column has no finite values) before
    fitting; this is only the ranking step's internal imputation and does not
    affect downstream model code.
    """
    X_arr, y_arr, names = _validate_inputs(X, y, feature_names)

    col_means = np.nanmean(np.where(np.isfinite(X_arr), X_arr, np.nan), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    mask = ~np.isfinite(X_arr)
    if mask.any():
        idx_rows, idx_cols = np.where(mask)
        X_arr = X_arr.copy()
        X_arr[idx_rows, idx_cols] = col_means[idx_cols]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_arr)

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(X_scaled, y_arr)

    coefs = np.asarray(clf.coef_, dtype=float).ravel()
    if coefs.shape[0] != len(names):
        raise RuntimeError(
            f"LogisticRegression returned {coefs.shape[0]} coefficients for "
            f"{len(names)} features."
        )
    return _ranked_frame(names, np.abs(coefs), method="lr_abs_coef")


def rank_features(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_names: list[str] | None = None,
    *,
    method: RankingMethod,
    **kwargs: object,
) -> pd.DataFrame:
    """Dispatch to a specific ranker by ``method`` name.

    Convenience entry point that lets callers iterate over methods without
    branching on the method string themselves.
    """
    if method == "univariate_auc":
        return rank_features_by_univariate_auc(X, y, feature_names)
    if method == "xgb_gain":
        return rank_features_by_xgb_gain(X, y, feature_names, **kwargs)  # type: ignore[arg-type]
    if method == "lr_abs_coef":
        return rank_features_by_lr_coef(X, y, feature_names, **kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"Unknown ranking method: {method!r}. "
        "Valid options: 'univariate_auc', 'xgb_gain', 'lr_abs_coef'."
    )


def top_k_features(
    rank_df: pd.DataFrame,
    k: int,
) -> list[str]:
    """Return the names of the top-``k`` features in rank order."""
    if "rank" not in rank_df.columns or "feature" not in rank_df.columns:
        raise ValueError(
            "rank_df must have 'rank' and 'feature' columns; got "
            f"{list(rank_df.columns)}."
        )
    if k <= 0:
        return []
    return (
        rank_df.sort_values("rank", kind="stable")
        .head(int(k))["feature"]
        .astype(str)
        .tolist()
    )
