#!/usr/bin/env python3
"""Train a classifier on already-extracted radiomics features.

Stage 2: Train a classifier on already-extracted radiomics features.

Uses the centralized evaluation framework (evaluation/) for metric
computation, per-subtype validation-summary reporting, and standard output
artefacts (metrics.json, predictions.csv, plots/roc_curve.png).
Radiomics-specific outputs (model.pkl, pr_curve.png, calibration_curve.png,
predictions_cv_train.csv) and extra metric fields (auc_train, auc_train_cv,
threshold, sensitivity/specificity, confusion matrix, etc.) are written
alongside the framework outputs so that run_experiment.py and the ablation
runner remain fully backward-compatible.

Inputs
------
- --train-features : CSV from radiomics_extract.py (rows = patients, cols = features)
- --test-features  : CSV from radiomics_extract.py
- --labels         : CSV with at least columns: patient_id,pcr[,subtype]
- --output         : output directory to write metrics, plots, and model

What this script does
---------------------
1) Load train/test feature CSVs and align with labels.
2) (Optional) Append a numeric subtype code as an extra feature.
3) Sanitize to numeric-only; drop all-NaN and zero-variance columns.
4) Build a sklearn Pipeline that includes — in order — median imputation,
   (optional) CorrelationPruner, (optional) SelectKBest, (optional) scaling,
   and the classifier.  Feature selection lives inside the pipeline so that it
   is re-fitted on only the training rows of every k-fold split, preventing
   label leakage into cross-validated AUC estimates.
5) Train the chosen classifier (logistic, RF, or XGBoost), optionally via GridSearchCV.
7) Run the evaluation framework:
   - Build a predictions DataFrame (with subtype column for automatic
     per-subtype AUC via evaluation.metrics.compute_metrics_by_group).
   - Call evaluator.save_results() to write:
       <output>/metrics.json          (framework structure + radiomics extras)
       <output>/predictions.csv       (patient_id, y_true, y_pred, y_prob[, subtype])
       <output>/plots/roc_curve.png   (seaborn ROC curve)
8) Augment metrics.json with all radiomics-specific fields so that
   run_experiment.py can still read flat keys (auc_test, auc_train,
   auc_train_cv, n_features_used) from the top level of the file.
9) Save additional radiomics outputs: pr_curve.png, calibration_curve.png, model.pkl.
   If --cv-folds > 1, also save predictions_cv_train.csv.

Example Usage
---------------------
python radiomics_train.py \
  --train-features experiments/extract_peri5_multiphase/features_train_final.csv \
  --test-features  experiments/extract_peri5_multiphase/features_test_final.csv \
  --labels         labels.csv \
  --output         outputs/elasticnet_corr0.9_k50_cv5 \
  --classifier     logistic \
  --logreg-penalty elasticnet \
  --logreg-l1-ratio 0.5 \
  --corr-threshold 0.9 \
  --k-best         50 \
  --grid-search \
  --cv-folds       5 \
  --include-subtype
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Centralized evaluation framework (evaluation/)
# ---------------------------------------------------------------------------
# Add the project root to sys.path so that
# `from evaluation import ...` resolves regardless of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation import Evaluator, FoldResults, TrainTestResults  # noqa: E402

MIN_CLASS_COUNT = 2
CONF_MATRIX_SIZE = 4
AUC_BASELINE = 0.5
COMBAT_EPS = 1e-8


# ---------------------------
# I/O helpers
# ---------------------------
def load_features(path: str) -> pd.DataFrame:
    """Load feature CSV with patient IDs in the index."""
    data = pd.read_csv(path, index_col=0)
    data.index = data.index.map(str)
    dup = data.index[data.index.duplicated()].unique()
    if len(dup) > 0:
        preview = ", ".join(map(str, dup[:5]))
        msg = (
            f"Feature table has duplicate patient IDs (n={len(dup)}): "
            f"{preview}{' ...' if len(dup) > 5 else ''}"
        )
        raise ValueError(msg)
    return data


def load_labels(labels_csv: str) -> pd.DataFrame:
    """Load labels CSV and ensure it contains a 'pcr' column."""
    lab = pd.read_csv(labels_csv).copy()
    if "patient_id" not in lab.columns:
        msg = "labels.csv must contain column 'patient_id'"
        raise ValueError(msg)
    dup = lab["patient_id"][lab["patient_id"].duplicated()].astype(str).unique()
    if len(dup) > 0:
        preview = ", ".join(dup[:5])
        msg = (
            f"labels.csv has duplicate patient_id values (n={len(dup)}): "
            f"{preview}{' ...' if len(dup) > 5 else ''}"
        )
        raise ValueError(msg)
    lab["patient_id"] = lab["patient_id"].astype(str)
    lab = lab.set_index("patient_id")
    if "pcr" not in lab.columns:
        msg = "labels.csv must contain column 'pcr'"
        raise ValueError(msg)
    return lab


# ---------------------------
# Sanitization & selection
# ---------------------------
def sanitize_numeric(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Build a train-time numeric feature matrix (train-only schema selection).

    All columns are coerced to numeric (non-numeric values -> NaN), then
    all-NaN and zero-variance columns are dropped. The resulting column set is
    the reference schema that should be applied to test data.
    """
    raw = df.shape
    coerced = df.apply(pd.to_numeric, errors="coerce")
    num = coerced.copy()
    # drop all-NaN (train-only)
    all_nan = num.columns[num.isna().all()].tolist()
    num = num.drop(columns=all_nan, errors="ignore")
    # drop zero-var (train-only)
    nunique = num.nunique(dropna=True)
    zero_var = nunique[nunique <= 1].index.tolist()
    num = num.drop(columns=zero_var, errors="ignore")
    print(
        f"[DEBUG] {tag}: raw={raw} -> numeric={num.shape} "
        f"(all-NaN={len(all_nan)}, zero-var={len(zero_var)})",
    )
    return num


def align_numeric_to_reference(
    df: pd.DataFrame,
    reference_columns: list[str],
    tag: str,
) -> pd.DataFrame:
    """Coerce to numeric and align to a train-derived column schema.

    Unlike ``sanitize_numeric``, this function never drops columns based on
    test-set variance. Any missing reference columns are added as NaN so they
    can be imputed by the training-set pipeline.
    """
    raw = df.shape
    coerced = df.apply(pd.to_numeric, errors="coerce")
    ref = list(reference_columns)
    extra_cols = [c for c in coerced.columns if c not in ref]
    missing_cols = [c for c in ref if c not in coerced.columns]
    aligned = coerced.reindex(columns=ref, fill_value=np.nan)
    all_nan_after_align = int(aligned.isna().all(axis=0).sum())
    print(
        f"[DEBUG] {tag}: raw={raw} -> aligned={aligned.shape} "
        f"(missing={len(missing_cols)}, extra_ignored={len(extra_cols)}, "
        f"all-NaN-after-align={all_nan_after_align})",
    )
    return aligned


def append_categorical_feature(
    Xtr_raw: pd.DataFrame,
    Xte_raw: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    column: str,
    prefix: str,
    encoding: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Append one categorical label column as model features.

    Categories are learned from the *training* split only and applied to test.
    This avoids introducing extra columns from test-only categories.
    """
    tr_series = labels.loc[Xtr_raw.index, column]
    te_series = labels.loc[Xte_raw.index, column]
    cats = sorted(pd.Series(tr_series.dropna().unique()).tolist())

    if encoding == "ordinal":
        code_map = {cat: i for i, cat in enumerate(cats)}
        col = f"{prefix}_code"
        Xtr_new = Xtr_raw.assign(**{col: tr_series.map(code_map)})
        Xte_new = Xte_raw.assign(**{col: te_series.map(code_map)})
        return Xtr_new, Xte_new, [col]

    # default: one-hot (including an explicit NaN bucket)
    cat_dtype = pd.CategoricalDtype(categories=cats)
    tr_cat = tr_series.astype(cat_dtype)
    te_cat = te_series.astype(cat_dtype)
    tr_dummies = pd.get_dummies(tr_cat, prefix=prefix, dummy_na=True).astype(float)
    te_dummies = pd.get_dummies(te_cat, prefix=prefix, dummy_na=True).astype(float)
    te_dummies = te_dummies.reindex(columns=tr_dummies.columns, fill_value=0.0)

    Xtr_new = Xtr_raw.join(tr_dummies)
    Xte_new = Xte_raw.join(te_dummies)
    added_cols = tr_dummies.columns.tolist()
    if Xtr_new[added_cols].isna().all(axis=0).any():
        msg = (
            "one-hot categorical join produced all-NaN columns on train; "
            "check patient_id index alignment"
        )
        raise RuntimeError(msg)
    return Xtr_new, Xte_new, tr_dummies.columns.tolist()


def _normalize_covariate_list(values: list[str]) -> list[str]:
    """Return a de-duplicated list of non-empty covariate names."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        parts = [p.strip() for p in str(raw).split(",")]
        for p in parts:
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def _aprior(delta: np.ndarray) -> float:
    """Method-of-moments prior for inverse-gamma shape."""
    m = float(np.mean(delta))
    s2 = float(np.var(delta, ddof=1)) if len(delta) > 1 else 0.0
    if s2 <= COMBAT_EPS:
        return 100.0
    return (2.0 * s2 + m * m) / s2


def _bprior(delta: np.ndarray) -> float:
    """Method-of-moments prior for inverse-gamma scale."""
    m = float(np.mean(delta))
    s2 = float(np.var(delta, ddof=1)) if len(delta) > 1 else 0.0
    if s2 <= COMBAT_EPS:
        return m * 99.0
    return (m * s2 + m * m * m) / s2


def _build_covariate_matrix(
    labels_slice: pd.DataFrame,
    covariate_cols: list[str],
    *,
    fitted_categories: dict[str, list[str]] | None = None,
    fitted_medians: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, list[str]], dict[str, float]]:
    """Build a numeric covariate design matrix from labels."""
    if not covariate_cols:
        return np.zeros((len(labels_slice), 0), dtype=float), {}, {}

    blocks: list[np.ndarray] = []
    categories_out: dict[str, list[str]] = {}
    medians_out: dict[str, float] = {}

    for col in covariate_cols:
        if col not in labels_slice.columns:
            msg = f"Harmonization covariate '{col}' not found in labels."
            raise ValueError(msg)

        s = labels_slice[col]
        if pd.api.types.is_numeric_dtype(s):
            if fitted_medians is not None and col in fitted_medians:
                med = float(fitted_medians[col])
            else:
                med = float(pd.to_numeric(s, errors="coerce").median())
                if np.isnan(med):
                    med = 0.0
            medians_out[col] = med
            arr = pd.to_numeric(s, errors="coerce").fillna(med).to_numpy(dtype=float)
            blocks.append(arr.reshape(-1, 1))
        else:
            clean = s.fillna("__NA__").astype(str)
            if fitted_categories is not None and col in fitted_categories:
                cats = fitted_categories[col]
            else:
                cats = sorted(pd.Series(clean).dropna().unique().tolist())
                if "__NA__" not in cats:
                    cats.append("__NA__")
            categories_out[col] = cats

            cat_to_idx = {v: i for i, v in enumerate(cats)}
            mat = np.zeros((len(clean), len(cats)), dtype=float)
            for i, val in enumerate(clean.to_numpy()):
                idx = cat_to_idx.get(val, cat_to_idx.get("__NA__", 0))
                mat[i, idx] = 1.0
            blocks.append(mat)

    design = (
        np.concatenate(blocks, axis=1)
        if blocks
        else np.zeros((len(labels_slice), 0), dtype=float)
    )
    return design, categories_out, medians_out


class FeatureHarmonizer:
    """Fold-safe feature harmonization fitted on training rows only."""

    def __init__(
        self,
        *,
        mode: str,
        batch_col: str,
        covariate_cols: list[str],
    ) -> None:
        self.mode = mode
        self.batch_col = batch_col
        self.covariate_cols = covariate_cols

    def _fit_zscore(
        self,
        X_fit: pd.DataFrame,
        batch_fit: pd.Series,
    ) -> None:
        self.global_mean_ = X_fit.mean(axis=0).to_numpy(dtype=float)
        self.global_std_ = (
            X_fit.std(axis=0, ddof=1).replace(0, 1).fillna(1).to_numpy(dtype=float)
        )
        self.batch_stats_: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for batch in sorted(batch_fit.unique()):
            idx = batch_fit == batch
            xb = X_fit.loc[idx]
            mean = xb.mean(axis=0).to_numpy(dtype=float)
            std = xb.std(axis=0, ddof=1).replace(0, 1).fillna(1).to_numpy(dtype=float)
            self.batch_stats_[str(batch)] = (mean, std)

    def _transform_zscore(
        self,
        X_apply: pd.DataFrame,
        batch_apply: pd.Series,
    ) -> tuple[pd.DataFrame, int]:
        arr = X_apply.to_numpy(dtype=float).copy()
        unknown_batches = 0
        for batch in pd.Series(batch_apply.astype(str)).unique():
            idx = np.where(batch_apply.astype(str).to_numpy() == str(batch))[0]
            if str(batch) not in self.batch_stats_:
                unknown_batches += len(idx)
                continue
            b_mean, b_std = self.batch_stats_[str(batch)]
            arr[idx, :] = (
                (arr[idx, :] - b_mean) / b_std
            ) * self.global_std_ + self.global_mean_
        return pd.DataFrame(
            arr, index=X_apply.index, columns=X_apply.columns
        ), unknown_batches

    def _fit_combat(
        self,
        X_fit: pd.DataFrame,
        labels_fit: pd.DataFrame,
    ) -> None:
        dat = X_fit.to_numpy(dtype=float).T  # p x n
        p, n = dat.shape

        batch_fit = labels_fit[self.batch_col].astype(str)
        batch_levels = sorted(batch_fit.unique().tolist())
        self.batch_levels_ = batch_levels
        self.batch_to_idx_ = {b: i for i, b in enumerate(batch_levels)}

        batch_design = np.zeros((n, len(batch_levels)), dtype=float)
        batch_codes = batch_fit.to_numpy()
        for i, b in enumerate(batch_codes):
            batch_design[i, self.batch_to_idx_[str(b)]] = 1.0

        covar_design, cats, meds = _build_covariate_matrix(
            labels_fit,
            self.covariate_cols,
        )
        self.covar_categories_ = cats
        self.covar_medians_ = meds

        design = np.concatenate([batch_design, covar_design], axis=1)
        b_hat = np.linalg.pinv(design.T @ design) @ design.T @ dat.T  # k x p

        n_per_batch = batch_design.sum(axis=0)
        grand_mean = (n_per_batch / float(n)) @ b_hat[: len(batch_levels), :]  # (p,)
        var_pooled = np.sum((dat.T - (design @ b_hat)) ** 2, axis=0) / float(n)
        var_pooled = np.where(var_pooled <= COMBAT_EPS, 1.0, var_pooled)

        if covar_design.shape[1] > 0:
            stand_mean = (
                grand_mean[:, None] + (covar_design @ b_hat[len(batch_levels) :, :]).T
            )
        else:
            stand_mean = np.repeat(grand_mean[:, None], n, axis=1)

        s_data = (dat - stand_mean) / np.sqrt(var_pooled)[:, None]

        gamma_hat = np.zeros((len(batch_levels), p), dtype=float)
        delta_hat = np.zeros((len(batch_levels), p), dtype=float)
        batch_indices: dict[str, np.ndarray] = {}
        for bi, batch in enumerate(batch_levels):
            idx = np.where(batch_codes == batch)[0]
            batch_indices[batch] = idx
            sb = s_data[:, idx]
            gamma_hat[bi, :] = np.mean(sb, axis=1)
            delta_hat[bi, :] = np.var(sb, axis=1, ddof=1)
            delta_hat[bi, :] = np.where(
                delta_hat[bi, :] <= COMBAT_EPS, 1.0, delta_hat[bi, :]
            )

        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        if self.mode == "combat_nonparam":
            # Practical non-param approximation: no EB shrinkage.
            gamma_star = gamma_hat.copy()
            delta_star = delta_hat.copy()
        else:
            for bi, batch in enumerate(batch_levels):
                idx = batch_indices[batch]
                n_i = len(idx)
                g_hat = gamma_hat[bi, :]
                d_hat = delta_hat[bi, :]

                g_bar = float(np.mean(g_hat))
                t2 = float(np.var(g_hat, ddof=1)) if len(g_hat) > 1 else 0.0
                if t2 <= COMBAT_EPS:
                    t2 = 1.0
                a_prior = _aprior(d_hat)
                b_prior = _bprior(d_hat)

                g_old = g_hat.copy()
                d_old = d_hat.copy()
                sb = s_data[:, idx]
                for _ in range(200):
                    g_new = (t2 * n_i * g_hat + d_old * g_bar) / (t2 * n_i + d_old)
                    sum2 = np.sum((sb - g_new[:, None]) ** 2, axis=1)
                    d_new = (0.5 * sum2 + b_prior) / (n_i / 2.0 + a_prior - 1.0)
                    d_new = np.where(d_new <= COMBAT_EPS, 1.0, d_new)
                    if (
                        np.max(np.abs(g_new - g_old)) < 1e-6
                        and np.max(np.abs(d_new - d_old)) < 1e-6
                    ):
                        g_old, d_old = g_new, d_new
                        break
                    g_old, d_old = g_new, d_new
                gamma_star[bi, :] = g_old
                delta_star[bi, :] = d_old

        self.var_pooled_ = var_pooled
        self.grand_mean_ = grand_mean
        self.b_hat_nonbatch_ = (
            b_hat[len(batch_levels) :, :]
            if covar_design.shape[1] > 0
            else np.zeros((0, p))
        )
        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star
        self.n_features_ = p

    def _transform_combat(
        self,
        X_apply: pd.DataFrame,
        labels_apply: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int]:
        dat = X_apply.to_numpy(dtype=float).T  # p x n
        n = dat.shape[1]
        if dat.shape[0] != self.n_features_:
            msg = "ComBat transform received mismatched feature dimension."
            raise ValueError(msg)

        covar_design, _, _ = _build_covariate_matrix(
            labels_apply,
            self.covariate_cols,
            fitted_categories=getattr(self, "covar_categories_", {}),
            fitted_medians=getattr(self, "covar_medians_", {}),
        )

        if (
            covar_design.shape[1] > 0
            and self.b_hat_nonbatch_.shape[0] == covar_design.shape[1]
        ):
            stand_mean = (
                self.grand_mean_[:, None] + (covar_design @ self.b_hat_nonbatch_).T
            )
        else:
            stand_mean = np.repeat(self.grand_mean_[:, None], n, axis=1)

        s_data = (dat - stand_mean) / np.sqrt(self.var_pooled_)[:, None]

        batch_apply = labels_apply[self.batch_col].astype(str).to_numpy()
        unknown_batches = 0
        adjusted = s_data.copy()
        for b in np.unique(batch_apply):
            idx = np.where(batch_apply == b)[0]
            if b not in self.batch_to_idx_:
                unknown_batches += len(idx)
                continue
            bi = self.batch_to_idx_[b]
            adjusted[:, idx] = (
                adjusted[:, idx] - self.gamma_star_[bi, :][:, None]
            ) / np.sqrt(self.delta_star_[bi, :])[:, None]

        out = adjusted * np.sqrt(self.var_pooled_)[:, None] + stand_mean
        return pd.DataFrame(
            out.T, index=X_apply.index, columns=X_apply.columns
        ), unknown_batches

    def fit(
        self,
        X_fit: pd.DataFrame,
        labels_fit: pd.DataFrame,
    ) -> FeatureHarmonizer:
        """Fit harmonization parameters from training features and labels."""
        if self.mode == "none":
            return self

        if self.batch_col not in labels_fit.columns:
            msg = f"Harmonization batch column '{self.batch_col}' not found in labels."
            raise ValueError(msg)

        batch_fit = labels_fit[self.batch_col].astype(str)
        if batch_fit.nunique() < 2:
            warnings.warn(
                f"Harmonization mode '{self.mode}' requested"
                " with <2 batches in fit data. "
                "No harmonization will be applied.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.mode = "none"
            return self

        if self.mode == "zscore_site":
            self._fit_zscore(X_fit, batch_fit)
            return self

        self._fit_combat(X_fit, labels_fit)
        return self

    def transform(
        self,
        X_apply: pd.DataFrame,
        labels_apply: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int]:
        """Apply fitted harmonization to features.

        Returns transformed data and unknown-batch count.
        """
        if self.mode == "none":
            return X_apply.copy(), 0

        if self.mode == "zscore_site":
            return self._transform_zscore(
                X_apply, labels_apply[self.batch_col].astype(str)
            )

        return self._transform_combat(X_apply, labels_apply)


def fit_apply_harmonization(
    *,
    X_fit_raw: pd.DataFrame,
    X_apply_raw: pd.DataFrame,
    labels: pd.DataFrame,
    fit_index: pd.Index,
    apply_index: pd.Index,
    mode: str,
    batch_col: str,
    covariate_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Fit harmonization on fit_index and apply to fit/apply matrices."""
    if mode == "none":
        return X_fit_raw.copy(), X_apply_raw.copy(), {"unknown_batches_apply": 0}

    # Harmonization requires complete matrices; impute with train medians only.
    medians = X_fit_raw.median(axis=0)
    X_fit_imp = X_fit_raw.fillna(medians)
    X_apply_imp = X_apply_raw.fillna(medians)

    labels_fit = labels.loc[fit_index]
    labels_apply = labels.loc[apply_index]

    harmonizer = FeatureHarmonizer(
        mode=mode,
        batch_col=batch_col,
        covariate_cols=covariate_cols,
    ).fit(X_fit_imp, labels_fit)

    X_fit_h, unknown_fit = harmonizer.transform(X_fit_imp, labels_fit)
    X_apply_h, unknown_apply = harmonizer.transform(X_apply_imp, labels_apply)
    info = {
        "unknown_batches_fit": int(unknown_fit),
        "unknown_batches_apply": int(unknown_apply),
    }
    return X_fit_h, X_apply_h, info


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop one of any highly-correlated pair of features (fit on train only).

    Operates on numpy arrays (i.e. after the imputer step) so NaNs are already
    filled.  Uses a greedy column-scan: for each pair (i, j) with i < j whose
    absolute Pearson correlation >= threshold, column j is marked for removal
    unless it has already been marked.
    """

    def __init__(self, threshold: float = 0.9) -> None:
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> CorrelationPruner:
        """Identify features to keep by pruning highly correlated pairs."""
        if self.threshold <= 0 or self.threshold >= 1 or X.shape[1] < MIN_CLASS_COUNT:
            self.keep_mask_ = np.ones(X.shape[1], dtype=bool)
            return self

        corr = np.abs(np.corrcoef(X.T))
        to_drop: set[int] = set()
        for i in range(corr.shape[0]):
            if i in to_drop:
                continue
            for j in range(i + 1, corr.shape[1]):
                if j not in to_drop and corr[i, j] >= self.threshold:
                    to_drop.add(j)

        self.keep_mask_ = np.array(
            [idx not in to_drop for idx in range(X.shape[1])], dtype=bool
        )
        print(
            f"[DEBUG] CorrelationPruner @ {self.threshold:.2f}: "
            f"dropped={len(to_drop)}, kept={self.keep_mask_.sum()}",
        )
        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return features filtered to the unpruned subset."""
        return X[:, self.keep_mask_]


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Select top-K features using mRMR (minimum-Redundancy Maximum-Relevance).

    Wraps ``mrmr-selection``'s ``mrmr_classif`` function into an
    sklearn-compatible transformer.  Operates on numpy arrays (i.e. after the
    imputer step); columns are addressed by integer index internally.

    Parameters
    ----------
    k : int
        Number of features to select.  Clamped to the actual number of
        available features so the selector never raises on small inputs.
    """

    def __init__(self, k: int = 20) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> MRMRSelector:
        """Select top-k features using minimum redundancy maximum relevance."""
        try:
            from mrmr import mrmr_classif
        except ImportError as exc:
            msg = (
                "mrmr-selection is not installed. "
                "Install it with 'pip install mrmr-selection' to use "
                "--feature-selection mrmr."
            )
            raise ImportError(msg) from exc

        k = min(self.k, X.shape[1])
        df = pd.DataFrame(X)
        selected = mrmr_classif(X=df, y=pd.Series(y), K=k, show_progress=False)
        self.selected_indices_ = np.array(selected, dtype=int)
        print(
            f"[DEBUG] MRMRSelector: requested k={self.k}, "
            f"selected={len(self.selected_indices_)} features",
        )
        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return features filtered to the mRMR-selected subset."""
        return X[:, self.selected_indices_]


# ---------------------------
# Additional plots
# (beyond what the evaluation framework's VISUALIZATION_REGISTRY provides)
# ---------------------------
def plot_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    outpath: Path,
) -> None:
    """Plot precision-recall curve and save to disk."""
    if len(np.unique(y_true)) < MIN_CLASS_COUNT:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_calib(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    outpath: Path,
) -> None:
    """Plot calibration curve and save to disk."""
    if len(np.unique(y_true)) < MIN_CLASS_COUNT:
        return
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=min(10, len(y_true)),
        strategy="quantile",
    )
    plt.figure()
    plt.plot(prob_pred, prob_true, "o-")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------------------------
# Model builders
# ---------------------------
def normalize_rf_max_features(
    val: int | float | str | None,
) -> int | float | str | None:
    """Parse --rf-max-features into a valid value."""
    if val is None:
        return None
    if isinstance(val, int | float):
        return val
    if val in ("sqrt", "log2"):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except (TypeError, ValueError):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def build_estimator(
    args: argparse.Namespace,
) -> RandomForestClassifier | LogisticRegression | GridSearchCV:
    """Return a classifier or a GridSearchCV wrapping the classifier."""
    # ---------------- Logistic regression ----------------
    if args.classifier == "logistic":
        solver = "saga" if args.logreg_penalty in ("l1", "elasticnet") else "lbfgs"
        base = LogisticRegression(
            penalty=args.logreg_penalty,
            l1_ratio=(
                args.logreg_l1_ratio if args.logreg_penalty == "elasticnet" else None
            ),
            C=args.logreg_C,
            solver=solver,
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        )
        if args.grid_search:
            param_grid: dict[str, list[float]] = {"C": [0.05, 0.1, 0.2, 0.5, 1.0]}
            if args.logreg_penalty == "elasticnet":
                param_grid["l1_ratio"] = [0.1, 0.3, 0.5, 0.7]
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return GridSearchCV(
                base,
                param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )
        return base

    # ---------------- Random forest ----------------
    if args.classifier == "rf":
        base = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            min_samples_split=args.rf_min_samples_split,
            max_features=normalize_rf_max_features(args.rf_max_features),
            ccp_alpha=args.rf_ccp_alpha,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        if args.grid_search:
            param_grid_rf: dict[str, list[float | int | str]] = {
                "n_estimators": [300, 400, 500],
                "max_depth": [6, 8, 10],
                "min_samples_leaf": [5, 10, 20],
                "max_features": [0.2, 0.3, 0.5, "sqrt"],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return GridSearchCV(
                base,
                param_grid_rf,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )
        return base

    # ---------------- XGBoost ----------------
    if args.classifier == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # optional dependency
            msg = (
                "XGBoost is not installed. Install it with "
                "'pip install xgboost' or "
                "'conda install -c conda-forge xgboost' to use "
                "--classifier xgb."
            )
            raise ImportError(msg) from exc

        base = XGBClassifier(
            objective="binary:logistic",
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_lambda=args.xgb_reg_lambda,
            reg_alpha=args.xgb_reg_alpha,
            scale_pos_weight=args.xgb_scale_pos_weight,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss",  # avoids warning; logistic loss for binary clf
        )
        if args.grid_search:
            param_grid_xgb: dict[str, list[float | int]] = {
                "n_estimators": [200, 300, 400],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return GridSearchCV(
                base,
                param_grid_xgb,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )
        return base

    # ---------------- Fallback ----------------
    msg = f"Unknown classifier: {args.classifier!r}"
    raise ValueError(msg)


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    """Parse arguments, train the model, and save predictions/metrics."""
    ap = argparse.ArgumentParser(
        description="Train classifier on already-extracted radiomics CSVs.",
    )
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--test-features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument(
        "--classifier", choices=["logistic", "rf", "xgb"], default="logistic"
    )

    # Logistic options
    ap.add_argument("--logreg-C", type=float, default=1.0)
    ap.add_argument(
        "--logreg-penalty",
        choices=["l2", "l1", "elasticnet"],
        default="l2",
    )
    ap.add_argument("--logreg-l1-ratio", type=float, default=0.0)

    # RF options
    ap.add_argument("--rf-n-estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--rf-min-samples-split", type=int, default=2)
    ap.add_argument("--rf-max-features", default="sqrt")
    ap.add_argument("--rf-ccp-alpha", type=float, default=0.0)

    # XGBoost options
    ap.add_argument("--xgb-n-estimators", type=int, default=300)
    ap.add_argument("--xgb-max-depth", type=int, default=4)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.05)
    ap.add_argument("--xgb-subsample", type=float, default=0.8)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    ap.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    ap.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    ap.add_argument(
        "--xgb-scale-pos-weight",
        type=float,
        default=1.0,
        help=(
            "Scale positive class in XGBoost; "
            "e.g. n_negative / n_positive for imbalance."
        ),
    )

    # Feature handling
    ap.add_argument(
        "--include-subtype",
        action="store_true",
        help=(
            "Append subtype-derived model features from labels "
            "(if 'subtype' or 'tumor_subtype' present)."
        ),
    )
    ap.add_argument(
        "--include-site",
        action="store_true",
        help=(
            "Append site-derived model features from labels "
            "(if 'site' column is present)."
        ),
    )
    ap.add_argument(
        "--categorical-encoding",
        choices=["onehot", "ordinal"],
        default="onehot",
        help=(
            "Encoding for --include-subtype/--include-site. "
            "'onehot' (default) avoids imposing ordinal structure; "
            "'ordinal' reproduces the old single-code behavior."
        ),
    )
    ap.add_argument(
        "--subtype-filter",
        type=str,
        default=None,
        help=(
            "If set, restrict train and test sets to patients whose subtype "
            "matches this value exactly (e.g. 'triple_negative', 'luminal_a'). "
            "Case-sensitive; must match a value in the labels subtype column."
        ),
    )
    ap.add_argument(
        "--site-filter",
        type=str,
        default=None,
        help=(
            "If set, restrict train and test sets to patients whose clinical site "
            "matches this value exactly (e.g. 'DUKE', 'ISPY2'). "
            "Case-sensitive; must match a value in the labels 'site' column."
        ),
    )

    ap.add_argument(
        "--corr-threshold",
        type=float,
        default=0.0,
        help="If >0, drop one of any pair with |rho| >= threshold on TRAIN only.",
    )
    ap.add_argument(
        "--k-best",
        type=int,
        default=0,
        help="If >0, keep top-K features by ANOVA F-test on TRAIN only.",
    )
    ap.add_argument(
        "--feature-selection",
        choices=["kbest", "mrmr"],
        default="kbest",
        help=(
            "Filter method applied after optional correlation pruning: "
            "'kbest' = ANOVA F-test SelectKBest (default); "
            "'mrmr' = minimum-Redundancy Maximum-Relevance. "
            "Only active when --k-best > 0."
        ),
    )
    ap.add_argument(
        "--exclude-feature-regex",
        action="append",
        default=[],
        help=(
            "Regex pattern for feature names to exclude before training. "
            "Can be passed multiple times."
        ),
    )
    ap.add_argument(
        "--grid-search",
        action="store_true",
        help="If set, run a small GridSearchCV on the classifier using train only.",
    )
    ap.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help=(
            "If > 1, run Stratified K-fold CV on the training set and "
            "save cross-validated metrics."
        ),
    )
    ap.add_argument(
        "--harmonization-mode",
        choices=["none", "zscore_site", "combat_param", "combat_nonparam"],
        default="none",
        help=(
            "Optional feature harmonization fitted on training rows only. "
            "'zscore_site' applies per-batch location/scale normalization; "
            "'combat_param' uses empirical-Bayes ComBat; "
            "'combat_nonparam' uses a non-shrinkage ComBat approximation."
        ),
    )
    ap.add_argument(
        "--harmonization-batch-col",
        default="site",
        help=(
            "Column in labels.csv used as harmonization batch variable "
            "(default: site)."
        ),
    )
    ap.add_argument(
        "--harmonization-covariate",
        action="append",
        default=[],
        help=(
            "Biological covariate column to preserve during ComBat. "
            "Can be passed multiple times or as comma-separated values."
        ),
    )
    ap.add_argument(
        "--cv-only",
        action="store_true",
        help=(
            "If set, skip test-set evaluation and write metrics from CV only. "
            "Use this when test split is not independent from validation."
        ),
    )

    args = ap.parse_args()
    harm_covariates = _normalize_covariate_list(args.harmonization_covariate)
    if "pcr" in harm_covariates:
        msg = "Do not include target label 'pcr' as a harmonization covariate."
        raise ValueError(msg)
    if args.cv_only and not (args.cv_folds and args.cv_folds > 1):
        msg = "--cv-only requires --cv-folds > 1."
        raise ValueError(msg)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load data ----------------
    Xtr_raw = load_features(args.train_features)
    Xte_raw = load_features(args.test_features)
    print(f"[DEBUG] Xtr_raw shape: {Xtr_raw.shape}, Xte_raw shape: {Xte_raw.shape}")

    labels = load_labels(args.labels)
    missing_tr = Xtr_raw.index.difference(labels.index)
    missing_te = Xte_raw.index.difference(labels.index)
    if len(missing_tr) > 0 or len(missing_te) > 0:
        preview_tr = ", ".join(map(str, missing_tr[:5]))
        preview_te = ", ".join(map(str, missing_te[:5]))
        msg = (
            "Feature rows contain patient IDs missing from labels.csv. "
            f"missing_train={len(missing_tr)}"
            f"{' [' + preview_tr + (' ...' if len(missing_tr) > 5 else '') + ']' if len(missing_tr) else ''}"  # noqa: E501
            ", "
            f"missing_test={len(missing_te)}"
            f"{' [' + preview_te + (' ...' if len(missing_te) > 5 else '') + ']' if len(missing_te) else ''}"  # noqa: E501
        )
        raise ValueError(msg)

    # Make sure indices line up and y are extracted
    ytr = labels.loc[Xtr_raw.index, "pcr"].astype(int).to_numpy()
    yte = labels.loc[Xte_raw.index, "pcr"].astype(int).to_numpy()

    # Identify subtype column if present
    subtype_col: str | None = None
    for candidate in ("subtype", "tumor_subtype"):
        if candidate in labels.columns:
            subtype_col = candidate
            break

    # Identify site column if present
    site_col: str | None = "site" if "site" in labels.columns else None

    if args.harmonization_mode != "none":
        if args.harmonization_batch_col not in labels.columns:
            msg = (
                f"--harmonization-batch-col '{args.harmonization_batch_col}' "
                "not found in labels."
            )
            raise ValueError(msg)
    for col in harm_covariates:
        if col not in labels.columns:
            msg = f"--harmonization-covariate '{col}' not found in labels."
            raise ValueError(msg)

    # Optional: restrict to a single subtype
    if args.subtype_filter:
        if subtype_col is None:
            print(
                "[WARN] --subtype-filter set but no subtype"
                " column found in labels; ignoring.",
            )
        else:
            tr_mask = (
                labels.loc[Xtr_raw.index, subtype_col] == args.subtype_filter
            ).to_numpy()
            te_mask = (
                labels.loc[Xte_raw.index, subtype_col] == args.subtype_filter
            ).to_numpy()
            Xtr_raw = Xtr_raw.iloc[tr_mask]
            Xte_raw = Xte_raw.iloc[te_mask]
            ytr = labels.loc[Xtr_raw.index, "pcr"].astype(int).to_numpy()
            yte = labels.loc[Xte_raw.index, "pcr"].astype(int).to_numpy()
            print(
                f"[DEBUG] subtype_filter='{args.subtype_filter}': "
                f"train={len(Xtr_raw)}, test={len(Xte_raw)}",
            )

    # Optional: restrict to a single clinical site
    if args.site_filter:
        if site_col is None:
            print(
                "[WARN] --site-filter set but no 'site' column"
                " found in labels; ignoring.",
            )
        else:
            tr_mask = (
                labels.loc[Xtr_raw.index, site_col] == args.site_filter
            ).to_numpy()
            te_mask = (
                labels.loc[Xte_raw.index, site_col] == args.site_filter
            ).to_numpy()
            Xtr_raw = Xtr_raw.iloc[tr_mask]
            Xte_raw = Xte_raw.iloc[te_mask]
            ytr = labels.loc[Xtr_raw.index, "pcr"].astype(int).to_numpy()
            yte = labels.loc[Xte_raw.index, "pcr"].astype(int).to_numpy()
            print(
                f"[DEBUG] site_filter='{args.site_filter}': "
                f"train={len(Xtr_raw)}, test={len(Xte_raw)}",
            )

    # Optional: subtype-derived features
    if args.include_subtype:
        if subtype_col is None:
            print(
                "[WARN] --include-subtype set but no 'subtype' or 'tumor_subtype' "
                "in labels; skipping.",
            )
        else:
            Xtr_raw, Xte_raw, added = append_categorical_feature(
                Xtr_raw,
                Xte_raw,
                labels,
                column=subtype_col,
                prefix="subtype",
                encoding=args.categorical_encoding,
            )
            print(
                f"[DEBUG] appended subtype features from '{subtype_col}' "
                f"using {args.categorical_encoding}: +{len(added)} columns",
            )

    # Optional: site-derived features
    if args.include_site:
        if site_col is None:
            print(
                "[WARN] --include-site set but no 'site' column in labels; skipping.",
            )
        else:
            Xtr_raw, Xte_raw, added = append_categorical_feature(
                Xtr_raw,
                Xte_raw,
                labels,
                column=site_col,
                prefix="site",
                encoding=args.categorical_encoding,
            )
            print(
                f"[DEBUG] appended site features from '{site_col}' "
                f"using {args.categorical_encoding}: +{len(added)} columns",
            )

    if args.exclude_feature_regex:
        drop_cols: set[str] = set()
        for pattern in args.exclude_feature_regex:
            rx = re.compile(pattern)
            drop_cols.update(c for c in Xtr_raw.columns if rx.search(str(c)))
        if drop_cols:
            cols_sorted = sorted(drop_cols)
            Xtr_raw = Xtr_raw.drop(columns=cols_sorted, errors="ignore")
            Xte_raw = Xte_raw.drop(columns=cols_sorted, errors="ignore")
            print(
                f"[DEBUG] excluded features by regex: dropped={len(cols_sorted)} "
                f"remaining={Xtr_raw.shape[1]}",
            )
        else:
            print(
                "[DEBUG] --exclude-feature-regex provided"
                " but no matching features found.",
            )

    if len(Xtr_raw) == 0 or len(Xte_raw) == 0:
        msg = (
            "Empty split after filtering: " f"train={len(Xtr_raw)}, test={len(Xte_raw)}"
        )
        raise ValueError(msg)

    if len(np.unique(ytr)) < MIN_CLASS_COUNT:
        msg = (
            "Training split has fewer than 2 classes after filtering; "
            "cannot train a classifier."
        )
        raise ValueError(msg)

    if args.cv_folds and args.cv_folds > 1:
        class_counts = np.bincount(ytr.astype(int), minlength=2)
        nonzero_counts = class_counts[class_counts > 0]
        min_class_count = int(nonzero_counts.min()) if len(nonzero_counts) else 0
        if min_class_count < args.cv_folds:
            msg = (
                f"--cv-folds={args.cv_folds} is invalid after filtering: "
                f"smallest training class has {min_class_count} samples"
            )
            raise ValueError(msg)

    if len(np.unique(yte)) < MIN_CLASS_COUNT:
        print(
            "[WARN] test split has fewer than 2 classes after filtering; "
            "AUC_test will be NaN.",
        )

    # ---------------- Sanitize ----------------
    Xtr = sanitize_numeric(Xtr_raw, "train")
    # Apply the train-derived schema to test; do not drop test columns by
    # test-only variance/NaN checks.
    Xte = align_numeric_to_reference(Xte_raw, Xtr.columns.tolist(), "test")

    print(f"[DEBUG] sanitized train/test shapes: {Xtr.shape} / {Xte.shape}")

    # ---------------- Optional harmonization (fit on train only) ------------
    Xtr_model = Xtr
    Xte_model = Xte
    harm_info = {"unknown_batches_fit": 0, "unknown_batches_apply": 0}
    if args.harmonization_mode != "none":
        Xtr_model, Xte_model, harm_info = fit_apply_harmonization(
            X_fit_raw=Xtr,
            X_apply_raw=Xte,
            labels=labels,
            fit_index=Xtr.index,
            apply_index=Xte.index,
            mode=args.harmonization_mode,
            batch_col=args.harmonization_batch_col,
            covariate_cols=harm_covariates,
        )
        print(
            "[DEBUG] harmonization "
            f"mode={args.harmonization_mode} batch_col={args.harmonization_batch_col} "
            f"covariates={harm_covariates or []} "
            f"unknown_apply_batches={harm_info['unknown_batches_apply']}",
        )

    # ---------------- Build pipeline (feature selection runs inside each fit) --
    # CorrelationPruner and SelectKBest are included as pipeline steps so that
    # they are re-fitted on only the training rows of every k-fold split,
    # preventing label leakage into cross-validated AUC estimates.
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if args.corr_threshold > 0:
        steps.append(("corr_prune", CorrelationPruner(threshold=args.corr_threshold)))
    if args.k_best > 0:
        if args.feature_selection == "mrmr":
            steps.append(("mrmr", MRMRSelector(k=args.k_best)))
        else:
            steps.append(("kbest", SelectKBest(score_func=f_classif, k=args.k_best)))
    if args.classifier == "logistic":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", build_estimator(args)))
    pipe = Pipeline(steps)

    # ---------------- Train ----------------
    pipe.fit(Xtr_model, ytr)

    # Number of features reaching the classifier after pipeline selection steps.
    if "mrmr" in pipe.named_steps:
        n_features_used = len(pipe.named_steps["mrmr"].selected_indices_)
    elif "kbest" in pipe.named_steps:
        n_features_used = int(pipe.named_steps["kbest"].get_support().sum())
    elif "corr_prune" in pipe.named_steps:
        n_features_used = int(pipe.named_steps["corr_prune"].keep_mask_.sum())
    else:
        n_features_used = int(Xtr_model.shape[1])
    print(f"[DEBUG] n_features_used (entering classifier): {n_features_used}")

    clf_step = pipe.named_steps["clf"]

    # predict_proba and classes_ should be available (GridSearchCV delegates)
    classes_ = clf_step.classes_
    pos_idx = int(np.where(classes_ == 1)[0][0])

    p_tr = pipe.predict_proba(Xtr_model)[:, pos_idx]
    p_te = pipe.predict_proba(Xte_model)[:, pos_idx]

    auc_tr = (
        float(roc_auc_score(ytr, p_tr))
        if len(np.unique(ytr)) > MIN_CLASS_COUNT - 1
        else float("nan")
    )
    auc_te = (
        float(roc_auc_score(yte, p_te))
        if len(np.unique(yte)) > MIN_CLASS_COUNT - 1
        else float("nan")
    )

    fpr, tpr, thr = roc_curve(ytr, p_tr)
    thr_opt = float(thr[np.argmax(tpr - fpr)]) if len(thr) else AUC_BASELINE

    # ---------------- Evaluator (shared by CV k-fold and test evaluation) -----
    # Created here so evaluator.create_kfold_splits() can be used in the CV
    # block below, and evaluator.save_results() used for both outputs later.
    # model_name=outdir.name drives the output path:
    #   save_results(..., outdir.parent)          → outdir/  (test results)
    #   save_results(..., outdir.parent, run_name="cv") → outdir/cv/ (k-fold)
    evaluator = Evaluator(
        X=Xtr,
        y=ytr,
        patient_ids=Xtr.index,
        model_name=outdir.name,
        random_state=42,
    )

    # ---------------- K-fold CV on training set (evaluation framework) -------
    auc_cv = float("nan")
    auc_cv_std = float("nan")
    auc_cv_by_subtype: dict[str, float] | None = None
    if args.cv_folds and args.cv_folds > 1:
        # Each fold gets a fresh clone of the full pipeline (imputer,
        # CorrelationPruner, SelectKBest, scaler, classifier) so that every
        # step — including feature selection — is fitted on only that fold's
        # training rows.  This prevents label leakage through feature selection.
        # Note: if --grid-search is enabled, each fold runs its own
        # GridSearchCV (correct but slower than the old cross_val_predict
        # approach, which ran one grid search on all of Xtr).
        fold_splits = evaluator.create_kfold_splits(
            n_splits=args.cv_folds,
            stratify=True,
            shuffle=True,
        )
        fold_results_list = []
        for split in fold_splits:
            X_fold_tr = Xtr.iloc[split.train_indices]
            y_fold_tr = ytr[split.train_indices]
            X_fold_val = Xtr.iloc[split.val_indices]
            y_fold_val = ytr[split.val_indices]

            X_fold_tr_model = X_fold_tr
            X_fold_val_model = X_fold_val
            if args.harmonization_mode != "none":
                X_fold_tr_model, X_fold_val_model, _ = fit_apply_harmonization(
                    X_fit_raw=X_fold_tr,
                    X_apply_raw=X_fold_val,
                    labels=labels,
                    fit_index=X_fold_tr.index,
                    apply_index=X_fold_val.index,
                    mode=args.harmonization_mode,
                    batch_col=args.harmonization_batch_col,
                    covariate_cols=harm_covariates,
                )

            fold_pipe = clone(pipe)
            fold_pipe.fit(X_fold_tr_model, y_fold_tr)

            fold_clf = fold_pipe.named_steps["clf"]
            fold_pos_idx = int(np.where(fold_clf.classes_ == 1)[0][0])
            y_prob_val = fold_pipe.predict_proba(X_fold_val_model)[:, fold_pos_idx]
            # Use 0.5 threshold for per-fold binary predictions; the Youden-J
            # threshold is only meaningful on the full training set.
            y_pred_val = (y_prob_val >= 0.5).astype(int)

            fold_pred_df = pd.DataFrame(
                {
                    "patient_id": split.val_patient_ids,
                    "y_true": y_fold_val,
                    "y_pred": y_pred_val,
                    "y_prob": y_prob_val,
                }
            )
            if subtype_col is not None:
                fold_pred_df["subtype"] = labels.loc[
                    Xtr.index[split.val_indices], subtype_col
                ].to_numpy()

            fold_results_list.append(
                FoldResults(fold_idx=split.fold_idx, predictions=fold_pred_df)
            )

        kfold_results = evaluator.aggregate_kfold_results(fold_results_list)

        # Save framework k-fold outputs to outdir/cv/:
        #   outdir/cv/metrics.json          (mean ± std AUC, per-fold metrics,
        #                                    validation_summary if subtype present)
        #   outdir/cv/metrics_per_fold.json (per-fold AUC list)
        #   outdir/cv/predictions.csv       (all OOF predictions, fold-labelled)
        #   outdir/cv/plots/roc_curve.png
        evaluator.save_results(kfold_results, outdir.parent, run_name="cv")

        auc_cv = kfold_results.aggregated_metrics.get("auc", {}).get(
            "mean", float("nan")
        )
        auc_cv_std = kfold_results.aggregated_metrics.get("auc", {}).get(
            "std", float("nan")
        )

        # Per-subtype CV AUC computed from the pooled OOF predictions
        # (kfold_results.predictions concatenates all fold val rows).
        if subtype_col is not None:
            cv_preds = kfold_results.predictions
            auc_cv_by_subtype = {}
            for sub_val in sorted(cv_preds["subtype"].dropna().unique()):
                mask = cv_preds["subtype"] == sub_val
                y_sub = cv_preds.loc[mask, "y_true"].to_numpy()
                p_sub = cv_preds.loc[mask, "y_prob"].to_numpy()
                if len(np.unique(y_sub)) < MIN_CLASS_COUNT:
                    auc_cv_by_subtype[str(sub_val)] = float("nan")
                else:
                    auc_cv_by_subtype[str(sub_val)] = float(roc_auc_score(y_sub, p_sub))

    auc_test_by_subtype: dict[str, float] | None = None
    auc_test_by_site: dict[str, float] | None = None
    sens = spec = None
    tn = fp = fn = tp = None
    calib_status = "none"
    metrics_path = outdir / "metrics.json"

    if args.cv_only:
        auc_te = float("nan")
        cv_metrics_path = outdir / "cv" / "metrics.json"
        if cv_metrics_path.exists():
            with cv_metrics_path.open(encoding="utf-8") as f:
                augmented = json.load(f)
        else:
            augmented = {
                "evaluation_type": "cv_only",
                "model_name": outdir.name,
                "metrics": {"auc": auc_cv},
                "n_samples": int(len(Xtr)),
                "n_features": int(Xtr.shape[1]),
            }
        augmented["evaluation_type"] = "cv_only"
        augmented["model_name"] = outdir.name
        augmented["metrics"] = {"auc": auc_cv}
        augmented["n_samples"] = int(len(Xtr))
        augmented["n_features"] = int(Xtr.shape[1])
    else:
        ypred_te = (p_te >= thr_opt).astype(int)
        cm = confusion_matrix(yte, ypred_te, labels=[0, 1])
        if cm.size == CONF_MATRIX_SIZE:
            tn, fp, fn, tp = cm.ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

        # ---------------- AUC by subtype on test set ----------------
        if subtype_col is not None:
            sub_te = labels.loc[Xte.index, subtype_col]
            auc_test_by_subtype = {}
            for sub_val in sorted(sub_te.dropna().unique()):
                mask = sub_te == sub_val
                mask_array = mask.to_numpy()
                y_true_sub = yte[mask_array]
                y_prob_sub = p_te[mask_array]
                if len(np.unique(y_true_sub)) < MIN_CLASS_COUNT:
                    auc_sub_test = float("nan")
                else:
                    auc_sub_test = float(roc_auc_score(y_true_sub, y_prob_sub))
                auc_test_by_subtype[str(sub_val)] = auc_sub_test

        # ---------------- AUC by site on test set ----------------
        if site_col is not None:
            site_te = labels.loc[Xte.index, site_col]
            auc_test_by_site = {}
            for site_val in sorted(site_te.dropna().unique()):
                mask = site_te == site_val
                mask_array = mask.to_numpy()
                y_true_site = yte[mask_array]
                y_prob_site = p_te[mask_array]
                if len(np.unique(y_true_site)) < MIN_CLASS_COUNT:
                    auc_site_test = float("nan")
                else:
                    auc_site_test = float(roc_auc_score(y_true_site, y_prob_site))
                auc_test_by_site[str(site_val)] = auc_site_test

        # Additional radiomics-specific plots not in VISUALIZATION_REGISTRY.
        plot_pr(yte, p_te, outdir / "pr_curve.png")
        try:
            plot_calib(yte, p_te, outdir / "calibration_curve.png")
            calib_status = "ok"
        except Exception:  # noqa: BLE001
            calib_status = "none"

        predictions_df = pd.DataFrame(
            {
                "patient_id": Xte.index,
                "y_true": yte,
                "y_pred": ypred_te,
                "y_prob": p_te,
            }
        )
        if subtype_col is not None:
            predictions_df["subtype"] = labels.loc[Xte.index, subtype_col].to_numpy()

        tt_results = TrainTestResults(
            metrics={"auc": auc_te},
            predictions=predictions_df,
            model_name=outdir.name,
            run_name=None,
        )
        evaluator.save_results(tt_results, outdir.parent)
        with metrics_path.open(encoding="utf-8") as f:
            augmented = json.load(f)

    augmented.update(
        {
            # Keys read by run_experiment.py and run_ablations.py
            "auc_test": auc_te,
            "auc_train": auc_tr,
            "auc_train_cv": auc_cv,
            "auc_train_cv_std": auc_cv_std,
            "n_features_used": n_features_used,
            # Additional radiomics diagnostics
            "auc_train_cv_by_subtype": auc_cv_by_subtype,
            "auc_test_by_subtype": auc_test_by_subtype,
            "auc_test_by_site": auc_test_by_site,
            "classifier_type": {
                "logistic": "logistic",
                "rf": "random_forest",
                "xgb": "xgboost",
            }.get(args.classifier, str(args.classifier)),
            "class_order": classes_.tolist(),
            "threshold_train_youdenJ": thr_opt,
            "sensitivity_test": sens,
            "specificity_test": spec,
            "tn_fp_fn_tp_test": [
                int(x) if x is not None else None for x in [tn, fp, fn, tp]
            ],
            "calibration": calib_status,
            "subtype_filter": args.subtype_filter,
            "site_filter": args.site_filter,
            "corr_threshold": args.corr_threshold,
            "k_best": int(args.k_best),
            "feature_selection": args.feature_selection,
            "grid_search": bool(args.grid_search),
            "cv_folds": int(args.cv_folds),
            "cv_only": bool(args.cv_only),
            "harmonization_mode": args.harmonization_mode,
            "harmonization_batch_col": args.harmonization_batch_col,
            "harmonization_covariates": harm_covariates,
            "harmonization_unknown_batches_fit": harm_info["unknown_batches_fit"],
            "harmonization_unknown_batches_apply": harm_info["unknown_batches_apply"],
        }
    )

    if not np.isnan(auc_te) and not (auc_te > AUC_BASELINE):
        augmented["commentary"] = (
            "AUC_test ≤ 0.5. Try stronger regularization (Elastic-Net), "
            "correlation pruning, K-best, or DCE kinetic deltas."
        )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2)

    # ---------------- Remaining radiomics-specific outputs ----------------
    joblib.dump(pipe, outdir / "model.pkl")
    print(json.dumps(augmented, indent=2))


if __name__ == "__main__":
    main()
