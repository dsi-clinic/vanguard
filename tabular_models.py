"""Tabular estimator construction and nested-tuning helpers.

This module owns the sklearn-side logic for the tabular pipeline:
- fold-safe numeric feature selection
- model pipeline construction
- inner-CV candidate generation and scoring
- logging of per-fold selector diagnostics

Keeping this separate from train_tabular.py makes feature work safer. Students can
change cohort assembly and feature blocks without also editing estimator code.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation.kfold import FoldSplit
from features._common import safe_float

MIN_BINARY_CLASSES = 2


class NumericTopKAUCSelector(BaseEstimator, TransformerMixin):
    """Fold-safe train-fold top-K numeric feature selection by univariate AUC."""

    def __init__(
        self,
        feature_names: list[str] | None = None,
        enabled: bool = False,
        k: int = 128,
        min_non_na_rate: float = 0.2,
        min_n_unique: int = 2,
        max_abs_corr: float | None = None,
        max_zero_rate: float | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.enabled = enabled
        self.k = k
        self.min_non_na_rate = min_non_na_rate
        self.min_n_unique = min_n_unique
        self.max_abs_corr = max_abs_corr
        self.max_zero_rate = max_zero_rate

    def fit(self, X: Any, y: Any = None) -> NumericTopKAUCSelector:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_features = int(arr.shape[1])

        feature_names = list(self.feature_names or [])
        if len(feature_names) != n_features:
            feature_names = [f"f{i}" for i in range(n_features)]
        self.feature_names_in_ = feature_names

        if not self.enabled or n_features == 0:
            self.keep_indices_ = np.arange(n_features, dtype=int)
            self.n_input_features_ = n_features
            self.n_selected_features_ = n_features
            self.n_dropped_low_info_ = 0
            self.n_dropped_zero_heavy_ = 0
            self.n_dropped_collinear_ = 0
            self.k_effective_ = n_features
            return self

        if y is None:
            raise ValueError("NumericTopKAUCSelector requires y during fit.")
        y_arr = np.asarray(y).astype(int, copy=False)
        if y_arr.ndim != 1 or y_arr.shape[0] != arr.shape[0]:
            raise ValueError("y must be a 1D array aligned with X rows.")

        scores = np.full(n_features, -np.inf, dtype=float)
        non_na_rate = np.isfinite(arr).mean(axis=0)
        zero_rate = np.full(n_features, np.nan, dtype=float)
        low_info = np.zeros(n_features, dtype=bool)
        zero_heavy = np.zeros(n_features, dtype=bool)
        zero_thr = self.max_zero_rate
        use_zero_gate = (
            zero_thr is not None
            and np.isfinite(zero_thr)
            and 0.0 <= float(zero_thr) < 1.0
        )

        for j in range(n_features):
            valid = np.isfinite(arr[:, j])
            if float(non_na_rate[j]) < float(self.min_non_na_rate):
                low_info[j] = True
                continue

            xv = arr[valid, j]
            if xv.size > 0:
                zero_rate[j] = float(np.mean(xv == 0.0))
            if (
                use_zero_gate
                and np.isfinite(zero_rate[j])
                and float(zero_rate[j]) > float(zero_thr)
            ):
                zero_heavy[j] = True
                continue

            yv = y_arr[valid]
            if (
                np.unique(xv).size < int(self.min_n_unique)
                or np.unique(yv).size < MIN_BINARY_CLASSES
            ):
                low_info[j] = True
                continue

            try:
                auc = float(roc_auc_score(yv, xv))
            except Exception:
                low_info[j] = True
                continue
            if np.isfinite(auc):
                scores[j] = float(abs(auc - 0.5) * 2.0)
            else:
                low_info[j] = True

        valid_idx = np.where(np.isfinite(scores))[0]
        if valid_idx.size == 0:
            fallback = int(np.argmax(non_na_rate)) if n_features > 0 else 0
            valid_idx = np.array([fallback], dtype=int)
            scores[fallback] = 0.0

        k_cfg = max(1, int(self.k))
        k_eff = min(k_cfg, int(valid_idx.size))
        order = sorted(
            valid_idx.tolist(),
            key=lambda j: (-float(scores[j]), feature_names[j]),
        )

        selected_list: list[int] = []
        dropped_collinear = 0
        corr_thr = self.max_abs_corr
        use_corr_gate = (
            corr_thr is not None
            and np.isfinite(corr_thr)
            and 0.0 < float(corr_thr) < 1.0
        )

        if use_corr_gate and len(order) > 1:
            arr_order = arr[:, order].copy()
            med = np.nanmedian(arr_order, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            nan_mask = ~np.isfinite(arr_order)
            if np.any(nan_mask):
                arr_order[nan_mask] = np.take(med, np.where(nan_mask)[1])

            corr = np.corrcoef(arr_order, rowvar=False)
            corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=1.0, neginf=1.0)
            np.fill_diagonal(corr, 0.0)

            selected_local: list[int] = []
            for local_i in range(len(order)):
                if len(selected_local) >= k_eff:
                    break
                if any(
                    corr[local_i, local_j] > float(corr_thr)
                    for local_j in selected_local
                ):
                    dropped_collinear += 1
                    continue
                selected_local.append(local_i)

            if len(selected_local) < k_eff:
                selected_set = set(selected_local)
                for local_i in range(len(order)):
                    if len(selected_local) >= k_eff:
                        break
                    if local_i in selected_set:
                        continue
                    selected_local.append(local_i)
            selected_list = [order[i] for i in selected_local[:k_eff]]
        else:
            selected_list = order[:k_eff]

        selected = np.array(selected_list, dtype=int)
        self.keep_indices_ = np.sort(selected)
        self.n_input_features_ = n_features
        self.n_selected_features_ = int(self.keep_indices_.size)
        self.n_dropped_low_info_ = int(low_info.sum())
        self.n_dropped_zero_heavy_ = int(zero_heavy.sum())
        self.n_dropped_collinear_ = int(dropped_collinear)
        self.k_effective_ = int(k_eff)
        self.feature_scores_ = scores
        self.feature_zero_rate_ = zero_rate
        self.kept_feature_names_ = [feature_names[i] for i in self.keep_indices_]
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if not hasattr(self, "keep_indices_"):
            raise RuntimeError("NumericTopKAUCSelector must be fit before transform.")
        return arr[:, self.keep_indices_]


class NumericKinematicAddonSelector(BaseEstimator, TransformerMixin):
    """Keep all non-kinematic numeric features and add selected kinematic features."""

    def __init__(
        self,
        feature_names: list[str] | None = None,
        enabled: bool = False,
        k_kin: int = 0,
        kinematic_prefixes: list[str] | None = None,
        method: str = "topk_auc",
        min_non_na_rate: float = 0.2,
        min_n_unique: int = 2,
        max_abs_corr: float | None = None,
        max_zero_rate: float | None = None,
        mrmr_redundancy_weight: float = 1.0,
        mrmr_include_baseline: bool = True,
        corr_gate_against_baseline: bool = True,
    ) -> None:
        self.feature_names = feature_names
        self.enabled = enabled
        self.k_kin = k_kin
        self.kinematic_prefixes = kinematic_prefixes
        self.method = method
        self.min_non_na_rate = min_non_na_rate
        self.min_n_unique = min_n_unique
        self.max_abs_corr = max_abs_corr
        self.max_zero_rate = max_zero_rate
        self.mrmr_redundancy_weight = mrmr_redundancy_weight
        self.mrmr_include_baseline = mrmr_include_baseline
        self.corr_gate_against_baseline = corr_gate_against_baseline

    def fit(self, X: Any, y: Any = None) -> NumericKinematicAddonSelector:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_features = int(arr.shape[1])

        feature_names = list(self.feature_names or [])
        if len(feature_names) != n_features:
            feature_names = [f"f{i}" for i in range(n_features)]
        self.feature_names_in_ = feature_names

        prefixes = normalize_prefixes(
            self.kinematic_prefixes,
            default_prefixes=["kinematic_"],
        )
        kin_mask = np.asarray(
            [
                any(str(name).startswith(pref) for pref in prefixes)
                for name in feature_names
            ],
            dtype=bool,
        )
        kin_idx = np.where(kin_mask)[0]
        base_idx = np.where(~kin_mask)[0]

        self.n_input_features_ = n_features
        self.n_base_features_ = int(base_idx.size)
        self.n_kin_features_total_ = int(kin_idx.size)

        if not self.enabled or n_features == 0:
            self.keep_indices_ = np.arange(n_features, dtype=int)
            self.n_selected_features_ = n_features
            self.n_kin_selected_ = int(kin_idx.size)
            self.n_kin_eligible_ = int(kin_idx.size)
            self.n_dropped_low_info_ = 0
            self.n_dropped_zero_heavy_ = 0
            self.n_dropped_collinear_ = 0
            self.k_effective_ = int(kin_idx.size)
            self.kept_feature_names_ = [feature_names[i] for i in self.keep_indices_]
            self.selected_kin_feature_names_ = [feature_names[i] for i in kin_idx]
            return self

        if y is None:
            raise ValueError("NumericKinematicAddonSelector requires y during fit.")
        y_arr = np.asarray(y).astype(int, copy=False)
        if y_arr.ndim != 1 or y_arr.shape[0] != arr.shape[0]:
            raise ValueError("y must be a 1D array aligned with X rows.")

        if kin_idx.size == 0:
            self.keep_indices_ = np.sort(base_idx)
            self.n_selected_features_ = int(self.keep_indices_.size)
            self.n_kin_selected_ = 0
            self.n_kin_eligible_ = 0
            self.n_dropped_low_info_ = 0
            self.n_dropped_zero_heavy_ = 0
            self.n_dropped_collinear_ = 0
            self.k_effective_ = 0
            self.kept_feature_names_ = [feature_names[i] for i in self.keep_indices_]
            self.selected_kin_feature_names_ = []
            return self

        k_cfg = max(0, int(self.k_kin))
        if k_cfg == 0:
            self.keep_indices_ = np.sort(base_idx)
            self.n_selected_features_ = int(self.keep_indices_.size)
            self.n_kin_selected_ = 0
            self.n_kin_eligible_ = 0
            self.n_dropped_low_info_ = 0
            self.n_dropped_zero_heavy_ = 0
            self.n_dropped_collinear_ = 0
            self.k_effective_ = 0
            self.kept_feature_names_ = [feature_names[i] for i in self.keep_indices_]
            self.selected_kin_feature_names_ = []
            return self

        kin_arr = arr[:, kin_idx]
        kin_scores = np.full(kin_idx.size, -np.inf, dtype=float)
        kin_zero_rate = np.full(kin_idx.size, np.nan, dtype=float)
        kin_non_na_rate = np.isfinite(kin_arr).mean(axis=0)

        low_info = np.zeros(kin_idx.size, dtype=bool)
        zero_heavy = np.zeros(kin_idx.size, dtype=bool)
        zero_thr = self.max_zero_rate
        use_zero_gate = (
            zero_thr is not None
            and np.isfinite(zero_thr)
            and 0.0 <= float(zero_thr) < 1.0
        )

        for local_j in range(kin_idx.size):
            valid = np.isfinite(kin_arr[:, local_j])
            if float(kin_non_na_rate[local_j]) < float(self.min_non_na_rate):
                low_info[local_j] = True
                continue

            xv = kin_arr[valid, local_j]
            if xv.size > 0:
                kin_zero_rate[local_j] = float(np.mean(xv == 0.0))
            if use_zero_gate and np.isfinite(kin_zero_rate[local_j]):
                if float(kin_zero_rate[local_j]) > float(zero_thr):
                    zero_heavy[local_j] = True
                    continue

            yv = y_arr[valid]
            if (
                np.unique(xv).size < int(self.min_n_unique)
                or np.unique(yv).size < MIN_BINARY_CLASSES
            ):
                low_info[local_j] = True
                continue

            try:
                auc = float(roc_auc_score(yv, xv))
            except Exception:
                low_info[local_j] = True
                continue

            if np.isfinite(auc):
                kin_scores[local_j] = float(abs(auc - 0.5) * 2.0)
            else:
                low_info[local_j] = True

        eligible_local = np.where(np.isfinite(kin_scores))[0]
        self.n_kin_eligible_ = int(eligible_local.size)

        if eligible_local.size == 0:
            selected_kin_global = np.array([], dtype=int)
            dropped_collinear = 0
            k_eff = 0
        else:
            k_eff = min(k_cfg, int(eligible_local.size))
            order = sorted(
                eligible_local.tolist(),
                key=lambda j: (-float(kin_scores[j]), feature_names[int(kin_idx[j])]),
            )

            kin_z = impute_standardize_matrix(kin_arr)
            base_z = (
                impute_standardize_matrix(arr[:, base_idx])
                if base_idx.size
                else np.empty((arr.shape[0], 0))
            )
            denom = float(max(1, kin_z.shape[0]))
            corr_kin_kin = np.abs((kin_z.T @ kin_z) / denom)
            np.fill_diagonal(corr_kin_kin, 0.0)
            corr_kin_base = (
                np.abs((kin_z.T @ base_z) / denom)
                if base_z.shape[1] > 0
                else np.empty((kin_idx.size, 0))
            )

            corr_thr = self.max_abs_corr
            use_corr_gate = (
                corr_thr is not None
                and np.isfinite(corr_thr)
                and 0.0 < float(corr_thr) < 1.0
            )
            dropped_collinear_set: set[int] = set()
            selected_local: list[int] = []

            method = str(self.method).strip().lower()
            if method == "mrmr":
                remaining = set(order)
                while remaining and len(selected_local) < k_eff:
                    best_local = None
                    best_score = -np.inf
                    for local_j in sorted(
                        remaining,
                        key=lambda j: (
                            -float(kin_scores[j]),
                            feature_names[int(kin_idx[j])],
                        ),
                    ):
                        if use_corr_gate:
                            blocked = False
                            if selected_local and np.any(
                                corr_kin_kin[
                                    local_j, np.asarray(selected_local, dtype=int)
                                ]
                                > float(corr_thr)
                            ):
                                blocked = True
                            if (
                                not blocked
                                and self.corr_gate_against_baseline
                                and corr_kin_base.shape[1] > 0
                                and float(np.max(corr_kin_base[local_j, :]))
                                > float(corr_thr)
                            ):
                                blocked = True
                            if blocked:
                                dropped_collinear_set.add(local_j)
                                continue

                        redundancy_terms: list[float] = []
                        if selected_local:
                            redundancy_terms.append(
                                float(
                                    np.mean(
                                        corr_kin_kin[
                                            local_j,
                                            np.asarray(selected_local, dtype=int),
                                        ]
                                    )
                                )
                            )
                        if self.mrmr_include_baseline and corr_kin_base.shape[1] > 0:
                            redundancy_terms.append(
                                float(np.mean(corr_kin_base[local_j, :]))
                            )

                        redundancy = (
                            float(np.mean(redundancy_terms))
                            if redundancy_terms
                            else 0.0
                        )
                        score = (
                            float(kin_scores[local_j])
                            - float(self.mrmr_redundancy_weight) * redundancy
                        )

                        if score > best_score or (
                            np.isclose(score, best_score)
                            and best_local is not None
                            and feature_names[int(kin_idx[local_j])]
                            < feature_names[int(kin_idx[best_local])]
                        ):
                            best_score = score
                            best_local = local_j

                    if best_local is None:
                        break
                    selected_local.append(best_local)
                    remaining.remove(best_local)

                if len(selected_local) < k_eff:
                    selected_set = set(selected_local)
                    for local_j in order:
                        if len(selected_local) >= k_eff:
                            break
                        if local_j in selected_set:
                            continue
                        selected_local.append(local_j)
            else:
                for local_j in order:
                    if len(selected_local) >= k_eff:
                        break
                    if use_corr_gate:
                        blocked = False
                        if selected_local and np.any(
                            corr_kin_kin[local_j, np.asarray(selected_local, dtype=int)]
                            > float(corr_thr)
                        ):
                            blocked = True
                        if (
                            not blocked
                            and self.corr_gate_against_baseline
                            and corr_kin_base.shape[1] > 0
                            and float(np.max(corr_kin_base[local_j, :]))
                            > float(corr_thr)
                        ):
                            blocked = True
                        if blocked:
                            dropped_collinear_set.add(local_j)
                            continue
                    selected_local.append(local_j)

                if len(selected_local) < k_eff:
                    selected_set = set(selected_local)
                    for local_j in order:
                        if len(selected_local) >= k_eff:
                            break
                        if local_j in selected_set:
                            continue
                        selected_local.append(local_j)

            dropped_collinear = int(len(dropped_collinear_set))
            selected_local = selected_local[:k_eff]
            selected_kin_global = kin_idx[np.asarray(selected_local, dtype=int)]

        keep = np.sort(np.concatenate([base_idx, selected_kin_global]).astype(int))
        self.keep_indices_ = keep
        self.n_selected_features_ = int(keep.size)
        self.n_kin_selected_ = int(selected_kin_global.size)
        self.n_dropped_low_info_ = int(low_info.sum())
        self.n_dropped_zero_heavy_ = int(zero_heavy.sum())
        self.n_dropped_collinear_ = int(dropped_collinear)
        self.k_effective_ = int(k_eff)
        self.feature_scores_ = kin_scores
        self.feature_zero_rate_ = kin_zero_rate
        self.kept_feature_names_ = [feature_names[i] for i in keep]
        self.selected_kin_feature_names_ = [
            feature_names[i] for i in np.sort(selected_kin_global).astype(int).tolist()
        ]
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if not hasattr(self, "keep_indices_"):
            raise RuntimeError(
                "NumericKinematicAddonSelector must be fit before transform."
            )
        return arr[:, self.keep_indices_]


def normalize_prefixes(prefixes: Any, default_prefixes: list[str]) -> list[str]:
    """Normalize prefix config to a non-empty list of strings."""
    if prefixes is None:
        out = list(default_prefixes)
    elif isinstance(prefixes, str):
        cleaned = prefixes.strip()
        out = [cleaned] if cleaned else list(default_prefixes)
    elif isinstance(prefixes, list | tuple):
        out = [str(p).strip() for p in prefixes if str(p).strip()]
        out = out or list(default_prefixes)
    else:
        out = list(default_prefixes)
    deduped: list[str] = []
    seen: set[str] = set()
    for pref in out:
        if pref in seen:
            continue
        seen.add(pref)
        deduped.append(pref)
    return deduped


def impute_standardize_matrix(values: np.ndarray) -> np.ndarray:
    """Median-impute and standardize columns for robust correlation estimates."""
    arr = np.asarray(values, dtype=float).copy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    nan_mask = ~np.isfinite(arr)
    if np.any(nan_mask):
        arr[nan_mask] = np.take(med, np.where(nan_mask)[1])

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    return (arr - mean) / std


def build_numeric_selector(
    *,
    numeric_cols: list[str],
    feature_select_mode: str,
    feature_select_k: int,
    feature_select_k_kin: int,
    feature_select_kin_method: str,
    feature_select_kinematic_prefixes: list[str],
    feature_select_min_non_na_rate: float,
    feature_select_min_n_unique: int,
    feature_select_max_abs_corr: float | None,
    feature_select_max_zero_rate: float | None,
    feature_select_mrmr_redundancy_weight: float,
    feature_select_mrmr_include_baseline: bool,
    feature_select_corr_gate_against_baseline: bool,
) -> BaseEstimator:
    """Build numeric selector according to configured feature-selection mode."""
    mode = str(feature_select_mode).strip().lower()
    if mode in {"block_kinematic", "kinematic_addon", "kin_addon"}:
        return NumericKinematicAddonSelector(
            feature_names=numeric_cols,
            enabled=True,
            k_kin=feature_select_k_kin,
            kinematic_prefixes=feature_select_kinematic_prefixes,
            method=feature_select_kin_method,
            min_non_na_rate=feature_select_min_non_na_rate,
            min_n_unique=feature_select_min_n_unique,
            max_abs_corr=feature_select_max_abs_corr,
            max_zero_rate=feature_select_max_zero_rate,
            mrmr_redundancy_weight=feature_select_mrmr_redundancy_weight,
            mrmr_include_baseline=feature_select_mrmr_include_baseline,
            corr_gate_against_baseline=feature_select_corr_gate_against_baseline,
        )

    return NumericTopKAUCSelector(
        feature_names=numeric_cols,
        enabled=True,
        k=feature_select_k,
        min_non_na_rate=feature_select_min_non_na_rate,
        min_n_unique=feature_select_min_n_unique,
        max_abs_corr=feature_select_max_abs_corr,
        max_zero_rate=feature_select_max_zero_rate,
    )


def build_model_pipeline(
    model_type: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: dict[str, Any],
    random_state: int,
    model_params_override: dict[str, Any] | None = None,
) -> Pipeline:
    """Construct the sklearn pipeline for a configured tabular model."""
    model_params = deepcopy(config.model_params)
    if model_params_override:
        model_params.update(model_params_override)

    feature_select_enabled = bool(model_params.feature_select_enabled)
    feature_select_k = int(model_params.feature_select_k)
    feature_select_mode = str(model_params.feature_select_mode)
    feature_select_k_kin = int(model_params.feature_select_k_kin)
    feature_select_kin_method = str(model_params.feature_select_kin_method)
    feature_select_kinematic_prefixes = normalize_prefixes(
        model_params.feature_select_kinematic_prefixes,
        default_prefixes=["kinematic_"],
    )
    feature_select_mrmr_redundancy_weight = float(
        model_params.feature_select_mrmr_redundancy_weight
    )
    feature_select_mrmr_include_baseline = bool(
        model_params.feature_select_mrmr_include_baseline
    )
    feature_select_corr_gate_against_baseline = bool(
        model_params.feature_select_corr_gate_against_baseline
    )
    feature_select_min_non_na_rate = float(model_params.feature_select_min_non_na_rate)
    feature_select_min_n_unique = int(model_params.feature_select_min_n_unique)
    feature_select_max_abs_corr_raw = model_params.feature_select_max_abs_corr
    feature_select_max_abs_corr = (
        None
        if feature_select_max_abs_corr_raw in {None, "", "none", "null"}
        else float(feature_select_max_abs_corr_raw)
    )
    feature_select_max_zero_rate_raw = model_params.feature_select_max_zero_rate
    feature_select_max_zero_rate = (
        None
        if feature_select_max_zero_rate_raw in {None, "", "none", "null"}
        else float(feature_select_max_zero_rate_raw)
    )

    if model_type == "rf":
        numeric_steps: list[tuple[str, Any]] = []
        if feature_select_enabled and numeric_cols:
            numeric_steps.append(
                (
                    "feature_selector",
                    build_numeric_selector(
                        numeric_cols=numeric_cols,
                        feature_select_mode=feature_select_mode,
                        feature_select_k=feature_select_k,
                        feature_select_k_kin=feature_select_k_kin,
                        feature_select_kin_method=feature_select_kin_method,
                        feature_select_kinematic_prefixes=feature_select_kinematic_prefixes,
                        feature_select_min_non_na_rate=feature_select_min_non_na_rate,
                        feature_select_min_n_unique=feature_select_min_n_unique,
                        feature_select_max_abs_corr=feature_select_max_abs_corr,
                        feature_select_max_zero_rate=feature_select_max_zero_rate,
                        feature_select_mrmr_redundancy_weight=feature_select_mrmr_redundancy_weight,
                        feature_select_mrmr_include_baseline=feature_select_mrmr_include_baseline,
                        feature_select_corr_gate_against_baseline=feature_select_corr_gate_against_baseline,
                    ),
                )
            )
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        numeric_transformer = Pipeline(steps=numeric_steps)
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", cat_transformer, categorical_cols),
            ],
            remainder="drop",
        )

        model = RandomForestClassifier(
            n_estimators=int(model_params.n_estimators),
            max_depth=model_params.max_depth,
            min_samples_leaf=int(model_params.min_samples_leaf),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        numeric_steps: list[tuple[str, Any]] = []
        if feature_select_enabled and numeric_cols:
            numeric_steps.append(
                (
                    "feature_selector",
                    build_numeric_selector(
                        numeric_cols=numeric_cols,
                        feature_select_mode=feature_select_mode,
                        feature_select_k=feature_select_k,
                        feature_select_k_kin=feature_select_k_kin,
                        feature_select_kin_method=feature_select_kin_method,
                        feature_select_kinematic_prefixes=feature_select_kinematic_prefixes,
                        feature_select_min_non_na_rate=feature_select_min_non_na_rate,
                        feature_select_min_n_unique=feature_select_min_n_unique,
                        feature_select_max_abs_corr=feature_select_max_abs_corr,
                        feature_select_max_zero_rate=feature_select_max_zero_rate,
                        feature_select_mrmr_redundancy_weight=feature_select_mrmr_redundancy_weight,
                        feature_select_mrmr_include_baseline=feature_select_mrmr_include_baseline,
                        feature_select_corr_gate_against_baseline=feature_select_corr_gate_against_baseline,
                    ),
                )
            )
        numeric_steps.extend(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        numeric_transformer = Pipeline(steps=numeric_steps)
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", cat_transformer, categorical_cols),
            ],
            remainder="drop",
        )

        penalty = str(model_params.penalty).lower()
        solver_default = "saga" if penalty == "elasticnet" else "lbfgs"
        solver = str(model_params.solver or solver_default)

        model_kwargs: dict[str, Any] = {
            "class_weight": "balanced",
            "random_state": random_state,
            "max_iter": int(model_params.max_iter),
            "solver": solver,
            "penalty": penalty,
            "C": float(model_params.C),
        }
        if penalty == "elasticnet":
            model_kwargs["l1_ratio"] = float(model_params.l1_ratio)

        model = LogisticRegression(**model_kwargs)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def grid_values(value: Any, fallback: Any) -> list[Any]:
    """Normalize scalar/list config values into a list for grid iteration."""
    source = fallback if value is None else value
    if isinstance(source, list | tuple | np.ndarray | pd.Series):
        return list(source)
    return [source]


def parse_optional_float_param(value: Any) -> float | None:
    """Parse optional float config values, allowing None/null-like strings."""
    if value in {None, "", "none", "null"}:
        return None
    maybe = safe_float(value)
    if maybe is None or not np.isfinite(maybe):
        return None
    return float(maybe)


def build_nested_candidate_overrides(
    model_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build inner-CV candidate override dictionaries from config grids."""
    c_vals = [float(v) for v in grid_values(model_params.nested_c_grid, model_params.C)]
    l1_vals = [
        float(v)
        for v in grid_values(
            model_params.nested_l1_ratio_grid,
            model_params.l1_ratio,
        )
    ]
    mode_vals = [
        str(v)
        for v in grid_values(
            model_params.nested_feature_select_mode_grid,
            model_params.feature_select_mode,
        )
    ]
    k_kin_vals = [
        int(v)
        for v in grid_values(
            model_params.nested_k_kin_grid,
            model_params.feature_select_k_kin,
        )
    ]
    kin_method_vals = [
        str(v)
        for v in grid_values(
            model_params.nested_kin_method_grid,
            model_params.feature_select_kin_method,
        )
    ]
    max_corr_vals = [
        parse_optional_float_param(v)
        for v in grid_values(
            model_params.nested_max_abs_corr_grid,
            model_params.feature_select_max_abs_corr,
        )
    ]
    max_zero_vals = [
        parse_optional_float_param(v)
        for v in grid_values(
            model_params.nested_max_zero_rate_grid,
            model_params.feature_select_max_zero_rate,
        )
    ]

    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[tuple[str, str], ...]] = set()

    for mode in mode_vals:
        mode_name = str(mode).strip().lower()
        use_kin_grid = mode_name in {"block_kinematic", "kinematic_addon", "kin_addon"}
        mode_k_kin_vals = k_kin_vals if use_kin_grid else [0]
        mode_kin_method_vals = (
            kin_method_vals
            if use_kin_grid
            else [str(model_params.feature_select_kin_method)]
        )

        for c_val, l1_val, k_kin, kin_method, max_corr, max_zero in product(
            c_vals,
            l1_vals,
            mode_k_kin_vals,
            mode_kin_method_vals,
            max_corr_vals,
            max_zero_vals,
        ):
            override: dict[str, Any] = {
                "C": float(c_val),
                "l1_ratio": float(l1_val),
                "feature_select_enabled": bool(model_params.feature_select_enabled),
                "feature_select_mode": str(mode),
            }

            if use_kin_grid:
                override["feature_select_k_kin"] = int(k_kin)
                override["feature_select_kin_method"] = str(kin_method)

            override["feature_select_max_abs_corr"] = (
                None if max_corr is None else float(max_corr)
            )
            override["feature_select_max_zero_rate"] = (
                None if max_zero is None else float(max_zero)
            )

            key = tuple(sorted((k, str(v)) for k, v in override.items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(override)

    return candidates


def score_inner_cv_candidate(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: dict[str, Any],
    random_state: int,
    inner_splits: int,
    candidate_override: dict[str, Any],
) -> tuple[float, float, int]:
    """Evaluate one nested-CV candidate and return mean/std AUC and count."""
    y_arr = y_train.astype(int).to_numpy()
    cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state,
    )

    aucs: list[float] = []
    for inner_tr, inner_va in cv.split(np.zeros(len(y_arr)), y_arr):
        clf = build_model_pipeline(
            model_type=model_type,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            config=config,
            random_state=random_state,
            model_params_override=candidate_override,
        )
        clf.fit(X_train.iloc[inner_tr], y_train.iloc[inner_tr])
        yv = y_arr[inner_va]
        if len(np.unique(yv)) < MIN_BINARY_CLASSES:
            continue
        prob = clf.predict_proba(X_train.iloc[inner_va])[:, 1]
        aucs.append(float(roc_auc_score(yv, prob)))

    if not aucs:
        return float("nan"), float("nan"), 0
    arr = np.asarray(aucs, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def pick_nested_candidate_for_outer_fold(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: dict[str, Any],
    random_state: int,
    outer_fold_idx: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the best inner-CV candidate for one outer fold."""
    model_params = config.model_params
    inner_splits = max(2, int(model_params.nested_inner_splits))

    candidates = build_nested_candidate_overrides(model_params)
    if not candidates:
        return {}, []

    rows: list[dict[str, Any]] = []
    inner_seed_base = int(model_params.nested_inner_random_state)
    fold_seed = inner_seed_base + int(outer_fold_idx) * 997

    for cand in candidates:
        try:
            mean_auc, std_auc, n_valid = score_inner_cv_candidate(
                X_train=X_train,
                y_train=y_train,
                model_type=model_type,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                config=config,
                random_state=fold_seed,
                inner_splits=inner_splits,
                candidate_override=cand,
            )
        except Exception as exc:
            logging.warning(
                "Nested inner-CV candidate failed on outer fold %d with %s: %s",
                outer_fold_idx,
                cand,
                exc,
            )
            mean_auc, std_auc, n_valid = float("nan"), float("nan"), 0

        row = {
            "outer_fold": int(outer_fold_idx),
            "inner_auc_mean": mean_auc,
            "inner_auc_std": std_auc,
            "inner_auc_n_folds": int(n_valid),
        }
        row.update(cand)
        rows.append(row)

    valid_rows = [r for r in rows if np.isfinite(r["inner_auc_mean"])]
    if not valid_rows:
        logging.warning(
            "No valid nested candidates for outer fold %d; using first candidate.",
            outer_fold_idx,
        )
        return candidates[0], rows

    def rank_key(row: dict[str, Any]) -> tuple[float, int, int, float, float]:
        method = str(row.get("feature_select_kin_method", "topk_auc")).strip().lower()
        method_rank = 0 if method == "topk_auc" else 1
        return (
            -float(row["inner_auc_mean"]),
            int(row.get("feature_select_k_kin", 0)),
            method_rank,
            float(row.get("C", 1.0)),
            float(row.get("l1_ratio", 0.5)),
        )

    best_row = sorted(valid_rows, key=rank_key)[0]
    best_override = {
        k: best_row[k]
        for k in [
            "C",
            "l1_ratio",
            "feature_select_enabled",
            "feature_select_mode",
            "feature_select_k_kin",
            "feature_select_kin_method",
            "feature_select_max_abs_corr",
            "feature_select_max_zero_rate",
        ]
        if k in best_row
    }
    return best_override, rows


def log_feature_selector_stats(
    *,
    clf: Pipeline,
    split: FoldSplit,
    feature_select_enabled: bool,
    numeric_cols: list[str],
) -> None:
    """Log per-fold selector statistics when available."""
    if not (feature_select_enabled and numeric_cols):
        return
    try:
        preprocessor = clf.named_steps.get("preprocessor")
        if preprocessor is None or not hasattr(preprocessor, "named_transformers_"):
            return
        num_pipe = preprocessor.named_transformers_.get("num")
        if num_pipe is None or not hasattr(num_pipe, "named_steps"):
            return
        fs = num_pipe.named_steps.get("feature_selector")
        if fs is None:
            fs = num_pipe.named_steps.get("topk_selector")
        if fs is None or not hasattr(fs, "n_selected_features_"):
            return
        if hasattr(fs, "n_kin_selected_"):
            logging.info(
                "Fold %d numeric selector kept %d/%d "
                "(base=%d, kin_selected=%d/%d, k_effective=%d, "
                "drop_low_info=%d, drop_zero_heavy=%d, drop_collinear=%d)",
                split.fold_idx,
                int(getattr(fs, "n_selected_features_", np.nan)),
                int(getattr(fs, "n_input_features_", np.nan)),
                int(getattr(fs, "n_base_features_", 0)),
                int(getattr(fs, "n_kin_selected_", 0)),
                int(getattr(fs, "n_kin_features_total_", 0)),
                int(getattr(fs, "k_effective_", np.nan)),
                int(getattr(fs, "n_dropped_low_info_", 0)),
                int(getattr(fs, "n_dropped_zero_heavy_", 0)),
                int(getattr(fs, "n_dropped_collinear_", 0)),
            )
        else:
            logging.info(
                "Fold %d numeric top-k kept %d/%d "
                "(k_effective=%d, drop_low_info=%d, drop_zero_heavy=%d, drop_collinear=%d)",
                split.fold_idx,
                int(getattr(fs, "n_selected_features_", np.nan)),
                int(getattr(fs, "n_input_features_", np.nan)),
                int(getattr(fs, "k_effective_", np.nan)),
                int(getattr(fs, "n_dropped_low_info_", 0)),
                int(getattr(fs, "n_dropped_zero_heavy_", 0)),
                int(getattr(fs, "n_dropped_collinear_", 0)),
            )
    except Exception as exc:  # noqa: BLE001
        logging.debug("Unable to log feature selector stats: %s", exc)
