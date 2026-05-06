#!/usr/bin/env python3
r"""XGBoost interaction-aware pCR prediction with near-duplicate feature pruning.

Companion to the marginal/linear elastic-net baseline in feature_selection.py.
This pipeline tests whether shallow-tree XGBoost, trained on near-duplicate-pruned
features (unsupervised only), improves held-out pCR prediction.

**Key differences from the elastic-net pipeline:**

- No Mann-Whitney / BH univariate screening -- XGBoost receives features after
  unsupervised cleanup only, so it can discover joint/nonlinear effects.
- Near-duplicate pruning uses a high threshold (|rho| >= 0.97, Spearman or Pearson)
  and is block-aware: only features within the same semantic block are grouped.
- Shallow trees (max_depth 2-3) learn low-order feature interactions.

**Scientific context:** The elastic-net pipeline is a marginal/linear baseline that
can miss features important only jointly or nonlinearly.  This pipeline serves as
the interaction-aware comparator (see issue #152).

Usage::

    python scripts/xgboost_interaction_pipeline.py \
        experiments/second_order_features/features_engineered_labeled.csv \
        -o experiments/xgboost_interaction
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import connected_components
from scipy.stats import false_discovery_control
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import feature_block_for_column

matplotlib.use("Agg")

# ── Constants ────────────────────────────────────────────────────────────

_NON_FEATURE_COLUMNS = frozenset(
    {
        "case_id",
        "dataset",
        "has_centerline_file",
        "site",
        "bilateral",
        "tumor_subtype",
        "menopausal_status",
        "pcr",
    }
)

_BLOCK_PREFIXES: list[tuple[str, str]] = [
    ("tumor_size_", "tumor_size"),
    ("morph_", "morph"),
    ("graph_", "graph"),
    ("kinematic_", "kinematic"),
    # Graph sub-group prefixes (not recognized by features.graph.matches_column)
    ("tumor_burden_", "graph"),
    ("boundary_crossing_", "graph"),
    ("per_shell_topology_", "graph"),
    ("caliber_heterogeneity_", "graph"),
    ("directional_", "graph"),
    ("length_weighted_shape_stats_", "graph"),
    ("normalized_ratios_", "graph"),
]

_NEAR_DUPLICATE_THRESHOLD = 0.97
_NEAR_CONSTANT_STD = 1e-10
_MIN_COVERAGE_FRACTION = 0.5
_MIN_POSITIVE_SAMPLES = 3
_MIN_PDP_FEATURES = 2
_MIN_MANNWHITNEY_SAMPLES = 3
_UNIVARIATE_P_THRESHOLD = 0.2
_FALLBACK_TOP_K = 30
_CV_FOLDS = 5
_N_OUTER_REPEATS = 3
_RANDOM_STATE = 42
_N_RANDOM_SEARCH_ITER = 50
_MAX_ROC_LEGEND_FOLDS = 20
_STABILITY_HIGHLIGHT_THRESHOLD = 0.5
_ELASTIC_NET_L1_RATIO = 0.5
_TOP_PDP_PAIRS = 3

_XGBOOST_PARAM_DIST: dict = {
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [100, 300],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [1, 5, 10],
    "min_child_weight": [1, 5],
}


# ── Feature block detection ─────────────────────────────────────────────


def detect_feature_block(col: str) -> str:
    """Map a feature column name to its semantic block.

    Uses the canonical ``feature_block_for_column`` from the features package,
    with a prefix-based fallback for features not recognized by the package.
    """
    canonical = feature_block_for_column(col)
    if canonical is not None:
        return canonical
    # Fallback: prefix-based heuristic
    for prefix, block in _BLOCK_PREFIXES:
        if col.startswith(prefix):
            return block
    return "clinical"


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are not metadata/labels."""
    return [
        c
        for c in df.columns
        if c not in _NON_FEATURE_COLUMNS
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]


# ── Step 1: Basic unsupervised cleanup ───────────────────────────────────


def basic_cleanup(
    features: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    """Drop all-NaN, near-constant, low-coverage, exact-duplicate features.

    Also removes single-unique-value columns.  No correlation pruning here.

    Must be called with training-fold data only.
    """
    kept: list[str] = []
    dropped_reasons: dict[str, list[str]] = {
        "all_nan": [],
        "low_coverage": [],
        "near_constant": [],
        "single_unique": [],
    }

    for col in feature_cols:
        series = features[col]
        non_null = series.dropna()
        if len(non_null) == 0:
            dropped_reasons["all_nan"].append(col)
        elif series.notna().mean() < _MIN_COVERAGE_FRACTION:
            dropped_reasons["low_coverage"].append(col)
        elif non_null.std() < _NEAR_CONSTANT_STD:
            dropped_reasons["near_constant"].append(col)
        elif len(set(non_null)) <= 1:
            dropped_reasons["single_unique"].append(col)
        else:
            kept.append(col)

    # Remove exact duplicate columns
    sub = features[kept]
    dup_groups = sub.T.duplicated(keep="first")
    exact_dups = [kept[i] for i, is_dup in enumerate(dup_groups) if is_dup]
    if exact_dups:
        dropped_reasons["exact_duplicate"] = exact_dups
        kept = [c for c in kept if c not in set(exact_dups)]

    for reason, cols in dropped_reasons.items():
        if cols:
            logging.info(
                "  Basic cleanup — dropped %d features (%s)", len(cols), reason
            )

    logging.info(
        "  Basic cleanup: %d → %d features",
        len(feature_cols),
        len(kept),
    )
    return kept


# ── Univariate screening (for elastic-net baseline arm only) ────────────


def _stage2_univariate(
    features: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> list[str]:
    """Mann-Whitney U screening with BH FDR correction.

    Used only for the elastic-net baseline arm to replicate the current
    marginal/linear pipeline.  The XGBoost arm does NOT use this.
    """
    grp0 = features[features[label_col] == 0]
    grp1 = features[features[label_col] == 1]

    results = []
    for col in feature_cols:
        v0 = grp0[col].dropna()
        v1 = grp1[col].dropna()
        if len(v0) < _MIN_MANNWHITNEY_SAMPLES or len(v1) < _MIN_MANNWHITNEY_SAMPLES:
            continue
        try:
            _, pval = stats.mannwhitneyu(v0, v1, alternative="two-sided")
            results.append({"feature": col, "p_value": pval})
        except Exception:  # noqa: BLE001, S110
            pass

    if not results:
        return []

    results_df = pd.DataFrame(results).sort_values("p_value")
    raw_pvals = results_df["p_value"].to_numpy()
    adjusted_pvals = false_discovery_control(raw_pvals, method="bh")
    results_df["p_adjusted"] = adjusted_pvals

    kept_df = results_df[results_df["p_adjusted"] < _UNIVARIATE_P_THRESHOLD]
    if kept_df.empty:
        kept_df = results_df.head(_FALLBACK_TOP_K)

    return list(kept_df["feature"])


# ── Step 2-4: Block-aware near-duplicate pruning ────────────────────────


def _unsupervised_rep_score(features: pd.DataFrame, col: str) -> tuple:
    """Score a feature for representative selection (higher = more preferred).

    Priority: coverage, MAD (signal variance), preference for normalized/compact
    names, preference against log1p (redundant for trees), column name tie-break.
    """
    s = features[col]
    coverage = s.notna().mean()
    mad = float((s - s.median()).abs().median())

    # Prefer normalized / ratio / density / CV features
    is_normalized = any(
        kw in col for kw in ("normalized", "density", "ratio", "cv_", "_cv")
    )
    # Prefer shorter (more compact/interpretable) names
    name_compactness = -len(col)
    # Penalize log1p (redundant with raw for trees)
    is_log = "log1p" in col

    return (coverage, not is_log, is_normalized, mad, name_compactness, col)


def prune_near_duplicates(
    features: pd.DataFrame,
    feature_cols: list[str],
    fold_idx: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Block-aware near-duplicate pruning using Spearman and Pearson correlations.

    Groups features within the same block if |Spearman| >= threshold OR
    |Pearson| >= threshold, then keeps one representative per group.

    Returns the kept feature list and a pruning report DataFrame.
    """
    # Assign blocks
    col_blocks = {col: detect_feature_block(col) for col in feature_cols}
    blocks = sorted(set(col_blocks.values()))

    all_kept: list[str] = []
    report_rows: list[dict] = []

    for block in blocks:
        block_cols = [c for c in feature_cols if col_blocks[c] == block]
        if len(block_cols) <= 1:
            all_kept.extend(block_cols)
            continue

        block_data = features[block_cols]

        # Compute Spearman correlation via pandas rank correlation (much faster
        # than scipy.stats.spearmanr with nan_policy="omit" for large matrices)
        spearman_corr = block_data.rank().corr().abs().to_numpy()

        # Compute Pearson correlation
        pearson_corr = block_data.corr().abs().to_numpy()

        # Adjacency: connected if Spearman >= threshold OR Pearson >= threshold
        adjacency = (
            (spearman_corr >= _NEAR_DUPLICATE_THRESHOLD)
            | (pearson_corr >= _NEAR_DUPLICATE_THRESHOLD)
        ).astype(int)
        np.fill_diagonal(adjacency, 0)

        n_components, labels = connected_components(adjacency, directed=False)

        for comp in range(n_components):
            member_idxs = [i for i in range(len(block_cols)) if labels[i] == comp]
            members = [block_cols[i] for i in member_idxs]

            if len(members) <= 1:
                all_kept.extend(members)
                continue

            # Pick best representative
            best = max(members, key=lambda c: _unsupervised_rep_score(features, c))
            all_kept.append(best)

            # Record pruning report entries
            for m in members:
                if m == best:
                    continue
                # Compute correlation of dropped feature to kept representative
                corr_to_kept = features[[m, best]].corr().iloc[0, 1]
                report_rows.append(
                    {
                        "fold": fold_idx,
                        "original_feature": m,
                        "kept_feature": best,
                        "feature_block": block,
                        "correlation_to_kept": round(corr_to_kept, 4),
                        "drop_reason": "near_duplicate",
                        "semantic_group": f"{block}_component_{comp}",
                    }
                )

        block_dropped = len(block_cols) - sum(
            1 for c in all_kept if col_blocks.get(c) == block
        )
        if block_dropped > 0:
            logging.info(
                "  Near-duplicate pruning [%s]: %d → %d features",
                block,
                len(block_cols),
                len(block_cols) - block_dropped,
            )

    report_df = pd.DataFrame(report_rows)
    logging.info(
        "  Near-duplicate pruning total: %d → %d features",
        len(feature_cols),
        len(all_kept),
    )
    return all_kept, report_df


# ── XGBoost fitting ─────────────────────────────────────────────────────


def _detect_gpu() -> str | None:
    """Return 'cuda' if a CUDA GPU is available for XGBoost, else None."""
    try:
        import subprocess

        result = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _fit_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> tuple[XGBClassifier, dict]:
    """Fit XGBoost with randomized hyperparameter search."""
    device = _detect_gpu() or "cpu"
    if device == "cuda":
        logging.info("  XGBoost: using GPU (CUDA)")
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model,
        _XGBOOST_PARAM_DIST,
        n_iter=_N_RANDOM_SEARCH_ITER,
        cv=cv,
        scoring="roc_auc",
        random_state=_RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    logging.info(
        "  XGBoost: best params = %s, inner CV AUC = %.3f",
        search.best_params_,
        search.best_score_,
    )
    return model, search.best_params_


# ── Elastic-net fitting (baseline arm) ──────────────────────────────────


def _fit_elastic_net(
    x_scaled: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
) -> tuple[SGDClassifier, list[str], pd.DataFrame]:
    """Fit elastic-net logistic regression via SGDClassifier with CV-tuned alpha."""
    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)

    base_model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        l1_ratio=_ELASTIC_NET_L1_RATIO,
        class_weight="balanced",
        max_iter=5000,
        random_state=_RANDOM_STATE,
    )
    param_grid = {"alpha": np.logspace(-5, -1, 20)}
    search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_scaled, y)
    model = search.best_estimator_

    coefs = model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": coefs}).sort_values(
        "coefficient", key=abs, ascending=False
    )
    selected = list(coef_df[coef_df["coefficient"].abs() > 0]["feature"])

    logging.info(
        "  Elastic net: best α=%.4g, %d/%d features selected",
        search.best_params_["alpha"],
        len(selected),
        len(feature_cols),
    )
    return model, selected, coef_df


# ── Nested cross-validation ─────────────────────────────────────────────


def run_comparison_cv(
    features: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    outdir: Path,
) -> dict:
    """Run both arms (XGBoost + elastic-net) inside the same nested CV.

    Outer loop: RepeatedStratifiedKFold for unbiased performance estimation.
    Both arms share the same outer folds for a fair paired comparison.
    """
    y = features[label_col].to_numpy()
    outer_cv = RepeatedStratifiedKFold(
        n_splits=_CV_FOLDS,
        n_repeats=_N_OUTER_REPEATS,
        random_state=_RANDOM_STATE,
    )
    total_folds = _CV_FOLDS * _N_OUTER_REPEATS

    # Per-arm results
    xgb_aucs: list[float] = []
    xgb_aps: list[float] = []
    xgb_roc_curves: list[dict] = []
    xgb_feature_counter: Counter[str] = Counter()
    xgb_best_params: list[dict] = []
    xgb_importances_all: list[dict] = []

    enet_aucs: list[float] = []
    enet_aps: list[float] = []
    enet_roc_curves: list[dict] = []
    enet_feature_counter: Counter[str] = Counter()

    all_pruning_reports: list[pd.DataFrame] = []
    fold_summaries: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(features, y)):
        train_df = features.iloc[train_idx]
        test_df = features.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        n_pos_train = int(y_train.sum())
        n_pos_test = int(y_test.sum())
        if n_pos_train < _MIN_POSITIVE_SAMPLES:
            logging.warning(
                "Fold %d/%d: only %d positives in train — skipping",
                fold_idx + 1,
                total_folds,
                n_pos_train,
            )
            continue
        if n_pos_test == 0:
            logging.warning(
                "Fold %d/%d: no positives in test — skipping",
                fold_idx + 1,
                total_folds,
            )
            continue

        logging.info("=== Fold %d/%d ===", fold_idx + 1, total_folds)

        # ── Shared: basic cleanup on train ──
        cleaned_cols = basic_cleanup(train_df, feature_cols)

        # ── XGBoost arm: near-duplicate pruning → XGBoost ──
        xgb_cols, pruning_report = prune_near_duplicates(
            train_df, cleaned_cols, fold_idx=fold_idx
        )
        all_pruning_reports.append(pruning_report)

        if not xgb_cols:
            logging.warning(
                "Fold %d/%d: no features after pruning", fold_idx + 1, total_folds
            )
            continue

        imputer_xgb = SimpleImputer(strategy="median")
        x_train_xgb = imputer_xgb.fit_transform(train_df[xgb_cols])
        x_test_xgb = imputer_xgb.transform(test_df[xgb_cols])

        xgb_model, best_params = _fit_xgboost(x_train_xgb, y_train, xgb_cols)
        xgb_best_params.append(best_params)

        y_proba_xgb = xgb_model.predict_proba(x_test_xgb)[:, 1]

        try:
            auc_xgb = roc_auc_score(y_test, y_proba_xgb)
            ap_xgb = average_precision_score(y_test, y_proba_xgb)
        except ValueError:
            logging.warning("Fold %d/%d: XGBoost AUC failed", fold_idx + 1, total_folds)
            continue

        xgb_aucs.append(auc_xgb)
        xgb_aps.append(ap_xgb)
        fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
        xgb_roc_curves.append(
            {"fpr": fpr, "tpr": tpr, "auc": auc_xgb, "fold": fold_idx}
        )

        # Permutation importance on test fold
        perm_imp = permutation_importance(
            xgb_model,
            x_test_xgb,
            y_test,
            n_repeats=10,
            random_state=_RANDOM_STATE,
            scoring="roc_auc",
        )
        for feat_idx, feat_name in enumerate(xgb_cols):
            xgb_feature_counter[feat_name] += 1
            xgb_importances_all.append(
                {
                    "fold": fold_idx,
                    "feature": feat_name,
                    "importance_mean": perm_imp.importances_mean[feat_idx],
                    "importance_std": perm_imp.importances_std[feat_idx],
                }
            )

        # ── Elastic-net arm: cleanup → Stage 2 univariate → elastic-net ──
        # Replicates the current marginal/linear pipeline for a fair comparison.
        s2_cols = _stage2_univariate(train_df, cleaned_cols, label_col)
        if not s2_cols:
            logging.warning(
                "Fold %d/%d: no features after Stage 2 for elastic-net — skipping",
                fold_idx + 1,
                total_folds,
            )
            continue

        imputer_enet = SimpleImputer(strategy="median")
        scaler_enet = StandardScaler()
        x_train_enet = scaler_enet.fit_transform(
            imputer_enet.fit_transform(train_df[s2_cols])
        )
        x_test_enet = scaler_enet.transform(imputer_enet.transform(test_df[s2_cols]))

        enet_model, enet_selected, _ = _fit_elastic_net(x_train_enet, y_train, s2_cols)
        y_proba_enet = enet_model.predict_proba(x_test_enet)[:, 1]

        try:
            auc_enet = roc_auc_score(y_test, y_proba_enet)
            ap_enet = average_precision_score(y_test, y_proba_enet)
        except ValueError:
            logging.warning(
                "Fold %d/%d: elastic-net AUC failed", fold_idx + 1, total_folds
            )
            continue

        enet_aucs.append(auc_enet)
        enet_aps.append(ap_enet)
        fpr_e, tpr_e, _ = roc_curve(y_test, y_proba_enet)
        enet_roc_curves.append(
            {"fpr": fpr_e, "tpr": tpr_e, "auc": auc_enet, "fold": fold_idx}
        )
        for feat in enet_selected:
            enet_feature_counter[feat] += 1

        # Fold summary
        block_counts_before = Counter(detect_feature_block(c) for c in cleaned_cols)
        block_counts_after = Counter(detect_feature_block(c) for c in xgb_cols)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "features_before_cleanup": len(feature_cols),
                "features_after_cleanup": len(cleaned_cols),
                "features_after_pruning": len(xgb_cols),
                "enet_after_stage2": len(s2_cols),
                "enet_selected": len(enet_selected),
                "xgb_auc": auc_xgb,
                "xgb_ap": ap_xgb,
                "enet_auc": auc_enet,
                "enet_ap": ap_enet,
                **{
                    f"before_{block}": block_counts_before.get(block, 0)
                    for block in [
                        "clinical",
                        "tumor_size",
                        "morph",
                        "graph",
                        "kinematic",
                    ]
                },
                **{
                    f"kept_{block}": block_counts_after.get(block, 0)
                    for block in [
                        "clinical",
                        "tumor_size",
                        "morph",
                        "graph",
                        "kinematic",
                    ]
                },
                "xgb_best_params": str(best_params),
            }
        )

        logging.info(
            "Fold %d/%d: XGBoost AUC=%.3f AP=%.3f | Elastic-net AUC=%.3f AP=%.3f "
            "(pruned: %d → %d features)",
            fold_idx + 1,
            total_folds,
            auc_xgb,
            ap_xgb,
            auc_enet,
            ap_enet,
            len(cleaned_cols),
            len(xgb_cols),
        )

    # Aggregate
    xgb_mean_auc = np.mean(xgb_aucs) if xgb_aucs else float("nan")
    xgb_std_auc = np.std(xgb_aucs) if xgb_aucs else float("nan")
    xgb_mean_ap = np.mean(xgb_aps) if xgb_aps else float("nan")
    xgb_std_ap = np.std(xgb_aps) if xgb_aps else float("nan")
    enet_mean_auc = np.mean(enet_aucs) if enet_aucs else float("nan")
    enet_std_auc = np.std(enet_aucs) if enet_aucs else float("nan")
    enet_mean_ap = np.mean(enet_aps) if enet_aps else float("nan")
    enet_std_ap = np.std(enet_aps) if enet_aps else float("nan")

    logging.info("=== Nested CV Results ===")
    logging.info(
        "  XGBoost:     AUC = %.3f ± %.3f,  AP = %.3f ± %.3f",
        xgb_mean_auc,
        xgb_std_auc,
        xgb_mean_ap,
        xgb_std_ap,
    )
    logging.info(
        "  Elastic-net: AUC = %.3f ± %.3f,  AP = %.3f ± %.3f",
        enet_mean_auc,
        enet_std_auc,
        enet_mean_ap,
        enet_std_ap,
    )

    # Combine pruning reports
    if all_pruning_reports:
        full_pruning_report = pd.concat(all_pruning_reports, ignore_index=True)
    else:
        full_pruning_report = pd.DataFrame()

    return {
        "xgb_aucs": xgb_aucs,
        "xgb_aps": xgb_aps,
        "xgb_roc_curves": xgb_roc_curves,
        "xgb_feature_counter": xgb_feature_counter,
        "xgb_importances": pd.DataFrame(xgb_importances_all),
        "xgb_best_params": xgb_best_params,
        "enet_aucs": enet_aucs,
        "enet_aps": enet_aps,
        "enet_roc_curves": enet_roc_curves,
        "enet_feature_counter": enet_feature_counter,
        "pruning_report": full_pruning_report,
        "fold_summaries": pd.DataFrame(fold_summaries),
        "total_folds": total_folds,
        "xgb_mean_auc": xgb_mean_auc,
        "xgb_std_auc": xgb_std_auc,
        "xgb_mean_ap": xgb_mean_ap,
        "xgb_std_ap": xgb_std_ap,
        "enet_mean_auc": enet_mean_auc,
        "enet_std_auc": enet_std_auc,
        "enet_mean_ap": enet_mean_ap,
        "enet_std_ap": enet_std_ap,
    }


# ── Visualizations ──────────────────────────────────────────────────────


def _shorten_feature_name(name: str) -> str:
    """Abbreviate long feature prefixes for plot labels."""
    return (
        name.replace("tumor_size_", "ts_")
        .replace("kinematic_", "kin_")
        .replace("morph_", "m_")
        .replace("graph_", "g_")
        .replace("boundary_crossing_", "bc_")
        .replace("caliber_heterogeneity_", "cal_")
        .replace("per_shell_topology_", "pst_")
        .replace("directional_", "dir_")
        .replace("normalized_shell_burdens_", "nsb_")
    )


def plot_comparison_roc(results: dict, outdir: Path) -> None:
    """Plot per-fold ROC curves for both arms side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, arm, curves, mean_auc, std_auc in [
        (
            axes[0],
            "XGBoost",
            results["xgb_roc_curves"],
            results["xgb_mean_auc"],
            results["xgb_std_auc"],
        ),
        (
            axes[1],
            "Elastic-net",
            results["enet_roc_curves"],
            results["enet_mean_auc"],
            results["enet_std_auc"],
        ),
    ]:
        for rc in curves:
            ax.plot(
                rc["fpr"],
                rc["tpr"],
                alpha=0.25,
                linewidth=1,
                label=f'Fold {rc["fold"]} (AUC={rc["auc"]:.2f})',
            )
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{arm}\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f}", fontsize=12)
        if len(curves) <= _MAX_ROC_LEGEND_FOLDS:
            ax.legend(fontsize=6, loc="lower right", ncol=2)

    fig.savefig(outdir / "comparison_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", outdir / "comparison_roc_curves.png")


def plot_comparison_auc_ap(results: dict, outdir: Path) -> None:
    """Box + strip plot comparing AUC and AP distributions for both arms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, metric, xgb_vals, enet_vals in [
        (axes[0], "ROC AUC", results["xgb_aucs"], results["enet_aucs"]),
        (axes[1], "Average Precision", results["xgb_aps"], results["enet_aps"]),
    ]:
        data = [xgb_vals, enet_vals]
        bp = ax.boxplot(
            data,
            labels=["XGBoost", "Elastic-net"],
            vert=True,
            widths=0.4,
            patch_artist=True,
            boxprops={"edgecolor": "#333"},
            medianprops={"color": "#E1812C", "linewidth": 2},
        )
        bp["boxes"][0].set_facecolor("#D5E8D4")
        bp["boxes"][1].set_facecolor("#DAE8FC")

        for i, vals in enumerate(data):
            ax.scatter(
                np.full(len(vals), i + 1),
                vals,
                alpha=0.5,
                color="#3274A1",
                zorder=3,
                s=30,
            )
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_ylabel(metric)
        ax.set_title(
            f"{metric} Distribution\n"
            f"XGBoost: {np.mean(xgb_vals):.3f}±{np.std(xgb_vals):.3f}  |  "
            f"Elastic-net: {np.mean(enet_vals):.3f}±{np.std(enet_vals):.3f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)

    fig.savefig(
        outdir / "comparison_auc_ap_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    logging.info("Saved: %s", outdir / "comparison_auc_ap_distribution.png")


def plot_permutation_importance(results: dict, outdir: Path) -> None:
    """Horizontal bar chart of mean permutation importance across folds."""
    imp_df = results["xgb_importances"]
    if imp_df.empty:
        return

    agg = (
        imp_df.groupby("feature")["importance_mean"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    top_n = min(30, len(agg))
    top = agg.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)), constrained_layout=True)
    short_names = [_shorten_feature_name(f) for f in top.index]
    ax.barh(short_names, top["mean"], xerr=top["std"], color="#3274A1", alpha=0.8)
    ax.set_xlabel("Mean Permutation Importance (ΔAUC)")
    ax.set_title(
        f"XGBoost Permutation Importance (top {top_n})\n"
        f"Averaged across {int(top['count'].max())} CV folds",
        fontsize=11,
    )
    ax.tick_params(labelsize=8)
    fig.savefig(outdir / "xgb_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", outdir / "xgb_permutation_importance.png")


def plot_feature_stability(
    feature_counter: Counter, total_folds: int, arm_name: str, outdir: Path
) -> None:
    """Horizontal bar chart of feature selection frequency across folds."""
    if not feature_counter:
        return

    top_n = min(40, len(feature_counter))
    most_common = feature_counter.most_common(top_n)
    names = [_shorten_feature_name(f) for f, _ in reversed(most_common)]
    freqs = [c / total_folds for _, c in reversed(most_common)]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)), constrained_layout=True)
    colors = [
        "#3274A1" if f >= _STABILITY_HIGHLIGHT_THRESHOLD else "#A0C4E8" for f in freqs
    ]
    ax.barh(names, freqs, color=colors)
    ax.set_xlabel("Selection Frequency (fraction of folds)")
    ax.set_title(
        f"{arm_name} Feature Stability (top {top_n})\n"
        f"Across {total_folds} nested CV folds",
        fontsize=11,
    )
    ax.axvline(x=0.5, color="#E1812C", linestyle="--", linewidth=1, label="50%")
    ax.set_xlim(0, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=9)
    fig.savefig(
        outdir / f"{arm_name.lower().replace('-', '_')}_feature_stability.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_partial_dependence_top_pairs(
    model: XGBClassifier,
    x_all: np.ndarray,
    feature_cols: list[str],
    importance_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """2D partial dependence plots for the top feature pairs."""
    if importance_df.empty or len(feature_cols) < _MIN_PDP_FEATURES:
        return

    # Get top features by mean importance
    top_features = (
        importance_df.groupby("feature")["importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(2 * _TOP_PDP_PAIRS)
        .index.tolist()
    )
    # Filter to features that are in the current feature set
    top_features = [f for f in top_features if f in feature_cols]

    if len(top_features) < _MIN_PDP_FEATURES:
        return

    # Generate pairs
    pairs = []
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            pairs.append((top_features[i], top_features[j]))
            if len(pairs) >= _TOP_PDP_PAIRS:
                break
        if len(pairs) >= _TOP_PDP_PAIRS:
            break

    for f1, f2 in pairs:
        idx1 = feature_cols.index(f1)
        idx2 = feature_cols.index(f2)
        try:
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            PartialDependenceDisplay.from_estimator(
                model,
                x_all,
                [(idx1, idx2)],
                feature_names=[_shorten_feature_name(f) for f in feature_cols],
                ax=ax,
                kind="average",
            )
            short1 = _shorten_feature_name(f1)
            short2 = _shorten_feature_name(f2)
            ax.set_title(f"Partial Dependence: {short1} × {short2}", fontsize=11)
            fname = f"pdp_{short1}_x_{short2}.png".replace("/", "_")
            fig.savefig(outdir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logging.info("Saved: %s", outdir / fname)
        except Exception:  # noqa: BLE001
            logging.warning("PDP failed for %s × %s", f1, f2)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input_csv", type=Path, help="Path to features_engineered_labeled.csv"
    )
    ap.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/xgboost_interaction)",
    )
    ap.add_argument("--label-col", type=str, default="pcr", help="Label column name")
    args = ap.parse_args()

    outdir = args.outdir or Path("experiments/xgboost_interaction")
    outdir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.input_csv)
    logging.info(
        "Loaded %s: %d rows x %d cols",
        args.input_csv,
        len(features),
        len(features.columns),
    )
    logging.info(
        "Label distribution: %s",
        features[args.label_col].value_counts().to_dict(),
    )

    feature_cols = _get_feature_columns(features)
    logging.info("Starting with %d candidate features", len(feature_cols))

    # Block breakdown
    block_counts = Counter(detect_feature_block(c) for c in feature_cols)
    for block, count in sorted(block_counts.items()):
        logging.info("  Block '%s': %d features", block, count)

    # ── Step 1: Nested CV comparison ──
    cv_results = run_comparison_cv(features, feature_cols, args.label_col, outdir)

    # ── Step 2: Save pruning report ──
    if not cv_results["pruning_report"].empty:
        cv_results["pruning_report"].to_csv(
            outdir / "feature_pruning_report.csv", index=False
        )
        logging.info("Saved: %s", outdir / "feature_pruning_report.csv")

    # Save fold summaries
    if not cv_results["fold_summaries"].empty:
        cv_results["fold_summaries"].to_csv(outdir / "fold_summaries.csv", index=False)
        logging.info("Saved: %s", outdir / "fold_summaries.csv")

    # Save permutation importances
    if not cv_results["xgb_importances"].empty:
        cv_results["xgb_importances"].to_csv(
            outdir / "xgb_permutation_importances.csv", index=False
        )

    # ── Step 3: Visualize ──
    if cv_results["xgb_roc_curves"] and cv_results["enet_roc_curves"]:
        plot_comparison_roc(cv_results, outdir)
    if cv_results["xgb_aucs"] and cv_results["enet_aucs"]:
        plot_comparison_auc_ap(cv_results, outdir)

    plot_permutation_importance(cv_results, outdir)

    plot_feature_stability(
        cv_results["xgb_feature_counter"],
        cv_results["total_folds"],
        "XGBoost",
        outdir,
    )
    plot_feature_stability(
        cv_results["enet_feature_counter"],
        cv_results["total_folds"],
        "Elastic-net",
        outdir,
    )

    # ── Step 4: Final XGBoost model on all data for interaction readouts ──
    logging.info("=== Fitting final XGBoost on all data for interaction analysis ===")
    all_cleaned = basic_cleanup(features, feature_cols)
    all_pruned, final_pruning = prune_near_duplicates(features, all_cleaned)

    if all_pruned:
        imputer = SimpleImputer(strategy="median")
        x_all = imputer.fit_transform(features[all_pruned])
        y_all = features[args.label_col].to_numpy()

        xgb_final, _ = _fit_xgboost(x_all, y_all, all_pruned)

        # Partial dependence plots
        plot_partial_dependence_top_pairs(
            xgb_final, x_all, all_pruned, cv_results["xgb_importances"], outdir
        )

    # ── Step 5: Comparison table ──
    comparison = pd.DataFrame(
        {
            "arm": ["XGBoost (interaction-aware)", "Elastic-net (marginal/linear)"],
            "auc_mean": [cv_results["xgb_mean_auc"], cv_results["enet_mean_auc"]],
            "auc_std": [cv_results["xgb_std_auc"], cv_results["enet_std_auc"]],
            "ap_mean": [cv_results["xgb_mean_ap"], cv_results["enet_mean_ap"]],
            "ap_std": [cv_results["xgb_std_ap"], cv_results["enet_std_ap"]],
            "n_folds": [len(cv_results["xgb_aucs"]), len(cv_results["enet_aucs"])],
        }
    )
    comparison.to_csv(outdir / "comparison_table.csv", index=False)
    logging.info("Saved: %s", outdir / "comparison_table.csv")

    # ── Summary ──
    logging.info("=== Comparison Summary ===")
    logging.info(
        "  XGBoost:     AUC = %.3f ± %.3f,  AP = %.3f ± %.3f  (%d folds)",
        cv_results["xgb_mean_auc"],
        cv_results["xgb_std_auc"],
        cv_results["xgb_mean_ap"],
        cv_results["xgb_std_ap"],
        len(cv_results["xgb_aucs"]),
    )
    logging.info(
        "  Elastic-net: AUC = %.3f ± %.3f,  AP = %.3f ± %.3f  (%d folds)",
        cv_results["enet_mean_auc"],
        cv_results["enet_std_auc"],
        cv_results["enet_mean_ap"],
        cv_results["enet_std_ap"],
        len(cv_results["enet_aucs"]),
    )
    logging.info(
        "  NOTE: XGBoost uses unsupervised near-duplicate pruning only (no "
        "Mann-Whitney/BH screening). The elastic-net arm uses basic cleanup "
        "with L1/L2 regularization for feature selection."
    )
    logging.info(
        "Outputs saved to: %s",
        outdir,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
