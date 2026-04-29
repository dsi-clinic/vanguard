#!/usr/bin/env python3
"""Feature selection pipeline for pCR prediction.

Three-stage pipeline with nested cross-validation to prevent data leakage:
  1. Pre-filter: drop all-NaN, near-constant, and highly correlated features
  2. Univariate screening: Mann-Whitney U test with BH FDR correction
  3. Elastic-net logistic regression with cross-validated alpha

All supervised steps run inside each outer CV fold to produce unbiased
performance estimates.  A final model is then fit on all data to produce
the definitive feature list.

Usage::

    python scripts/feature_selection.py \
        experiments/second_order_features/features_engineered_labeled.csv \
        -o experiments/feature_selection
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
from scipy.stats import false_discovery_control
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

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

_CORRELATION_THRESHOLD = 0.8
_UNIVARIATE_P_THRESHOLD = 0.2
_ELASTIC_NET_L1_RATIO = 0.5
_MIN_MANNWHITNEY_SAMPLES = 3
_NEAR_CONSTANT_STD = 1e-10
_MIN_COVERAGE_FRACTION = 0.5
_STABILITY_HIGHLIGHT_THRESHOLD = 0.5
_MAX_ROC_LEGEND_FOLDS = 20
_CV_FOLDS = 5
_N_OUTER_REPEATS = 3
_FALLBACK_TOP_K = 30
_RANDOM_STATE = 42


def _get_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric columns that are not metadata/labels."""
    return [
        c
        for c in features.columns
        if c not in _NON_FEATURE_COLUMNS
        and features[c].dtype in ("float64", "float32", "int64", "int32")
    ]


def stage1_prefilter(
    features: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    """Drop all-NaN, near-constant, low-coverage, and correlated features.

    Correlation tie-breaking is fully unsupervised: prefer higher non-missing
    coverage, then higher MAD (median absolute deviation), then column name.
    """
    kept = []
    dropped_reasons: dict[str, list[str]] = {
        "all_nan": [],
        "low_coverage": [],
        "near_constant": [],
        "high_correlation": [],
    }

    # Pass 1: basic quality filters
    for col in feature_cols:
        series = features[col]
        if series.notna().sum() == 0:
            dropped_reasons["all_nan"].append(col)
        elif series.notna().mean() < _MIN_COVERAGE_FRACTION:
            dropped_reasons["low_coverage"].append(col)
        elif series.std() < _NEAR_CONSTANT_STD:
            dropped_reasons["near_constant"].append(col)
        else:
            kept.append(col)

    for reason, cols in dropped_reasons.items():
        if cols:
            logging.info("Stage 1 — dropped %d features (%s)", len(cols), reason)

    # Pass 2: remove one from each highly correlated pair (unsupervised)
    corr_matrix = features[kept].corr().abs()
    to_drop: set[str] = set()

    def _unsupervised_score(col: str) -> tuple[float, float, str]:
        """Higher is better: (coverage, MAD, name for deterministic tie-break)."""
        s = features[col]
        coverage = s.notna().mean()
        mad = (s - s.median()).abs().median()
        return (coverage, mad, col)

    for i in range(len(kept)):
        if kept[i] in to_drop:
            continue
        for j in range(i + 1, len(kept)):
            if kept[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > _CORRELATION_THRESHOLD:
                score_i = _unsupervised_score(kept[i])
                score_j = _unsupervised_score(kept[j])
                drop_col = kept[j] if score_i >= score_j else kept[i]
                to_drop.add(drop_col)

    dropped_reasons["high_correlation"] = list(to_drop)
    kept = [c for c in kept if c not in to_drop]

    logging.info(
        "Stage 1 summary: %d → %d features (dropped %d correlated)",
        len(feature_cols),
        len(kept),
        len(to_drop),
    )
    return kept


def stage2_univariate(
    features: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    *,
    fallback_top_k: int = _FALLBACK_TOP_K,
) -> tuple[list[str], pd.DataFrame]:
    """Univariate Mann-Whitney U screening with BH FDR correction.

    Keeps features whose BH-adjusted p-value is below the threshold.
    If no features survive correction, falls back to the top-k by raw p-value
    to ensure the pipeline can proceed.
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
            stat, pval = stats.mannwhitneyu(v0, v1, alternative="two-sided")
            results.append({"feature": col, "U_statistic": stat, "p_value": pval})
        except Exception:  # noqa: BLE001
            logging.debug("Mann-Whitney U test failed for %s", col)

    results_df = pd.DataFrame(results).sort_values("p_value")

    if results_df.empty:
        logging.warning("Stage 2 — no testable features; returning empty list")
        return [], results_df

    # Benjamini-Hochberg FDR correction
    raw_pvals = results_df["p_value"].to_numpy()
    adjusted_pvals = false_discovery_control(raw_pvals, method="bh")
    results_df["p_adjusted"] = adjusted_pvals

    kept_df = results_df[results_df["p_adjusted"] < _UNIVARIATE_P_THRESHOLD]
    if kept_df.empty:
        logging.warning(
            "Stage 2 — no features passed BH correction at α=%.2f; "
            "falling back to top %d by raw p-value",
            _UNIVARIATE_P_THRESHOLD,
            fallback_top_k,
        )
        kept_df = results_df.head(fallback_top_k)

    kept = list(kept_df["feature"])

    logging.info(
        "Stage 2 — univariate screening: %d → %d features "
        "(BH-adjusted p < %s, %d tested)",
        len(feature_cols),
        len(kept),
        _UNIVARIATE_P_THRESHOLD,
        len(results_df),
    )
    return kept, results_df


def _fit_elastic_net(
    x_scaled: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
) -> tuple[SGDClassifier, list[str], pd.DataFrame]:
    """Fit elastic-net logistic regression via SGDClassifier with CV-tuned alpha.

    Uses SGDClassifier with loss='log_loss' and penalty='elasticnet' to support
    a mix of L1 and L2 regularization.  The L1/L2 ratio is controlled by
    ``_ELASTIC_NET_L1_RATIO``.
    """
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
        "Elastic net: best α=%.4g, l1_ratio=%.1f, %d/%d features selected",
        search.best_params_["alpha"],
        _ELASTIC_NET_L1_RATIO,
        len(selected),
        len(feature_cols),
    )
    return model, selected, coef_df


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


# ── Nested cross-validation ─────────────────────────────────────────────


def run_nested_cv(
    features: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    outdir: Path,
) -> dict:
    """Run the full pipeline inside nested CV for unbiased evaluation.

    Outer loop: RepeatedStratifiedKFold for performance estimation.
    Inside each outer training fold:
        Stage 1 → Stage 2 (with BH) → impute → scale → elastic-net logistic regression
    Evaluate on the held-out outer test fold.

    Returns a dict with per-fold AUCs, feature selection frequencies, and
    aggregated ROC data for plotting.
    """
    y = features[label_col].to_numpy()
    outer_cv = RepeatedStratifiedKFold(
        n_splits=_CV_FOLDS,
        n_repeats=_N_OUTER_REPEATS,
        random_state=_RANDOM_STATE,
    )

    fold_aucs: list[float] = []
    feature_counter: Counter[str] = Counter()
    roc_curves: list[dict] = []
    total_folds = _CV_FOLDS * _N_OUTER_REPEATS

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(features, y)):
        train_df = features.iloc[train_idx]
        test_df = features.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        n_pos_train = int(y_train.sum())
        n_pos_test = int(y_test.sum())
        if n_pos_train < _MIN_MANNWHITNEY_SAMPLES:
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

        # Stage 1 on train only
        s1_cols = stage1_prefilter(train_df, feature_cols)

        # Stage 2 on train only (with BH correction)
        s2_cols, _ = stage2_univariate(train_df, s1_cols, label_col)
        if not s2_cols:
            logging.warning(
                "Fold %d/%d: no features after stage 2 — skipping",
                fold_idx + 1,
                total_folds,
            )
            continue

        # Impute + scale fitted on train only
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(train_df[s2_cols]))
        x_test = scaler.transform(imputer.transform(test_df[s2_cols]))

        # Stage 3: L1 logistic regression on train
        model, selected, _ = _fit_elastic_net(x_train, y_train, s2_cols)

        # Predict on held-out test fold using the CV-tuned model
        y_proba = model.predict_proba(x_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            logging.warning(
                "Fold %d/%d: AUC computation failed", fold_idx + 1, total_folds
            )
            continue

        fold_aucs.append(auc)
        for feat in selected:
            feature_counter[feat] += 1

        # Store ROC curve data for plotting
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append({"fpr": fpr, "tpr": tpr, "auc": auc, "fold": fold_idx})

        logging.info(
            "Fold %d/%d: AUC=%.3f, %d features selected " "(S1: %d → S2: %d → S3: %d)",
            fold_idx + 1,
            total_folds,
            auc,
            len(selected),
            len(s1_cols),
            len(s2_cols),
            len(selected),
        )

    cv_mean = np.mean(fold_aucs) if fold_aucs else float("nan")
    cv_std = np.std(fold_aucs) if fold_aucs else float("nan")
    logging.info(
        "=== Nested CV: AUC = %.3f ± %.3f (%d folds completed) ===",
        cv_mean,
        cv_std,
        len(fold_aucs),
    )

    return {
        "fold_aucs": fold_aucs,
        "feature_counter": feature_counter,
        "roc_curves": roc_curves,
        "total_folds": total_folds,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }


# ── Visualizations ──────────────────────────────────────────────────────


def plot_roc_curves(
    roc_curves: list[dict], cv_mean: float, cv_std: float, outdir: Path
) -> None:
    """Plot per-fold ROC curves with mean AUC annotation."""
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    for rc in roc_curves:
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
    ax.set_title(
        f"Nested CV ROC Curves\nMean AUC = {cv_mean:.3f} ± {cv_std:.3f}",
        fontsize=12,
    )
    if len(roc_curves) <= _MAX_ROC_LEGEND_FOLDS:
        ax.legend(fontsize=6, loc="lower right", ncol=2)
    fig.savefig(outdir / "nested_cv_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", outdir / "nested_cv_roc_curves.png")


def plot_auc_distribution(fold_aucs: list[float], outdir: Path) -> None:
    """Box + strip plot of per-fold AUC scores."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.boxplot(
        fold_aucs,
        vert=True,
        widths=0.4,
        patch_artist=True,
        boxprops={"facecolor": "#D5E8D4", "edgecolor": "#333"},
        medianprops={"color": "#E1812C", "linewidth": 2},
    )
    ax.scatter(
        np.ones(len(fold_aucs)), fold_aucs, alpha=0.5, color="#3274A1", zorder=3, s=30
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_ylabel("ROC AUC")
    ax.set_title(
        f"Nested CV AUC Distribution (n={len(fold_aucs)} folds)\n"
        f"Mean = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}",
        fontsize=11,
    )
    ax.set_xticks([])
    ax.legend(fontsize=9)
    fig.savefig(outdir / "nested_cv_auc_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", outdir / "nested_cv_auc_distribution.png")


def plot_feature_stability(
    feature_counter: Counter, total_folds: int, outdir: Path
) -> None:
    """Horizontal bar chart of feature selection frequency across folds."""
    if not feature_counter:
        logging.warning("No features selected across folds — skipping stability plot")
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
        f"Feature Selection Stability (top {top_n})\n"
        f"Across {total_folds} nested CV folds",
        fontsize=11,
    )
    ax.axvline(
        x=0.5, color="#E1812C", linestyle="--", linewidth=1, label="50% threshold"
    )
    ax.set_xlim(0, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=9)
    fig.savefig(outdir / "feature_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", outdir / "feature_stability.png")


def plot_pvalue_comparison(univariate_df: pd.DataFrame, outdir: Path) -> None:
    """Scatter plot comparing raw vs BH-adjusted p-values."""
    if "p_adjusted" not in univariate_df.columns or univariate_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    raw = univariate_df["p_value"].to_numpy()
    adj = univariate_df["p_adjusted"].to_numpy()

    passed = adj < _UNIVARIATE_P_THRESHOLD
    ax.scatter(
        raw[~passed],
        adj[~passed],
        alpha=0.4,
        s=15,
        color="#999999",
        label=f"Not significant (n={int((~passed).sum())})",
    )
    ax.scatter(
        raw[passed],
        adj[passed],
        alpha=0.6,
        s=25,
        color="#E1812C",
        label=f"Significant after BH (n={int(passed.sum())})",
    )

    lims = [0, max(raw.max(), adj.max()) * 1.05]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8, label="y = x")
    ax.axhline(
        y=_UNIVARIATE_P_THRESHOLD,
        color="#3274A1",
        linestyle=":",
        linewidth=1,
        label=f"α = {_UNIVARIATE_P_THRESHOLD}",
    )
    ax.set_xlabel("Raw p-value")
    ax.set_ylabel("BH-adjusted p-value")
    ax.set_title(
        "Effect of Benjamini-Hochberg FDR Correction\n"
        f"{int(passed.sum())} / {len(raw)} features survive correction",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.savefig(outdir / "pvalue_raw_vs_adjusted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", outdir / "pvalue_raw_vs_adjusted.png")


def plot_final_coefficients(
    coef_df: pd.DataFrame,
    cv_mean: float,
    cv_std: float,
    outdir: Path,
) -> None:
    """Horizontal bar chart of final model coefficients."""
    selected = coef_df[coef_df["coefficient"].abs() > 0].copy()
    if selected.empty:
        logging.warning("No non-zero coefficients — skipping coefficient plot")
        return

    top_n = min(30, len(selected))
    top = selected.head(top_n).copy()
    top = top.sort_values("coefficient", key=abs)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)), constrained_layout=True)
    colors = ["#E1812C" if c > 0 else "#3274A1" for c in top["coefficient"]]
    short_names = [_shorten_feature_name(f) for f in top["feature"]]
    ax.barh(short_names, top["coefficient"], color=colors)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title(
        f"Final Model: Top {top_n} Selected Features (Elastic Net)\n"
        f"Nested CV AUC: {cv_mean:.3f} ± {cv_std:.3f} | "
        f"{len(selected)} features selected\n"
        "(coefficients from full-data refit — not an unbiased estimate)",
        fontsize=10,
    )
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.tick_params(labelsize=8)
    fig.savefig(outdir / "lasso_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", outdir / "lasso_coefficients.png")


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
        help="Output directory (default: experiments/feature_selection)",
    )
    ap.add_argument("--label-col", type=str, default="pcr", help="Label column name")
    args = ap.parse_args()

    outdir = args.outdir or Path("experiments/feature_selection")
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

    # ── Step 1: Nested CV for unbiased performance estimation ──
    cv_results = run_nested_cv(features, feature_cols, args.label_col, outdir)

    # ── Step 2: Visualize nested CV results ──
    if cv_results["roc_curves"]:
        plot_roc_curves(
            cv_results["roc_curves"],
            cv_results["cv_mean"],
            cv_results["cv_std"],
            outdir,
        )
    if cv_results["fold_aucs"]:
        plot_auc_distribution(cv_results["fold_aucs"], outdir)
    plot_feature_stability(
        cv_results["feature_counter"], cv_results["total_folds"], outdir
    )

    # ── Step 3: Final model on all data (for definitive feature list) ──
    logging.info("=== Fitting final model on all data ===")
    stage1_cols = stage1_prefilter(features, feature_cols)
    stage2_cols, univariate_df = stage2_univariate(
        features, stage1_cols, args.label_col
    )
    univariate_df.to_csv(outdir / "univariate_pvalues.csv", index=False)

    # Visualize p-value correction
    plot_pvalue_comparison(univariate_df, outdir)

    if not stage2_cols:
        logging.error("No features survived — cannot fit final model")
        return

    # Impute + scale on all data for final model
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_all = scaler.fit_transform(imputer.fit_transform(features[stage2_cols]))
    y_all = features[args.label_col].to_numpy()

    _, selected_features, coef_df = _fit_elastic_net(x_all, y_all, stage2_cols)
    coef_df.to_csv(outdir / "lasso_coefficients.csv", index=False)

    # Save selected features
    selected_df = coef_df[coef_df["coefficient"].abs() > 0].copy()
    selected_df.to_csv(outdir / "selected_features.csv", index=False)

    # Visualize final coefficients
    plot_final_coefficients(
        coef_df, cv_results["cv_mean"], cv_results["cv_std"], outdir
    )

    # ── Summary ──
    logging.info("=== Feature Selection Summary ===")
    logging.info("  Input features: %d", len(feature_cols))
    logging.info("  After pre-filter (stage 1): %d", len(stage1_cols))
    logging.info("  After univariate screen (stage 2): %d", len(stage2_cols))
    logging.info("  After L1 logistic regression (stage 3): %d", len(selected_features))
    logging.info(
        "  Nested CV AUC: %.3f ± %.3f (unbiased estimate)",
        cv_results["cv_mean"],
        cv_results["cv_std"],
    )
    logging.info("Selected features saved to: %s", outdir / "selected_features.csv")
    logging.info(
        "Plots saved to: %s",
        ", ".join(
            str(outdir / f)
            for f in [
                "nested_cv_roc_curves.png",
                "nested_cv_auc_distribution.png",
                "feature_stability.png",
                "pvalue_raw_vs_adjusted.png",
                "lasso_coefficients.png",
            ]
        ),
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
