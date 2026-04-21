#!/usr/bin/env python3
"""Feature selection pipeline for pCR prediction.

Three-stage pipeline:
  1. Pre-filter: drop all-NaN, near-constant, and highly correlated features
  2. Univariate screening: Mann-Whitney U test, keep p < 0.1
  3. L1-regularized logistic regression with cross-validated C

Usage::

    python scripts/feature_selection.py \
        experiments/second_order_features/features_engineered_labeled.csv \
        -o experiments/feature_selection
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
_UNIVARIATE_P_THRESHOLD = 0.1
_MIN_MANNWHITNEY_SAMPLES = 3
_NEAR_CONSTANT_STD = 1e-10
_MIN_COVERAGE_FRACTION = 0.5
_CV_FOLDS = 5
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
    label_col: str,
) -> list[str]:
    """Drop all-NaN, near-constant, low-coverage, and correlated features."""
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

    # Pass 2: remove one from each highly correlated pair
    corr_matrix = features[kept].corr().abs()
    to_drop: set[str] = set()
    for i in range(len(kept)):
        if kept[i] in to_drop:
            continue
        for j in range(i + 1, len(kept)):
            if kept[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > _CORRELATION_THRESHOLD:
                # Keep the one with lower p-value against label
                grp0 = features[features[label_col] == 0]
                grp1 = features[features[label_col] == 1]
                _, pi = stats.mannwhitneyu(
                    grp0[kept[i]].dropna(),
                    grp1[kept[i]].dropna(),
                    alternative="two-sided",
                )
                _, pj = stats.mannwhitneyu(
                    grp0[kept[j]].dropna(),
                    grp1[kept[j]].dropna(),
                    alternative="two-sided",
                )
                drop_col = kept[j] if pi <= pj else kept[i]
                to_drop.add(drop_col)
                logging.info(
                    "Stage 1 — corr %.3f: %s (p=%.3g) vs %s (p=%.3g) → drop %s",
                    corr_matrix.iloc[i, j],
                    kept[i],
                    pi,
                    kept[j],
                    pj,
                    drop_col,
                )

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
) -> tuple[list[str], pd.DataFrame]:
    """Univariate Mann-Whitney U screening, keep features with p < threshold."""
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
    kept_df = results_df[results_df["p_value"] < _UNIVARIATE_P_THRESHOLD]
    kept = list(kept_df["feature"])

    logging.info(
        "Stage 2 — univariate screening: %d → %d features (p < %s)",
        len(feature_cols),
        len(kept),
        _UNIVARIATE_P_THRESHOLD,
    )
    return kept, results_df


def stage3_lasso_lr(
    features: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    outdir: Path,
) -> tuple[list[str], pd.DataFrame]:
    """L1-regularized logistic regression with cross-validated C."""
    x_raw = features[feature_cols].copy()
    y = features[label_col].to_numpy()

    # Impute remaining NaNs with median, then standardize
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x_raw)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)

    model = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        Cs=20,
        cv=cv,
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_scaled, y)

    best_c = model.C_[0]
    logging.info("Stage 3 — best C=%.4g (from %d candidates)", best_c, 20)

    # Extract coefficients
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": coefs}).sort_values(
        "coefficient", key=abs, ascending=False
    )

    selected = coef_df[coef_df["coefficient"].abs() > 0]
    selected_features = list(selected["feature"])

    logging.info(
        "Stage 3 — L1 logistic regression: %d → %d features (non-zero coefs)",
        len(feature_cols),
        len(selected_features),
    )

    # Cross-validated performance
    y_proba = model.predict_proba(x_scaled)[:, 1]
    y_pred = model.predict(x_scaled)
    auc = roc_auc_score(y, y_proba)
    bal_acc = balanced_accuracy_score(y, y_pred)
    logging.info("Stage 3 — in-sample AUC=%.3f, balanced accuracy=%.3f", auc, bal_acc)

    # CV scores
    cv_scores = model.scores_[1]  # scores for class=1
    best_c_idx = list(model.Cs_).index(best_c)
    cv_mean = cv_scores[:, best_c_idx].mean()
    cv_std = cv_scores[:, best_c_idx].std()
    logging.info("Stage 3 — CV AUC: %.3f ± %.3f", cv_mean, cv_std)

    # Plot coefficient magnitudes
    top_n = min(30, len(selected))
    top = selected.head(top_n).copy()
    top["abs_coef"] = top["coefficient"].abs()
    top = top.sort_values("abs_coef")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)), constrained_layout=True)
    colors = ["#E1812C" if c > 0 else "#3274A1" for c in top["coefficient"]]
    short_names = [
        f.replace("tumor_size_", "ts_")
        .replace("kinematic_", "kin_")
        .replace("morph_", "m_")
        .replace("graph_", "g_")
        .replace("boundary_crossing_", "bc_")
        .replace("caliber_heterogeneity_", "cal_")
        .replace("per_shell_topology_", "pst_")
        .replace("directional_", "dir_")
        .replace("normalized_shell_burdens_", "nsb_")
        for f in top["feature"]
    ]
    ax.barh(short_names, top["coefficient"], color=colors)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title(
        f"Top {top_n} Selected Features (L1 Logistic Regression)\n"
        f"CV AUC: {cv_mean:.3f} ± {cv_std:.3f} | "
        f"Best C: {best_c:.4g} | "
        f"{len(selected_features)} features selected",
        fontsize=11,
    )
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.tick_params(labelsize=8)
    fig.savefig(outdir / "lasso_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return selected_features, coef_df


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

    # Stage 1
    stage1_cols = stage1_prefilter(features, feature_cols, args.label_col)

    # Stage 2
    stage2_cols, univariate_df = stage2_univariate(
        features, stage1_cols, args.label_col
    )
    univariate_df.to_csv(outdir / "univariate_pvalues.csv", index=False)

    # Stage 3
    selected_features, coef_df = stage3_lasso_lr(
        features, stage2_cols, args.label_col, outdir
    )
    coef_df.to_csv(outdir / "lasso_coefficients.csv", index=False)

    # Save selected features
    selected_df = coef_df[coef_df["coefficient"].abs() > 0].copy()
    selected_df.to_csv(outdir / "selected_features.csv", index=False)

    # Summary
    logging.info("=== Feature Selection Summary ===")
    logging.info("  Input features: %d", len(feature_cols))
    logging.info("  After pre-filter (stage 1): %d", len(stage1_cols))
    logging.info("  After univariate screen (stage 2): %d", len(stage2_cols))
    logging.info("  After L1 logistic regression (stage 3): %d", len(selected_features))
    logging.info("Selected features saved to: %s", outdir / "selected_features.csv")
    logging.info("Done.")


if __name__ == "__main__":
    main()
