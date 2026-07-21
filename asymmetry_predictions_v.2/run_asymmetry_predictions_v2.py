#!/usr/bin/env python3
"""Build and evaluate asymmetry_predictions_v.2.

Model naming convention:

- asymmetry_predictions_v.1: original all-feature vessel asymmetry model,
  previously reported as ``logreg_all``.
- asymmetry_predictions_v.2: nuisance-filtered vessel asymmetry model built by
  this script.

This is the PR-ready selected asymmetry feature update:

- start from the shared 273-case DUKE tumor/vessel comparison input table
- keep vessel/asymmetry features only
- compute each feature's maximum absolute Spearman correlation to nuisance
  variables: xy spacing, z spacing, and tumor size
- remove features with max nuisance |r| >= 0.25
- train the same balanced logistic pCR model used in the earlier comparisons

Outputs are written next to this script by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import Evaluator, FoldResults

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "vessel_tumor_comparisons"
    / "shared_273_case_comparison"
    / "inputs"
    / "tumor_size_plus_vessel_asymmetry.csv"
)
DEFAULT_SPACING = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/outputs/data_viz/spacing_by_hospital.csv"
)
DEFAULT_OUTDIR = THIS_DIR
MODEL_NAME = "asymmetry_predictions_v.2"
PREVIOUS_MODEL_NAME = "asymmetry_predictions_v.1"
RANDOM_STATE = 42
NUISANCE_THRESHOLD = 0.25
MIN_FEATURE_COMPLETENESS = 0.75
MIN_UNIQUE_VALUES = 2
N_SPLITS = 5
PREDICTION_THRESHOLD = 0.5
NUISANCE_COLS = ("xy_spacing_mm", "z_spacing_mm", "tumor_size_tumor_voxels")
VESSEL_PREFIXES = (
    "ipsilateral_",
    "contralateral_",
    "ic_diff_",
    "ic_abs_asymmetry_",
    "log2_ic_ratio_",
    "ic_ratio_",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the v2 asymmetry model run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--spacing", type=Path, default=DEFAULT_SPACING)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_input(input_path: Path, spacing_path: Path) -> pd.DataFrame:
    """Load the pCR feature table and merge scanner spacing nuisance variables."""
    feature_table = pd.read_csv(input_path)
    feature_table["case_id"] = feature_table["case_id"].astype(str)
    labeled_table = feature_table[feature_table["pcr"].isin([0, 1, 0.0, 1.0])].copy()

    spacing = pd.read_csv(spacing_path).rename(columns={"patient_id": "case_id"})
    spacing["case_id"] = spacing["case_id"].astype(str)
    keep = [
        col for col in ["case_id", "xy_spacing_mm", "z_spacing_mm"] if col in spacing
    ]
    return labeled_table.merge(
        spacing[keep], on="case_id", how="left", validate="one_to_one"
    )


def is_vessel_feature(column: str) -> bool:
    """Return whether a column is one of the vessel asymmetry predictors."""
    return column.startswith(VESSEL_PREFIXES)


def preprocess_vessel_features(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Build the numeric vessel feature matrix used before nuisance filtering."""
    columns = [col for col in feature_table.columns if is_vessel_feature(col)]
    features = feature_table[columns].apply(pd.to_numeric, errors="coerce")

    raw_ratio_cols = [col for col in features.columns if col.startswith("ic_ratio_")]
    for col in raw_ratio_cols:
        values = features[col].where(features[col] > 0.0)
        features[f"log2_{col}"] = np.log2(values)
    if raw_ratio_cols:
        features = features.drop(columns=raw_ratio_cols)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.loc[
        :, features.notna().mean(axis=0) >= MIN_FEATURE_COMPLETENESS
    ]
    non_constant = features.apply(
        lambda column: len(column.dropna().unique()) >= MIN_UNIQUE_VALUES
    )
    return features.loc[:, non_constant]


def spearman(x: pd.Series, y: pd.Series, min_n: int = 20) -> tuple[float, float, int]:
    """Compute Spearman correlation after dropping invalid paired values."""
    paired = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(paired))
    if (
        n < min_n
        or paired["x"].nunique() < MIN_UNIQUE_VALUES
        or paired["y"].nunique() < MIN_UNIQUE_VALUES
    ):
        return np.nan, np.nan, n
    result = stats.spearmanr(paired["x"], paired["y"])
    return float(result.statistic), float(result.pvalue), n


def nuisance_scores(
    feature_table: pd.DataFrame, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate feature correlations to configured nuisance variables."""
    rows = []
    for feature in features.columns:
        values = pd.to_numeric(features[feature], errors="coerce")
        abs_rs = []
        for target in NUISANCE_COLS:
            target_values = pd.to_numeric(feature_table[target], errors="coerce")
            r, p, n = spearman(values, target_values)
            abs_rs.append(abs(r) if np.isfinite(r) else np.nan)
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "spearman_r": r,
                    "p_value": p,
                    "n": n,
                }
            )
        rows.append(
            {
                "feature": feature,
                "target": "max_abs_nuisance_r",
                "spearman_r": float(np.nanmax(abs_rs)),
                "p_value": np.nan,
                "n": int(values.notna().sum()),
            }
        )
    long_df = pd.DataFrame(rows)
    max_df = (
        long_df[long_df["target"] == "max_abs_nuisance_r"]
        .rename(columns={"spearman_r": "max_abs_nuisance_r"})[
            ["feature", "max_abs_nuisance_r", "n"]
        ]
        .sort_values("max_abs_nuisance_r", ascending=False)
    )
    return long_df, max_df


def make_model() -> Pipeline:
    """Create the balanced logistic regression pipeline for pCR prediction."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def run_cv(
    features: pd.DataFrame, y: pd.Series, case_ids: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 5-fold cross-validation and return metrics plus out-of-fold predictions."""
    evaluator = Evaluator(
        X=features,
        y=y,
        case_ids=case_ids,
        model_name=MODEL_NAME,
        random_state=RANDOM_STATE,
    )
    splits = evaluator.create_kfold_splits(n_splits=N_SPLITS)
    model = make_model()
    fold_results: list[FoldResults] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for split in splits:
        clf = clone(model)
        train_idx = split.train_indices
        test_idx = split.val_indices
        clf.fit(features.iloc[train_idx], y.iloc[train_idx])
        y_prob = clf.predict_proba(features.iloc[test_idx])[:, 1]
        y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)
        y_test = y.iloc[test_idx].to_numpy()
        pred_df = pd.DataFrame(
            {
                "case_id": case_ids.iloc[test_idx].to_numpy(),
                "y_true": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }
        )
        metrics = evaluator.compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
        fold_results.append(FoldResults(split.fold_idx, pred_df, metrics))
        metric_rows.append(
            {
                "fold": split.fold_idx,
                "n_test": int(len(test_idx)),
                "auc": metrics.get("auc", float("nan")),
                "average_precision": metrics.get("ap", float("nan")),
            }
        )
        for row in pred_df.itertuples(index=False):
            prediction_rows.append(
                {
                    "fold": split.fold_idx,
                    "case_id": row.case_id,
                    "y_true": int(row.y_true),
                    "y_prob": float(row.y_prob),
                    "y_pred": int(row.y_pred),
                }
            )

    results = evaluator.aggregate_kfold_results(fold_results)
    for stat_name in ["mean", "std"]:
        metric_rows.append(
            {
                "fold": stat_name,
                "n_test": int(len(y)),
                "auc": results.aggregated_metrics["auc"].get(stat_name, float("nan")),
                "average_precision": results.aggregated_metrics["ap"].get(
                    stat_name, float("nan")
                ),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def plot_auc(metrics: pd.DataFrame, out_path: Path) -> None:
    """Save a compact AUC summary plot for the v2 pCR model."""
    fold_df = metrics[~metrics["fold"].astype(str).isin(["mean", "std"])].copy()
    mean_row = metrics[metrics["fold"].astype(str) == "mean"].iloc[0]
    std_row = metrics[metrics["fold"].astype(str) == "std"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.bar([0], [mean_row["auc"]], yerr=[std_row["auc"]], color="#4c78a8", capsize=5)
    ax.scatter(
        np.linspace(-0.08, 0.08, len(fold_df)),
        fold_df["auc"],
        color="black",
        s=24,
        zorder=3,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0], ["asymmetry\npredictions v.2"])
    ax.set_ylim(0.0, 0.8)
    ax.set_ylabel("5-fold mean ROC AUC")
    ax.set_title("asymmetry_predictions_v.2 pCR model")
    ax.text(0, float(mean_row["auc"]) + 0.035, f"{mean_row['auc']:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_nuisance_heatmap(
    before_scores: pd.DataFrame,
    after_scores: pd.DataFrame,
    removed: pd.DataFrame,
    out_path: Path,
) -> None:
    """Save the before/after nuisance-correlation heatmap."""
    before = before_scores[before_scores["target"].isin(NUISANCE_COLS)]
    after = after_scores[after_scores["target"].isin(NUISANCE_COLS)]
    before_matrix = before.pivot_table(
        index="feature",
        columns="target",
        values="spearman_r",
        aggfunc="first",
    )
    after_matrix = after.pivot_table(
        index="feature",
        columns="target",
        values="spearman_r",
        aggfunc="first",
    )
    max_abs_before = before_matrix.abs().max(axis=1).rename("max_abs_before")
    top_features = max_abs_before.sort_values(ascending=False).head(35).index.tolist()

    heatmap_df = pd.DataFrame(index=top_features)
    for target in NUISANCE_COLS:
        heatmap_df[f"Before {target}"] = before_matrix.reindex(top_features)[target]
        heatmap_df[f"After {target}"] = (
            after_matrix.reindex(top_features)[target].fillna(0.0)
            if target in after_matrix.columns
            else 0.0
        )
    heatmap_df = heatmap_df.rename(
        columns={
            "Before xy_spacing_mm": "Before xy",
            "After xy_spacing_mm": "After xy",
            "Before z_spacing_mm": "Before z",
            "After z_spacing_mm": "After z",
            "Before tumor_size_tumor_voxels": "Before tumor size",
            "After tumor_size_tumor_voxels": "After tumor size",
        }
    )
    removed_features = set(removed["feature"])
    yticklabels = [
        f"{feature} (removed)" if feature in removed_features else feature
        for feature in heatmap_df.index
    ]

    fig, ax = plt.subplots(figsize=(9, max(8, 0.32 * len(heatmap_df))))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Spearman r"},
        ax=ax,
    )
    ax.set_title("Nuisance correlations before vs after filtering")
    ax.set_xlabel("")
    ax.set_ylabel("Top pre-filter nuisance-sensitive features")
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(outdir: Path, metadata: dict[str, Any], metrics: pd.DataFrame) -> None:
    """Write a short provenance README describing the v2 model artifacts."""
    mean_row = metrics[metrics["fold"].astype(str) == "mean"].iloc[0]
    std_row = metrics[metrics["fold"].astype(str) == "std"].iloc[0]
    lines = [
        "# asymmetry_predictions_v.2",
        "",
        "`asymmetry_predictions_v.2` is the selected nuisance-filtered DUKE",
        "vessel/asymmetry pCR model.",
        "",
        "`asymmetry_predictions_v.1` refers to the original all-feature vessel",
        "asymmetry model previously reported as `logreg_all`.",
        "",
        "## Selection Rule",
        "",
        f"Remove vessel features with max absolute Spearman correlation >= `{NUISANCE_THRESHOLD}`",
        "to any of: `xy_spacing_mm`, `z_spacing_mm`, `tumor_size_tumor_voxels`.",
        "",
        "## Result",
        "",
        f"- Cases: `{metadata['n_labeled_cases']}`",
        f"- Features kept: `{metadata['n_features_kept']}`",
        f"- Features removed: `{metadata['n_features_removed']}`",
        f"- Mean AUC: `{mean_row['auc']:.3f} +/- {std_row['auc']:.3f}`",
        f"- Mean AP: `{mean_row['average_precision']:.3f} +/- {std_row['average_precision']:.3f}`",
        f"- Max nuisance |r| after filtering: `{metadata['max_abs_nuisance_r_after_filtering']:.3f}`",
        "",
        "## Files",
        "",
        "- `run_asymmetry_predictions_v2.py`: build/evaluate script.",
        "- `asymmetry_predictions_v2_features.csv`: filtered model input table.",
        "- `removed_nuisance_features.csv`: removed features and nuisance scores.",
        "- `all_feature_nuisance_scores.csv`: full nuisance audit.",
        "- `filtered_feature_nuisance_scores.csv`: nuisance audit after filtering.",
        "- `cv_metrics.csv`: fold-level and mean/std pCR metrics.",
        "- `oof_predictions.csv`: out-of-fold predictions.",
        "- `auc_asymmetry_predictions_v2.png`: AUC plot.",
        "- `nuisance_before_after_heatmap.png`: nuisance before/after heatmap.",
        "- `asymmetry_predictions_v1_nuisance_heatmap.png`: nuisance heatmap for",
        "  `asymmetry_predictions_v.1`.",
        "- `asymmetry_predictions_v2_nuisance_heatmap.png`: matching nuisance heatmap",
        "  for `asymmetry_predictions_v.2`.",
        "- `run_metadata.json`: exact paths and feature counts.",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run feature filtering, model evaluation, and artifact export."""
    args = parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.overwrite:
        allowed_existing = {Path(__file__).resolve()}
        existing = [
            p for p in args.outdir.iterdir() if p.resolve() not in allowed_existing
        ]
        if existing:
            raise FileExistsError(
                f"Output directory exists and is non-empty: {args.outdir}"
            )
    args.outdir.mkdir(parents=True, exist_ok=True)

    feature_table = load_input(args.input, args.spacing)
    y = feature_table["pcr"].astype(int)
    case_ids = feature_table["case_id"].astype(str)
    full_features = preprocess_vessel_features(feature_table)
    score_long, score_max = nuisance_scores(feature_table, full_features)
    removed = score_max[score_max["max_abs_nuisance_r"] >= NUISANCE_THRESHOLD].copy()
    kept_cols = [
        col for col in full_features.columns if col not in set(removed["feature"])
    ]
    updated_features = full_features[kept_cols].copy()

    metrics, predictions = run_cv(updated_features, y, case_ids)
    updated_table = pd.concat(
        [
            feature_table[["case_id", "dataset", "pcr"]].reset_index(drop=True),
            updated_features.reset_index(drop=True),
        ],
        axis=1,
    )

    updated_table.to_csv(
        args.outdir / "asymmetry_predictions_v2_features.csv", index=False
    )
    removed.to_csv(args.outdir / "removed_nuisance_features.csv", index=False)
    score_long.to_csv(args.outdir / "all_feature_nuisance_scores.csv", index=False)
    metrics.to_csv(args.outdir / "cv_metrics.csv", index=False)
    predictions.to_csv(args.outdir / "oof_predictions.csv", index=False)
    plot_auc(metrics, args.outdir / "auc_asymmetry_predictions_v2.png")

    after_score_long, after_score_max = nuisance_scores(feature_table, updated_features)
    after_score_long.to_csv(
        args.outdir / "filtered_feature_nuisance_scores.csv", index=False
    )
    plot_nuisance_heatmap(
        score_long,
        after_score_long,
        removed,
        args.outdir / "nuisance_before_after_heatmap.png",
    )
    metadata = {
        "input": str(args.input),
        "spacing": str(args.spacing),
        "outdir": str(args.outdir),
        "model_name": MODEL_NAME,
        "previous_model_name": PREVIOUS_MODEL_NAME,
        "previous_model_definition": "Original all-feature vessel asymmetry logreg_all model.",
        "nuisance_threshold": NUISANCE_THRESHOLD,
        "n_labeled_cases": int(len(feature_table)),
        "n_pcr_positive": int(y.sum()),
        "n_pcr_negative": int((y == 0).sum()),
        "n_features_original": int(full_features.shape[1]),
        "n_features_kept": int(updated_features.shape[1]),
        "n_features_removed": int(len(removed)),
        "max_abs_nuisance_r_after_filtering": float(
            after_score_max["max_abs_nuisance_r"].max()
        ),
        "kept_features": kept_cols,
        "removed_features": list(removed["feature"]),
    }
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(args.outdir, metadata, metrics)

    print(f"Wrote {MODEL_NAME} artifacts to {args.outdir}")
    print(metrics.to_string(index=False))
    print(
        f"Removed {len(removed)} features; kept {updated_features.shape[1]} features."
    )


if __name__ == "__main__":
    main()
