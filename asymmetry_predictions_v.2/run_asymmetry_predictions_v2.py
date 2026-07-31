#!/usr/bin/env python3
"""Build and evaluate asymmetry_predictions_v.2.

Model naming convention:

- asymmetry_predictions_v.1: original all-feature vessel asymmetry model,
  previously reported as ``logreg_all``.
- asymmetry_predictions_v.2: lower maximum marginal feature_confounder
  correlation vessel asymmetry model built by this script.

This is the PR-ready selected asymmetry feature update:

- start from the shared 273-case DUKE tumor/vessel comparison input table
- keep vessel/asymmetry features only
- compute each feature's maximum absolute Spearman correlation to audited
  variables: xy spacing, z spacing, and tumor size
- remove features with max audited covariate |r| >= 0.25
- train the same balanced logistic pCR model used in the earlier comparisons
- audit out-of-fold predicted probabilities against xy spacing, z spacing, and
  tumor size

Outputs are written next to this script by default.
"""

from __future__ import annotations

import argparse
import hashlib
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
SHARED_FEATURE_INPUT_COPY = Path(
    "/gpfs/data/karczmar-lab/workspaces/aakrithiram/"
    "asymmetry_predictions_v2_inputs/tumor_size_plus_vessel_asymmetry.csv"
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
TECHNICAL_CONFOUNDER_COLS = ("xy_spacing_mm", "z_spacing_mm")
BIOLOGICAL_COVARIATE_COLS = ("tumor_size_tumor_voxels",)
AUDITED_COVARIATE_COLS = TECHNICAL_CONFOUNDER_COLS + BIOLOGICAL_COVARIATE_COLS
NUISANCE_COLS = AUDITED_COVARIATE_COLS
VESSEL_PREFIXES = (
    "ipsilateral_",
    "contralateral_",
    "ic_diff_",
    "ic_abs_asymmetry_",
    "log2_ic_ratio_",
    "ic_ratio_",
)
RUN_COMMAND = (
    "micromamba run -n vanguard python "
    "asymmetry_predictions_v.2/run_asymmetry_predictions_v2.py --overwrite"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _covariate_category(column: str) -> str:
    if column in TECHNICAL_CONFOUNDER_COLS:
        return "technical_confounder"
    return "biological_clinical_covariate"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the v2 asymmetry model run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--spacing", type=Path, default=DEFAULT_SPACING)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_input(input_path: Path, spacing_path: Path) -> pd.DataFrame:
    """Load the pCR feature table and merge scanner spacing covariates."""
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
    """Build the numeric vessel feature matrix used before covariate filtering."""
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
    """Calculate feature correlations to configured audited covariates."""
    rows = []
    for feature in features.columns:
        values = pd.to_numeric(features[feature], errors="coerce")
        abs_rs = []
        for target in AUDITED_COVARIATE_COLS:
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


def model_covariate_audit(
    feature_table: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Correlate out-of-fold predicted pCR probability with audited covariates."""
    audit_table = predictions.merge(
        feature_table[["case_id", *AUDITED_COVARIATE_COLS]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for covariate in AUDITED_COVARIATE_COLS:
        r, p, n = spearman(
            audit_table["y_prob"],
            pd.to_numeric(audit_table[covariate], errors="coerce"),
        )
        rows.append(
            {
                "model_output": "oof_y_prob",
                "covariate": covariate,
                "covariate_category": _covariate_category(covariate),
                "spearman_r": r,
                "abs_spearman_r": abs(r) if np.isfinite(r) else np.nan,
                "p_value": p,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


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
    before = before_scores[before_scores["target"].isin(AUDITED_COVARIATE_COLS)]
    after = after_scores[after_scores["target"].isin(AUDITED_COVARIATE_COLS)]
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
    for target in AUDITED_COVARIATE_COLS:
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
    ax.set_title("Audited covariate correlations before vs after filtering")
    ax.set_xlabel("")
    ax.set_ylabel("Top pre-filter nuisance-sensitive features")
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(
    outdir: Path,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    model_audit: pd.DataFrame,
) -> None:
    """Write a short provenance README describing the v2 model artifacts."""
    mean_row = metrics[metrics["fold"].astype(str) == "mean"].iloc[0]
    std_row = metrics[metrics["fold"].astype(str) == "std"].iloc[0]
    xy_audit = model_audit[model_audit["covariate"] == "xy_spacing_mm"].iloc[0]
    z_audit = model_audit[model_audit["covariate"] == "z_spacing_mm"].iloc[0]
    tumor_audit = model_audit[
        model_audit["covariate"] == "tumor_size_tumor_voxels"
    ].iloc[0]
    lines = [
        "# asymmetry_predictions_v.2",
        "",
        "`asymmetry_predictions_v.2` is the selected DUKE vessel/asymmetry pCR",
        "model with lower maximum marginal feature_confounder correlation.",
        "",
        "`asymmetry_predictions_v.1` refers to the original all-feature vessel",
        "asymmetry model previously reported as `logreg_all`.",
        "",
        "## Selection Rule",
        "",
        f"Remove vessel features with max absolute Spearman correlation >= `{NUISANCE_THRESHOLD}`",
        "to any audited covariate. The technical confounders are `xy_spacing_mm`",
        "and `z_spacing_mm`. `tumor_size_tumor_voxels` is treated separately as a",
        "biological/clinical covariate used to test whether vessel signal is",
        "independent of tumor extent.",
        "",
        "This supports the narrower claim of lower maximum marginal",
        "feature_confounder correlation for retained features. It does not prove",
        "that the multivariate model has lower technical-confounder sensitivity or",
        "will generalize outside Duke.",
        "",
        "## Result",
        "",
        f"- Cases: `{metadata['n_labeled_cases']}`",
        f"- Features kept: `{metadata['n_features_kept']}`",
        f"- Features removed: `{metadata['n_features_removed']}`",
        f"- Mean AUC: `{mean_row['auc']:.3f} +/- {std_row['auc']:.3f}`",
        f"- Mean AP: `{mean_row['average_precision']:.3f} +/- {std_row['average_precision']:.3f}`",
        "- Max marginal feature/covariate |r| after filtering:",
        f"  `{metadata['max_abs_audited_covariate_r_after_filtering']:.3f}`",
        "- Max marginal feature/technical-confounder |r| after filtering:",
        f"  `{metadata['max_abs_technical_confounder_r_after_filtering']:.3f}`",
        "- Max marginal feature/tumor-size |r| after filtering:",
        f"  `{metadata['max_abs_biological_covariate_r_after_filtering']:.3f}`",
        "",
        "## OOF Model-Level Covariate Audit",
        "",
        "Out-of-fold predicted pCR probabilities were correlated with the audited",
        "covariates as a small model-level check.",
        "",
        f"- `y_prob` vs `xy_spacing_mm`: Spearman r `{xy_audit['spearman_r']:.3f}`",
        f"  (`n={int(xy_audit['n'])}`)",
        f"- `y_prob` vs `z_spacing_mm`: Spearman r `{z_audit['spearman_r']:.3f}`",
        f"  (`n={int(z_audit['n'])}`)",
        "- `y_prob` vs `tumor_size_tumor_voxels`",
        f"  (biological/clinical covariate): Spearman r `{tumor_audit['spearman_r']:.3f}`",
        f"  (`n={int(tumor_audit['n'])}`)",
        "",
        "See `oof_prediction_covariate_audit.csv` for p-values and absolute",
        "correlations.",
        "",
        "## Derived Inputs",
        "",
        "### Tumor Size Plus Vessel Asymmetry Feature Table",
        "",
        "- Contains: 273 shared DUKE cases with `case_id`, `dataset`, `pcr`, 11",
        "  tumor-size features, and 115 vessel/asymmetry features before v2",
        "  filtering.",
        "- Produced by: the shared 273-case DUKE tumor/vessel comparison workflow",
        "  documented in",
        "  `vessel_tumor_comparisons/shared_273_case_comparison/README.md`, using",
        "  `tabular/duke_final_vessel_asymmetry_features.csv` plus",
        "  `vessel_tumor_comparisons/duke_tumor_size_only_ablation/runs/tumor_size_only/features_engineered_labeled.csv`.",
        f"- Local source path: `{metadata['input']}`",
        f"- Durable shared copy: `{metadata['shared_feature_input_copy']}`",
        f"- SHA256: `{metadata['input_sha256']}`",
        "",
        "### Spacing Table",
        "",
        "- Contains: per-case scanner spacing covariates, including",
        "  `xy_spacing_mm` and `z_spacing_mm` after `patient_id` is renamed to",
        "  `case_id`.",
        "- Produced by: Sarit's data-visualization spacing export workflow.",
        f"- Shared path: `{metadata['spacing']}`",
        f"- SHA256: `{metadata['spacing_sha256']}`",
        "",
        "## Exact Command",
        "",
        "```bash",
        metadata["run_command"],
        "```",
        "",
        "## Files",
        "",
        "- `run_asymmetry_predictions_v2.py`: build/evaluate script.",
        "- `asymmetry_predictions_v2_features.csv`: filtered model input table.",
        "- `removed_nuisance_features.csv`: removed features and audited",
        "  covariate scores.",
        "- `all_feature_nuisance_scores.csv`: full feature/covariate audit.",
        "- `filtered_feature_nuisance_scores.csv`: feature/covariate audit after",
        "  filtering.",
        "- `cv_metrics.csv`: fold-level and mean/std pCR metrics.",
        "- `oof_predictions.csv`: out-of-fold predictions.",
        "- `oof_prediction_covariate_audit.csv`: model-level OOF y_prob",
        "  correlation audit against spacing and tumor size.",
        "- `auc_asymmetry_predictions_v2.png`: AUC plot.",
        "- `nuisance_before_after_heatmap.png`: audited covariate before/after",
        "  heatmap.",
        "- `asymmetry_predictions_v1_nuisance_heatmap.png`: covariate heatmap for",
        "  `asymmetry_predictions_v.1`; separately generated exploratory artifact.",
        "- `asymmetry_predictions_v2_nuisance_heatmap.png`: matching covariate heatmap",
        "  for `asymmetry_predictions_v.2`; separately generated exploratory artifact.",
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
    model_audit = model_covariate_audit(feature_table, predictions)
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
    model_audit.to_csv(args.outdir / "oof_prediction_covariate_audit.csv", index=False)
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
        "shared_feature_input_copy": str(SHARED_FEATURE_INPUT_COPY),
        "spacing": str(args.spacing),
        "outdir": str(args.outdir),
        "run_command": RUN_COMMAND,
        "model_name": MODEL_NAME,
        "previous_model_name": PREVIOUS_MODEL_NAME,
        "previous_model_definition": "Original all-feature vessel asymmetry logreg_all model.",
        "nuisance_threshold": NUISANCE_THRESHOLD,
        "input_sha256": _sha256(args.input),
        "shared_feature_input_copy_sha256": _sha256(SHARED_FEATURE_INPUT_COPY),
        "spacing_sha256": _sha256(args.spacing),
        "n_labeled_cases": int(len(feature_table)),
        "n_pcr_positive": int(y.sum()),
        "n_pcr_negative": int((y == 0).sum()),
        "n_features_original": int(full_features.shape[1]),
        "n_features_kept": int(updated_features.shape[1]),
        "n_features_removed": int(len(removed)),
        "max_abs_audited_covariate_r_after_filtering": float(
            after_score_max["max_abs_nuisance_r"].max()
        ),
        "max_abs_nuisance_r_after_filtering": float(
            after_score_max["max_abs_nuisance_r"].max()
        ),
        "max_abs_technical_confounder_r_after_filtering": float(
            after_score_long[
                after_score_long["target"].isin(TECHNICAL_CONFOUNDER_COLS)
            ]["spearman_r"]
            .abs()
            .max()
        ),
        "max_abs_biological_covariate_r_after_filtering": float(
            after_score_long[
                after_score_long["target"].isin(BIOLOGICAL_COVARIATE_COLS)
            ]["spearman_r"]
            .abs()
            .max()
        ),
        "oof_prediction_covariate_audit": model_audit.to_dict(orient="records"),
        "kept_features": kept_cols,
        "removed_features": list(removed["feature"]),
    }
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(args.outdir, metadata, metrics, model_audit)

    print(f"Wrote {MODEL_NAME} artifacts to {args.outdir}")
    print(metrics.to_string(index=False))
    print(
        f"Removed {len(removed)} features; kept {updated_features.shape[1]} features."
    )


if __name__ == "__main__":
    main()
