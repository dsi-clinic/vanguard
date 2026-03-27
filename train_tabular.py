"""MAMA-MIA pCR prediction pipeline using the evaluation framework.

Outputs:
- features_raw.csv
- features_engineered_labeled.csv
- evaluator metrics/predictions/plots under experiment output dir
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation import FoldResults
from evaluation.build_splits import create_splits_for_dataframe
from evaluation.kfold import FoldSplit
from load_cohort import (
    load_config,
    resolve_run_output_dir,
    write_config_snapshot,
)
from tabular_cohort import prepare_data
from tabular_models import (
    build_model_pipeline,
    log_feature_selector_stats,
    pick_nested_candidate_for_outer_fold,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="pCR prediction with evaluator")
    ap.add_argument(
        "--config",
        type=str,
        default="configs/ispy2.yaml",
        help="Path to YAML config",
    )
    ap.add_argument("--outdir", type=Path, help="Override output directory")
    return ap.parse_args()


def run_evaluation_pipeline(
    df: pd.DataFrame, config: dict[str, Any], outdir: Path
) -> None:
    """Run evaluator-based cross-validation over configured model/features."""
    context = prepare_evaluation_context(df, config)
    fold_results_list, nested_rows = run_cross_validation_from_context(context)

    if nested_rows:
        nested_df = pd.DataFrame(nested_rows)
        nested_df.to_csv(outdir / "nested_tuning_summary.csv", index=False)
        if not nested_df.empty and "inner_auc_mean" in nested_df.columns:
            best_per_fold = (
                nested_df.sort_values(
                    ["outer_fold", "inner_auc_mean", "feature_select_k_kin"],
                    ascending=[True, False, True],
                )
                .groupby("outer_fold", as_index=False)
                .head(1)
            )
            best_per_fold.to_csv(
                outdir / "nested_tuning_best_per_fold.csv", index=False
            )

    logging.info("Aggregating fold metrics...")
    kfold_results = context["evaluator"].aggregate_kfold_results(fold_results_list)

    logging.info("Saving evaluator outputs to: %s", outdir)
    context["evaluator"].save_results(kfold_results, outdir)

    print("\n" + "=" * 48)
    print(f"Plots saved in: {outdir / context['evaluator'].model_name / 'plots'}")
    print("=" * 48 + "\n")


def prepare_evaluation_context(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prepare evaluator inputs and deterministic fold splits for a config."""
    label_col = config.data_paths.label_column
    model_params = config.model_params
    toggles = config.feature_toggles
    model_type = str(model_params.model).lower()
    random_state = int(model_params.random_state)
    use_clinical_features = bool(toggles.use_clinical)
    feature_select_enabled = bool(model_params.feature_select_enabled)
    nested_tune_enabled = bool(model_params.nested_tune_enabled)

    y = df[label_col].astype(int)
    case_ids = df["case_id"]

    drop_cols = {
        "case_id",
        label_col,
        "has_centerline_file",
        "dataset",
        "bilateral",
        "tumor_subtype",
    }
    drop_cols.update({c for c in df.columns if "variant" in c and c != label_col})
    if not use_clinical_features:
        drop_cols.update(
            {
                "age",
                "menopausal_status",
                "breast_density",
                "site",
                "scanner_manufacturer",
                "scanner_model",
                "field_strength",
                "echo_time",
                "repetition_time",
            }
        )
    group_col = str(model_params.group_col)
    if bool(model_params.use_group_split):
        drop_cols.discard(group_col)
    stratum_col = model_params.stratum_col
    if stratum_col:
        drop_cols.add(str(stratum_col))

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if X.empty:
        raise ValueError("No feature columns remain after dropping ID/label columns.")

    categorical_cols = [
        c
        for c in X.columns
        if pd.api.types.is_object_dtype(X[c])
        or isinstance(X[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(X[c])
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    evaluator, splits, stratum_col = create_splits_for_dataframe(
        X=X,
        y=y,
        case_ids=case_ids,
        cohort_df=df,
        config=config,
        model_name=config.experiment_setup.name,
    )

    return {
        "config": config,
        "df": df,
        "label_col": label_col,
        "model_type": model_type,
        "random_state": random_state,
        "feature_select_enabled": feature_select_enabled,
        "nested_tune_enabled": nested_tune_enabled,
        "stratum_col": stratum_col,
        "X": X,
        "y": y,
        "case_ids": case_ids,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "evaluator": evaluator,
        "splits": splits,
    }


def run_single_fold_from_context(
    context: dict[str, Any],
    split: FoldSplit,
) -> tuple[FoldResults, list[dict[str, Any]], dict[str, Any] | None]:
    """Run one outer fold from a prepared evaluation context."""
    X = context["X"]
    y = context["y"]
    case_ids = context["case_ids"]
    cohort_df = context["df"]
    config = context["config"]
    model_type = context["model_type"]
    numeric_cols = context["numeric_cols"]
    categorical_cols = context["categorical_cols"]
    random_state = context["random_state"]
    nested_tune_enabled = context["nested_tune_enabled"]
    feature_select_enabled = context["feature_select_enabled"]
    stratum_col = context["stratum_col"]

    logging.info("Processing fold %d", split.fold_idx)
    train_idx = split.train_indices
    val_idx = split.val_indices

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model_params_override: dict[str, Any] | None = None
    nested_rows: list[dict[str, Any]] = []
    if nested_tune_enabled:
        model_params_override, nested_rows = pick_nested_candidate_for_outer_fold(
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            config=config,
            random_state=random_state,
            outer_fold_idx=split.fold_idx,
        )
        if model_params_override:
            logging.info(
                "Fold %d nested selected override: %s",
                split.fold_idx,
                model_params_override,
            )

    clf = build_model_pipeline(
        model_type=model_type,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        config=config,
        random_state=random_state,
        model_params_override=model_params_override,
    )
    clf.fit(X_train, y_train)
    log_feature_selector_stats(
        clf=clf,
        split=split,
        feature_select_enabled=feature_select_enabled,
        numeric_cols=numeric_cols,
    )

    y_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = clf.predict(X_val)
    val_pids = case_ids.iloc[val_idx].to_numpy()

    pred_df = pd.DataFrame(
        {
            "case_id": val_pids,
            "y_true": y_val.to_numpy(),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    )
    if stratum_col and stratum_col in cohort_df.columns:
        pred_df["stratum"] = cohort_df.iloc[val_idx][stratum_col].astype(str).to_numpy()

    return (
        FoldResults(fold_idx=split.fold_idx, predictions=pred_df),
        nested_rows,
        model_params_override,
    )


def run_cross_validation_from_context(
    context: dict[str, Any],
    *,
    selected_fold_indices: set[int] | None = None,
) -> tuple[list[FoldResults], list[dict[str, Any]]]:
    """Run cross-validation from a prepared context, optionally on selected folds."""
    splits = context["splits"]
    n_splits = len(splits)
    logging.info("Starting %d-fold cross-validation...", n_splits)
    fold_results_list: list[FoldResults] = []
    nested_rows: list[dict[str, Any]] = []
    for split in splits:
        if (
            selected_fold_indices is not None
            and split.fold_idx not in selected_fold_indices
        ):
            continue
        fold_result, fold_nested_rows, _ = run_single_fold_from_context(context, split)
        fold_results_list.append(fold_result)
        nested_rows.extend(fold_nested_rows)
    return fold_results_list, nested_rows


def run_pipeline_from_config(
    config: dict[str, Any],
    outdir: Path,
    *,
    config_source: Path | None = None,
) -> None:
    """Run the full feature-build + evaluation pipeline for a loaded config."""
    write_config_snapshot(config=config, outdir=outdir, config_source=config_source)

    try:
        merged_data = prepare_data(config, outdir)
        run_evaluation_pipeline(merged_data, config, outdir)
    except Exception as exc:  # noqa: BLE001
        logging.error("Pipeline failed: %s", exc, exc_info=True)


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    config = load_config(Path(args.config))
    outdir = resolve_run_output_dir(config=config, outdir_override=args.outdir)
    run_pipeline_from_config(config, outdir, config_source=Path(args.config))


if __name__ == "__main__":
    main()
