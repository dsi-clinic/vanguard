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

from cohorts.base import DatasetAdapter
from cohorts.factory import (
    build_adapter_from_config,
    resolve_folds,
    resolve_split_policy,
)
from evaluation import FoldResults
from evaluation.build_splits import create_splits_for_dataframe
from evaluation.kfold import FoldSplit
from load_cohort import (
    load_config,
    resolve_run_output_dir,
    write_config_snapshot,
)
from tabular.cohort import prepare_data
from tabular.models import (
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
    df: pd.DataFrame,
    config: dict[str, Any],
    outdir: Path,
    report_by: str | None = None,
    group_col_from_adapter: bool = False,
) -> None:
    """Run evaluator-based cross-validation over configured model/features.

    ``report_by`` (a dataset adapter's ``report_by`` column, e.g. UChicago's
    ``dataset`` sub-source) drives the per-subgroup QC breakdown (Step 4); when
    ``None`` -- every non-adapter run -- behavior is unchanged.

    ``group_col_from_adapter`` must be ``True`` only when the caller actually
    populated ``model_params.group_col`` from an adapter identity key (via
    ``_apply_group_keys`` -- e.g. UChicago's ``patient_key``). See
    :func:`prepare_evaluation_context` for why this can't be inferred from
    ``config`` alone.
    """
    context = prepare_evaluation_context(
        df,
        config,
        report_by=report_by,
        group_col_from_adapter=group_col_from_adapter,
    )
    fold_results_list, nested_rows = run_cross_validation_from_context(context)

    # Nested tuning CSVs must live under the per-run directory (same leaf as metrics.json)
    # so concurrent ablation / matrix runs do not overwrite each other.
    run_subdir = outdir / str(config.experiment_setup.name)
    run_subdir.mkdir(parents=True, exist_ok=True)

    if nested_rows:
        nested_df = pd.DataFrame(nested_rows)
        nested_df.to_csv(run_subdir / "nested_tuning_summary.csv", index=False)
        if not nested_df.empty and "inner_auc_mean" in nested_df.columns:
            sort_cols = ["outer_fold", "inner_auc_mean"]
            ascending = [True, False]
            if "feature_select_k_kin" in nested_df.columns:
                sort_cols.append("feature_select_k_kin")
                ascending.append(True)
            best_per_fold = (
                nested_df.sort_values(
                    sort_cols,
                    ascending=ascending,
                )
                .groupby("outer_fold", as_index=False)
                .head(1)
            )
            best_per_fold.to_csv(
                run_subdir / "nested_tuning_best_per_fold.csv", index=False
            )

    logging.info("Aggregating fold metrics...")
    kfold_results = context["evaluator"].aggregate_kfold_results(fold_results_list)

    logging.info("Saving evaluator outputs to: %s", outdir)
    context["evaluator"].save_results(
        kfold_results, outdir, report_by=context.get("report_by")
    )

    print("\n" + "=" * 48)
    print(f"Plots saved in: {outdir / context['evaluator'].model_name / 'plots'}")
    print("=" * 48 + "\n")
    return kfold_results


def prepare_evaluation_context(
    df: pd.DataFrame,
    config: dict[str, Any],
    report_by: str | None = None,
    group_col_from_adapter: bool = False,
) -> dict[str, Any]:
    """Prepare evaluator inputs and deterministic fold splits for a config.

    ``report_by`` names an optional per-case subgroup column (a dataset adapter's
    ``report_by``, Step 4) attached to each fold's predictions for the QC
    breakdown; ``None`` (every non-adapter run) leaves predictions unchanged.
    When it *is* set, the column must exist in ``df``: a missing one raises
    rather than being skipped, since evaluation would otherwise fall back to
    reporting by a different column under the requested column's name.

    ``group_col_from_adapter`` must be ``True`` only when the caller populated
    ``model_params.group_col`` from an adapter identity key (``_apply_group_keys``
    -- e.g. UChicago's ``patient_key``), never merely because a dataset is
    configured: some callers (e.g. ``modeling/ablation.py``) build features with
    an adapter but never call ``_apply_group_keys``, so ``group_col`` there can be
    a genuine feature (e.g. MAMA-MIA's clinical ``site``), not an identity.
    Deciding the drop from ``config.dataset.name`` alone conflated those cases and
    silently stripped ``site`` from such runs; this flag is the caller's explicit
    signal instead.
    """
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
    if group_col_from_adapter:
        # group_col was populated from the adapter's identity key (e.g.
        # UChicago's patient_key) by _apply_group_keys -- always drop it so it
        # can't leak into X.
        drop_cols.add(group_col)
    elif bool(model_params.use_group_split):
        drop_cols.discard(group_col)
    stratum_col = model_params.stratum_col
    if stratum_col:
        drop_cols.add(str(stratum_col))
    # Eval bookkeeping, not a feature -- must not leak into X even when unset
    # (harmless to drop a column that isn't present).
    drop_cols.add(str(model_params.split_col))

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
        "report_by": report_by,
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
    report_by = context.get("report_by")

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
    # QC subgroup column from the dataset adapter's report_by (Step 4): attach it
    # so save_results can break metrics down by sub-source. None for non-adapter
    # runs, so predictions are unchanged.
    #
    # Missing here is fatal, and deliberately raised at *this* seam rather than
    # left to evaluation: the cohort frame is where the column actually went
    # missing, so this is the only place that can say so. Downstream,
    # evaluation._stratum_column would also raise -- but it can only see that
    # the column is absent from the predictions frame, which sends whoever is
    # debugging to the wrong file. Skipping silently is not an option either:
    # _stratum_column would then fall back to an alias and report the QC
    # breakdown by some other column while the run still looked like it had
    # produced the requested sub-source breakdown.
    if report_by and report_by not in pred_df.columns:
        if report_by not in cohort_df.columns:
            raise KeyError(
                f"The dataset adapter requested a QC breakdown by {report_by!r}, "
                f"but that column is not in the cohort frame (columns: "
                f"{sorted(cohort_df.columns)}). It must be attached upstream, "
                "where the cohort table is built -- see prepare_data() and the "
                "adapter's load_labels(). Refusing to run the fold without it, "
                "because evaluation would otherwise report by a different column."
            )
        pred_df[report_by] = cohort_df.iloc[val_idx][report_by].astype(str).to_numpy()

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


#: Cap on how many case ids to list in a fail-closed fold error before truncating.
_MAX_MISSING_CASES_SHOWN = 10


def _preview(case_ids: list[str]) -> str:
    """Format a truncated ``case_id`` list for error messages."""
    head = case_ids[:_MAX_MISSING_CASES_SHOWN]
    return f"{head}{' ...' if len(case_ids) > _MAX_MISSING_CASES_SHOWN else ''}"


def _apply_provided_folds(
    feats_df: pd.DataFrame, config: dict[str, Any], adapter: DatasetAdapter
) -> pd.DataFrame:
    """Attach the adapter's provided CV folds onto the feature table, fail-closed.

    ``resolve_folds`` returns a ``(case_id, fold)`` table when the resolved split
    policy is "provided", or ``None`` when it's "compute" -- in which case
    ``feats_df`` is returned unchanged. Otherwise the folds are attached under
    ``model_params.split_col`` (default ``"fold"``), consumed downstream by the
    predefined-fold path in ``create_splits_for_dataframe``.

    Because a bad fold attachment silently corrupts cross-validation (a missing
    fold becomes an empty validation split; a duplicate mapping puts one case in
    both train and validation), this validates rather than trusts, and raises on
    anything unsafe:

    - ``split_col`` must not already be a column in ``feats_df`` (it would
      silently drop and overwrite whatever that column held -- ``case_id`` and
      the label column included, but not limited to them: any real feature or
      metadata column of the same name is just as unsafe to clobber);
    - the provided folds must map each ``case_id`` at most once, and the merge is
      ``validate="one_to_one"`` so a duplicate on either side is an error, not a
      row-exploding leak;
    - every modeled case must receive exactly one non-null fold.

    Raises:
        ValueError: On a ``split_col`` collision, a duplicate fold mapping, a
            non-unique feature-table ``case_id``, or any modeled case left
            without a fold.
    """
    folds = resolve_folds(config, adapter)
    if folds is None:
        return feats_df

    split_col = str(config.model_params.split_col)
    if split_col in feats_df.columns:
        raise ValueError(
            f"model_params.split_col={split_col!r} collides with an existing "
            "column in the feature table; attaching provided folds under that "
            "name would silently drop and overwrite it. Choose a different "
            "model_params.split_col."
        )

    dup_folds = folds["case_id"][folds["case_id"].duplicated()].unique().tolist()
    if dup_folds:
        raise ValueError(
            "Provided folds map these case ids more than once: "
            f"{_preview([str(c) for c in dup_folds])}"
        )

    folds = folds.rename(columns={"fold": split_col})
    # validate="one_to_one": a duplicate case_id on either side raises MergeError
    # rather than fanning a case across folds.
    feats_df = feats_df.merge(folds, on="case_id", how="left", validate="one_to_one")

    missing = feats_df.loc[feats_df[split_col].isna(), "case_id"].tolist()
    if missing:
        raise ValueError(
            f"{len(missing)} modeled case(s) have no provided fold: "
            f"{_preview([str(c) for c in missing])}. Provided folds must cover every "
            "modeled case; use split_policy: compute to build splits instead."
        )
    return feats_df


def _apply_group_keys(
    feats_df: pd.DataFrame, config: dict[str, Any], adapter: DatasetAdapter
) -> pd.DataFrame:
    """Populate the split grouping column from the adapter, so computed CV groups.

    When splits are *computed* (``split_policy: compute``), the pipeline must
    keep a case's group together across folds -- for UChicago that is the patient
    (multiple exams per patient), via ``adapter.group_key``. Without this the
    grouping column is absent and ``create_splits_for_dataframe`` silently falls
    back to case-level CV, leaking same-patient exams across train/validation.

    Writes ``model_params.group_col`` from ``adapter.group_key(case_id)`` only
    when that column is not already present, so a dataset whose group column is
    supplied by an earlier stage (e.g. MAMA-MIA's clinical ``site``) is left
    untouched. Callers should invoke this only when :func:`_needs_group_keys` is
    true, and pass ``group_col_from_adapter=True`` to
    :func:`prepare_evaluation_context` **only when this actually wrote the column**
    (i.e. it was absent) -- that flag is what excludes the identity key from model
    features, so setting it for a pre-existing genuine feature would wrongly drop
    it.
    """
    group_col = str(config.model_params.group_col)
    if group_col in feats_df.columns:
        return feats_df
    feats_df = feats_df.copy()
    feats_df[group_col] = feats_df["case_id"].map(adapter.group_key)
    return feats_df


def _needs_group_keys(config: dict[str, Any], adapter: DatasetAdapter) -> bool:
    """Whether this run should populate the grouping column from the adapter.

    ``adapter.group_key(case_id)`` can require real I/O (e.g. ``MamaMiaDataset``
    reads clinical data per case), so only call it when the result will be used:
    the resolved split policy is ``"compute"`` (grouping only matters when the
    pipeline builds its own splits -- a "provided" run uses the shipped fold
    column instead, design decision 3 in cohorts/README.md) and
    ``model_params.use_group_split`` is on. Otherwise the column would be
    populated -- and then dropped -- for nothing, at the cost of an unnecessary,
    possibly-failing adapter call.
    """
    return resolve_split_policy(config, adapter) == "compute" and bool(
        config.model_params.use_group_split
    )


def _adapter_populates_group_col(
    merged_data: pd.DataFrame, config: dict[str, Any], adapter: DatasetAdapter
) -> bool:
    """Whether this run will fill ``group_col`` from the adapter's identity key.

    True only when grouping is needed (:func:`_needs_group_keys`) *and* the
    column is not already present as a genuine feature. This is exactly the
    condition under which :func:`_apply_group_keys` writes the column, so it also
    decides ``group_col_from_adapter`` -- setting that flag when the column
    pre-existed would make :func:`prepare_evaluation_context` drop a real feature
    (e.g. MAMA-MIA's clinical ``site``) that the adapter never touched.
    """
    if not _needs_group_keys(config, adapter):
        return False
    return str(config.model_params.group_col) not in merged_data.columns


def run_pipeline_from_config(
    config: dict[str, Any],
    outdir: Path,
    *,
    config_source: Path | None = None,
) -> None:
    """Run the full feature-build + evaluation pipeline for a loaded config."""
    write_config_snapshot(config=config, outdir=outdir, config_source=config_source)

    # Build the dataset adapter from run config (Step 2 of the multi-dataset
    # migration). Returns None for every config without a `dataset:` block, so
    # existing runs are unchanged; a configured dataset routes cohort identity
    # through the adapter. See cohorts/README.md.
    adapter = build_adapter_from_config(config)
    if adapter is not None:
        logging.info("Using dataset adapter: %s", type(adapter).__name__)

    report_by = adapter.report_by if adapter is not None else None
    group_col_from_adapter = False
    try:
        merged_data = prepare_data(config, outdir, adapter=adapter)
        if adapter is not None:
            merged_data = _apply_provided_folds(merged_data, config, adapter)
            # Populate the grouping column from the adapter only when it will be
            # used and isn't already a genuine feature; the flag mirrors that so
            # prepare_evaluation_context drops the identity key but never a real
            # pre-existing feature.
            group_col_from_adapter = _adapter_populates_group_col(
                merged_data, config, adapter
            )
            if group_col_from_adapter:
                merged_data = _apply_group_keys(merged_data, config, adapter)
        run_evaluation_pipeline(
            merged_data,
            config,
            outdir,
            report_by=report_by,
            group_col_from_adapter=group_col_from_adapter,
        )
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
