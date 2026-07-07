"""Shared split-building helpers for modeling entrypoints."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from evaluation.evaluator import Evaluator
from evaluation.kfold import FoldSplit, create_predefined_splits


def create_splits_for_dataframe(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    case_ids: pd.Series | np.ndarray,
    cohort_df: pd.DataFrame,
    config: dict[str, Any],
    model_name: str,
) -> tuple[Evaluator, list[FoldSplit], str | None]:
    """Create an evaluator and configured splits from a cohort dataframe.

    ``cohort_df`` should contain any group or stratum columns referenced by the
    config, such as ``site`` or ``tumor_subtype``. When
    ``model_params.split_mode == "predefined"``, splits instead come from a
    fixed fold column (``model_params.split_col``) already present on
    ``cohort_df`` -- one fold per unique value, leave-one-fold-out -- so every
    model family can share the same fold definition.
    """
    model_params = config.model_params
    random_state = int(model_params.random_state)
    stratum_col = model_params.stratum_col

    evaluator = Evaluator(
        X=X,
        y=y,
        case_ids=case_ids,
        model_name=model_name,
        random_state=random_state,
    )

    if str(model_params.split_mode) == "predefined":
        split_col = str(model_params.split_col)
        if split_col not in cohort_df.columns:
            raise ValueError(
                f"split_mode='predefined' requires column {split_col!r} in "
                f"cohort_df; available columns: {sorted(cohort_df.columns)}"
            )
        split_dicts = create_predefined_splits(
            cohort_df[split_col].to_numpy(),
            case_ids=np.asarray(case_ids),
        )
        splits = [
            FoldSplit(
                fold_idx=d["fold_idx"],
                train_indices=d["train_indices"],
                val_indices=d["val_indices"],
                train_case_ids=d["train_case_ids"],
                val_case_ids=d["val_case_ids"],
            )
            for d in split_dicts
        ]
        logging.info(
            "Using predefined fold column '%s' (%d folds).", split_col, len(splits)
        )
        return evaluator, splits, str(stratum_col) if stratum_col else None

    use_group_split = bool(model_params.use_group_split)
    group_col = str(model_params.group_col)
    n_splits = int(model_params.n_splits)
    if use_group_split and group_col in cohort_df.columns:
        groups = cohort_df[group_col].fillna("UNKNOWN").astype(str).to_numpy()
        if stratum_col and str(stratum_col) in cohort_df.columns:
            stratify_labels = cohort_df[str(stratum_col)].astype(str).to_numpy()
        else:
            stratify_labels = y.to_numpy()

        splits = evaluator.create_kfold_splits(
            n_splits=n_splits,
            groups=groups,
            stratify_labels=stratify_labels,
            validate_exclusivity=True,
        )
        logging.info(
            "Using group-stratified k-fold with group column '%s' (%d splits).",
            group_col,
            n_splits,
        )
    else:
        splits = evaluator.create_kfold_splits(n_splits=n_splits)
        if use_group_split and group_col not in cohort_df.columns:
            logging.warning(
                "Requested group split on '%s' but column was not found. "
                "Falling back to standard stratified k-fold.",
                group_col,
            )

    return evaluator, splits, str(stratum_col) if stratum_col else None
