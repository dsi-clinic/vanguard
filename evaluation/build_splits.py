"""Shared split-building helpers for modeling entrypoints."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from evaluation import Evaluator
from evaluation.kfold import FoldSplit


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
    config, such as ``site`` or ``tumor_subtype``.
    """
    model_params = config.model_params
    random_state = int(model_params.random_state)
    use_group_split = bool(model_params.use_group_split)
    group_col = str(model_params.group_col)
    stratum_col = model_params.stratum_col

    evaluator = Evaluator(
        X=X,
        y=y,
        case_ids=case_ids,
        model_name=model_name,
        random_state=random_state,
    )

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
