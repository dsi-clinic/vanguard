"""Shared split-building helpers for modeling entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.evaluator import Evaluator
from evaluation.kfold import FoldSplit


def _load_fixed_splits(
    fixed_splits_csv: str,
    case_ids: np.ndarray,
    n_splits: int,
) -> list[FoldSplit]:
    """Build FoldSplit objects from a pre-saved case_id → val_fold CSV.

    The CSV must have columns ``case_id`` and ``val_fold`` (integer 0..n_splits-1).
    Only patients present in both the CSV and ``case_ids`` are included.
    Patients in ``case_ids`` but absent from the CSV are excluded from all folds
    with a warning. This lets the native run operate on the intersection without
    raising errors when patient coverage differs slightly.

    Parameters
    ----------
    fixed_splits_csv : str
        Path to CSV with columns ``case_id``, ``val_fold``.
    case_ids : np.ndarray
        Ordered array of case IDs from the current dataset (same order as X/y).
    n_splits : int
        Expected number of folds. Used to enumerate fold indices.

    Returns
    -------
    list[FoldSplit]
        One FoldSplit per fold (0..n_splits-1), with train/val indices into
        the ``case_ids`` array.
    """
    splits_df = pd.read_csv(fixed_splits_csv, dtype={"case_id": str, "val_fold": int})
    split_map: dict[str, int] = dict(
        zip(splits_df["case_id"].astype(str), splits_df["val_fold"].astype(int))
    )

    case_ids_str = [str(cid) for cid in case_ids]
    missing = [cid for cid in case_ids_str if cid not in split_map]
    if missing:
        logging.warning(
            "fixed_splits_csv: %d patients in this dataset are not in the splits CSV "
            "and will be excluded from all folds. First 10: %s",
            len(missing),
            missing[:10],
        )

    valid_indices = [i for i, cid in enumerate(case_ids_str) if cid in split_map]
    logging.info(
        "fixed_splits_csv: using %d/%d patients present in %s",
        len(valid_indices),
        len(case_ids_str),
        fixed_splits_csv,
    )

    fold_splits: list[FoldSplit] = []
    for fold_k in range(n_splits):
        val_indices = np.array(
            [i for i in valid_indices if split_map[case_ids_str[i]] == fold_k],
            dtype=np.intp,
        )
        train_indices = np.array(
            [i for i in valid_indices if split_map[case_ids_str[i]] != fold_k],
            dtype=np.intp,
        )
        fold_splits.append(
            FoldSplit(
                fold_idx=fold_k,
                train_indices=train_indices,
                val_indices=val_indices,
                train_case_ids=case_ids[train_indices],
                val_case_ids=case_ids[val_indices],
            )
        )
        logging.info(
            "  Fold %d: %d train, %d val", fold_k, len(train_indices), len(val_indices)
        )

    return fold_splits


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

    If ``config.data_paths.fixed_splits_csv`` is set to a non-empty path, the
    splits are loaded from that CSV instead of being generated from scratch.
    The CSV must have columns ``case_id`` and ``val_fold``. This locks the
    train/test assignment so that runs with different input sources (e.g.
    resampled vs. native centerlines) are evaluated on the same folds.
    """
    model_params = config.model_params
    random_state = int(model_params.random_state)
    use_group_split = bool(model_params.use_group_split)
    group_col = str(model_params.group_col)
    stratum_col = model_params.stratum_col
    n_splits = int(model_params.n_splits)

    evaluator = Evaluator(
        X=X,
        y=y,
        case_ids=case_ids,
        model_name=model_name,
        random_state=random_state,
    )

    # Check for a pre-saved split assignment (locked-split mode).
    fixed_splits_csv: str = str(config.data_paths.get("fixed_splits_csv", "") or "")
    if fixed_splits_csv and Path(fixed_splits_csv).is_file():
        logging.info(
            "Loading fixed splits from %s (locked-split mode; "
            "random_state and group_split settings are ignored).",
            fixed_splits_csv,
        )
        case_ids_arr = np.asarray(case_ids).ravel()
        splits = _load_fixed_splits(fixed_splits_csv, case_ids_arr, n_splits)
        return evaluator, splits, str(stratum_col) if stratum_col else None

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
