"""Scaffold for a future graph neural network pCR training pipeline.

This file is intentionally a skeleton. It shows how a GNN runner should plug
into the shared evaluation framework without forcing the model into the tabular
``train_tabular.py`` path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation import FoldResults
from evaluation.build_splits import create_splits_for_dataframe
from evaluation.utils import prepare_predictions_df
from load_cohort import load_config, resolve_run_output_dir, write_config_snapshot

REQUIRED_MANIFEST_COLUMNS = (
    "case_id",
    "patient_id",
)
OPTIONAL_SPLIT_COLUMNS = (
    "site",
    "tumor_subtype",
    "dataset",
    "bilateral",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scaffold GNN pCR training entrypoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ispy2.yaml"),
        help="Path to YAML config. Copy configs/ispy2.yaml and add graph-specific paths.",
    )
    parser.add_argument("--outdir", type=Path, help="Override output directory")
    return parser.parse_args()


def describe_required_gnn_config() -> dict[str, str]:
    """Return the config keys students are expected to add for a GNN run."""
    return {
        "data_paths.graph_manifest_csv": (
            "CSV with one row per case or patient, labels, and paths to graph data."
        ),
        "data_paths.graph_data_root": (
            "Root directory containing serialized graph objects or per-case graph files."
        ),
        "model_params.batch_size": "Mini-batch size for graph training.",
        "model_params.epochs": "Maximum number of training epochs per fold.",
        "model_params.learning_rate": "Optimizer learning rate.",
    }


def validate_graph_manifest(manifest_df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Validate that the graph manifest contains the minimum required columns."""
    label_col = config["data_paths"]["label_column"]
    required_columns = set(REQUIRED_MANIFEST_COLUMNS) | {label_col}
    missing = sorted(required_columns.difference(manifest_df.columns))
    if missing:
        raise ValueError(
            "Graph manifest is missing required columns: "
            f"{missing}. Required columns are {sorted(required_columns)}."
        )

    if manifest_df["case_id"].isna().any():
        raise ValueError("Graph manifest has missing case_id values.")
    if manifest_df["patient_id"].isna().any():
        raise ValueError("Graph manifest has missing patient_id values.")


def load_graph_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Load the patient-level manifest needed for graph-based modeling.

    Implement this to return one row per patient/study with at
    least:
    - ``case_id``
    - label column from ``config['data_paths']['label_column']``
    - any split metadata columns used by the config, such as ``site`` or
      ``tumor_subtype``
    - paths or identifiers needed to construct graph objects
    """
    _ = describe_required_gnn_config()
    raise NotImplementedError(
        "Implement graph manifest loading for the GNN pipeline."
    )


def build_graph_dataset(manifest_df: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Build graph objects or dataset wrappers from the manifest.

    This can return any object you want here: a list of PyG Data objects,
    a DGL dataset, or a custom dataset wrapper. The evaluator does not depend on
    this object directly.
    """
    raise NotImplementedError("Implement graph dataset construction for the GNN.")


def build_fold_prediction_table(
    *,
    fold_patient_ids: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    fold_idx: int,
    manifest_df: pd.DataFrame,
    stratum_col: str | None,
) -> pd.DataFrame:
    """Build the evaluator-ready prediction table for one fold.

    Students should call this after their model-specific training loop returns
    validation predictions.
    """
    pred_df = prepare_predictions_df(
        patient_ids=fold_patient_ids,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fold=fold_idx,
    )
    if stratum_col and stratum_col in manifest_df.columns:
        pred_df["stratum"] = (
            manifest_df.set_index("case_id")
            .loc[pred_df["patient_id"].astype(str), stratum_col]
            .astype(str)
            .to_numpy()
        )
    return pred_df


def fit_predict_one_fold(
    *,
    dataset: Any,
    manifest_df: pd.DataFrame,
    split,
    config: dict[str, Any],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Train the GNN for one fold and return validation predictions.

    Expected return values:
    - ``patient_ids``
    - ``y_true``
    - ``y_pred``
    - ``y_prob``

    Expected implementation pattern:
    1. build train/validation subsets from ``split.train_indices`` and ``split.val_indices``
    2. construct dataloaders from ``dataset``
    3. train the GNN on the training subset
    4. run validation inference
    5. return predictions aligned to the validation cases
    """
    raise NotImplementedError("Implement one-fold GNN training and prediction.")


def run_gnn_pipeline(config: dict[str, Any], outdir: Path) -> None:
    """Run the scaffolded GNN training/evaluation pipeline."""
    manifest_df = load_graph_manifest(config)
    validate_graph_manifest(manifest_df, config)
    label_col = config["data_paths"]["label_column"]

    dataset = build_graph_dataset(manifest_df, config)
    y = manifest_df[label_col].astype(int)
    patient_ids = manifest_df["patient_id"].astype(str)

    # The evaluator only needs an aligned sample table for splits and output
    # bookkeeping. It does not care that the actual model consumes graph objects.
    split_frame = manifest_df.copy()
    evaluator, splits, stratum_col = create_splits_for_dataframe(
        X=split_frame[["case_id"]].copy(),
        y=y,
        patient_ids=patient_ids,
        cohort_df=split_frame,
        config=config,
        model_name=config["experiment_setup"].get("name", "gnn_model"),
    )

    fold_results: list[FoldResults] = []
    for split in splits:
        fold_patient_ids, y_true, y_pred, y_prob = fit_predict_one_fold(
            dataset=dataset,
            manifest_df=manifest_df,
            split=split,
            config=config,
        )
        pred_df = build_fold_prediction_table(
            fold_patient_ids=fold_patient_ids,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            fold_idx=split.fold_idx,
            manifest_df=manifest_df,
            stratum_col=stratum_col,
        )
        fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))

    kfold_results = evaluator.aggregate_kfold_results(fold_results)
    evaluator.save_results(kfold_results, outdir)


def main() -> None:
    """Entry point for the GNN template.

    This script is expected to fail with ``NotImplementedError`` until students
    fill in the data-loading and fold-training hooks above.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    outdir = resolve_run_output_dir(config=config, outdir_override=args.outdir)
    write_config_snapshot(config=config, outdir=outdir, config_source=args.config)
    run_gnn_pipeline(config, outdir)


if __name__ == "__main__":
    main()
