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

import pandas as pd

from evaluation import FoldResults
from evaluation.build_splits import create_splits_for_dataframe
from evaluation.utils import prepare_predictions_df
from load_cohort import load_config, resolve_run_output_dir, write_config_snapshot


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scaffold GNN pCR training entrypoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ispy2.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--outdir", type=Path, help="Override output directory")
    return parser.parse_args()


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
    """
    raise NotImplementedError("Implement one-fold GNN training and prediction.")


def run_gnn_pipeline(config: dict[str, Any], outdir: Path) -> None:
    """Run the scaffolded GNN training/evaluation pipeline."""
    manifest_df = load_graph_manifest(config)
    label_col = config["data_paths"]["label_column"]

    if "case_id" not in manifest_df.columns:
        raise ValueError("Graph manifest must include a case_id column.")
    if label_col not in manifest_df.columns:
        raise ValueError(f"Graph manifest must include label column {label_col!r}.")

    dataset = build_graph_dataset(manifest_df, config)
    y = manifest_df[label_col].astype(int)
    patient_ids = manifest_df["case_id"].astype(str)

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
        pred_df = prepare_predictions_df(
            patient_ids=fold_patient_ids,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            fold=split.fold_idx,
        )
        if stratum_col and stratum_col in manifest_df.columns:
            pred_df["stratum"] = (
                manifest_df.set_index("case_id")
                .loc[pred_df["patient_id"].astype(str), stratum_col]
                .astype(str)
                .to_numpy()
            )
        fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))

    kfold_results = evaluator.aggregate_kfold_results(fold_results)
    evaluator.save_results(kfold_results, outdir)


def main() -> None:
    """Entry point."""
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
