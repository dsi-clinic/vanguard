"""Run one ablation arm and one outer fold from a cached labeled feature table."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from config import load_config, to_plain_data
from run_ablation_matrix import _normalize_ablation_arms
from tabular_cohort import select_features
from train_tabular import prepare_evaluation_context, run_single_fold_from_context


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--arm-index", type=int, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()

    config = load_config(args.config)

    arms = _normalize_ablation_arms(config)
    if args.arm_index < 0 or args.arm_index >= len(arms):
        raise IndexError(f"arm_index out of range: {args.arm_index}")
    arm = arms[args.arm_index]
    arm_name = arm["name"]
    blocks = list(arm["selected_features"])

    full_df = pd.read_csv(args.features_csv)
    label_col = config["data_paths"]["label_column"]
    arm_df = select_features(
        full_df,
        selected_blocks=blocks,
        label_col=label_col,
    )

    arm_config = deepcopy(config)
    arm_config.setdefault("experiment_setup", {})["name"] = arm_name
    toggles = arm_config.setdefault("feature_toggles", {})
    toggles["selected_features"] = blocks
    toggles["use_clinical"] = "clinical" in blocks
    toggles["use_vascular"] = any(block != "clinical" for block in blocks)
    toggles.update(arm.get("feature_toggles_override", {}))
    arm_config.setdefault("model_params", {}).update(
        arm.get("model_params_override", {})
    )

    context = prepare_evaluation_context(arm_df, arm_config)
    split_map = {split.fold_idx: split for split in context["splits"]}
    if args.fold_index not in split_map:
        raise IndexError(f"fold_index out of range: {args.fold_index}")

    fold_result, nested_rows, model_override = run_single_fold_from_context(
        context,
        split_map[args.fold_index],
    )

    fold_dir = args.out_root / "fold_results" / arm_name / f"fold_{args.fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_result.predictions.to_csv(fold_dir / "predictions.csv", index=False)
    metrics = context["evaluator"].compute_metrics(
        fold_result.predictions["y_true"].to_numpy(),
        fold_result.predictions["y_pred"].to_numpy(),
        fold_result.predictions["y_prob"].to_numpy(),
    )
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "arm_name": arm_name,
                "fold": int(args.fold_index),
                "metrics": metrics,
                "model_params_override": model_override,
            },
            handle,
            indent=2,
        )
    if nested_rows:
        pd.DataFrame(nested_rows).to_csv(
            fold_dir / "nested_tuning_summary.csv",
            index=False,
        )
    with (fold_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_data(arm_config), handle, sort_keys=False)


if __name__ == "__main__":
    main()
