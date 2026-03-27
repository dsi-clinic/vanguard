"""Merge per-arm per-fold ablation outputs into evaluator-style summaries."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation import Evaluator, FoldResults
from run_ablation_matrix import (
    _add_baseline_deltas,
    _metrics_summary_row,
    _normalize_ablation_arms,
)
from tabular_cohort import select_features
from train_tabular import prepare_evaluation_context


def _write_auc_summary_plot(summary_df: pd.DataFrame, out_root: Path) -> None:
    """Write a one-dot-per-arm AUC plot with error bars for fold std."""
    if summary_df.empty:
        return
    plot_df = summary_df.loc[
        summary_df["status"].astype(str) == "ok",
        ["arm_name", "auc_mean", "auc_std"],
    ].copy()
    plot_df = plot_df.dropna(subset=["auc_mean"])
    plot_df = plot_df.sort_values("auc_mean", ascending=True).reset_index(drop=True)
    if plot_df.empty:
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logging.warning("matplotlib not available; skipping ablation summary plot")
        return

    y_pos = np.arange(len(plot_df))
    auc_mean = plot_df["auc_mean"].astype(float).to_numpy()
    auc_std = plot_df["auc_std"].fillna(0.0).astype(float).to_numpy()
    display_labels = (
        plot_df["arm_name"]
        .astype(str)
        .str.replace("_plus_", " + ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.lower()
        .tolist()
    )

    fig_height = max(4.0, 0.7 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.errorbar(
        auc_mean,
        y_pos,
        xerr=auc_std,
        fmt="o",
        color="tab:blue",
        ecolor="tab:gray",
        elinewidth=1.5,
        capsize=4,
        markersize=6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("validation auc")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    data_min = float((auc_mean - auc_std).min())
    data_max = float((auc_mean + auc_std).max())
    data_span = max(data_max - data_min, 0.02)
    left_pad = max(0.01, 0.15 * data_span)
    right_pad = max(0.05, 0.45 * data_span)
    x_min = max(0.0, data_min - left_pad)
    x_max = min(1.0, data_max + right_pad)
    if x_max <= x_min:
        x_min = max(0.0, data_min - 0.02)
        x_max = min(1.0, data_max + 0.08)
    ax.set_xlim(x_min, x_max)

    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(
            x_max - 0.002,
            idx,
            f"{float(row['auc_mean']):.3f} +/- {float(row['auc_std'] or 0.0):.3f}",
            ha="right",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_root / "ablation_auc_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    arms = _normalize_ablation_arms(config)
    full_df = pd.read_csv(args.features_csv)
    label_col = config["data_paths"]["label_column"]

    summary_rows: list[dict[str, Any]] = []
    fold_auc_rows: list[dict[str, Any]] = []

    runs_root = args.out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        arm_name = arm["name"]
        blocks = list(arm["selected_features"])
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
        fold_results: list[FoldResults] = []
        nested_frames: list[pd.DataFrame] = []
        for split in context["splits"]:
            fold_dir = (
                args.out_root / "fold_results" / arm_name / f"fold_{split.fold_idx}"
            )
            pred_path = fold_dir / "predictions.csv"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing fold predictions: {pred_path}")
            pred_df = pd.read_csv(pred_path)
            fold_results.append(
                FoldResults(fold_idx=split.fold_idx, predictions=pred_df)
            )
            nested_path = fold_dir / "nested_tuning_summary.csv"
            if nested_path.exists():
                nested_frames.append(pd.read_csv(nested_path))

        evaluator = Evaluator(
            X=context["X"],
            y=context["y"],
            case_ids=context["case_ids"],
            model_name=arm_name,
            random_state=context["random_state"],
        )
        kfold_results = evaluator.aggregate_kfold_results(fold_results)
        evaluator.save_results(kfold_results, runs_root)

        arm_results_dir = runs_root / arm_name
        with (arm_results_dir / "config_used.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(arm_config, handle, sort_keys=False)
        arm_df.to_csv(arm_results_dir / "features_engineered_labeled.csv", index=False)

        if nested_frames:
            nested_df = pd.concat(nested_frames, ignore_index=True)
            nested_df.to_csv(arm_results_dir / "nested_tuning_summary.csv", index=False)

        summary_rows.append(
            _metrics_summary_row(
                arm_name=arm_name,
                blocks=blocks,
                results_dir=arm_results_dir,
            )
        )

        metrics_path = arm_results_dir / "metrics_per_fold.json"
        payload = json.loads(metrics_path.read_text())
        for row in payload:
            fold_auc_rows.append(
                {
                    "arm_name": arm_name,
                    "fold": int(row["fold"]),
                    "auc": float(row["auc"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_auc_rows)
    summary_df, fold_df = _add_baseline_deltas(
        summary_df,
        fold_df,
        baseline_arm_name=config.get("baseline_arm_name"),
    )
    summary_df.to_csv(args.out_root / "ablation_summary.csv", index=False)
    if not fold_df.empty:
        fold_df.to_csv(args.out_root / "ablation_fold_auc.csv", index=False)
    _write_auc_summary_plot(summary_df, args.out_root)


if __name__ == "__main__":
    main()
