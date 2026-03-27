"""Run a reproducible pCR feature-block ablation matrix."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from config import DEFAULT_ABLATION_ARMS, load_config, to_plain_data
from features import FEATURE_BLOCK_DESCRIPTIONS, normalize_selected_features
from tabular_cohort import build_modular_features, load_labels, select_features
from train_tabular import run_evaluation_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run feature-block ablation matrix.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to base YAML config with optional ablation_arms block.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Root output directory for shared features and per-arm runs.",
    )
    return parser.parse_args()


def _normalize_ablation_arms(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ablation arms from config or defaults."""
    raw_arms = config["ablation_arms"] or DEFAULT_ABLATION_ARMS
    arms: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for raw_arm in raw_arms:
        name = str(raw_arm["name"]).strip()
        if not name:
            raise ValueError("Each ablation arm must define a non-empty name.")
        if name in seen_names:
            raise ValueError(f"Duplicate ablation arm name: {name}")
        seen_names.add(name)

        selected_blocks = raw_arm.get("selected_features")
        if selected_blocks is None:
            raise ValueError(f"Ablation arm {name} is missing selected_features.")
        normalized_blocks = normalize_selected_features(selected_blocks)
        if not normalized_blocks:
            raise ValueError(f"Ablation arm {name} selected no valid feature blocks.")

        arm = {
            "name": name,
            "selected_features": normalized_blocks,
            "model_params_override": deepcopy(raw_arm.get("model_params_override", {})),
            "feature_toggles_override": deepcopy(
                raw_arm.get("feature_toggles_override", {})
            ),
        }
        arms.append(arm)
        logging.info(
            "Ablation arm %s uses blocks %s (%s)",
            name,
            normalized_blocks,
            {
                block: FEATURE_BLOCK_DESCRIPTIONS.get(block, block)
                for block in normalized_blocks
            },
        )

    return arms


def _prepare_full_dataset(
    base_config: dict[str, Any],
    arms: list[dict[str, Any]],
    outdir: Path,
) -> pd.DataFrame:
    """Build the superset labeled dataset once for reuse across ablation arms."""
    full_config = deepcopy(base_config)
    toggles = full_config.setdefault("feature_toggles", {})
    toggles["use_vascular"] = True
    if any("clinical" in arm["selected_features"] for arm in arms):
        toggles["use_clinical"] = True
    toggles.pop("selected_features", None)

    features_df = build_modular_features(full_config)
    features_df.to_csv(outdir / "features_full_raw.csv", index=False)

    label_col = full_config["data_paths"]["label_column"]
    id_col = full_config["data_paths"]["id_column"]
    labels_df = load_labels(full_config["data_paths"]["labels_csv"], id_col, label_col)

    merged_df = features_df.merge(labels_df, on="case_id", how="inner")
    merged_df.to_csv(outdir / "features_full_labeled.csv", index=False)
    logging.info("Prepared full labeled dataset with shape %s", merged_df.shape)
    return merged_df


def _metrics_summary_row(
    *,
    arm_name: str,
    blocks: list[str],
    results_dir: Path,
) -> dict[str, Any]:
    """Extract a compact metrics summary row from evaluator outputs."""
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "arm_name": arm_name,
            "selected_features": ",".join(blocks),
            "status": "missing_metrics",
        }

    metrics = json.loads(metrics_path.read_text())
    aggregated = metrics.get("aggregated_metrics", {})

    def _metric_value(metric_name: str, field: str) -> float | None:
        payload = aggregated.get(metric_name)
        if not isinstance(payload, dict):
            return None
        value = payload.get(field)
        return None if value is None else float(value)

    return {
        "arm_name": arm_name,
        "selected_features": ",".join(blocks),
        "status": "ok",
        "n_features": metrics.get("n_features"),
        "n_samples": metrics.get("n_samples"),
        "auc_mean": _metric_value("auc", "mean"),
        "auc_std": _metric_value("auc", "std"),
        "ap_mean": _metric_value("ap", "mean"),
        "ap_std": _metric_value("ap", "std"),
        "accuracy_mean": _metric_value("accuracy", "mean"),
        "accuracy_std": _metric_value("accuracy", "std"),
    }


def _per_fold_auc_rows(*, arm_name: str, results_dir: Path) -> list[dict[str, Any]]:
    """Extract per-fold AUC rows from evaluator outputs when available."""
    metrics_path = results_dir / "metrics_per_fold.json"
    if not metrics_path.exists():
        return []

    payload = json.loads(metrics_path.read_text())
    if not isinstance(payload, list):
        return []

    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        fold = row.get("fold")
        auc = row.get("auc")
        if fold is None or auc is None:
            continue
        rows.append(
            {
                "arm_name": arm_name,
                "fold": int(fold),
                "auc": float(auc),
            }
        )
    return rows


def _add_baseline_deltas(
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    *,
    baseline_arm_name: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add summary and fold-wise AUC deltas versus a named baseline arm."""
    if not baseline_arm_name:
        return summary_df, fold_df
    if summary_df.empty or fold_df.empty:
        return summary_df, fold_df
    if baseline_arm_name not in set(summary_df["arm_name"].astype(str)):
        logging.warning(
            "baseline_arm_name=%s not present in summary table; skipping deltas",
            baseline_arm_name,
        )
        return summary_df, fold_df

    baseline_auc_series = summary_df.loc[
        summary_df["arm_name"].astype(str) == baseline_arm_name,
        "auc_mean",
    ]
    if baseline_auc_series.empty or pd.isna(baseline_auc_series.iloc[0]):
        return summary_df, fold_df
    baseline_auc_mean = float(baseline_auc_series.iloc[0])
    summary_df[f"auc_mean_delta_vs_{baseline_arm_name}"] = (
        summary_df["auc_mean"] - baseline_auc_mean
    )

    baseline_fold = fold_df.loc[
        fold_df["arm_name"].astype(str) == baseline_arm_name,
        ["fold", "auc"],
    ].rename(columns={"auc": "baseline_auc"})
    if baseline_fold.empty:
        return summary_df, fold_df

    fold_df = fold_df.merge(baseline_fold, on="fold", how="left")
    fold_df[f"auc_delta_vs_{baseline_arm_name}"] = (
        fold_df["auc"] - fold_df["baseline_auc"]
    )

    delta_mean = (
        fold_df.groupby("arm_name")[f"auc_delta_vs_{baseline_arm_name}"]
        .mean()
        .rename(f"auc_fold_delta_mean_vs_{baseline_arm_name}")
    )
    delta_std = (
        fold_df.groupby("arm_name")[f"auc_delta_vs_{baseline_arm_name}"]
        .std()
        .rename(f"auc_fold_delta_std_vs_{baseline_arm_name}")
    )
    summary_df = summary_df.merge(delta_mean, on="arm_name", how="left")
    summary_df = summary_df.merge(delta_std, on="arm_name", how="left")
    return summary_df, fold_df


def run_ablation_matrix(config: dict[str, Any], outdir: Path) -> None:
    """Run configured ablation arms and write a merged summary table."""
    outdir.mkdir(parents=True, exist_ok=True)
    arms = _normalize_ablation_arms(config)

    with (outdir / "ablation_arms_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"ablation_arms": to_plain_data(arms)}, handle, sort_keys=False)

    full_df = _prepare_full_dataset(config, arms, outdir)
    label_col = config["data_paths"]["label_column"]

    runs_root = outdir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_name = arm["name"]
        blocks = list(arm["selected_features"])
        logging.info("Running ablation arm %s with blocks %s", arm_name, blocks)

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

        arm_df = select_features(
            full_df,
            selected_blocks=blocks,
            label_col=label_col,
        )

        arm_dir = runs_root / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_df.to_csv(arm_dir / "features_engineered_labeled.csv", index=False)
        with (arm_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(to_plain_data(arm_config), handle, sort_keys=False)

        run_evaluation_pipeline(arm_df, arm_config, runs_root)
        summary_rows.append(
            _metrics_summary_row(
                arm_name=arm_name,
                blocks=blocks,
                results_dir=runs_root / arm_name,
            )
        )
        fold_rows.extend(
            _per_fold_auc_rows(
                arm_name=arm_name,
                results_dir=runs_root / arm_name,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    summary_df, fold_df = _add_baseline_deltas(
        summary_df,
        fold_df,
        baseline_arm_name=config["baseline_arm_name"],
    )
    summary_df.to_csv(outdir / "ablation_summary.csv", index=False)
    if not fold_df.empty:
        fold_df.to_csv(outdir / "ablation_fold_auc.csv", index=False)
    logging.info("Wrote ablation summary to %s", outdir / "ablation_summary.csv")


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    run_ablation_matrix(config, args.outdir)


if __name__ == "__main__":
    main()
