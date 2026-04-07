"""Run robustness checks for top tabular model families.

This script freezes selected family setups and compares:
- standard stratified CV
- site-exclusive group-aware CV

It writes merged summary tables with overall metrics and subtype AUCs.
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from config import load_config, to_plain_data
from features import FEATURE_BLOCK_DESCRIPTIONS, normalize_selected_features
from tabular_cohort import build_modular_features, load_labels, select_features
from train_tabular import run_evaluation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-family robustness matrix.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def _normalize_arms(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_arms = config.get("robustness_arms", [])
    if not raw_arms:
        raise ValueError("Config must define robustness_arms.")

    arms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_arm in raw_arms:
        name = str(raw_arm.get("name", "")).strip()
        if not name:
            raise ValueError("Each robustness arm needs a non-empty name.")
        if name in seen:
            raise ValueError(f"Duplicate robustness arm name: {name}")
        seen.add(name)

        selected = normalize_selected_features(raw_arm.get("selected_features"))
        if not selected:
            raise ValueError(f"Arm {name} selected no valid feature blocks.")

        arm = {
            "name": name,
            "selected_features": selected,
            "model_params_override": to_plain_data(
                deepcopy(raw_arm.get("model_params_override", {}))
            ),
            "feature_toggles_override": to_plain_data(
                deepcopy(raw_arm.get("feature_toggles_override", {}))
            ),
        }
        arms.append(arm)
        logging.info(
            "Robustness arm %s blocks=%s (%s)",
            name,
            selected,
            {b: FEATURE_BLOCK_DESCRIPTIONS.get(b, b) for b in selected},
        )
    return arms


def _normalize_model_families(config: dict[str, Any]) -> list[str]:
    raw = config.get("model_families", ["lr", "xgb"])
    if isinstance(raw, str):
        raw = [raw]
    families = [str(v).strip().lower() for v in raw if str(v).strip()]
    if not families:
        raise ValueError("model_families cannot be empty.")

    alias_map = {
        "logistic": "lr",
        "logreg": "lr",
        "lr": "lr",
        "xgb": "xgb",
        "xgboost": "xgb",
        "rf": "rf",
        "random_forest": "rf",
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for name in families:
        mapped = alias_map.get(name, name)
        if mapped not in {"lr", "xgb", "rf"}:
            raise ValueError(f"Unsupported model family {name!r}.")
        if mapped in seen:
            continue
        seen.add(mapped)
        normalized.append(mapped)
    return normalized


def _normalize_split_modes(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("robustness_split_modes", [])
    if not raw:
        raw = [
            {"name": "standard_cv", "use_group_split": False},
            {"name": "site_group_cv", "use_group_split": True},
        ]

    modes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each robustness split mode needs a non-empty name.")
        if name in seen:
            raise ValueError(f"Duplicate robustness split mode {name}")
        seen.add(name)
        modes.append(
            {
                "name": name,
                "use_group_split": bool(item.get("use_group_split", False)),
                "model_params_override": to_plain_data(
                    deepcopy(item.get("model_params_override", {}))
                ),
            }
        )
    return modes


def _prepare_full_dataset(config: dict[str, Any], outdir: Path) -> pd.DataFrame:
    full_cfg = deepcopy(config)
    toggles = full_cfg.feature_toggles
    toggles.use_vascular = True
    toggles.use_clinical = True
    toggles.pop("selected_features", None)

    features_df = build_modular_features(full_cfg)
    features_df.to_csv(outdir / "features_full_raw.csv", index=False)

    label_col = full_cfg.data_paths.label_column
    id_col = full_cfg.data_paths.id_column
    labels_df = load_labels(full_cfg.data_paths.labels_csv, id_col, label_col)
    merged_df = features_df.merge(labels_df, on="case_id", how="inner")
    merged_df.to_csv(outdir / "features_full_labeled.csv", index=False)
    logging.info("Prepared full labeled dataset shape=%s", merged_df.shape)
    return merged_df


def _extract_summary_row(
    *,
    run_name: str,
    arm_name: str,
    model_family: str,
    split_mode: str,
    selected_features: list[str],
    run_dir: Path,
    run_cfg: dict[str, Any],
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    row = {
        "run_name": run_name,
        "arm_name": arm_name,
        "model_family": model_family,
        "split_mode": split_mode,
        "use_group_split": bool(run_cfg.model_params.use_group_split),
        "group_col": str(run_cfg.model_params.group_col),
        "stratum_col": str(run_cfg.model_params.stratum_col),
        "selected_features": ",".join(selected_features),
        "feature_selection_mode": str(run_cfg.model_params.feature_select_mode),
        "nested_tuning_on": bool(run_cfg.model_params.nested_tune_enabled),
    }
    if not metrics_path.exists():
        row["status"] = "missing_metrics"
        return row

    payload = json.loads(metrics_path.read_text())
    aggregated = payload.get("aggregated_metrics", {})
    auc = aggregated.get("auc", {})
    ap = aggregated.get("average_precision", {})

    row.update(
        {
            "status": "ok",
            "n_features": payload.get("n_features"),
            "n_samples": payload.get("n_samples"),
            "auc_mean": auc.get("mean"),
            "auc_std": auc.get("std"),
            "ap_mean": ap.get("mean"),
            "ap_std": ap.get("std"),
        }
    )
    return row


def _extract_subtype_rows(
    *,
    run_name: str,
    arm_name: str,
    model_family: str,
    split_mode: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return []

    payload = json.loads(metrics_path.read_text())
    by_group = payload.get("validation_summary", {}).get("by_group", {})
    rows: list[dict[str, Any]] = []
    for subtype, metrics in by_group.items():
        rows.append(
            {
                "run_name": run_name,
                "arm_name": arm_name,
                "model_family": model_family,
                "split_mode": split_mode,
                "tumor_subtype": subtype,
                "auc": metrics.get("auc"),
            }
        )
    return rows


def run_model_family_robustness(config: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    arms = _normalize_arms(config)
    families = _normalize_model_families(config)
    split_modes = _normalize_split_modes(config)
    family_overrides = to_plain_data(
        deepcopy(config.get("model_family_overrides", {}))
    )

    with (outdir / "robustness_definition_used.yaml").open("w", encoding="utf-8") as h:
        yaml.safe_dump(
            {
                "robustness_arms": to_plain_data(arms),
                "model_families": families,
                "robustness_split_modes": to_plain_data(split_modes),
                "model_family_overrides": to_plain_data(family_overrides),
            },
            h,
            sort_keys=False,
        )

    full_df = _prepare_full_dataset(config, outdir)
    label_col = str(config.data_paths.label_column)
    runs_root = outdir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    subtype_rows: list[dict[str, Any]] = []

    for arm in arms:
        arm_name = str(arm["name"])
        blocks = list(arm["selected_features"])
        arm_df = select_features(full_df, selected_blocks=blocks, label_col=label_col)

        for family in families:
            for split_mode in split_modes:
                split_name = str(split_mode["name"])
                run_name = f"{arm_name}__{family}__{split_name}"
                logging.info(
                    "Running robustness cell arm=%s family=%s split=%s",
                    arm_name,
                    family,
                    split_name,
                )
                run_cfg = deepcopy(config)
                run_cfg.experiment_setup.name = run_name
                run_cfg.feature_toggles.selected_features = blocks
                run_cfg.feature_toggles.use_clinical = "clinical" in blocks
                run_cfg.feature_toggles.use_vascular = any(b != "clinical" for b in blocks)
                run_cfg.feature_toggles.update(arm.get("feature_toggles_override", {}))

                run_cfg.model_params.model = family
                run_cfg.model_params.update(arm.get("model_params_override", {}))
                run_cfg.model_params.update(family_overrides.get(family, {}))
                run_cfg.model_params.use_group_split = bool(split_mode["use_group_split"])
                run_cfg.model_params.update(split_mode.get("model_params_override", {}))

                run_dir = runs_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                arm_df.to_csv(run_dir / "features_engineered_labeled.csv", index=False)
                with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as h:
                    yaml.safe_dump(to_plain_data(run_cfg), h, sort_keys=False)

                run_evaluation_pipeline(arm_df, run_cfg, runs_root)
                summary_rows.append(
                    _extract_summary_row(
                        run_name=run_name,
                        arm_name=arm_name,
                        model_family=family,
                        split_mode=split_name,
                        selected_features=blocks,
                        run_dir=runs_root / run_name,
                        run_cfg=run_cfg,
                    )
                )
                subtype_rows.extend(
                    _extract_subtype_rows(
                        run_name=run_name,
                        arm_name=arm_name,
                        model_family=family,
                        split_mode=split_name,
                        run_dir=run_dir,
                    )
                )

    pd.DataFrame(summary_rows).to_csv(
        outdir / "model_family_robustness_summary.csv", index=False
    )
    pd.DataFrame(subtype_rows).to_csv(
        outdir / "model_family_robustness_subtype_summary.csv", index=False
    )
    logging.info(
        "Wrote robustness summaries to %s and %s",
        outdir / "model_family_robustness_summary.csv",
        outdir / "model_family_robustness_subtype_summary.csv",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    run_model_family_robustness(cfg, args.outdir)


if __name__ == "__main__":
    main()
