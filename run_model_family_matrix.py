"""Run a reproducible model-family x feature-arm matrix for tabular pCR models."""

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run model-family x feature-arm matrix."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config describing matrix arms/families.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Root output directory for shared features and per-run outputs.",
    )
    return parser.parse_args()


def _normalize_arms(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return model-family matrix arms from config."""
    raw_arms = config.get("model_family_arms", [])
    if not raw_arms:
        raise ValueError(
            "Config must define model_family_arms for model-family matrix runs."
        )

    arms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_arm in raw_arms:
        name = str(raw_arm.get("name", "")).strip()
        if not name:
            raise ValueError("Each model_family_arms entry needs a non-empty name.")
        if name in seen:
            raise ValueError(f"Duplicate model_family_arms name: {name}")
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
            "Matrix arm %s blocks=%s (%s)",
            name,
            selected,
            {b: FEATURE_BLOCK_DESCRIPTIONS.get(b, b) for b in selected},
        )
    return arms


def _normalize_model_families(config: dict[str, Any]) -> list[str]:
    """Return normalized model family names."""
    raw = config.get("model_families", ["lr", "rf", "xgb"])
    if isinstance(raw, str):
        raw = [raw]
    families = [str(v).strip().lower() for v in raw if str(v).strip()]
    if not families:
        raise ValueError("model_families cannot be empty.")

    alias_map = {
        "logistic": "lr",
        "logreg": "lr",
        "lr": "lr",
        "rf": "rf",
        "random_forest": "rf",
        "xgb": "xgb",
        "xgboost": "xgb",
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for name in families:
        mapped = alias_map.get(name, name)
        if mapped not in {"lr", "rf", "xgb"}:
            raise ValueError(
                f"Unsupported model family {name!r}; expected one of lr/rf/xgb."
            )
        if mapped in seen:
            continue
        seen.add(mapped)
        normalized.append(mapped)
    return normalized


def _prepare_full_dataset(config: dict[str, Any], outdir: Path) -> pd.DataFrame:
    """Build the superset labeled dataset once for all matrix runs."""
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
    arm_name: str,
    model_family: str,
    run_name: str,
    selected_features: list[str],
    run_dir: Path,
    arm_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Read evaluator outputs and map them to compact matrix columns."""
    metrics_path = run_dir / "metrics.json"
    base = {
        "run_name": run_name,
        "arm_name": arm_name,
        "model_family": model_family,
        "selected_features": ",".join(selected_features),
        "feature_selection_mode": str(arm_cfg.model_params.feature_select_mode),
        "feature_select_enabled": bool(arm_cfg.model_params.feature_select_enabled),
        "nested_tuning_on": bool(arm_cfg.model_params.nested_tune_enabled),
    }
    if not metrics_path.exists():
        base["status"] = "missing_metrics"
        return base

    payload = json.loads(metrics_path.read_text())
    aggregated = payload.get("aggregated_metrics", {})

    def _metric(metric_name: str, field: str) -> float | None:
        obj = aggregated.get(metric_name)
        if not isinstance(obj, dict):
            return None
        value = obj.get(field)
        return None if value is None else float(value)

    base.update(
        {
            "status": "ok",
            "n_features": payload.get("n_features"),
            "n_samples": payload.get("n_samples"),
            "auc_mean": _metric("auc", "mean"),
            "auc_std": _metric("auc", "std"),
            "ap_mean": _metric("ap", "mean"),
            "ap_std": _metric("ap", "std"),
        }
    )
    return base


def run_model_family_matrix(config: dict[str, Any], outdir: Path) -> None:
    """Run a focused model-family x feature-arm matrix and save merged summary."""
    outdir.mkdir(parents=True, exist_ok=True)
    arms = _normalize_arms(config)
    families = _normalize_model_families(config)
    family_overrides = to_plain_data(deepcopy(config.get("model_family_overrides", {})))

    with (outdir / "matrix_definition_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "model_families": families,
                "model_family_arms": to_plain_data(arms),
                "model_family_overrides": to_plain_data(family_overrides),
            },
            handle,
            sort_keys=False,
        )

    full_df = _prepare_full_dataset(config, outdir)
    label_col = str(config.data_paths.label_column)
    runs_root = outdir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_name = str(arm["name"])
        blocks = list(arm["selected_features"])
        arm_df = select_features(
            full_df,
            selected_blocks=blocks,
            label_col=label_col,
        )

        for family in families:
            run_name = f"{arm_name}__{family}"
            logging.info("Running matrix cell arm=%s model=%s", arm_name, family)

            run_cfg = deepcopy(config)
            run_cfg.experiment_setup.name = run_name
            run_cfg.feature_toggles.selected_features = blocks
            run_cfg.feature_toggles.use_clinical = "clinical" in blocks
            run_cfg.feature_toggles.use_vascular = any(b != "clinical" for b in blocks)
            run_cfg.feature_toggles.update(arm.get("feature_toggles_override", {}))

            run_cfg.model_params.model = family
            run_cfg.model_params.update(arm.get("model_params_override", {}))
            run_cfg.model_params.update(family_overrides.get(family, {}))

            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            arm_df.to_csv(run_dir / "features_engineered_labeled.csv", index=False)
            with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(to_plain_data(run_cfg), handle, sort_keys=False)

            run_evaluation_pipeline(arm_df, run_cfg, runs_root)
            summary_rows.append(
                _extract_summary_row(
                    arm_name=arm_name,
                    model_family=family,
                    run_name=run_name,
                    selected_features=blocks,
                    run_dir=runs_root / run_name,
                    arm_cfg=run_cfg,
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "model_family_matrix_summary.csv", index=False)
    logging.info(
        "Wrote model-family summary to %s",
        outdir / "model_family_matrix_summary.csv",
    )


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    run_model_family_matrix(cfg, args.outdir)


if __name__ == "__main__":
    main()

