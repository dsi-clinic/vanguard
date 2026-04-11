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
    raw_arms = config.ablation_arms
    if not raw_arms:
        raw_arms = config.get("model_family_arms")
    if not raw_arms:
        raw_arms = DEFAULT_ABLATION_ARMS

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


def _normalize_model_families(config: dict[str, Any]) -> list[str]:
    """Model families to run (defaults to model_params.model)."""
    raw = config.get("model_families")
    if raw is None:
        return [str(config.model_params.model).strip().lower()]
    if isinstance(raw, str):
        raw = [raw]
    out = [str(v).strip().lower() for v in raw if str(v).strip()]
    if not out:
        raise ValueError("model_families, if set, cannot be empty.")
    return out


def _normalize_split_modes(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Split/evaluation modes (standard vs site-group CV, etc.)."""
    raw = config.get("split_mode_matrix") or config.get("robustness_split_modes")
    if not raw:
        return [
            {
                "name": "cv",
                "use_group_split": bool(config.model_params.use_group_split),
                "model_params_override": {},
            }
        ]
    modes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each split_mode_matrix entry needs a non-empty name.")
        if name in seen:
            raise ValueError(f"Duplicate split mode name: {name}")
        seen.add(name)
        modes.append(
            {
                "name": name,
                "use_group_split": bool(item.get("use_group_split", False)),
                "model_params_override": deepcopy(
                    item.get("model_params_override", {})
                ),
            }
        )
    return modes


def _multi_combo(families: list[str], modes: list[dict[str, Any]]) -> bool:
    """Whether run IDs must include family and split mode (more than one combo)."""
    return len(families) * len(modes) > 1


def _run_name(arm_name: str, family: str, mode_name: str, *, multi: bool) -> str:
    if not multi:
        return arm_name
    return f"{arm_name}__{family}__{mode_name}"


def _prepare_full_dataset(
    base_config: dict[str, Any],
    arms: list[dict[str, Any]],
    outdir: Path,
) -> pd.DataFrame:
    """Build the superset labeled dataset once for reuse across ablation arms."""
    full_config = deepcopy(base_config)
    toggles = full_config.feature_toggles
    toggles.use_vascular = True
    if any("clinical" in arm["selected_features"] for arm in arms):
        toggles.use_clinical = True
    toggles.pop("selected_features", None)

    features_df = build_modular_features(full_config)
    features_df.to_csv(outdir / "features_full_raw.csv", index=False)

    label_col = full_config.data_paths.label_column
    id_col = full_config.data_paths.id_column
    labels_df = load_labels(full_config.data_paths.labels_csv, id_col, label_col)

    merged_df = features_df.merge(labels_df, on="case_id", how="inner")
    merged_df.to_csv(outdir / "features_full_labeled.csv", index=False)
    logging.info("Prepared full labeled dataset with shape %s", merged_df.shape)
    return merged_df


def _metrics_summary_row(
    *,
    run_name: str,
    arm_name: str,
    model_family: str,
    split_mode: str,
    blocks: list[str],
    results_dir: Path,
) -> dict[str, Any]:
    """Extract a compact metrics summary row from evaluator outputs."""
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "run_name": run_name,
            "arm_name": arm_name,
            "model_family": model_family,
            "split_mode": split_mode,
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
        "run_name": run_name,
        "arm_name": arm_name,
        "model_family": model_family,
        "split_mode": split_mode,
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


def _per_fold_auc_rows(
    *,
    run_name: str,
    arm_name: str,
    results_dir: Path,
) -> list[dict[str, Any]]:
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
                "run_name": run_name,
                "arm_name": arm_name,
                "fold": int(fold),
                "auc": float(auc),
            }
        )
    return rows


def _subtype_rows_from_metrics(
    *,
    run_name: str,
    arm_name: str,
    model_family: str,
    split_mode: str,
    results_dir: Path,
) -> list[dict[str, Any]]:
    """Pull per-subtype AUC from validation_summary in metrics.json."""
    metrics_path = results_dir / "metrics.json"
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


def _add_baseline_deltas(
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    *,
    baseline_run_name: str | None,
    baseline_arm_name: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add summary and fold-wise AUC deltas versus a baseline row."""
    baseline_key_col = "run_name"
    baseline_value: str | None = baseline_run_name
    if not baseline_value and baseline_arm_name:
        baseline_key_col = "arm_name"
        baseline_value = baseline_arm_name

    if not baseline_value or summary_df.empty:
        return summary_df, fold_df

    if baseline_key_col not in summary_df.columns:
        return summary_df, fold_df

    match = summary_df.loc[summary_df[baseline_key_col].astype(str) == baseline_value]
    if match.empty:
        logging.warning(
            "Baseline %s=%r not found in summary; skipping deltas",
            baseline_key_col,
            baseline_value,
        )
        return summary_df, fold_df
    if len(match) > 1:
        logging.warning(
            "Multiple summary rows match baseline %s=%r; using the first",
            baseline_key_col,
            baseline_value,
        )

    baseline_auc_mean = match.iloc[0].get("auc_mean")
    if baseline_auc_mean is None or (
        isinstance(baseline_auc_mean, float) and pd.isna(baseline_auc_mean)
    ):
        return summary_df, fold_df
    baseline_auc_mean = float(baseline_auc_mean)
    delta_col = f"auc_mean_delta_vs_{baseline_value}"
    summary_df = summary_df.copy()
    summary_df[delta_col] = summary_df["auc_mean"] - baseline_auc_mean

    if fold_df.empty or baseline_key_col not in fold_df.columns:
        return summary_df, fold_df

    baseline_fold = fold_df.loc[
        fold_df[baseline_key_col].astype(str) == baseline_value,
        ["fold", "auc"],
    ].rename(columns={"auc": "baseline_auc"})
    if baseline_fold.empty:
        return summary_df, fold_df

    fold_df = fold_df.merge(baseline_fold, on="fold", how="left")
    fold_df[f"auc_delta_vs_{baseline_value}"] = fold_df["auc"] - fold_df["baseline_auc"]

    grp_col = "run_name" if "run_name" in fold_df.columns else "arm_name"
    delta_mean = (
        fold_df.groupby(grp_col)[f"auc_delta_vs_{baseline_value}"]
        .mean()
        .rename(f"auc_fold_delta_mean_vs_{baseline_value}")
    )
    delta_std = (
        fold_df.groupby(grp_col)[f"auc_delta_vs_{baseline_value}"]
        .std()
        .rename(f"auc_fold_delta_std_vs_{baseline_value}")
    )
    summary_df = summary_df.merge(
        delta_mean,
        left_on=grp_col,
        right_index=True,
        how="left",
    )
    summary_df = summary_df.merge(
        delta_std,
        left_on=grp_col,
        right_index=True,
        how="left",
    )
    return summary_df, fold_df


def run_ablation_matrix(config: dict[str, Any], outdir: Path) -> None:
    """Run configured ablation arms and write merged summary table(s)."""
    outdir.mkdir(parents=True, exist_ok=True)
    arms = _normalize_ablation_arms(config)
    families = _normalize_model_families(config)
    modes = _normalize_split_modes(config)
    multi = _multi_combo(families, modes)
    family_overrides = dict(config.get("model_family_overrides") or {})

    matrix_meta = {
        "ablation_arms": to_plain_data(arms),
        "model_families": families,
        "split_mode_matrix": to_plain_data(modes),
        "model_family_overrides": to_plain_data(family_overrides),
        "multi_combo": multi,
    }
    with (outdir / "ablation_matrix_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(matrix_meta, handle, sort_keys=False)

    full_df = _prepare_full_dataset(config, arms, outdir)
    label_col = config.data_paths.label_column

    runs_root = outdir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    subtype_rows: list[dict[str, Any]] = []

    for arm in arms:
        arm_name = arm["name"]
        blocks = list(arm["selected_features"])
        arm_df = select_features(
            full_df,
            selected_blocks=blocks,
            label_col=label_col,
        )

        for family in families:
            for mode in modes:
                run_name = _run_name(arm_name, family, mode["name"], multi=multi)
                logging.info(
                    "Running ablation cell run_name=%s (arm=%s family=%s mode=%s)",
                    run_name,
                    arm_name,
                    family,
                    mode["name"],
                )

                arm_config = deepcopy(config)
                arm_config.experiment_setup.name = run_name
                toggles = arm_config.feature_toggles
                toggles.selected_features = blocks
                toggles.use_clinical = "clinical" in blocks
                toggles.use_vascular = any(block != "clinical" for block in blocks)
                toggles.update(arm.get("feature_toggles_override", {}))

                arm_config.model_params.model = family
                arm_config.model_params.update(arm.get("model_params_override", {}))
                arm_config.model_params.update(family_overrides.get(family, {}))
                arm_config.model_params.use_group_split = bool(mode["use_group_split"])
                arm_config.model_params.update(mode.get("model_params_override", {}))

                run_dir = runs_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                arm_df.to_csv(run_dir / "features_engineered_labeled.csv", index=False)
                with (run_dir / "config_used.yaml").open(
                    "w", encoding="utf-8"
                ) as handle:
                    yaml.safe_dump(to_plain_data(arm_config), handle, sort_keys=False)

                run_evaluation_pipeline(arm_df, arm_config, runs_root)
                summary_rows.append(
                    _metrics_summary_row(
                        run_name=run_name,
                        arm_name=arm_name,
                        model_family=family,
                        split_mode=mode["name"],
                        blocks=blocks,
                        results_dir=runs_root / run_name,
                    )
                )
                fold_rows.extend(
                    _per_fold_auc_rows(
                        run_name=run_name,
                        arm_name=arm_name,
                        results_dir=runs_root / run_name,
                    )
                )
                if bool(config.get("export_subtype_summary")):
                    subtype_rows.extend(
                        _subtype_rows_from_metrics(
                            run_name=run_name,
                            arm_name=arm_name,
                            model_family=family,
                            split_mode=mode["name"],
                            results_dir=runs_root / run_name,
                        )
                    )

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    baseline_run = config.get("baseline_run_name")
    baseline_arm = config.get("baseline_arm_name")
    summary_df, fold_df = _add_baseline_deltas(
        summary_df,
        fold_df,
        baseline_run_name=str(baseline_run) if baseline_run else None,
        baseline_arm_name=str(baseline_arm) if baseline_arm else None,
    )
    summary_df.to_csv(outdir / "ablation_summary.csv", index=False)
    if not fold_df.empty:
        fold_df.to_csv(outdir / "ablation_fold_auc.csv", index=False)
    if subtype_rows:
        pd.DataFrame(subtype_rows).to_csv(
            outdir / "ablation_subtype_summary.csv", index=False
        )
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
