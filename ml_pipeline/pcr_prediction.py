"""Config-driven pCR prediction pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_pipeline.feature_factor import get_clinical_features
from ml_pipeline.utils.config_utils import load_pipeline_config

logger = logging.getLogger(__name__)
MIN_CASE_ID_TOKENS = 2


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("PCR Prediction Pipeline")
    parser.add_argument("--config", type=Path, default="ml_pipeline/config_pcr.yaml")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def _to_num(val: object) -> float | None:
    """Convert a JSON scalar to float when possible."""
    if isinstance(val, bool):
        return float(int(val))
    if isinstance(val, int | float):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            return None
    return None


def _infer_case_id(path: Path) -> str:
    """Infer a case id from a JSON filename."""
    tokens = path.stem.split("_")
    if len(tokens) >= MIN_CASE_ID_TOKENS and tokens[1].isdigit():
        return f"{tokens[0]}_{tokens[1]}"
    return path.stem


def build_features_from_feature_jsons(feature_dir: Path) -> pd.DataFrame:
    """Aggregate numeric stats from per-case JSON morphometrics."""
    rows: list[dict[str, Any]] = []
    json_files = sorted(feature_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON feature files found in {feature_dir}")

    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping feature JSON %s: %s", path, exc)
            continue

        if not isinstance(data, dict):
            continue

        case_id = str(
            data.get("case_id")
            or data.get("patient_id")
            or data.get("study_id")
            or _infer_case_id(path)
        )
        feats: dict[str, Any] = {"case_id": case_id}

        for group in data.values():
            if not isinstance(group, dict):
                continue
            for vessel_name, items in group.items():
                prefix = str(vessel_name).replace(" ", "_")

                if isinstance(items, dict):
                    for field, value in items.items():
                        num = _to_num(value)
                        if num is not None:
                            feats[f"{prefix}__{field}"] = num
                    continue

                if not isinstance(items, list):
                    continue

                per_item_vals: dict[str, list[float]] = {}
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    for field, value in item.items():
                        num = _to_num(value)
                        if num is not None:
                            per_item_vals.setdefault(str(field), []).append(num)

                for field, values in per_item_vals.items():
                    feats[f"{prefix}__{field}__mean"] = float(np.mean(values))
                    feats[f"{prefix}__{field}__std"] = (
                        float(np.std(values)) if len(values) > 1 else 0.0
                    )
                    feats[f"{prefix}__{field}__count"] = float(len(values))

        rows.append(feats)

    features_df = pd.DataFrame(rows)
    if features_df.empty or "case_id" not in features_df.columns:
        raise ValueError(f"No usable feature rows were produced from {feature_dir}")
    return features_df


def _prefix_feature_columns(
    frame: pd.DataFrame,
    *,
    prefix: str,
    skip_columns: set[str],
) -> pd.DataFrame:
    """Prefix feature columns to avoid collisions between sources."""
    rename_map = {
        column: f"{prefix}{column}"
        for column in frame.columns
        if column not in skip_columns
    }
    return frame.rename(columns=rename_map)


def build_radiomics_features(paths: dict[str, Any], id_column: str) -> pd.DataFrame:
    """Load radiomics features from CSV."""
    radiomics_csv = Path(paths["radiomics_csv"])
    radiomics_df = pd.read_csv(radiomics_csv)
    if id_column not in radiomics_df.columns and "patient_id" in radiomics_df.columns:
        radiomics_df = radiomics_df.rename(columns={"patient_id": id_column})
    if id_column not in radiomics_df.columns:
        raise ValueError(f"Radiomics CSV missing id column '{id_column}'")
    label_column = str(paths.get("label_column", "pcr"))
    radiomics_df = radiomics_df.drop(columns=[label_column], errors="ignore")
    radiomics_df = _prefix_feature_columns(
        radiomics_df,
        prefix="radiomics__",
        skip_columns={id_column},
    )
    return radiomics_df.set_index(id_column)


def run_pipeline() -> None:
    """Orchestrate the pipeline via YAML config."""
    args = parse_args()
    config = load_pipeline_config(str(args.config))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    toggles = config["feature_toggles"]
    paths = config["data_paths"]
    model_cfg = config["model_params"]
    setup = config["experiment_setup"]
    id_column = str(paths["id_column"])
    label_column = str(paths["label_column"])

    outdir = args.outdir if args.outdir else Path(setup["base_outdir"]) / setup["name"]
    outdir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(paths["labels_csv"]).set_index(id_column)
    y_target = labels_df[[label_column]]

    feature_sets: list[pd.DataFrame] = []

    if toggles.get("use_vascular", False):
        logger.info("Loading vascular features")
        vasc_df = build_features_from_feature_jsons(Path(paths["feature_dir"]))
        feature_sets.append(vasc_df.set_index("case_id"))

    if toggles.get("use_clinical", False):
        logger.info("Loading clinical features")
        clinical_df = get_clinical_features(config)
        if id_column not in clinical_df.columns and "patient_id" in clinical_df.columns:
            clinical_df = clinical_df.rename(columns={"patient_id": id_column})
        clinical_df = clinical_df.drop(columns=[label_column], errors="ignore")
        clinical_df = _prefix_feature_columns(
            clinical_df,
            prefix="clinical__",
            skip_columns={id_column},
        )
        feature_sets.append(clinical_df.set_index(id_column))

    if toggles.get("use_radiomics", False):
        logger.info("Loading radiomics features")
        feature_sets.append(build_radiomics_features(paths, id_column))

    if not feature_sets:
        raise ValueError("No feature sets selected in config_pcr.yaml")

    x_features = pd.concat(feature_sets, axis=1, join="inner")
    final_df = x_features.join(y_target, how="inner")
    final_df.to_csv(outdir / "features_complete.csv")

    y = final_df[label_column].astype(int).to_numpy()
    xmat = x_features.to_numpy()
    model_name = str(model_cfg["model"])
    clf = RandomForestClassifier() if model_name == "rf" else LogisticRegression()
    clf.fit(xmat, y)

    dump(clf, outdir / f"model_{model_name}.pkl")
    logger.info("Model saved to %s", outdir)


if __name__ == "__main__":
    run_pipeline()
