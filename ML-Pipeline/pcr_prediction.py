"""Module for running the MAMA-MIA baseline PCR prediction pipeline."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from .utils.config_utils import load_pipeline_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser("PCR Prediction Pipeline")
    ap.add_argument("--config", type=Path, default="ML-Pipeline/config_pcr.yaml")
    return ap.parse_args()


def build_features_from_feature_jsons(feature_dir: Path) -> pd.DataFrame:
    """Aggregate numeric stats from per-case JSON morphometrics."""
    rows: list[dict[str, float]] = []

    def to_num(val: bool | float | int | str) -> float | None:
        if isinstance(val, bool | float | int):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val.strip())
            except ValueError:
                return None
        return None

    json_files = sorted(feature_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON feature files found in {feature_dir}")

    for p in json_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping feature JSON %s: %s", p, e)
            continue

        case_id = "_".join(p.stem.split("_")[:2])
        feats: dict[str, float] = {"case_id": float(case_id)}

        if isinstance(data, dict):
            for _, group in data.items():
                if not isinstance(group, dict):
                    continue
                for vessel_name, items in group.items():
                    if not isinstance(items, list):
                        continue
                    per_item_vals: dict[str, list[float]] = {}
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        for k, v in item.items():
                            val = to_num(v)
                            if val is not None:
                                per_item_vals.setdefault(k, []).append(val)

                    vprefix = vessel_name.replace(" ", "_")
                    for fld, vals in per_item_vals.items():
                        feats[f"{vprefix}__{fld}__mean"] = float(np.mean(vals))
                        feats[f"{vprefix}__{fld}__std"] = (
                            float(np.std(vals)) if len(vals) > 1 else 0.0
                        )
        rows.append(feats)

    return pd.DataFrame(rows)


def train_baseline(
    features_csv: Path,
    labels_source: Path,
    id_col: str,
    label_col: str,
    outdir: Path,
    model_type: str,
) -> None:
    """Train the model and evaluate performance."""
    features_df = pd.read_csv(features_csv)
    labels_df = pd.read_csv(labels_source)

    if id_col != "case_id":
        labels_df = labels_df.rename(columns={id_col: "case_id"})

    merged_df = features_df.merge(labels_df, on="case_id", how="inner")
    y = merged_df[label_col].astype(int).to_numpy()
    Xmat = merged_df.drop(columns=["case_id", label_col], errors="ignore").to_numpy()

    clf = RandomForestClassifier() if model_type == "rf" else LogisticRegression()
    clf.fit(Xmat, y)

    dump(clf, outdir / f"model_{model_type}.pkl")
    logger.info("Model saved to %s", outdir / f"model_{model_type}.pkl")


def run_pipeline() -> None:
    """Orchestrate the pipeline via YAML config with conditional feature toggles."""
    args = parse_args()
    config = load_pipeline_config(str(args.config))

    toggles = config["feature_toggles"]
    paths = config["data_paths"]
    model_cfg = config["model_params"]
    setup = config["experiment_setup"]

    outdir = Path(setup["base_outdir"]) / setup["name"]
    outdir.mkdir(parents=True, exist_ok=True)

    feature_sets = []

    if toggles.get("use_vascular", False):
        logger.info("Loading Vascular features...")
        feature_sets.append(
            build_features_from_feature_jsons(Path(paths["feature_dir"]))
        )

    if toggles.get("use_clinical", False):
        logger.info("Loading Clinical features...")
        feature_sets.append(pd.read_csv(paths["labels_csv"]))

    if toggles.get("use_radiomics", False):
        logger.info("Loading Radiomics features...")

    if not feature_sets:
        raise ValueError("No feature sets selected in config_pcr.yaml")

    features_df = pd.concat(feature_sets, axis=1)
    features_path = outdir / "features.csv"
    features_df.to_csv(features_path, index=False)

    train_baseline(
        features_csv=features_path,
        labels_source=Path(paths["labels_csv"]),
        id_col=paths["id_column"],
        label_col=paths["label_column"],
        outdir=outdir,
        model_type=model_cfg["model"],
    )


if __name__ == "__main__":
    run_pipeline()
