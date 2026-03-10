"""Module for running the MAMA-MIA baseline PCR prediction pipeline."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_pipeline.utils.config_utils import load_pipeline_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser("PCR Prediction Pipeline")
    ap.add_argument("--config", type=Path, default="ml_pipeline/config_pcr.yaml")
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


def train_and_save(X: np.ndarray, y: np.ndarray, model_cfg: dict, outdir: Path) -> None:
    """Helper to handle model training and serialization."""
    model_map = {"rf": RandomForestClassifier(), "lr": LogisticRegression()}

    clf = model_map.get(model_cfg["model"], RandomForestClassifier())
    clf.fit(X, y)

    save_path = outdir / f"model_{model_cfg['model']}.pkl"
    dump(clf, save_path)
    logger.info("Model saved to %s", save_path)


def run_pipeline() -> None:
    """Orchestrate the pipeline via YAML config with robust index-based merging."""
    args = parse_args()
    config = load_pipeline_config(str(args.config))

    toggles = config["feature_toggles"]
    paths = config["data_paths"]
    model_cfg = config["model_params"]
    setup = config["experiment_setup"]

    outdir = Path(setup["base_outdir"]) / setup["name"]
    outdir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(paths["labels_csv"]).set_index(paths["id_column"])
    y_target = labels_df[[paths["label_column"]]]

    feature_sets = []

    if toggles.get("use_vascular", False):
        logger.info("Loading Vascular features:")
        vasc_df = build_features_from_feature_jsons(Path(paths["feature_dir"]))
        feature_sets.append(vasc_df.set_index("case_id"))

    if toggles.get("use_clinical", False):
        logger.info("Loading Clinical features:")
        clin_df = pd.read_excel(paths["clinical_excel"]).set_index(paths["id_column"])
        clin_df = clin_df.drop(columns=[paths["label_column"]], errors="ignore")
        feature_sets.append(clin_df)

    if toggles.get("use_radiomics", False):
        logger.warning("Radiomics integration is currently a stub.")

    if not feature_sets:
        raise ValueError("No feature sets selected in config_pcr.yaml")

    X_features = pd.concat(feature_sets, axis=1, join="inner")

    final_df = X_features.join(y_target, how="inner")
    final_df.to_csv(outdir / "features_complete.csv")

    y = final_df[paths["label_column"]].astype(int).to_numpy()
    Xmat = X_features.to_numpy()

    clf = (
        RandomForestClassifier() if model_cfg["model"] == "rf" else LogisticRegression()
    )
    clf.fit(Xmat, y)

    dump(clf, outdir / f"model_{model_cfg['model']}.pkl")
    logger.info("Model saved to %s", outdir)


if __name__ == "__main__":
    run_pipeline()
