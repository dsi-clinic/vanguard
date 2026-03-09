"""Module for running the MAMA-MIA baseline PCR prediction pipeline."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

# Constants
ROC_FLIP_THRESHOLD: float = 0.5
DEFAULT_PROBA_THRESHOLD: float = 0.5

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser("JSON feature extraction + baseline classification")
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--id-column", default="case_id")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--random-baseline", action="store_true")
    ap.add_argument("--bootstrap-n", type=int, default=0)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--model", choices=["rf", "lr"], default="rf")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--delong", action="store_true")
    ap.add_argument("--ensemble-runs", type=int, default=0)
    ap.add_argument("--ensemble-hist", action="store_true")
    return ap.parse_args()


def build_features_from_feature_jsons(feature_dir: Path) -> pd.DataFrame:
    """Aggregate numeric stats from per-case JSON morphometrics."""
    rows: list[dict[str, float]] = []

    def to_num(val: bool | float | int | str) -> float | None:
        """Safely convert inputs to float."""
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


def to_int_label(val: dict | bool | np.bool_ | str | int) -> int:
    """Map fetal dict/bool/str/int to {0,1}."""
    if isinstance(val, dict):
        return int(any(bool(v) for v in val.values()))
    if isinstance(val, bool | np.bool_):
        return int(val)
    s = str(val).strip().lower()
    return 1 if s in {"true", "yes", "1"} else 0


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard binary classification metrics."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def engineer_features(in_csv: Path, out_csv: Path) -> None:
    """Clean and select relevant morphometric features from raw extraction CSV."""
    features_df = pd.read_csv(in_csv)
    drop_cols = [
        "source_file",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "bbox_volume",
        "n_points",
        "n_cells",
    ]
    features_df = features_df.drop(
        columns=[c for c in drop_cols if c in features_df.columns]
    )
    features_df.to_csv(out_csv, index=False)
    logger.info("Engineered features saved to %s", out_csv)


def plot_confusion_matrix_clean(cm: np.ndarray, outdir: Path) -> None:
    """Plot and save a clean confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="viridis")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    ax.set_title("Confusion Matrix @ tuned threshold")
    plt.tight_layout()
    fig.savefig(outdir / "confusion_matrix_clean.png", dpi=220)
    plt.close(fig)


def train_baseline(
    feats_engineered_csv: Path,
    labels_source: Path,
    id_col: str,
    label_col: str,
    outdir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    model: str = "rf",
    bootstrap_n: int = 0,
    make_plots: bool = False,
) -> None:
    """Train the model and evaluate performance."""
    features_df = pd.read_csv(feats_engineered_csv)
    labels_df = pd.read_csv(labels_source)
    if id_col != "case_id":
        labels_df = labels_df.rename(columns={id_col: "case_id"})

    merged_df = features_df.merge(labels_df, on="case_id", how="inner")
    merged_df = merged_df[merged_df[label_col].notna()].copy()

    y = merged_df[label_col].astype(int).to_numpy()
    Xmat = merged_df.drop(columns=["case_id", label_col], errors="ignore").to_numpy()

    clf = (
        RandomForestClassifier(n_estimators=100, random_state=random_state)
        if model == "rf"
        else LogisticRegression()
    )
    clf.fit(Xmat, y)

    dump(clf, outdir / f"model_{model}.pkl")
    logger.info("Model saved to %s/model_%s.pkl", outdir, model)


def main() -> None:
    """Run the main entry point."""
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    features_df = build_features_from_feature_jsons(args.feature_dir)
    features_path = args.outdir / "features.csv"
    features_df.to_csv(features_path, index=False)

    engineer_features(features_path, args.outdir / "features_engineered.csv")

    if args.labels and args.label_column:
        train_baseline(
            args.outdir / "features_engineered.csv",
            args.labels,
            args.id_column,
            args.label_column,
            args.outdir,
            model=args.model,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
