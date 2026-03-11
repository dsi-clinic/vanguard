"""Module for training an ElasticNet Logistic Regression model to predict pCR from PVD features."""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline.utils.config_utils import load_pipeline_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_vmodel(config_path: str = "ml_pipeline/config_pcr.yaml"):
    """Train ElasticNet model using paths and params from config."""
    config = load_pipeline_config(config_path)
    paths = config["data_paths"]
    setup = config["experiment_setup"]
    
    outdir = Path(setup["base_outdir"]) / setup["name"]
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        vanguard_df = pd.read_csv(paths["labels_csv"]) 
    except FileNotFoundError:
        logger.error("Could not find input file. Ensure pcr_prediction.py has run.")
        return

    vanguard_df = vanguard_df.dropna(subset=[paths["label_column"]])

    X = vanguard_df[["pvd"]] if "pvd" in vanguard_df.columns else vanguard_df.iloc[:, :-1]
    y = vanguard_df[paths["label_column"]]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            l1_ratios=[0.5, 0.7, 0.9],
            penalty="elasticnet",
            solver="saga",
            cv=5,
            scoring="roc_auc",
            max_iter=10000,
            random_state=42,
        )),
    ])

    cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Running Cross-Validation...")
    auc_scores = cross_val_score(pipeline, X, y, cv=cv_split, scoring="roc_auc")
    y_probas = cross_val_predict(pipeline, X, y, cv=cv_split, method="predict_proba")[:, 1]
    
    # Metrics
    fpr, tpr, _ = roc_curve(y, y_probas)
    roc_auc_pooled = auc(fpr, tpr)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc_pooled:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title(f"Vanguard Model: PVD Prediction of pCR\n(N={len(vanguard_df)}, Mean AUC={np.mean(auc_scores):.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_path = outdir / "EN_roc_curve.png"
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")

    print("\nElasticNet Results ---")
    print(f"Total Patients (N): {len(vanguard_df)}")
    print(f"Mean AUC:          {np.mean(auc_scores):.3f}")
    print(f"Std Deviation:     {np.std(auc_scores):.3f}")

if __name__ == "__main__":
    train_vmodel()