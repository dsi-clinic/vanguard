"""Module for training an ElasticNet Logistic Regression model to predict pCR from PVD features."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
DATA_PATH = Path(__file__).parent / "combined_vanguard_features.csv"
vanguard_df = pd.read_csv(DATA_PATH)
vanguard_df = vanguard_df.dropna(subset=["pcr"])

X = vanguard_df[["pvd"]]
y = vanguard_df["pcr"]

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegressionCV(
                l1_ratios=[0.5, 0.7, 0.9],
                penalty="elasticnet",
                solver="saga",
                cv=5,
                scoring="roc_auc",
                max_iter=10000,
                random_state=42,
            ),
        ),
    ]
)

cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = cross_val_score(pipeline, X, y, cv=cv_split, scoring="roc_auc")

y_probas = cross_val_predict(pipeline, X, y, cv=cv_split, method="predict_proba")[:, 1]
fpr, tpr, _ = roc_curve(y, y_probas)
roc_auc_pooled = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(
    fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc_pooled:.2f})"
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.title(
    f"Vanguard Model: PVD Prediction of pCR\n"
    f"(N={len(vanguard_df)}, Mean AUC={np.mean(auc_scores):.3f})"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("EN_roc_curve.png")

print("\nElasticNet Results ---")
print(f"Total Patients (N): {len(vanguard_df)}")
print(f"AUC per Fold:      {auc_scores}")
print(f"Mean AUC:          {np.mean(auc_scores):.3f}")
print(f"Std Deviation:     {np.std(auc_scores):.3f}")
