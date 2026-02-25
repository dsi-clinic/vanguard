import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

df = pd.read_csv("combined_vanguard_features.csv")

initial_count = len(df)
df = df.dropna(subset=['pcr'])
print(f"Dropped {initial_count - len(df)} rows with missing pCR labels.")

X = df[['pvd']]  
y = df['pcr']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegressionCV(
        l1_ratios=[0.5, 0.7, 0.9], 
        penalty='elasticnet',
        solver='saga',
        cv=5, 
        scoring='roc_auc',
        max_iter=10000,
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_probas = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]

fpr, tpr, _ = roc_curve(y, y_probas)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Vanguard Model: PVD Prediction of pCR\n(Combined Cohort N={len(df)})')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("vanguard_roc_curve.png")

print(f"\n--- Model Results ---")
print(f"Final N: {len(df)}")
print(f"Final AUC: {roc_auc:.3f}")
print(f"ROC Curve saved to EN_roc_curve.png")