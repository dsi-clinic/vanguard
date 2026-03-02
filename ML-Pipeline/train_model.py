import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

df = pd.read_csv("combined_vanguard_features.csv")

df = df.dropna(subset=['pcr'])

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

auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

y_probas = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]
fpr, tpr, _ = roc_curve(y, y_probas)
roc_auc_pooled = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_pooled:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title(f'Vanguard Model: PVD Prediction of pCR\n(N={len(df)}, Mean AUC={np.mean(auc_scores):.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("EN_roc_curve.png")

print(f"\nElasticNet Results ---")
print(f"Total Patients (N): {len(df)}")
print(f"AUC per Fold:      {auc_scores}")
print(f"Mean AUC:          {np.mean(auc_scores):.3f}")
print(f"Std Deviation:     {np.std(auc_scores):.3f}")