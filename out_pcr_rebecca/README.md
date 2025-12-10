# Model Output: pCR Prediction (Run: out_pcr_jose)

This directory contains outputs from Rebecca's extraction method, as mentioned in `pcr_prediction.py` model. It contains evaluation metrics, visualizations, and feature importance data by the pCR prediction pipeline.

* **`confusion_matrix.png` / `confusion_matrix_labeled.png`**
    * Model's predictions versus actual ground truth.
    * Counts for True Positives, True Negatives, False Positives, and False Negatives.
    * *Useful for:* Identifying if the model is biased toward a specific class (e.g., ignoring non-responders).

* **`roc_comparison.png`**
    * Receiver Operating Characteristic (ROC) Curve.
    * Plots the True Positive Rate (Sensitivity) against the False Positive Rate (1-Specificity) at various threshold settings.
    * *Metric:* Look for the AUC (Area Under the Curve) score in the legend; closer to 1.0 is better.

* **`pr_comparison.png`**
    * Precision-Recall Curve.
    * Plots Precision against Recall.
    * *Useful for:* Evaluating performance when the classes are imbalanced (e.g., if there are far fewer pCR cases than non-pCR cases).

### Feature Analysis
* **`feature_importance.csv`**
    * Raw data file listing all input features and their calculated importance scores (Gini impurity or permutation importance).
    * Sorted by contribution to the model's decision-making process.

* **`feature_importance_top20.png`**
    * Bar chart visualizing the top 20 most influential features from the CSV file.
    * *Useful for:* Quickly identifying which imaging biomarkers are driving predictions.

### Metrics & Logs
* **`metrics_rf.json`**
    * A JSON file containing the exact numerical scores for the Random Forest model.
    *
