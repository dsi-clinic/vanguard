"""
Baseline pCR (0/1) prediction with minimal metadata features

Outputs (written to --output):
  - metrics.json: {"auc_train": float, "auc_test": float, "n_features": int, "n_train": int, "n_test": int}
  - predictions.csv: columns [patient_id, split, y_true, y_pred_score]
  - roc_test.png: ROC curve plot
  - model.pkl: saved logistic regression model

Usage:
  python baseline_pcr_simple.py
    --json-dir /path/to/jsons
    --split-csv splits_v1.csv
    --output outdir
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt