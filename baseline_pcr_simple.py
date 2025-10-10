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

def get_patient_id(path: Path, js: Dict[str, Any]) -> str:
    return js.get("patient_id", path.stem)

def get_age(js):
    age = js.get("clinical_data", {}).get("age", None)
    return float(age) if age not in (None, "") else None
  
def get_subtype(js: Dict[str, Any]) -> str:
  subtype = js.get("primary_lesion", {}).get("tumor_subtype", "")
  s = str(subtype).strip()
  return s

def get_label(js: Dict[str, Any]) -> int:
    lab = js.get("primary_lesion", {}).get("pcr", None)
    if lab in (None, ""):
        raise KeyError("primary_lesion.pcr missing or blank")
    return int(lab)

def get_bbox_volume(js: Dict[str, Any]) -> Optional[float]:
  bc = js.get("primary_lesion", {}).get("breast_coordinates", {})
  try:
    x_min = float(bc.get("x_min"))
    x_max = float(bc.get("x_max"))
    y_min = float(bc.get("y_min"))
    y_max = float(bc.get("y_max"))
    z_min = float(bc.get("z_min"))
    z_max = float(bc.get("z_max"))
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    vol = dx * dy * dz
    return vol if (dx > 0 and dy > 0 and dz > 0) else None
  except Exception:
      return None
  
# DATA LOADING

def load_dataset(json_dir: Path, split_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
  splits = pd.read_csv(split_csv)
  if not {"patient_id", "split"}.issubset(set(splits.columns)):
    raise ValueError("split CSV must have columns: patient_id, split")
  
  def get_splits(s: str) -> str: #getting splits from the split column in csv
      s = str(s).strip().lower()
      if s == "train":
          return "train"
      if s in "val":
          return "val"
      return "test"

  splits["split"] = splits["split"].map(get_splits)
  split_map = dict(zip(splits["patient_id"].astype(str), splits["split"]))

  rows = []
  for p in sorted(Path(json_dir).glob("*.json")):
      js = json.loads(Path(p).read_text())

      pid = get_patient_id(p, js)
      y = get_label(js) 
      age = get_age(js)
      subtype = get_subtype(js)
      bbox_volume = get_bbox_volume(js)

      rows.append({
            "patient_id": pid,
            "split": split_map[pid],
            "age": age,
            "tumor_subtype": subtype,
            "bbox_volume": bbox_volume,
            "y": y,
        })
      
  df = pd.DataFrame(rows)

      
  df_train = df[df["split"] == "train"].copy()
  df_eval  = df[df["split"] != "train"].copy()

  return df_train, df_eval






