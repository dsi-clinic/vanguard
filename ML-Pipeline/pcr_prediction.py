"""MAMA-MIA data loader and classifier using the Evaluation Framework.

Outputs:
- experiment_results/ (Plots, Metrics JSON, Predictions CSV)
"""

import yaml
import json
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import Evaluator, FoldResults
from .feature_factor import get_clinical_features


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="PCR Prediction using Luna Evaluator")
    ap.add_argument("--config", type=str, default="ML-Pipeline/config_pcr.yaml", help="Path to YAML config")
    ap.add_argument("--outdir", type=Path, help="Override output directory")
    return ap.parse_args()

def build_modular_features(config):
    """Extracts features from JSON and integrates clinical data."""
    feature_dir = Path(config['data_paths']['feature_dir'])
    rows = []
    
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    for p in sorted(feature_dir.glob("*.json")):
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            case_id = data.get("patient_id")

            feats = {"case_id": case_id, "morph_feature_1": np.random.rand()} 
            rows.append(feats)
        except Exception as e:
            logging.warning(f"Failed to load {p}: {e}")
    
    df_json = pd.DataFrame(rows)
    
    if config.get('feature_toggles', {}).get('use_clinical', False):
        logging.info("Merging clinical features...")
        try:
            clinical_df = get_clinical_features(config)
            df_json = df_json.merge(clinical_df, left_on='case_id', right_on='patient_id', how='inner')
            df_json = df_json.drop(columns=['patient_id'], errors='ignore')
        except Exception as e:
            logging.error(f"Clinical merge failed: {e}")

    return df_json


def load_labels(path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    """Load labels from CSV or JSON and normalize to integer {0, 1}."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df_labels = pd.read_csv(path)
    else:
        if path.is_dir():
            rows = []
            for jp in sorted(path.glob("*.json")):
                try:
                    obj = json.loads(jp.read_text())
                    rows.append(obj)
                except:
                    continue
            df_labels = pd.DataFrame(rows)
        else:
            obj = json.loads(path.read_text())
            df_labels = pd.DataFrame(obj)
    if id_col not in df_labels.columns and "patient_id" in df_labels.columns:
        df_labels = df_labels.rename(columns={"patient_id": id_col})

    df_labels = df_labels.dropna(subset=[label_col])
    mapping = {"true": 1, "false": 0, "yes": 1, "no": 0}
    
    def clean_val(v):
        s = str(v).strip().lower()
        return mapping.get(s, v)

    df_labels[label_col] = pd.to_numeric(
        df_labels[label_col].map(clean_val), 
        errors='coerce'
    )

    df_labels = df_labels.dropna(subset=[label_col])
    df_labels[label_col] = df_labels[label_col].astype(int)
    return df_labels[[id_col, label_col]].rename(columns={id_col: "case_id"})


def prepare_data(config, outdir):
    """Orchestrates loading features and labels, and merging them."""
    
    feats_df = build_modular_features(config)
    feats_df.to_csv(outdir / "features_raw.csv", index=False)
    
    labels_path = config['data_paths']['labels_csv']
    label_col = config['data_paths']['label_column']
    id_col = config['data_paths'].get('id_column', 'case_id')
    
    labels_df = load_labels(labels_path, id_col, label_col)
    
    merged_df = feats_df.merge(labels_df, on="case_id", how="inner")
    merged_df = merged_df.fillna(0.0)
    
    merged_df.to_csv(outdir / "features_engineered_labeled.csv", index=False)
    
    logging.info(f"Final shape: {merged_df.shape}")
    return merged_df

def run_evaluation_pipeline(df, config, outdir):
    """
    Runs the model using Daniel Luna's Evaluator framework.
    """
    label_col = config['data_paths']['label_column']
    model_type = config['model_params'].get('model', 'rf')
    random_state = config['model_params'].get('random_state', 42)
    
    y = df[label_col].astype(int)
    patient_ids = df['case_id']

    drop_cols = ['case_id', label_col] + [c for c in df.columns if 'variant' in c and c != label_col]
    X = df.drop(columns=drop_cols, errors='ignore')
    
    evaluator = Evaluator(
        X=X,
        y=y,
        patient_ids=patient_ids,
        model_name=config['experiment_setup'].get('name', 'PCR_Model'),
        random_state=random_state
    )

    n_splits = config['model_params'].get('n_splits', 5)
    splits = evaluator.create_kfold_splits(n_splits=n_splits)
    
    fold_results_list = []
    logging.info(f"Starting {n_splits}-Fold Cross Validation...")

    for split in splits:
        print(f"  > Processing Fold {split.fold_idx}...")
        X_train, y_train = X.iloc[split.train_indices], y.iloc[split.train_indices]
        X_val, y_val = X.iloc[split.val_indices], y.iloc[split.val_indices]
        
        if model_type == 'rf':
            clf = RandomForestClassifier(
                n_estimators=config['model_params'].get('n_estimators', 800),
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", random_state=random_state))
            ])
            
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)
        val_pids = patient_ids.iloc[split.val_indices].values
        
        pred_df = pd.DataFrame({
            "patient_id": val_pids,
            "y_true": y_val.values,
            "y_pred": y_pred,
            "y_prob": y_prob
        })
        
        fold_results_list.append(
            FoldResults(fold_idx=split.fold_idx, predictions=pred_df)
        )

    logging.info("Aggregating results...")
    kfold_results = evaluator.aggregate_kfold_results(fold_results_list)
    
    logging.info(f"Saving results to: {outdir}")
    evaluator.save_results(kfold_results, outdir)
    
    print("\n" + "="*40)
    print(f"Visualization plots saved in: {outdir / evaluator.model_name / 'plots'}")
    print("="*40 + "\n")

def main() -> None:

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = args.outdir if args.outdir else Path(config['experiment_setup']['base_outdir'])
    run_name = f"{config['experiment_setup']['name']}_{timestamp}"
    outdir = base_out / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(args.config, outdir / "config_used.yaml")
    
    try:
        merged_data = prepare_data(config, outdir)
        run_evaluation_pipeline(merged_data, config, outdir)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()