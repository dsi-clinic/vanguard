import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from evaluation import Evaluator, FoldResults

def train_baseline_mock(train_df, val_df, features):
    """Simple trainer for the rerun script."""
    y_train = train_df['pcr'].astype(int)
    X_train = train_df[features]
    X_val = val_df[features]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return FoldResults(
        fold_idx=0, 
        predictions=pd.DataFrame({
            "patient_id": val_df['patient_id'],
            "y_true": val_df['pcr'].values,
            "y_pred": clf.predict(X_val),
            "y_prob": clf.predict_proba(X_val)[:, 1]
        })
    )

def prepare_data():
    path = "/net/projects2/vanguard/MAMA-MIA-syn60868042/"
    labels = pd.read_csv(f"{path}pcr_labels.csv")
    clinical = pd.read_excel(f"{path}clinical_and_imaging_info.xlsx")
    
    labels = labels.rename(columns={'case_id': 'patient_id'})
    df = pd.merge(labels, clinical, on='patient_id', how='inner')
    df = df.rename(columns={'pcr_x': 'pcr'})

    def clean_spacing(x):
        if isinstance(x, str):
            return float(x.replace('[', '').replace(']', '').split(',')[0])
        return float(x)

    for col in ['pixel_spacing', 'slice_thickness', 'image_rows', 'image_columns', 'num_slices']:
        df[col] = df[col].apply(clean_spacing)

    df['tumor_volume'] = (
        (df['image_rows'] * df['pixel_spacing']) * (df['image_columns'] * df['pixel_spacing']) * (df['num_slices'] * df['slice_thickness'])
    )
    return df

def run_experiment(df, dataset, features, stratifiers):
    print(f"Running: {dataset} | Features: {features}")
    
    test_df = df.copy() if dataset == 'all' else df[df['dataset'].str.lower() == dataset.lower()].copy()
    
    cols_to_check = features + stratifiers + ['pcr']
    test_df = test_df.dropna(subset=cols_to_check)
    
    if test_df.empty:
        print(f"Warning: No data left for {dataset} after dropping NaNs.")
        return

    evaluator = Evaluator(
        X=test_df[features], 
        y=test_df['pcr'], 
        patient_ids=test_df['patient_id'],
        model_name=f"{dataset}_baseline"
    )
    
    splits = evaluator.create_kfold_splits(n_splits=5) 
    
    results = [train_baseline_mock(test_df.iloc[f.train_indices], test_df.iloc[f.val_indices], features) for f in splits]
    
    kfold_results = evaluator.aggregate_kfold_results(results)
    evaluator.save_results(kfold_results, Path(f"results/{dataset}_{'_'.join(features)}"))

if __name__ == "__main__":
    master_df = prepare_data()

    run_experiment(master_df, 'ispy2', ['age'], ['tumor_subtype'])
    run_experiment(master_df, 'duke', ['age', 'tumor_volume'], ['tumor_subtype', 'bilateral_mri'])