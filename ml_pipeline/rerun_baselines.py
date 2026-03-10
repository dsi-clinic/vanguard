"""Module for rerunning baseline models using the Evaluator framework."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluation import Evaluator, FoldResults

ROOT_PATH: Path = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))


def train_baseline_mock(
    train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]
) -> FoldResults:
    """Train a baseline Random Forest model on the provided data splits.

    Args:
        train_df: DataFrame for training the model.
        val_df: DataFrame for validating the model.
        features: List of feature column names to use.

    Returns:
        FoldResults object containing fold metrics and predictions.
    """
    y_train: pd.Series = train_df["pcr"].astype(int)
    X_train: pd.DataFrame = train_df[features]
    X_val: pd.DataFrame = val_df[features]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    return FoldResults(
        fold_idx=0,
        predictions=pd.DataFrame(
            {
                "patient_id": val_df["patient_id"],
                "y_true": val_df["pcr"].to_numpy(),
                "y_pred": clf.predict(X_val),
                "y_prob": clf.predict_proba(X_val)[:, 1],
            }
        ),
    )


def prepare_data() -> pd.DataFrame:
    """Load labels and clinical data, then engineer tumor volume features.

    Returns:
        A merged and cleaned DataFrame ready for experimentation.
    """
    path: Path = Path("/net/projects2/vanguard/MAMA-MIA-syn60868042/")
    labels_df: pd.DataFrame = pd.read_csv(path / "pcr_labels.csv")
    clinical_df: pd.DataFrame = pd.read_excel(path / "clinical_and_imaging_info.xlsx")

    labels_df = labels_df.rename(columns={"case_id": "patient_id"})
    merged_data: pd.DataFrame = labels_df.merge(
        clinical_df, on="patient_id", how="inner"
    )
    merged_data = merged_data.rename(columns={"pcr_x": "pcr"})

    def clean_spacing(val: str | float | int) -> float:
        if isinstance(val, str):
            return float(val.replace("[", "").replace("]", "").split(",")[0])
        return float(val)

    cols = [
        "pixel_spacing",
        "slice_thickness",
        "image_rows",
        "image_columns",
        "num_slices",
    ]
    for col in cols:
        merged_data[col] = merged_data[col].apply(clean_spacing)

    merged_data["tumor_volume"] = (
        (merged_data["image_rows"] * merged_data["pixel_spacing"])
        * (merged_data["image_columns"] * merged_data["pixel_spacing"])
        * (merged_data["num_slices"] * merged_data["slice_thickness"])
    )
    return merged_data


def run_experiment(
    data_df: pd.DataFrame, dataset: str, features: list[str], stratifiers: list[str]
) -> None:
    """Run baseline experiment and save results."""
    print(f"Running: {dataset} | Features: {features}")

    test_df: pd.DataFrame = (
        data_df.copy()
        if dataset == "all"
        else data_df[data_df["dataset"].str.lower() == dataset.lower()].copy()
    )

    cols_to_check = features + stratifiers + ["pcr"]
    test_df = test_df.dropna(subset=cols_to_check)

    if test_df.empty:
        print(f"Warning: No data left for {dataset} after dropping NaNs.")
        return

    evaluator = Evaluator(
        X=test_df[features],
        y=test_df["pcr"],
        patient_ids=test_df["patient_id"],
        model_name=f"{dataset}_baseline",
    )

    splits = evaluator.create_kfold_splits(n_splits=5)
    results = [
        train_baseline_mock(
            test_df.iloc[f.train_indices], test_df.iloc[f.val_indices], features
        )
        for f in splits
    ]

    kfold_results = evaluator.aggregate_kfold_results(results)
    evaluator.save_results(
        kfold_results, Path(f"results/{dataset}_{'_'.join(features)}")
    )


if __name__ == "__main__":
    master_data = prepare_data()
    run_experiment(master_data, "ispy2", ["age"], ["tumor_subtype"])
    run_experiment(
        master_data, "duke", ["age", "tumor_volume"], ["tumor_subtype", "bilateral_mri"]
    )
