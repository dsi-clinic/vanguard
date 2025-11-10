"""Utility functions for the Vanguard project.

Includes:
- Data loading and cleaning
- Splitting dataframes for ML pipelines
- Helper functions for sample splitting and report generation
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# PARAMETERS
INPUT_DIR = Path("/net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files/")
OUTPUT_CSV = Path("../../output/split_sample/splits_v1.csv")
REPORT_MD = Path("../../output/split_sample/split_report.md")
SEED = 42  # Fixed seed for reproducibility
FLOAT_TOL = 1e-6  # Magic constant for float comparisons


def load_and_clean_patient_data(
    input_dir: Path,
    output_csv: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load patient-level JSON metadata files into a pandas DataFrame.

    Clean key variables (subtype and pCR), and save to CSV.

    Parameters
    ----------
    input_dir : Path
        Path to the folder containing .json files.
    output_csv : Path
        Path where the output CSV file will be saved.
    verbose : bool, default=True
        Whether to print progress and summary info.

    Returns:
    -------
    patients : pd.DataFrame
        Cleaned DataFrame with columns:
        ['patient_id', 'pcr', 'subtype', 'site']
    """
    # Step 1. Load patient microdata
    records: list[dict] = []
    if verbose:
        print("Loading patient JSON files...")

    # Load files and extract relevant features
    for file_path in tqdm(
        input_dir.iterdir(),
        desc="Loading JSON files",
        unit="file",
        disable=not verbose,
    ):
        if file_path.suffix == ".json":
            with file_path.open(encoding="utf-8") as f:
                data = json.load(f)
            patient_id = data.get("patient_id")
            pcr = data.get("primary_lesion", {}).get("pcr")
            subtype = data.get("primary_lesion", {}).get("tumor_subtype")
            site = data.get("imaging_data", {}).get("site")
            records.append(
                {
                    "patient_id": patient_id,
                    "pcr": pcr,
                    "subtype": subtype,
                    "site": site,
                },
            )

    patients = pd.DataFrame(records)
    if verbose:
        print(f"Loaded {len(patients)} patients")

    # Step 2. Clean subtype and pCR before stratification

    # 1. Drop observations with missing subtype or pcr
    patients = patients.dropna(subset=["subtype", "pcr"]).copy()

    # 2. Normalize strings (remove whitespace, lowercase)
    patients["subtype"] = patients["subtype"].astype(str).str.strip().str.lower()

    # 3. Group HER2 variants
    patients["subtype"] = patients["subtype"].replace(
        {
            "her2_enriched": "her2_pure",
            "her2+": "her2_pure",
        },
    )

    # 4. Group luminal variants
    patients["subtype"] = patients["subtype"].replace(
        {
            "luminal_a": "luminal",
            "luminal_b": "luminal",
        },
    )

    # Step 3. Optional sanity check
    if verbose:
        print("\nSubtype counts after cleaning:")
        print(patients["subtype"].value_counts(dropna=False))

    # Step 4. Save output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(output_csv, index=False)
    if verbose:
        print(f"\nCleaned dataset saved to: {output_csv}")

    return patients


def create_dataset_splits(
    df: pd.DataFrame,
    stratify_vars: list[str],
    seed: int = SEED,
    split_percents: dict[str, float] | None = None,
    external_validation: bool = False,
    external_site: str | None = None,
    site_col: str = "site",
) -> pd.DataFrame:
    """Create reproducible stratified splits for train/val/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least `patient_id` and stratification
        variables.
    stratify_vars : list[str]
        List of column names to stratify on (e.g., ["pcr", "subtype"]).
    seed : int, default=42
        Random seed for reproducibility.
    split_percents : dict, optional
        Dictionary specifying split proportions,
        e.g. {"train": 0.7, "val": 0.1, "test": 0.2}.
        Default = {"train": 0.7, "val": 0.1, "test": 0.2}.
    external_validation : bool, default=False
        If True, the test set will consist entirely of samples from the
        selected external site.
    external_site : str, optional
        Site name to hold out for testing if `external_validation=True`.
    site_col : str, default="site"
        Column name containing site identifiers.

    Returns:
    -------
    df_splits : pd.DataFrame
        Original dataframe with an added column
        'split' ∈ {"train", "val", "test"}.
    """
    # Step 1. Input validation
    patients = df.copy()
    missing_vars = [v for v in stratify_vars if v not in patients.columns]
    if missing_vars:
        raise ValueError(f"Missing strat columns in patients: {missing_vars}")

    if split_percents is None:
        split_percents = {"train": 0.7, "val": 0.1, "test": 0.2}

    if external_validation and external_site is None:
        raise ValueError(
            "When external_validation=True, you must specify external_site.",
        )

    # Step 2. Prepare stratification key
    patients["strat_key"] = patients[stratify_vars].astype(str).agg("_".join, axis=1)

    # Step 3: External validation logic
    if external_validation:
        test_df = patients[patients[site_col] == external_site].copy()
        remaining_df = (
            patients[patients[site_col] != external_site].copy().reset_index(drop=True)
            )
        total = split_percents["train"] + split_percents["val"]
        train_ratio = split_percents["train"] / total

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(1 - train_ratio),
            random_state=seed,
        )
        train_idx, val_idx = next(
            splitter.split(remaining_df, remaining_df["strat_key"])
            )
        remaining_df.loc[train_idx, "split"] = "train"
        remaining_df.loc[val_idx, "split"] = "val"
        test_df["split"] = "test"
        df_splits = pd.concat([remaining_df, test_df], axis=0)

    else:
        # Step 4. Internal split (train/val/test)
        train_size = split_percents["train"]
        val_size = split_percents["val"]
        test_size = split_percents["test"]

        if abs(train_size + val_size + test_size - 1.0) >= FLOAT_TOL:
            raise ValueError("Splits must sum to 1.0.")

        splitter1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(1 - train_size),
            random_state=seed,
        )
        train_idx, temp_idx = next(splitter1.split(patients, patients["strat_key"]))
        train_df = patients.iloc[train_idx].copy()
        temp_df = patients.iloc[temp_idx].copy()

        test_ratio = test_size / (val_size + test_size)
        splitter2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=seed,
        )
        val_idx, test_idx = next(splitter2.split(temp_df, temp_df["strat_key"]))
        val_df = temp_df.iloc[val_idx].copy()
        test_df = temp_df.iloc[test_idx].copy()

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        df_splits = pd.concat([train_df, val_df, test_df], axis=0)

    df_splits = df_splits.drop(columns=["strat_key"]).reset_index(drop=True)
    return df_splits


def print_split_report(df_splits: pd.DataFrame) -> None:
    """Print a summary report of dataset splits.

    Includes:
    - Number of patients per split
    - pCR rate per split
    - Subtype distribution per split

    Parameters
    ----------
    df_splits : pd.DataFrame
        DataFrame with columns ['patient_id', 'pcr', 'subtype', 'split'].
    """
    required_cols = {"split", "pcr", "subtype"}
    missing = required_cols - set(df_splits.columns)
    if missing:
        raise ValueError(f"Missing required columns in df_splits: {missing}")

    print("\n=== Sanity Check: Split Summary ===")

    summary_counts = df_splits.groupby("split").size()
    print("\nPatients per split:")
    print(summary_counts)

    summary_pcr = df_splits.groupby("split")["pcr"].mean()
    print("\npCR rate per split:")
    print(summary_pcr.round(3))

    summary_subtype = df_splits.pivot_table(
        index="split", columns="subtype", aggfunc="size", fill_value=0
        ).sort_index()
    print("\nSubtype distribution per split:")
    print(summary_subtype)

    print("\nTotal patients:", len(df_splits))
    print("Report generated successfully.\n")
