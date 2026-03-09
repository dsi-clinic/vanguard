"""Module for loading and cleaning clinical feature datasets."""

from pathlib import Path
from typing import Any

import pandas as pd


def get_clinical_features(config: dict[str, Any]) -> pd.DataFrame:
    """Load and clean the high-value clinical features from an Excel file."""
    path = Path(config["data_paths"]["clinical_excel"])
    clinical_data = pd.read_excel(path)

    cols = [
        "patient_id",
        "age",
        "menopause",
        "tumor_subtype",
        "hr",
        "er",
        "pr",
        "her2",
        "nottingham_grade",
        "bmi_group",
    ]

    feature_df = clinical_data[cols].copy()
    feature_df = pd.get_dummies(
        feature_df, columns=["tumor_subtype", "menopause", "bmi_group"]
    )

    return feature_df
