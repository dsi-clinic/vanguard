"""Tests for clinic metadata loading and annotation building."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.clinic_metadata import (
    align_metadata_to_patient_ids,
    build_split_annotations,
    load_clinic_metadata_excel,
)


@pytest.fixture
def sample_metadata_df() -> pd.DataFrame:
    """DataFrame fixture matching expected Excel structure."""
    return pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(1, 31)],
            "site": ["SiteA"] * 10 + ["SiteB"] * 10 + ["SiteC"] * 10,
            "dataset": ["DS1"] * 15 + ["DS2"] * 15,
            "subtype": ["Type1"] * 10 + ["Type2"] * 10 + ["Type1"] * 10,
        }
    )


@pytest.fixture
def mock_excel_path(tmp_path: Path, sample_metadata_df: pd.DataFrame) -> Path:
    """Create a temporary Excel file with known structure."""
    excel_path = tmp_path / "test_metadata.xlsx"
    sample_metadata_df.to_excel(excel_path, index=False, engine="openpyxl")
    return excel_path


# Excel Loading Tests


def test_load_clinic_metadata_excel(
    mock_excel_path: Path, sample_metadata_df: pd.DataFrame
) -> None:
    """Successfully loads Excel file, returns DataFrame with expected columns."""
    loaded_metadata = load_clinic_metadata_excel(mock_excel_path)

    assert isinstance(loaded_metadata, pd.DataFrame)
    assert len(loaded_metadata) == len(sample_metadata_df)
    assert list(loaded_metadata.columns) == list(sample_metadata_df.columns)


def test_load_missing_file(tmp_path: Path) -> None:
    """Raises FileNotFoundError for non-existent path."""
    missing_path = tmp_path / "nonexistent.xlsx"

    with pytest.raises(FileNotFoundError):
        load_clinic_metadata_excel(missing_path)


def test_load_excel_column_names(mock_excel_path: Path) -> None:
    """Column names are preserved."""
    metadata = load_clinic_metadata_excel(mock_excel_path)

    expected_cols = {"patient_id", "site", "dataset", "subtype"}
    assert set(metadata.columns) == expected_cols


# Annotation Building Tests


def test_build_split_annotations_default(sample_metadata_df: pd.DataFrame) -> None:
    """With default config, produces patient_id, group, stratum_key columns."""
    annotations = build_split_annotations(
        sample_metadata_df,
        id_col="patient_id",
        group_col="site",
        stratify_cols=["dataset"],
    )

    expected_cols = {"patient_id", "group", "stratum_key"}
    assert set(annotations.columns) == expected_cols
    assert len(annotations) == len(sample_metadata_df)


def test_build_split_annotations_custom_group(sample_metadata_df: pd.DataFrame) -> None:
    """Custom group_col works correctly."""
    # Create test data with different group column name
    test_df = sample_metadata_df.rename(columns={"site": "institution"})

    annotations = build_split_annotations(
        test_df, id_col="patient_id", group_col="institution", stratify_cols=["dataset"]
    )

    assert "group" in annotations.columns
    assert set(annotations["group"].unique()) == {"SiteA", "SiteB", "SiteC"}


def test_build_split_annotations_composite_stratum(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """Multiple stratify_cols → composite stratum_key."""
    annotations = build_split_annotations(
        sample_metadata_df,
        id_col="patient_id",
        group_col="site",
        stratify_cols=["dataset", "subtype"],
    )

    assert "stratum_key" in annotations.columns
    # Check composite keys exist
    unique_keys = annotations["stratum_key"].unique()
    assert any(
        "|" in str(key) for key in unique_keys
    ), "Should have composite keys with separator"


def test_build_split_annotations_missing_columns(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """Missing required columns raise ValueError with clear message."""
    # Remove required column
    test_df = sample_metadata_df.drop(columns=["site"])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_split_annotations(
            test_df, id_col="patient_id", group_col="site", stratify_cols=["dataset"]
        )


def test_build_split_annotations_handles_nulls() -> None:
    """Missing values in group/stratum columns handled."""
    test_df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "site": ["SiteA", "SiteB", None, "SiteA"],  # One null
            "dataset": ["DS1", "DS1", "DS2", None],  # One null
        }
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        annotations = build_split_annotations(
            test_df, id_col="patient_id", group_col="site", stratify_cols=["dataset"]
        )

        # Should drop rows with nulls
        assert len(annotations) < len(test_df)
        assert len(w) > 0  # Should have warnings


# Join/Alignment Tests


def test_align_metadata_to_patient_ids(sample_metadata_df: pd.DataFrame) -> None:
    """Given patient_ids array, returns aligned groups and stratify_labels arrays."""
    annotations = build_split_annotations(
        sample_metadata_df,
        id_col="patient_id",
        group_col="site",
        stratify_cols=["dataset"],
    )

    patient_ids = np.array([f"P{i}" for i in range(1, 26)])  # First 25
    groups, strata = align_metadata_to_patient_ids(
        annotations, patient_ids, id_col="patient_id"
    )

    assert len(groups) == len(patient_ids)
    assert len(strata) == len(patient_ids)
    assert groups.dtype == object
    assert strata.dtype == object


def test_align_metadata_missing_patients(sample_metadata_df: pd.DataFrame) -> None:
    """Patients in patient_ids but not in metadata → handled with warnings."""
    annotations = build_split_annotations(
        sample_metadata_df,
        id_col="patient_id",
        group_col="site",
        stratify_cols=["dataset"],
    )

    # Include some patient_ids not in metadata
    patient_ids = np.array(["P1", "P2", "P999", "P1000"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        groups, strata = align_metadata_to_patient_ids(
            annotations, patient_ids, id_col="patient_id", warn_missing=True
        )

        assert len(w) > 0  # Should warn about missing patients
        # Missing patients should have NaN
        assert pd.isna(groups[2]) or pd.isna(strata[2])


def test_align_metadata_extra_patients(sample_metadata_df: pd.DataFrame) -> None:
    """Patients in metadata but not in patient_ids → ignored."""
    annotations = build_split_annotations(
        sample_metadata_df,
        id_col="patient_id",
        group_col="site",
        stratify_cols=["dataset"],
    )

    # Use subset of patient_ids
    patient_ids = np.array([f"P{i}" for i in range(1, 21)])  # First 20 only

    groups, strata = align_metadata_to_patient_ids(
        annotations, patient_ids, id_col="patient_id"
    )

    # Should only return values for provided patient_ids
    assert len(groups) == len(patient_ids)
    assert len(strata) == len(patient_ids)


# Data Validation Tests


def test_validate_group_uniqueness(sample_metadata_df: pd.DataFrame) -> None:
    """Each patient_id maps to exactly one group value."""
    annotations = build_split_annotations(
        sample_metadata_df,
        id_col="patient_id",
        group_col="site",
        stratify_cols=["dataset"],
    )

    # Check: each patient_id appears once
    assert annotations["patient_id"].nunique() == len(annotations)
    # Check: each patient_id has exactly one group
    assert annotations.groupby("patient_id")["group"].nunique().all() == 1


def test_validate_stratum_values(sample_metadata_df: pd.DataFrame) -> None:
    """Stratum keys are non-empty strings."""
    annotations = build_split_annotations(
        sample_metadata_df,
        id_col="patient_id",
        group_col="site",
        stratify_cols=["dataset"],
    )

    # All stratum keys should be non-empty strings
    assert annotations["stratum_key"].dtype == object
    assert (annotations["stratum_key"].astype(str).str.len() > 0).all()
