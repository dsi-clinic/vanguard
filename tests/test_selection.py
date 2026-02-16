"""Tests for evaluation selection criteria and filtering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.selection import (
    SampleSelectionCriteria,
    apply_selection_criteria,
    load_selection_criteria_from_yaml,
)

# Path to MAMA-MIA clinical metadata Excel (used for integration tests when available).
# Update this constant if the dataset location changes.
MAMA_MIA_EXCEL_PATH = Path(
    "/net/projects2/vanguard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx"
)

# Expected row counts from sample_metadata_df fixture (30 rows: DS1/DS2 15 each, SiteA/B/C 10 each, etc.)
N_DS1 = 15
N_DS2 = 15
N_TOTAL_ROWS = 30
N_SITE_A = 10
N_TYPE1 = 20
N_UNILATERAL = 12
N_BILATERAL = 18
N_DS1_AND_SITEA = 10


@pytest.fixture
def sample_metadata_df() -> pd.DataFrame:
    """DataFrame with patient_id, site, dataset, subtype, laterality for selection tests."""
    return pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(1, 31)],
            "site": ["SiteA"] * 10 + ["SiteB"] * 10 + ["SiteC"] * 10,
            "dataset": ["DS1"] * 15 + ["DS2"] * 15,
            "subtype": ["Type1"] * 10 + ["Type2"] * 10 + ["Type1"] * 10,
            "laterality": ["unilateral"] * 12 + ["bilateral"] * 18,
        }
    )


@pytest.fixture
def mock_excel_path(tmp_path: Path, sample_metadata_df: pd.DataFrame) -> Path:
    """Create a temporary Excel file with sample_metadata_df structure."""
    excel_path = tmp_path / "test_metadata.xlsx"
    sample_metadata_df.to_excel(excel_path, index=False, engine="openpyxl")
    return excel_path


# ---------------------------------------------------------------------------
# 7.1 Unit tests for apply_selection_criteria
# ---------------------------------------------------------------------------


def test_apply_selection_criteria_empty_criteria(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """None or empty criteria returns unfiltered DataFrame."""
    result_none = apply_selection_criteria(sample_metadata_df, None)
    assert len(result_none) == len(sample_metadata_df)
    pd.testing.assert_frame_equal(result_none, sample_metadata_df)

    result_empty_dict = apply_selection_criteria(sample_metadata_df, {})
    assert len(result_empty_dict) == len(sample_metadata_df)
    pd.testing.assert_frame_equal(result_empty_dict, sample_metadata_df)


def test_apply_selection_criteria_datasets_only(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """datasets=['DS1'] keeps only DS1 rows; datasets=['DS1','DS2'] keeps both."""
    criteria_ds1 = SampleSelectionCriteria(datasets=["DS1"])
    result_ds1 = apply_selection_criteria(sample_metadata_df, criteria_ds1)
    assert len(result_ds1) == N_DS1
    assert set(result_ds1["dataset"].unique()) == {"DS1"}

    criteria_both = SampleSelectionCriteria(datasets=["DS1", "DS2"])
    result_both = apply_selection_criteria(sample_metadata_df, criteria_both)
    assert len(result_both) == N_TOTAL_ROWS
    assert set(result_both["dataset"].unique()) == {"DS1", "DS2"}


def test_apply_selection_criteria_sites_only(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """sites=['SiteA'] keeps only SiteA rows."""
    criteria = SampleSelectionCriteria(sites=["SiteA"])
    result = apply_selection_criteria(sample_metadata_df, criteria)
    assert len(result) == N_SITE_A
    assert set(result["site"].unique()) == {"SiteA"}


def test_apply_selection_criteria_tumor_types_only(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """tumor_types=['Type1'] keeps only matching subtype rows."""
    criteria = SampleSelectionCriteria(tumor_types=["Type1"])
    result = apply_selection_criteria(sample_metadata_df, criteria)
    assert len(result) == N_TYPE1
    assert set(result["subtype"].unique()) == {"Type1"}


def test_apply_selection_criteria_unilateral_only(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """unilateral_only=True keeps only rows where laterality = 'unilateral'."""
    criteria = SampleSelectionCriteria(unilateral_only=True)
    result = apply_selection_criteria(sample_metadata_df, criteria)
    assert len(result) == N_UNILATERAL
    assert (result["laterality"].str.lower() == "unilateral").all()


def test_apply_selection_criteria_bilateral_only(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """bilateral_only=True keeps only bilateral rows."""
    criteria = SampleSelectionCriteria(bilateral_only=True)
    result = apply_selection_criteria(sample_metadata_df, criteria)
    assert len(result) == N_BILATERAL
    assert (result["laterality"].str.lower() == "bilateral").all()


def test_apply_selection_criteria_and_logic(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """datasets=['DS1'] + sites=['SiteA'] keeps rows satisfying BOTH (AND)."""
    criteria = SampleSelectionCriteria(datasets=["DS1"], sites=["SiteA"])
    result = apply_selection_criteria(sample_metadata_df, criteria)
    # DS1 rows 1-15, SiteA rows 1-10; intersection = rows 1-10 (DS1 and SiteA)
    assert len(result) == N_DS1_AND_SITEA
    assert (result["dataset"] == "DS1").all()
    assert (result["site"] == "SiteA").all()


def test_apply_selection_criteria_missing_column(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """Missing criterion column is skipped; no error, no filtering for that criterion."""
    # DataFrame without 'subtype' column
    df_no_subtype = sample_metadata_df.drop(columns=["subtype"])
    criteria = SampleSelectionCriteria(tumor_types=["Type1"])
    # Expected: tumor_type_col 'subtype' not in df -> skipped, result unfiltered
    result = apply_selection_criteria(df_no_subtype, criteria)
    assert len(result) == len(df_no_subtype)


def test_apply_selection_criteria_dict_format(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """Dict {'dataset': ['DS1']} via column_filters works like SampleSelectionCriteria(datasets=['DS1'])."""
    dict_criteria = {"dataset": ["DS1"]}
    result_dict = apply_selection_criteria(sample_metadata_df, dict_criteria)
    # Dict with "dataset" goes to column_filters (generic column filter)
    assert len(result_dict) == N_DS1
    assert set(result_dict["dataset"].unique()) == {"DS1"}

    # Equivalent SampleSelectionCriteria
    criteria_obj = SampleSelectionCriteria(datasets=["DS1"])
    result_obj = apply_selection_criteria(sample_metadata_df, criteria_obj)
    pd.testing.assert_frame_equal(result_dict, result_obj)


# ---------------------------------------------------------------------------
# 7.2 Integration tests for Excel + selection
# ---------------------------------------------------------------------------


def test_create_splits_from_excel_with_datasets_filter(
    mock_excel_path: Path,
    sample_metadata_df: pd.DataFrame,
) -> None:
    """Create splits with selection_criteria=datasets=['DS1']; splits contain only DS1 patient_ids."""
    from evaluation.kfold import create_splits_from_excel

    patient_ids = sample_metadata_df["patient_id"].astype(str).to_numpy()
    criteria = SampleSelectionCriteria(datasets=["DS1"])
    splits = create_splits_from_excel(
        excel_path=mock_excel_path,
        patient_ids=patient_ids,
        n_splits=3,
        random_state=42,
        selection_criteria=criteria,
    )
    ds1_ids = set(
        sample_metadata_df[sample_metadata_df["dataset"] == "DS1"]["patient_id"].astype(
            str
        )
    )
    for split in splits:
        for pid in split.val_patient_ids:
            assert str(pid) in ds1_ids, f"Val patient {pid} should be in DS1"


def test_create_splits_from_excel_with_stacked_criteria(
    mock_excel_path: Path,
    sample_metadata_df: pd.DataFrame,
) -> None:
    """datasets=['DS1'] + sites=['SiteA']; sample count = intersection."""
    from evaluation.kfold import create_splits_from_excel

    patient_ids = sample_metadata_df["patient_id"].astype(str).to_numpy()
    criteria = SampleSelectionCriteria(datasets=["DS1"], sites=["SiteA"])
    splits = create_splits_from_excel(
        excel_path=mock_excel_path,
        patient_ids=patient_ids,
        n_splits=2,
        random_state=42,
        selection_criteria=criteria,
    )
    # DS1 AND SiteA = 10 rows
    expected = sample_metadata_df[
        (sample_metadata_df["dataset"] == "DS1")
        & (sample_metadata_df["site"] == "SiteA")
    ]
    n_expected = len(expected)
    total_val = sum(len(s.val_patient_ids) for s in splits)
    assert total_val == n_expected


def test_create_splits_from_excel_no_criteria_unchanged(
    mock_excel_path: Path,
    sample_metadata_df: pd.DataFrame,
) -> None:
    """selection_criteria=None yields same behavior as no filtering."""
    from evaluation.kfold import create_splits_from_excel

    patient_ids = sample_metadata_df["patient_id"].astype(str).to_numpy()
    splits_none = create_splits_from_excel(
        excel_path=mock_excel_path,
        patient_ids=patient_ids,
        n_splits=3,
        random_state=42,
        selection_criteria=None,
    )
    splits_default = create_splits_from_excel(
        excel_path=mock_excel_path,
        patient_ids=patient_ids,
        n_splits=3,
        random_state=42,
    )
    total_val_none = sum(len(s.val_patient_ids) for s in splits_none)
    total_val_default = sum(len(s.val_patient_ids) for s in splits_default)
    assert total_val_none == total_val_default == len(sample_metadata_df)


# ---------------------------------------------------------------------------
# 7.3 Edge cases
# ---------------------------------------------------------------------------


def test_apply_selection_criteria_all_excluded(
    sample_metadata_df: pd.DataFrame,
) -> None:
    """Criteria that match no rows returns empty DataFrame."""
    criteria = SampleSelectionCriteria(datasets=["DS99"])
    result = apply_selection_criteria(sample_metadata_df, criteria)
    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(sample_metadata_df.columns)


def test_apply_selection_criteria_unilateral_and_bilateral_mutually_exclusive() -> None:
    """Pass both unilateral_only and bilateral_only; expect ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        SampleSelectionCriteria(unilateral_only=True, bilateral_only=True)


# ---------------------------------------------------------------------------
# Optional: Integration tests with real MAMA-MIA Excel (skipped if file absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not MAMA_MIA_EXCEL_PATH.exists(),
    reason=f"MAMA-MIA Excel not found at {MAMA_MIA_EXCEL_PATH}",
)
def test_create_splits_from_excel_real_dataset_filter() -> None:
    """Create splits on real Excel with datasets filter; verify sample counts."""
    from evaluation.kfold import create_splits_from_excel
    from src.utils.clinic_metadata import load_clinic_metadata_excel

    metadata = load_clinic_metadata_excel(MAMA_MIA_EXCEL_PATH)
    patient_ids = metadata["patient_id"].astype(str).to_numpy()

    # Filter to iSpy2 only
    criteria = SampleSelectionCriteria(datasets=["ISPY2"])
    filtered = apply_selection_criteria(metadata, criteria)
    n_ispy2 = len(filtered)
    if n_ispy2 == 0:
        pytest.skip("No ISPY2 rows in Excel; dataset column may use different values")

    # Create splits with criteria; need patient_ids that include filtered set
    splits = create_splits_from_excel(
        excel_path=MAMA_MIA_EXCEL_PATH,
        patient_ids=patient_ids,
        n_splits=5,
        random_state=42,
        selection_criteria=criteria,
    )
    total_val = sum(len(s.val_patient_ids) for s in splits)
    assert total_val == n_ispy2


@pytest.mark.skipif(
    not MAMA_MIA_EXCEL_PATH.exists(),
    reason=f"MAMA-MIA Excel not found at {MAMA_MIA_EXCEL_PATH}",
)
def test_load_selection_criteria_from_yaml_and_apply_real() -> None:
    """Load criteria from YAML and apply to real Excel."""
    from pathlib import Path

    from src.utils.clinic_metadata import load_clinic_metadata_excel

    config_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "eval_selection_example.yaml"
    )
    if not config_path.exists():
        pytest.skip("eval_selection_example.yaml not found")

    criteria = load_selection_criteria_from_yaml(config_path)
    assert criteria is not None
    assert criteria.datasets == ["ISPY2", "DUKE"]

    metadata = load_clinic_metadata_excel(MAMA_MIA_EXCEL_PATH)
    filtered = apply_selection_criteria(metadata, criteria)
    assert len(filtered) <= len(metadata)
    # If Excel uses same casing as config, we get hits; otherwise filtered may be empty
    if len(filtered) > 0:
        unique_ds = {str(v).lower() for v in filtered["dataset"].unique()}
        assert unique_ds.issubset({"ispy2", "duke"})
