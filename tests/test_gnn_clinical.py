"""Unit tests for graph-level clinical feature encoding (``gnn.clinical``).

Exercises ``build_clinical_feature_matrix`` directly on small synthetic
clinical dataframes -- no cluster data required -- covering imputation,
one-hot encoding, ``menopausal_status`` normalization (grounded in the real
raw-string variants observed in the MAMA-MIA cohort, see
``AUDITING_RESULTS.md``), and the genuine-clinical vs. site/scanner column
split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnn.clinical import (
    DEFAULT_CLINICAL_COLUMNS,
    SITE_CLINICAL_COLUMNS,
    build_clinical_feature_matrix,
)


def _clinical_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["A", "B", "C", "D"],
            "age": [40, None, 60, 50],
            "menopausal_status": [
                "pre",
                "pre (<6 months since LMP AND no prior bilateral ovariectomy "
                "AND not on estrogen replacement)",
                "post",
                "",
            ],
            "tumor_subtype": ["luminal_a", "triple_negative", None, "luminal_a"],
            "breast_density": ["", "", "", ""],
            "field_strength": [1.5, 3.0, 1.5, 3.0],
            "scanner_manufacturer": ["SIEMENS", "GE", "SIEMENS", "GE"],
        }
    )


def test_numeric_column_mean_imputed() -> None:
    """A missing age is filled with the mean of the observed ages."""
    matrix, names = build_clinical_feature_matrix(
        _clinical_df(), ["A", "B", "C", "D"], ["age"]
    )
    assert names == ["num__age"]
    expected_mean = np.mean([40, 60, 50])
    assert matrix[:, 0].tolist() == pytest.approx([40.0, expected_mean, 60.0, 50.0])


def test_menopausal_status_variants_collapse_to_one_category() -> None:
    """The two wording variants of 'pre' land in the same one-hot column."""
    matrix, names = build_clinical_feature_matrix(
        _clinical_df(), ["A", "B", "C", "D"], ["menopausal_status"]
    )
    pre_col = names.index("cat__menopausal_status_pre")
    # A ("pre") and B (the long-wording variant) both encode as pre=1.
    assert matrix[0, pre_col] == 1.0
    assert matrix[1, pre_col] == 1.0
    # C ("post") does not.
    assert matrix[2, pre_col] == 0.0


def test_menopausal_status_missing_value_imputed_to_most_frequent() -> None:
    """Case D's blank menopausal_status is most-frequent-imputed (2/3 'pre')."""
    matrix, names = build_clinical_feature_matrix(
        _clinical_df(), ["A", "B", "C", "D"], ["menopausal_status"]
    )
    pre_col = names.index("cat__menopausal_status_pre")
    # Among A/B/C, "pre" (A, B-normalized) outnumbers "post" (C) 2-to-1, so
    # the most-frequent imputation for D's blank value is "pre".
    assert matrix[3, pre_col] == 1.0


def test_unrecognized_menopausal_status_value_raises() -> None:
    """A raw value outside the exhaustive normalization map is not silently bucketed."""
    clinical_df = _clinical_df()
    clinical_df.loc[0, "menopausal_status"] = "surgically induced"
    with pytest.raises(ValueError, match="Unrecognized menopausal_status"):
        build_clinical_feature_matrix(
            clinical_df, ["A", "B", "C", "D"], ["menopausal_status"]
        )


def test_breast_density_excluded_from_default_columns() -> None:
    """breast_density is not in the default set (96% missing in the real cohort)."""
    assert "breast_density" not in DEFAULT_CLINICAL_COLUMNS
    assert "age" in DEFAULT_CLINICAL_COLUMNS
    assert "menopausal_status" in DEFAULT_CLINICAL_COLUMNS
    assert "tumor_subtype" in DEFAULT_CLINICAL_COLUMNS


def test_breast_density_still_available_as_opt_in() -> None:
    """breast_density can still be requested explicitly when not 100% missing."""
    clinical_df = _clinical_df()
    clinical_df.loc[0, "breast_density"] = "c"  # one real value, matching reality
    matrix, names = build_clinical_feature_matrix(
        clinical_df, ["A", "B", "C", "D"], ["breast_density"]
    )
    assert len(names) >= 1
    assert matrix.shape == (4, len(names))


def test_breast_density_fully_missing_column_raises() -> None:
    """A column with zero non-missing values in the cohort can't be imputed -- raises."""
    with pytest.raises(ValueError):
        build_clinical_feature_matrix(
            _clinical_df(), ["A", "B", "C", "D"], ["breast_density"]
        )


def test_site_columns_available_but_not_default() -> None:
    """Site/scanner columns are a distinct, opt-in-only vocabulary."""
    assert not set(SITE_CLINICAL_COLUMNS) & set(DEFAULT_CLINICAL_COLUMNS)
    matrix, names = build_clinical_feature_matrix(
        _clinical_df(), ["A", "B", "C", "D"], ["field_strength", "scanner_manufacturer"]
    )
    assert "num__field_strength" in names
    assert matrix.shape == (4, len(names))


def test_unknown_column_raises() -> None:
    """A requested column outside the supported vocabulary raises."""
    with pytest.raises(ValueError, match="Unknown clinical columns"):
        build_clinical_feature_matrix(_clinical_df(), ["A"], ["not_a_real_column"])


def test_missing_case_id_raises() -> None:
    """A case_id absent from clinical_df is a caller bug, not an expected case."""
    with pytest.raises(KeyError, match="No clinical row"):
        build_clinical_feature_matrix(_clinical_df(), ["A", "ZZZ"], ["age"])


def test_output_row_order_matches_case_ids_order() -> None:
    """Rows are aligned to the requested case_ids order, not clinical_df's own order."""
    matrix, _names = build_clinical_feature_matrix(_clinical_df(), ["C", "A"], ["age"])
    assert matrix[:, 0].tolist() == [60.0, 40.0]
