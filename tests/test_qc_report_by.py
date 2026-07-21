"""Tests for the Step 4 QC-reporting adapter seam (report_by sub-source breakdown).

A dataset adapter's ``report_by`` names the column QC/eval metrics should be
broken down by (``None`` for MAMA-MIA; ``"dataset"`` -- the manifest sub-source
-- for UChicago). ``run_pipeline_from_config`` threads it into
``run_evaluation_pipeline``: each fold's predictions get the column attached, and
``Evaluator.save_results`` selects it (via ``_stratum_column``) for the
per-subgroup metric breakdown. When ``report_by`` is ``None`` -- every non-adapter
run -- selection and predictions are byte-for-byte unchanged.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import DEFAULT_CONFIG, ConfigNode, _deep_merge
from evaluation.evaluator import _stratum_column
from tabular.train import prepare_evaluation_context, run_single_fold_from_context


def test_stratum_column_none_report_by_is_unchanged_alias_behavior() -> None:
    """report_by=None selects the first alias, exactly as before."""
    preds = pd.DataFrame({"subtype": ["A", "B"], "y_true": [1, 0]})
    assert _stratum_column(preds) == "subtype"
    assert _stratum_column(preds, report_by=None) == "subtype"

    assert _stratum_column(pd.DataFrame({"y_true": [1]})) is None


def test_stratum_column_prefers_report_by_when_present() -> None:
    """report_by wins over the default aliases when it is a real column."""
    preds = pd.DataFrame({"subtype": ["A", "B"], "dataset": ["simbiosys", "uch_nac"]})
    assert _stratum_column(preds, report_by="dataset") == "dataset"


def test_stratum_column_raises_when_report_by_absent_from_predictions() -> None:
    """A requested report_by column that isn't present is a hard error.

    Falling back to an alias would silently produce a breakdown by some other
    column while the run still appears to have produced the requested UChicago
    sub-source QC.
    """
    preds = pd.DataFrame({"subtype": ["A", "B"]})
    with pytest.raises(KeyError, match="dataset"):
        _stratum_column(preds, report_by="dataset")

    # Also fatal when there is no alias to fall back to at all.
    with pytest.raises(KeyError, match="dataset"):
        _stratum_column(pd.DataFrame({"y_true": [1]}), report_by="dataset")


def _fold_config() -> ConfigNode:
    """Predefined 2-fold config, clinical off, so only the toy feature is used."""
    return ConfigNode._wrap(
        _deep_merge(
            DEFAULT_CONFIG,
            {
                "model_params": {"split_mode": "predefined", "split_col": "fold"},
                "feature_toggles": {"use_clinical": False},
            },
        )
    )


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "pcr": [1, 0, 1, 0],
            "dataset": ["simbiosys", "uch_nac", "simbiosys", "uch_nac"],
            "feature_a": [0.1, 0.9, 0.2, 0.8],
            "fold": [0, 0, 1, 1],
        }
    )


def test_report_by_stored_in_context_and_kept_out_of_features() -> None:
    """report_by flows into the context; the sub-source column is not a feature."""
    context = prepare_evaluation_context(
        _toy_frame(), _fold_config(), report_by="dataset"
    )
    assert context["report_by"] == "dataset"
    assert "dataset" not in context["X"].columns  # sub-source is not a predictor


def test_report_by_column_attached_to_fold_predictions() -> None:
    """A fold's predictions carry the report_by column, so save_results can split."""
    context = prepare_evaluation_context(
        _toy_frame(), _fold_config(), report_by="dataset"
    )
    fold_results, _nested, _override = run_single_fold_from_context(
        context, context["splits"][0]
    )
    preds = fold_results.predictions
    assert "dataset" in preds.columns
    assert set(preds["dataset"]).issubset({"simbiosys", "uch_nac"})


def test_report_by_missing_from_cohort_frame_names_that_frame() -> None:
    """A requested report_by absent from the cohort frame fails at the fold seam.

    Skipping the attach would leave evaluation to fall back to an alias and
    report by some other column. Evaluation's own guard would also catch this,
    but only as "absent from the predictions frame" -- pointing whoever is
    debugging at the wrong file. The cohort frame is where the column actually
    went missing, so the error has to name it.
    """
    frame = _toy_frame().drop(columns=["dataset"])
    context = prepare_evaluation_context(frame, _fold_config(), report_by="dataset")

    with pytest.raises(KeyError, match="cohort frame"):
        run_single_fold_from_context(context, context["splits"][0])


def test_no_report_by_leaves_predictions_without_subsource_column() -> None:
    """Without report_by (non-adapter run), no sub-source column is attached."""
    context = prepare_evaluation_context(_toy_frame(), _fold_config())
    fold_results, _nested, _override = run_single_fold_from_context(
        context, context["splits"][0]
    )
    assert "dataset" not in fold_results.predictions.columns
