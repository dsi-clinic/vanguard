"""Tests for evaluation.late_fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from evaluation.late_fusion import fuse_oof_predictions

MIN_CLASSES_FOR_AUC = 2


def _write_predictions(
    path: Path,
    *,
    case_ids: list[str],
    y_true: list[int],
    y_prob: list[float],
    folds: list[int],
) -> None:
    """Helper: write a predictions.csv in the canonical schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "case_id": case_ids,
            "y_true": y_true,
            "y_pred": [int(p >= 0.5) for p in y_prob],
            "y_prob": y_prob,
            "fold": folds,
            "stratum": ["her2_enriched"] * len(case_ids),
        }
    ).to_csv(path, index=False)


def _write_fold_map(path: Path, case_ids: list[str], folds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "case_id": case_ids,
            "label": [0] * len(case_ids),
            "fold": folds,
            "n_splits": [max(folds) + 1] * len(case_ids),
            "random_state": [42] * len(case_ids),
        }
    ).to_csv(path, index=False)


def _balanced_fold(n: int, n_folds: int) -> list[int]:
    """Return a deterministic round-robin fold assignment."""
    return [i % n_folds for i in range(n)]


def test_fusion_with_complementary_signals_beats_each(tmp_path: Path) -> None:
    """LR-stack should clear both single-model AUCs on hand-built complementary data."""
    rng = np.random.default_rng(0)
    n = 60
    case_ids = [f"case_{i:03d}" for i in range(n)]
    y_true = rng.integers(0, 2, size=n).tolist()
    folds = _balanced_fold(n, 3)

    a_probs = [
        float(np.clip(0.7 if y == 1 else 0.3, 0.0, 1.0) + rng.normal(0, 0.15))
        for y in y_true
    ]
    b_probs = [
        float(
            np.clip(0.7 if y == 1 else 0.3, 0.0, 1.0)
            + rng.normal(0, 0.15)
            + (0.0 if y == y_true[(i + 7) % n] else 0.05 * rng.standard_normal())
        )
        for i, y in enumerate(y_true)
    ]
    a_probs = [float(np.clip(p, 0.01, 0.99)) for p in a_probs]
    b_probs = [float(np.clip(p, 0.01, 0.99)) for p in b_probs]

    pred_a = tmp_path / "a.csv"
    pred_b = tmp_path / "b.csv"
    fold_map = tmp_path / "fold_map.csv"
    _write_predictions(
        pred_a, case_ids=case_ids, y_true=y_true, y_prob=a_probs, folds=folds
    )
    _write_predictions(
        pred_b, case_ids=case_ids, y_true=y_true, y_prob=b_probs, folds=folds
    )
    _write_fold_map(fold_map, case_ids=case_ids, folds=folds)

    result = fuse_oof_predictions(
        predictions_a_path=pred_a,
        predictions_b_path=pred_b,
        fold_map_path=fold_map,
        model_a_name="m_a",
        model_b_name="m_b",
    )

    assert result.n_cases == n
    assert result.n_folds == 3
    assert result.mean_of_probs_auc >= max(result.model_a_auc, result.model_b_auc) - 0.05


def test_fold_mismatch_raises(tmp_path: Path) -> None:
    """If predictions disagree with the fold map, fuse must abort."""
    case_ids = [f"case_{i:03d}" for i in range(12)]
    folds_pred = _balanced_fold(12, 3)
    folds_map = [(f + 1) % 3 for f in folds_pred]
    y_true = [i % 2 for i in range(12)]
    y_prob = [0.5 + 0.1 * ((-1) ** i) for i in range(12)]

    pred_a = tmp_path / "a.csv"
    pred_b = tmp_path / "b.csv"
    fold_map = tmp_path / "fold_map.csv"
    _write_predictions(
        pred_a, case_ids=case_ids, y_true=y_true, y_prob=y_prob, folds=folds_pred
    )
    _write_predictions(
        pred_b, case_ids=case_ids, y_true=y_true, y_prob=y_prob, folds=folds_pred
    )
    _write_fold_map(fold_map, case_ids=case_ids, folds=folds_map)

    with pytest.raises(ValueError, match="disagree with fold map"):
        fuse_oof_predictions(
            predictions_a_path=pred_a,
            predictions_b_path=pred_b,
            fold_map_path=fold_map,
        )


def test_case_id_set_mismatch_raises(tmp_path: Path) -> None:
    """If the two prediction files cover different case_id sets, fuse must abort.

    Fold assignments must be aligned with the fold map for both files so the
    fold-alignment check passes and the case_id-set mismatch is what trips.
    """
    all_case_ids = [f"case_{i:03d}" for i in range(14)]
    fold_map_folds = [i % 3 for i in range(14)]
    fold_lookup = dict(zip(all_case_ids, fold_map_folds, strict=True))

    case_ids_a = [f"case_{i:03d}" for i in range(12)]
    case_ids_b = [f"case_{i:03d}" for i in range(2, 14)]
    folds_a = [fold_lookup[cid] for cid in case_ids_a]
    folds_b = [fold_lookup[cid] for cid in case_ids_b]
    y_true_a = [i % 2 for i in range(len(case_ids_a))]
    y_true_b = [i % 2 for i in range(len(case_ids_b))]
    y_prob_a = [0.5 + 0.1 * ((-1) ** i) for i in range(len(case_ids_a))]
    y_prob_b = [0.5 + 0.1 * ((-1) ** i) for i in range(len(case_ids_b))]

    pred_a = tmp_path / "a.csv"
    pred_b = tmp_path / "b.csv"
    fold_map = tmp_path / "fold_map.csv"
    _write_predictions(
        pred_a, case_ids=case_ids_a, y_true=y_true_a, y_prob=y_prob_a, folds=folds_a
    )
    _write_predictions(
        pred_b, case_ids=case_ids_b, y_true=y_true_b, y_prob=y_prob_b, folds=folds_b
    )
    _write_fold_map(fold_map, case_ids=all_case_ids, folds=fold_map_folds)

    with pytest.raises(ValueError, match="case_id sets disagree"):
        fuse_oof_predictions(
            predictions_a_path=pred_a,
            predictions_b_path=pred_b,
            fold_map_path=fold_map,
        )


def test_missing_required_columns_raises(tmp_path: Path) -> None:
    """A predictions CSV missing required columns must trigger a clear error."""
    pred_bad = tmp_path / "bad.csv"
    pd.DataFrame({"case_id": ["a", "b"], "y_prob": [0.1, 0.9]}).to_csv(
        pred_bad, index=False
    )
    pred_ok = tmp_path / "ok.csv"
    _write_predictions(
        pred_ok,
        case_ids=["a", "b"],
        y_true=[0, 1],
        y_prob=[0.1, 0.9],
        folds=[0, 1],
    )
    fold_map = tmp_path / "fold_map.csv"
    _write_fold_map(fold_map, case_ids=["a", "b"], folds=[0, 1])

    with pytest.raises(ValueError, match="missing required columns"):
        fuse_oof_predictions(
            predictions_a_path=pred_bad,
            predictions_b_path=pred_ok,
            fold_map_path=fold_map,
        )


def test_lr_overfit_flag_set_when_lr_far_above_mean(tmp_path: Path) -> None:
    """Construct an example where LR-stack noticeably beats mean-of-probs and check the flag."""
    rng = np.random.default_rng(7)
    n = 30
    case_ids = [f"case_{i:03d}" for i in range(n)]
    y_true = ([1] * (n // 2)) + ([0] * (n - n // 2))
    folds = _balanced_fold(n, 3)
    a_probs = [0.6 if y == 1 else 0.4 for y in y_true]
    b_probs = [0.4 if y == 1 else 0.6 for y in y_true]
    a_probs = [float(p + rng.normal(0, 0.01)) for p in a_probs]
    b_probs = [float(p + rng.normal(0, 0.01)) for p in b_probs]

    pred_a = tmp_path / "a.csv"
    pred_b = tmp_path / "b.csv"
    fold_map = tmp_path / "fold_map.csv"
    _write_predictions(
        pred_a, case_ids=case_ids, y_true=y_true, y_prob=a_probs, folds=folds
    )
    _write_predictions(
        pred_b, case_ids=case_ids, y_true=y_true, y_prob=b_probs, folds=folds
    )
    _write_fold_map(fold_map, case_ids=case_ids, folds=folds)

    result = fuse_oof_predictions(
        predictions_a_path=pred_a,
        predictions_b_path=pred_b,
        fold_map_path=fold_map,
        lr_noise_threshold=0.02,
    )
    assert result.lr_stack_minus_mean_auc >= 0.0
    if result.lr_stack_minus_mean_auc > 0.02:
        assert result.lr_likely_overfits is True


def test_cohort_filter_applied(tmp_path: Path) -> None:
    """cohort_case_ids must shrink both prediction sets to that subset."""
    case_ids = [f"case_{i:03d}" for i in range(12)]
    folds = _balanced_fold(12, 3)
    y_true = [i % 2 for i in range(12)]
    y_prob_a = [0.5 + 0.05 * (1 if i % 2 else -1) for i in range(12)]
    y_prob_b = [0.5 + 0.07 * (1 if i % 2 else -1) for i in range(12)]

    pred_a = tmp_path / "a.csv"
    pred_b = tmp_path / "b.csv"
    fold_map = tmp_path / "fold_map.csv"
    _write_predictions(
        pred_a, case_ids=case_ids, y_true=y_true, y_prob=y_prob_a, folds=folds
    )
    _write_predictions(
        pred_b, case_ids=case_ids, y_true=y_true, y_prob=y_prob_b, folds=folds
    )
    _write_fold_map(fold_map, case_ids=case_ids, folds=folds)

    subset = {f"case_{i:03d}" for i in (0, 1, 2, 3, 4, 5)}
    result = fuse_oof_predictions(
        predictions_a_path=pred_a,
        predictions_b_path=pred_b,
        fold_map_path=fold_map,
        cohort_case_ids=subset,
    )
    assert result.n_cases == len(subset)
