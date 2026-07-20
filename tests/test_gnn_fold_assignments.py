"""Tests for the shared fold-assignment generator (gnn/make_fold_assignments.py).

The generator freezes the evaluation framework's seeded ``StratifiedKFold`` to a
``fold`` column so the voxel/segment/junction arms of the graph comparison all
consume identical per-case folds. These tests pin the three properties that make
that frozen file trustworthy: it covers every case exactly once, it is a
deterministic function of the seed, and it reproduces a direct
``StratifiedKFold(shuffle=True, random_state=...)`` reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from gnn.make_fold_assignments import assign_folds  # noqa: E402

N_CASES = 50
N_SPLITS = 5
SEED = 42
# Label every case with (i % 5) in {0, 1} positive -> a stable ~40% positive rate.
_POS_MOD = 5
_POS_CUTOFF = 2


def _write_labels(path: Path, n: int = N_CASES) -> None:
    """Write a small ``case_id,pcr`` labels CSV with a ~40% positive rate."""
    frame = pd.DataFrame(
        {
            "case_id": [f"C{i:03d}" for i in range(n)],
            "pcr": [int(i % _POS_MOD < _POS_CUTOFF) for i in range(n)],
        }
    )
    frame.to_csv(path, index=False)


def test_assign_folds_covers_every_case_once(tmp_path: Path) -> None:
    """Every case is assigned exactly one fold, and all n_splits folds are used."""
    labels = tmp_path / "labels.csv"
    _write_labels(labels)
    frame = assign_folds(labels, "case_id", "pcr", N_SPLITS, SEED)

    assert len(frame) == N_CASES
    assert frame["case_id"].is_unique
    assert set(frame["fold"].unique()) == set(range(N_SPLITS))


def test_assign_folds_is_deterministic(tmp_path: Path) -> None:
    """The same seed yields the same per-case fold assignment across runs."""
    labels = tmp_path / "labels.csv"
    _write_labels(labels)
    first = assign_folds(labels, "case_id", "pcr", N_SPLITS, SEED).set_index("case_id")[
        "fold"
    ]
    second = assign_folds(labels, "case_id", "pcr", N_SPLITS, SEED).set_index(
        "case_id"
    )["fold"]
    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())


def test_assign_folds_matches_stratified_kfold_reference(tmp_path: Path) -> None:
    """The frozen assignment reproduces a direct StratifiedKFold(seed) reference."""
    from sklearn.model_selection import StratifiedKFold

    labels = tmp_path / "labels.csv"
    _write_labels(labels)
    frame = (
        assign_folds(labels, "case_id", "pcr", N_SPLITS, SEED)
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    # Reproduce the framework's split on the same case_id-sorted order.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y = frame["pcr"].to_numpy()
    ref_fold = np.empty(len(frame), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros((len(frame), 1)), y)):
        ref_fold[val_idx] = fold_idx

    assert list(frame["fold"]) == list(ref_fold)
