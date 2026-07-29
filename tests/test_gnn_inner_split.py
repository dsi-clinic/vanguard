"""Unit tests for the inner train/early-stop split in ``gnn.train``.

``_stratified_inner_split`` fixes model-selection leakage in
:func:`gnn.train.fit_predict_one_fold`: the split must be carved out of a fold's
OWN training cases (never the outer held-out fold), be stratified on the binary
pCR label, be deterministic given a seed, and fail loudly on a fold too small to
hold out a stratified split.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")

from torch_geometric.data import Data  # noqa: E402

from gnn.train import _stratified_inner_split  # noqa: E402


def _graphs(labels: list[int]) -> list[Data]:
    """One trivial single-node graph per label; only ``.y`` matters here."""
    return [
        Data(x=torch.zeros(1, 1), y=torch.tensor([float(label)])) for label in labels
    ]


def test_split_partitions_without_overlap() -> None:
    """Inner-train and inner-val together cover every graph, with no overlap."""
    graphs = _graphs([1] * 8 + [0] * 24)
    inner_train, inner_val = _stratified_inner_split(
        graphs, random_state=0, val_fraction=0.2
    )
    train_ids = {id(g) for g in inner_train}
    val_ids = {id(g) for g in inner_val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {id(g) for g in graphs}


def test_split_is_stratified() -> None:
    """A 0.2 hold-out keeps the minority class in both inner splits."""
    graphs = _graphs([1] * 8 + [0] * 24)
    inner_train, inner_val = _stratified_inner_split(
        graphs, random_state=0, val_fraction=0.2
    )

    def positives(gs: list[Data]) -> int:
        return sum(int(g.y.item()) for g in gs)

    assert positives(inner_val) >= 1
    assert positives(inner_train) >= 1
    # ~20% held out (StratifiedKFold with n_splits=5).
    assert len(inner_val) == pytest.approx(len(graphs) * 0.2, abs=2)


def test_split_is_deterministic() -> None:
    """The same random_state reproduces the same inner-train / inner-val split."""
    graphs = _graphs([1] * 8 + [0] * 24)
    a_train, a_val = _stratified_inner_split(graphs, random_state=7, val_fraction=0.2)
    b_train, b_val = _stratified_inner_split(graphs, random_state=7, val_fraction=0.2)
    assert [id(g) for g in a_train] == [id(g) for g in b_train]
    assert [id(g) for g in a_val] == [id(g) for g in b_val]


def test_too_few_positives_fails_loudly() -> None:
    """A fold with only one positive raises instead of silently degrading."""
    graphs = _graphs([1] * 1 + [0] * 20)
    with pytest.raises(ValueError, match="stratified inner early-stopping split"):
        _stratified_inner_split(graphs, random_state=0, val_fraction=0.2)
