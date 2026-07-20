"""Unit tests for the tabular GNN-feature baseline's per-case summarization.

Focus: ``time_to_enhancement`` must be summarized over observed-arrival nodes
only (excluding ``TTE_NO_ARRIVAL_SENTINEL``), so its mean/quantiles measure
arrival timing rather than a blend of timing and missing-arrival prevalence --
the missingness is carried separately by ``tte_no_arrival_fraction``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("xgboost")

from torch_geometric.data import Data  # noqa: E402

from gnn.data_loader import TTE_NO_ARRIVAL_SENTINEL  # noqa: E402
from tabular.gnn_feature_baseline import summarize_graph  # noqa: E402

_NODE_FEATURES = ("peak_time", "time_to_enhancement", "radius")
_TTE = _NODE_FEATURES.index("time_to_enhancement")


def _graph(tte_values: list[float]) -> Data:
    """A tiny graph whose TTE column is ``tte_values`` (other columns are ones)."""
    n = len(tte_values)
    x = np.ones((n, len(_NODE_FEATURES)), dtype=np.float32)
    x[:, _TTE] = np.asarray(tte_values, dtype=np.float32)
    data = Data(x=torch.from_numpy(x))
    data.edge_index = torch.zeros((2, 0), dtype=torch.long)
    data.num_nodes = n
    data.case_id = "CASE_01"
    data.y = torch.tensor([1])
    data.dataset = "CASE"
    return data


def test_tte_summary_excludes_no_arrival_sentinel() -> None:
    """The sentinel is dropped before the TTE mean/quantiles are computed."""
    # Three observed arrivals at 0.2/0.4/0.6, plus two no-arrival sentinels.
    row = summarize_graph(
        _graph([0.2, 0.4, 0.6, TTE_NO_ARRIVAL_SENTINEL, TTE_NO_ARRIVAL_SENTINEL]),
        _NODE_FEATURES,
    )
    # Mean/median reflect only the observed arrivals, not the -1 sentinels.
    assert row["time_to_enhancement_mean"] == pytest.approx(0.4)
    assert row["time_to_enhancement_q50"] == pytest.approx(0.4)
    assert row["time_to_enhancement_q10"] >= 0.0  # never dragged below 0 by -1
    # Missingness is kept as its own explicit feature: 2 of 5 nodes.
    assert row["tte_no_arrival_fraction"] == pytest.approx(2 / 5)


def test_non_tte_features_still_summarize_all_nodes() -> None:
    """Masking is TTE-specific; other columns keep every node."""
    row = summarize_graph(_graph([0.5, TTE_NO_ARRIVAL_SENTINEL]), _NODE_FEATURES)
    # radius is all ones across both nodes regardless of TTE missingness.
    assert row["radius_mean"] == pytest.approx(1.0)
    assert row["radius_std"] == pytest.approx(0.0)


def test_all_no_arrival_case_yields_nan_timing_and_full_fraction() -> None:
    """A case where no node ever enhanced has undefined timing (NaN), fraction 1.0."""
    row = summarize_graph(
        _graph([TTE_NO_ARRIVAL_SENTINEL, TTE_NO_ARRIVAL_SENTINEL]), _NODE_FEATURES
    )
    assert np.isnan(row["time_to_enhancement_mean"])
    assert np.isnan(row["time_to_enhancement_q50"])
    assert row["tte_no_arrival_fraction"] == pytest.approx(1.0)
