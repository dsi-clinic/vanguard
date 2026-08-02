"""Baseline validity and graph-local S0 repair under the temporal contract."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from gnn.baseline_validity import observed_valid_mask, repair_baselines


def test_validity_threshold_uses_only_case_baselines() -> None:
    """Validity follows the exact precontrast-only case-relative threshold."""
    s0 = np.asarray([100.0, 4.0, np.nan, 10.0])
    valid, threshold = observed_valid_mask(s0)
    # finite median = 10, threshold = 0.5
    assert threshold == pytest.approx(0.5)
    assert valid.tolist() == [True, True, False, True]


def test_imputation_propagates_one_hop_per_simultaneous_round() -> None:
    """A filled value advances by at most one graph hop in each round."""
    graph = nx.path_graph([0, 1, 2])
    result = repair_baselines(
        graph,
        [0, 1, 2],
        np.asarray([10.0, 0.0, 0.0]),
        mode="impute",
    )
    assert result.observed_valid.tolist() == [True, False, False]
    np.testing.assert_allclose(result.s0_filled, [10.0, 10.0, 10.0])
    assert result.imputation_round.tolist() == [0, 1, 2]


def test_component_without_valid_seed_remains_unrepaired() -> None:
    """An all-invalid disconnected component stays masked and unfilled."""
    graph = nx.Graph([(0, 1)])
    graph.add_edge(2, 3)
    result = repair_baselines(
        graph,
        [0, 1, 2, 3],
        np.asarray([10.0, 0.0, 0.0, 0.0]),
        mode="impute",
    )
    assert result.has_baseline.tolist() == [True, True, False, False]
    assert np.isnan(result.s0_filled[2:]).all()
