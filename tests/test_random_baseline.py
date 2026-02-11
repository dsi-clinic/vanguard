"""Tests for evaluation.random_baseline."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.random_baseline import (
    compute_random_auc_distribution,
    empirical_p_value,
    generate_random_probs,
    z_score,
)

# Test constants (avoid PLR2004 magic values)
N_RUNS = 50
MEAN_LOW = 0.4
MEAN_HIGH = 0.6
P_MID_EXPECTED = 0.6


@pytest.fixture
def binary_y_true() -> np.ndarray:
    """Small binary labels: 20 positives, 30 negatives, fixed seed."""
    rng = np.random.default_rng(42)
    y = np.zeros(50, dtype=np.int64)
    y[:20] = 1
    rng.shuffle(y)
    return y


def test_generate_random_probs_shape(binary_y_true: np.ndarray) -> None:
    """Output length equals len(y_true); values in (0, 1)."""
    out = generate_random_probs(binary_y_true, seed=0)
    assert out.shape == (len(binary_y_true),)
    assert np.all(out > 0) and np.all(out < 1)


def test_compute_random_auc_distribution(binary_y_true: np.ndarray) -> None:
    """With n_runs=50 and balanced y_true, mean close to 0.5; std > 0; keys present."""
    dist = compute_random_auc_distribution(
        binary_y_true, n_runs=N_RUNS, random_state=42
    )
    assert "auc_values" in dist
    assert "mean" in dist
    assert "std" in dist
    assert "n_runs" in dist
    assert dist["n_runs"] == N_RUNS
    assert len(dist["auc_values"]) == N_RUNS
    assert MEAN_LOW <= dist["mean"] <= MEAN_HIGH
    assert dist["std"] > 0


def test_empirical_p_value() -> None:
    """Result in [0, 1]; above all -> ~0; below all -> 1."""
    auc_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    p = empirical_p_value(auc_values, observed_auc=0.8)
    assert 0 <= p <= 1
    assert p == 0.0  # no value >= 0.8

    p_low = empirical_p_value(auc_values, observed_auc=0.2)
    assert p_low == 1.0  # all >= 0.2

    p_mid = empirical_p_value(auc_values, observed_auc=0.5)
    assert 0 <= p_mid <= 1  # 0.5, 0.6, 0.7 are >= 0.5 -> 3/5 = 0.6
    assert p_mid == P_MID_EXPECTED


def test_z_score() -> None:
    """When observed_auc > mean, z_score > 0; std=0 does not crash."""
    assert z_score(0.7, 0.5, 0.1) > 0
    assert z_score(0.3, 0.5, 0.1) < 0
    # std 0 returns 0.0
    assert z_score(0.7, 0.5, 0.0) == 0.0
    assert z_score(0.7, 0.5, float("nan")) == 0.0
