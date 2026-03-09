#!/usr/bin/env python3
"""Unit tests for site_analysis.py using synthetic data.

Run with::

    python -m pytest radiomics_baseline/scripts/test_site_analysis.py -v
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest
from site_analysis import (
    _safe_auc,
    _safe_sens_spec,
    assign_sites,
    extract_site,
    loso_analysis,
    per_site_analysis,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    sites: dict[str, int],
    n_features: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Create synthetic features, labels, and site assignments.

    Parameters
    ----------
    sites : dict mapping site name to number of patients
    n_features : number of random features
    seed : random seed
    """
    rng = np.random.RandomState(seed)
    pids = []
    for site, n in sites.items():
        pids.extend([f"{site}_{i:03d}" for i in range(n)])

    X = pd.DataFrame(
        rng.randn(len(pids), n_features),
        index=pids,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Make labels correlated with first feature for non-trivial AUC
    probs = 1 / (1 + np.exp(-X["feat_0"].values))
    y = (rng.rand(len(pids)) < probs).astype(int)

    sites_series = assign_sites(X.index)
    return X, y, sites_series


def _default_args(**overrides) -> argparse.Namespace:
    """Create a minimal argparse Namespace for classifier construction."""
    defaults = {
        "classifier": "logistic",
        "logreg_C": 1.0,
        "logreg_penalty": "l2",
        "logreg_l1_ratio": 0.0,
        "corr_threshold": 0.0,
        "k_best": 0,
        "feature_selection": "kbest",
        "grid_search": False,
        "include_subtype": False,
        # RF defaults (not used but needed by build_estimator)
        "rf_n_estimators": 100,
        "rf_max_depth": None,
        "rf_min_samples_leaf": 1,
        "rf_min_samples_split": 2,
        "rf_max_features": "sqrt",
        "rf_ccp_alpha": 0.0,
        # XGB defaults
        "xgb_n_estimators": 100,
        "xgb_max_depth": 4,
        "xgb_learning_rate": 0.05,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
        "xgb_reg_lambda": 1.0,
        "xgb_reg_alpha": 0.0,
        "xgb_scale_pos_weight": 1.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Tests: extract_site
# ---------------------------------------------------------------------------


class TestExtractSite:
    def test_duke(self):
        assert extract_site("DUKE_001") == "DUKE"

    def test_ispy1(self):
        assert extract_site("ISPY1_0042") == "ISPY1"

    def test_ispy2(self):
        assert extract_site("ISPY2_1234") == "ISPY2"

    def test_nact(self):
        assert extract_site("NACT_007") == "NACT"

    def test_unknown(self):
        assert extract_site("12345") == "UNKNOWN"

    def test_lowercase(self):
        assert extract_site("duke_001") == "duke"


# ---------------------------------------------------------------------------
# Tests: assign_sites
# ---------------------------------------------------------------------------


class TestAssignSites:
    def test_mixed_index(self):
        idx = pd.Index(["DUKE_001", "ISPY1_002", "DUKE_003", "NACT_001"])
        sites = assign_sites(idx)
        assert list(sites) == ["DUKE", "ISPY1", "DUKE", "NACT"]

    def test_preserves_index(self):
        idx = pd.Index(["DUKE_001", "ISPY2_002"])
        sites = assign_sites(idx)
        assert list(sites.index) == ["DUKE_001", "ISPY2_002"]


# ---------------------------------------------------------------------------
# Tests: safe metrics
# ---------------------------------------------------------------------------


class TestSafeMetrics:
    def test_safe_auc_normal(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        auc = _safe_auc(y_true, y_prob)
        assert auc == 1.0

    def test_safe_auc_single_class(self):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3])
        assert np.isnan(_safe_auc(y_true, y_prob))

    def test_safe_sens_spec(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        sens, spec = _safe_sens_spec(y_true, y_pred)
        assert sens == 1.0
        assert spec == 1.0

    def test_safe_sens_spec_all_same(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 0])
        sens, spec = _safe_sens_spec(y_true, y_pred)
        # spec is NaN because no true negatives exist (0/0)
        assert 0 < sens <= 1.0


# ---------------------------------------------------------------------------
# Tests: per_site_analysis
# ---------------------------------------------------------------------------


class TestPerSiteAnalysis:
    def test_returns_all_sites(self):
        sites = {"DUKE": 40, "ISPY1": 30, "ISPY2": 50}
        X, y, s = _make_synthetic_data(sites, seed=1)

        # Split into train/test
        train_mask = np.array([i % 3 != 0 for i in range(len(X))])
        Xtr = X.iloc[train_mask]
        ytr = y[train_mask]
        Xte = X.iloc[~train_mask]
        yte = y[~train_mask]
        sites_test = assign_sites(Xte.index)

        args = _default_args()
        result = per_site_analysis(Xtr, ytr, Xte, yte, sites_test, args)

        # Should have a key per site + _overall
        assert "DUKE" in result["per_site"]
        assert "ISPY1" in result["per_site"]
        assert "ISPY2" in result["per_site"]
        assert "_overall" in result["per_site"]

    def test_n_counts_sum(self):
        sites = {"DUKE": 40, "ISPY1": 30}
        X, y, s = _make_synthetic_data(sites, seed=2)

        train_mask = np.array([i % 3 != 0 for i in range(len(X))])
        Xtr = X.iloc[train_mask]
        ytr = y[train_mask]
        Xte = X.iloc[~train_mask]
        yte = y[~train_mask]
        sites_test = assign_sites(Xte.index)

        args = _default_args()
        result = per_site_analysis(Xtr, ytr, Xte, yte, sites_test, args)

        # Per-site counts should sum to overall
        site_n = sum(m["n"] for k, m in result["per_site"].items() if k != "_overall")
        assert site_n == result["per_site"]["_overall"]["n"]

    def test_y_prob_length(self):
        sites = {"DUKE": 30, "NACT": 20}
        X, y, _ = _make_synthetic_data(sites, seed=3)

        train_mask = np.array([i % 2 == 0 for i in range(len(X))])
        Xtr, Xte = X.iloc[train_mask], X.iloc[~train_mask]
        ytr, yte = y[train_mask], y[~train_mask]
        sites_test = assign_sites(Xte.index)

        result = per_site_analysis(Xtr, ytr, Xte, yte, sites_test, _default_args())
        assert len(result["y_prob"]) == len(yte)


# ---------------------------------------------------------------------------
# Tests: loso_analysis
# ---------------------------------------------------------------------------


class TestLosoAnalysis:
    def test_covers_all_patients(self):
        sites = {"DUKE": 30, "ISPY1": 25, "NACT": 20}
        X, y, s = _make_synthetic_data(sites, seed=10)

        args = _default_args()
        result = loso_analysis(X, y, s, args)

        preds = result["predictions"]
        # Every patient should appear exactly once
        assert set(preds["patient_id"]) == set(X.index)
        assert len(preds) == len(X)

    def test_all_sites_in_results(self):
        sites = {"DUKE": 30, "ISPY2": 40, "NACT": 20}
        X, y, s = _make_synthetic_data(sites, seed=11)

        result = loso_analysis(X, y, s, _default_args())

        assert "DUKE" in result["loso"]
        assert "ISPY2" in result["loso"]
        assert "NACT" in result["loso"]

    def test_n_train_n_test_consistent(self):
        sites = {"DUKE": 30, "ISPY1": 25}
        X, y, s = _make_synthetic_data(sites, seed=12)

        result = loso_analysis(X, y, s, _default_args())

        for site, m in result["loso"].items():
            assert m["n_train"] + m["n_test"] == len(X)

    def test_held_out_site_in_predictions(self):
        sites = {"DUKE": 30, "ISPY1": 25}
        X, y, s = _make_synthetic_data(sites, seed=13)

        result = loso_analysis(X, y, s, _default_args())
        preds = result["predictions"]

        # DUKE patients should have site="DUKE"
        duke_preds = preds[preds["site"] == "DUKE"]
        for pid in duke_preds["patient_id"]:
            assert pid.startswith("DUKE_")

    def test_works_with_small_site(self):
        """Site with very few patients should not crash."""
        sites = {"DUKE": 40, "TINY": 5}
        X, y, s = _make_synthetic_data(sites, seed=14)

        result = loso_analysis(X, y, s, _default_args())
        # TINY may have NaN AUC (too few samples or single class), but shouldn't crash
        assert "TINY" in result["loso"]


# ---------------------------------------------------------------------------
# Tests: with feature selection
# ---------------------------------------------------------------------------


class TestWithFeatureSelection:
    def test_corr_prune_and_kbest(self):
        """Per-site analysis works with correlation pruning and k-best."""
        sites = {"DUKE": 50, "ISPY1": 40}
        X, y, _ = _make_synthetic_data(sites, n_features=30, seed=20)

        train_mask = np.array([i % 3 != 0 for i in range(len(X))])
        Xtr, Xte = X.iloc[train_mask], X.iloc[~train_mask]
        ytr, yte = y[train_mask], y[~train_mask]
        sites_test = assign_sites(Xte.index)

        args = _default_args(corr_threshold=0.8, k_best=10)
        result = per_site_analysis(Xtr, ytr, Xte, yte, sites_test, args)

        assert "_overall" in result["per_site"]
        assert not np.isnan(result["per_site"]["_overall"]["auc"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
