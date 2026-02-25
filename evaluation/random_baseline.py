"""Distribution of random-model AUCs for significance of performance.

This module implements the null distribution of AUCs from random predictors
on a fixed set of labels. Generate many random models (different seeds), compute
AUC for each, then compare a real model's AUC to this distribution via z-score
or empirical p-value. When the null distribution is roughly normal and well
within [0, 1], Gaussian tail probabilities (e.g. standard normal table) apply;
otherwise prefer the empirical p-value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# Same convention as evaluation.metrics: need both classes for valid AUC
MIN_CLASSES_FOR_BINARY = 2


def generate_random_probs(y_true: np.ndarray, seed: int) -> np.ndarray:
    """Generate random probabilities for positive class (same interface as model y_prob).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (length used for output size only).
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    np.ndarray
        Random probabilities in (0, 1), length len(y_true).
    """
    rng = np.random.default_rng(seed)
    n = len(np.asarray(y_true))
    return rng.random(n).astype(np.float64)


def compute_random_auc_distribution(
    y_true: np.ndarray,
    n_runs: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute the distribution of AUCs from random predictors on fixed labels.

    For each run, generate random probabilities (different seed), compute AUC
    with sklearn.metrics.roc_auc_score. Single-class y_true yields NaN for that run.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (both classes must be present for valid AUC).
    n_runs : int, default=1000
        Number of random predictor runs.
    random_state : int, default=42
        Base random seed; run i uses random_state + i.

    Returns:
    -------
    dict
        Keys: "auc_values" (list of float), "mean", "std", "min", "max", "n_runs".
    """
    y_true = np.asarray(y_true)
    unique_labels = np.unique(y_true)
    if len(unique_labels) < MIN_CLASSES_FOR_BINARY:
        return {
            "auc_values": [],
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_runs": n_runs,
        }

    auc_values = []
    for i in range(n_runs):
        seed = random_state + i
        y_prob = generate_random_probs(y_true, seed)
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        auc_values.append(auc)

    valid = [a for a in auc_values if not np.isnan(a)]
    mean_val = float(np.mean(valid)) if valid else float("nan")
    std_val = (
        float(np.std(valid)) if len(valid) > 1 else (0.0 if valid else float("nan"))
    )
    min_val = float(np.min(valid)) if valid else float("nan")
    max_val = float(np.max(valid)) if valid else float("nan")

    return {
        "auc_values": auc_values,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "n_runs": n_runs,
    }


def empirical_p_value(auc_values: list[float], observed_auc: float) -> float:
    """One-tailed probability that a random model's AUC is >= observed_auc.

    Fraction of auc_values that are >= observed_auc. Ties count as >=.

    Parameters
    ----------
    auc_values : list of float
        AUCs from the null distribution (e.g. from compute_random_auc_distribution).
    observed_auc : float
        AUC of the real model.

    Returns:
    -------
    float
        Value in [0, 1]. Lower means more inconsistent with random guessing.
    """
    if not auc_values:
        return float("nan")
    n = len(auc_values)
    count_ge = sum(1 for a in auc_values if a >= observed_auc)
    return count_ge / n


def z_score(
    observed_auc: float,
    mean_auc: float,
    std_auc: float,
) -> float:
    """Z-score of observed AUC relative to null distribution.

    (observed_auc - mean_auc) / std_auc. Gaussian interpretation is only valid
    when the null distribution is roughly normal and well within [0, 1];
    otherwise prefer empirical_p_value.

    Parameters
    ----------
    observed_auc : float
        AUC of the real model.
    mean_auc : float
        Mean of null AUC distribution.
    std_auc : float
        Standard deviation of null AUC distribution.

    Returns:
    -------
    float
        Z-score. Returns 0.0 when std_auc is 0 to avoid division by zero.
    """
    if std_auc <= 0 or np.isnan(std_auc):
        return 0.0
    return (observed_auc - mean_auc) / std_auc


def report_random_baseline(
    distribution: dict,
    observed_auc: float | None = None,
) -> None:
    """Print random baseline distribution and optional comparison to observed AUC.

    Prints mean ± std and n_runs. If observed_auc is provided, prints z-score
    and empirical p-value. Use empirical p-value when std is large or
    distribution is skewed; Gaussian (z-score) when std is small and mean near 0.5.

    Parameters
    ----------
    distribution : dict
        From compute_random_auc_distribution: mean, std, n_runs, optionally auc_values.
    observed_auc : float, optional
        AUC of the real model to compare.
    """
    mean_val = distribution["mean"]
    std_val = distribution["std"]
    n_runs = distribution["n_runs"]
    print("Random baseline AUC distribution:")
    print(f"  mean = {mean_val:.4f}, std = {std_val:.4f}, n_runs = {n_runs}")

    if observed_auc is not None:
        z = z_score(observed_auc, mean_val, std_val)
        auc_values = distribution.get("auc_values", [])
        p_val = (
            empirical_p_value(auc_values, observed_auc) if auc_values else float("nan")
        )
        print(
            f"  vs observed AUC {observed_auc:.4f}: z = {z:.3f}, P(random >= observed) = {p_val:.4f}"
        )


def save_random_baseline_distribution(distribution: dict, path: Path | str) -> None:
    """Write distribution summary to JSON.

    Saves mean, std, min, max, n_runs, and optionally auc_values. Caller can omit
    auc_values from the dict to keep the file small (e.g. pass a copy without
    auc_values).

    Parameters
    ----------
    distribution : dict
        From compute_random_auc_distribution.
    path : Path or str
        Output JSON path.
    """
    path = Path(path)
    out = {
        "mean": distribution["mean"],
        "std": distribution["std"],
        "min": distribution.get("min", float("nan")),
        "max": distribution.get("max", float("nan")),
        "n_runs": distribution["n_runs"],
    }
    if "auc_values" in distribution and distribution["auc_values"]:
        out["auc_values"] = distribution["auc_values"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f, indent=2)
