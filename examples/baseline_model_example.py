#!/usr/bin/env python3
r"""Baseline Model Example for Centralized Evaluation System (Phase 2).

This script demonstrates the complete workflow for using the evaluation system:
1. Data preparation and loading (synthetic by default, or from CSV)
2. Model definition: random predictor baseline and optional logistic model
3. K-fold cross-validation evaluation
4. Train/test split evaluation
5. Results interpretation and saving

The random model is a sanity-check baseline that outputs random class labels
and random probabilities (no training). It should yield AUC ~0.5 and validates
that the evaluation pipeline works correctly.

Usage:
    # Run random model with default synthetic data (recommended for testing)
    python examples/baseline_model_example.py --model random

    # Run logistic regression baseline with synthetic data
    python examples/baseline_model_example.py --model logistic

    # Run with custom data
    python examples/baseline_model_example.py \
        --model random \
        --features path/to/features.csv \
        --labels path/to/labels.csv \
        --output results/baseline_example

    # Run with Excel-driven splits (site-exclusive, stratified).
    # --excel-metadata alone: synthetic data is generated for the Excel cohort.
    python examples/baseline_model_example.py \
        --model random \
        --excel-metadata path/to/metadata.xlsx \
        --output results/baseline_example

    # Excel + your own features/labels (must cover all Excel patient_ids):
    python examples/baseline_model_example.py \
        --model random \
        --features path/to/features.csv \
        --labels path/to/labels.csv \
        --excel-metadata path/to/metadata.xlsx \
        --output results/baseline_example

Examples:
    python examples/baseline_model_example.py \
        --model random \
        --excel-metadata /net/projects2/vanguard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx \
        --output results/baseline_example_excel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root so we can import evaluation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import (  # noqa: E402
    Evaluator,
    FoldResults,
    FoldSplit,
    TrainTestResults,
    create_splits_from_excel,
)
from src.utils.clinic_metadata import get_patient_ids_from_excel  # noqa: E402
    TrainTestResults,
    plot_random_auc_distribution,
    report_random_baseline,
    save_random_baseline_distribution,
)

# ---------------------------------------------------------------------------
# Section 1: Synthetic data generation
# ---------------------------------------------------------------------------
# The evaluation system is configuration-agnostic: it accepts (X, y, patient_ids)
# regardless of how they were loaded (synthetic, CSV, config, etc.). Model systems
# are responsible for loading data; the evaluator only consumes arrays.


def generate_synthetic_data(
    n_samples: int = 200,
    n_features: int = 10,
    random_state: int = 42,
    patient_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data for testing.

    Why: Enables running the example without external data. The evaluation
    system accepts (X, y, patient_ids) regardless of source.

    Parameters
    ----------
    n_samples : int
        Number of samples (ignored if patient_ids is provided).
    n_features : int
        Number of features (ignored for random model; useful for logistic)
    random_state : int
        Seed for reproducibility
    patient_ids : np.ndarray or None, optional
        If provided, generate exactly len(patient_ids) samples and use these IDs.
        Allows aligning synthetic data to an external cohort (e.g. from Excel).

    Returns:
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Binary labels (0 or 1)
    patient_ids : np.ndarray
        Patient IDs (from argument or generated as "patient_0000", ...)
    """
    ids_provided = patient_ids is not None
    if ids_provided:
        patient_ids = np.asarray(patient_ids)
        n_samples = len(patient_ids)
    rng = np.random.default_rng(random_state)
    # Simple separable-ish data: random features, label from threshold on first feature
    X = rng.standard_normal((n_samples, n_features))
    # Create class imbalance ~30% positive for realism
    threshold = np.percentile(X[:, 0], 70)
    y = (X[:, 0] > threshold).astype(np.int64)
    # Flip a few to add noise
    noise_flip_prob = 0.1
    flip = rng.random(n_samples) < noise_flip_prob
    y[flip] = 1 - y[flip]
    if not ids_provided:
        patient_ids = np.array([f"patient_{i:04d}" for i in range(n_samples)])
    return X, y, patient_ids


def load_data_from_csv(
    features_path: Path,
    labels_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load features and labels from CSV files.

    Expects:
        - features_path: CSV with one row per sample, optional 'patient_id' column
        - labels_path: CSV with columns 'patient_id' (or index), 'label' (0/1),
          and optionally 'stratum' or 'subtype' for subgroup reporting

    Returns:
    -------
    X, y, patient_ids, stratum
        stratum is None if labels have no 'stratum' or 'subtype' column.
    """
    X_df = pd.read_csv(features_path)
    y_df = pd.read_csv(labels_path)

    if "patient_id" in X_df.columns:
        patient_ids = X_df["patient_id"].to_numpy()
        X = X_df.drop(columns=["patient_id"]).to_numpy()
    else:
        patient_ids = np.array([f"sample_{i}" for i in range(len(X_df))])
        X = X_df.to_numpy()

    stratum = None
    if "patient_id" in y_df.columns and "label" in y_df.columns:
        # Align by patient_id if present in both
        merged = X_df.merge(y_df, on="patient_id", how="inner")
        if "patient_id" in merged.columns:
            patient_ids = merged["patient_id"].to_numpy()
            drop_cols = ["patient_id", "label"]
            for col in ("stratum", "subtype"):
                if col in merged.columns:
                    stratum = merged[col].astype(str).to_numpy()
                    drop_cols.append(col)
                    break
            X = merged.drop(columns=drop_cols).to_numpy()
        y = merged["label"].to_numpy().astype(np.int64)
    else:
        y = y_df.iloc[:, 0].to_numpy().astype(np.int64)

    if len(y) != len(X):
        raise ValueError(f"Features and labels length mismatch: {len(X)} vs {len(y)}")
    if len(patient_ids) != len(X):
        patient_ids = np.array([f"sample_{i}" for i in range(len(X))])
    return X, y, patient_ids, stratum


# ---------------------------------------------------------------------------
# Section 2: Random model (Phase 2 baseline)
# ---------------------------------------------------------------------------
# The random model produces random class labels and random probabilities with no
# training. Use it as a sanity-check baseline: AUC should be ~0.5. If your real
# model does not beat the random baseline, something is wrong (data, pipeline, or
# metric). Reproducibility is ensured by passing random_state into the RNG.


def predict_random(
    n_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce random binary predictions and probabilities (no training).

    This is the random model: for each sample we output a random class (0 or 1)
    and a random probability in [0, 1]. Used as a sanity-check baseline (AUC ~0.5)
    and to validate the evaluation pipeline.

    Parameters
    ----------
    n_samples : int
        Number of predictions to generate
    random_state : int
        Seed for reproducibility

    Returns:
    -------
    y_pred : np.ndarray
        Random class labels (0 or 1)
    y_prob : np.ndarray
        Random probabilities for positive class in [0, 1]
    """
    rng = np.random.default_rng(random_state)
    y_pred = rng.integers(0, 2, size=n_samples)
    y_prob = rng.random(size=n_samples).astype(np.float64)
    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Section 3: K-fold evaluation
# ---------------------------------------------------------------------------
# K-fold: evaluator creates splits; we produce predictions per fold and pass
# FoldResults to aggregate_kfold_results(). Use k-fold when you have limited
# data and want stable metric estimates (mean ± std across folds). The evaluator
# never trains the model—we do. We only use split indices to get train/val data.


def run_kfold_random(
    evaluator: Evaluator,
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray | None,
    n_splits: int,
    random_state: int,
    stratum: np.ndarray | None = None,
) -> list[FoldResults]:
    """Run k-fold evaluation using the random model (no training).

    For each fold we only need the validation indices. We generate random
    predictions for the validation set and collect FoldResults. The evaluator
    then aggregates metrics across folds. If stratum is provided (aligned to X/y),
    it is added to predictions for subgroup (stratum-specific) reporting.
    """
    splits = evaluator.create_kfold_splits(
        n_splits=n_splits,
        stratify=True,
        shuffle=True,
    )
    fold_results = []
    for split in splits:
        n_val = len(split.val_indices)
        # Use fold index in seed so different folds get different random predictions
        fold_seed = random_state + split.fold_idx
        y_pred, y_prob = predict_random(n_val, random_state=fold_seed)

        y_true = y[split.val_indices]
        pid = (
            split.val_patient_ids
            if split.val_patient_ids is not None
            else np.array([f"val_{i}" for i in range(n_val)])
        )

        pred_df = pd.DataFrame(
            {
                "patient_id": pid,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }
        )
        if stratum is not None:
            pred_df["stratum"] = stratum[split.val_indices]
        fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))
    return fold_results


def run_kfold_random_with_splits(
    evaluator: Evaluator,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[FoldSplit],
    random_state: int,
    stratum: np.ndarray | None = None,
) -> list[FoldResults]:
    """Run k-fold evaluation using the random model with pre-defined splits (e.g. from Excel)."""
    fold_results = []
    for split in splits:
        n_val = len(split.val_indices)
        fold_seed = random_state + split.fold_idx
        y_pred, y_prob = predict_random(n_val, random_state=fold_seed)
        y_true = y[split.val_indices]
        pid = (
            split.val_patient_ids
            if split.val_patient_ids is not None
            else np.array([f"val_{i}" for i in range(n_val)])
        )
        pred_df = pd.DataFrame(
            {"patient_id": pid, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
        )
        if stratum is not None:
            pred_df["stratum"] = stratum[split.val_indices]
        fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))
    return fold_results


def run_kfold_logistic_with_splits(
    evaluator: Evaluator,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[FoldSplit],
    random_state: int,
    stratum: np.ndarray | None = None,
) -> list[FoldResults]:
    """Run k-fold evaluation with LogisticRegression using pre-defined splits (e.g. from Excel)."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=random_state, max_iter=500)
    fold_results = []
    for split in splits:
        X_train = X[split.train_indices]
        y_train = y[split.train_indices]
        X_val = X[split.val_indices]
        y_val = y[split.val_indices]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        pid = (
            split.val_patient_ids
            if split.val_patient_ids is not None
            else np.array([f"val_{i}" for i in range(len(y_val))])
        )
        pred_df = pd.DataFrame(
            {"patient_id": pid, "y_true": y_val, "y_pred": y_pred, "y_prob": y_prob}
        )
        if stratum is not None:
            pred_df["stratum"] = stratum[split.val_indices]
        fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))
    return fold_results


def run_kfold_logistic(
    evaluator: Evaluator,
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray | None,
    n_splits: int,
    random_state: int,
    stratum: np.ndarray | None = None,
) -> list[FoldResults]:
    """Run k-fold evaluation with LogisticRegression (for comparison)."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=random_state, max_iter=500)
    splits = evaluator.create_kfold_splits(
        n_splits=n_splits,
        stratify=True,
        shuffle=True,
    )
    fold_results = []
    for split in splits:
        # Train on this fold
        X_train = X[split.train_indices]
        y_train = y[split.train_indices]
        X_val = X[split.val_indices]
        y_val = y[split.val_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        pid = (
            split.val_patient_ids
            if split.val_patient_ids is not None
            else np.array([f"val_{i}" for i in range(len(y_val))])
        )
        pred_df = pd.DataFrame(
            {
                "patient_id": pid,
                "y_true": y_val,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }
        )
        if stratum is not None:
            pred_df["stratum"] = stratum[split.val_indices]
        fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))
    return fold_results


# ---------------------------------------------------------------------------
# Section 4: Train/test evaluation
# ---------------------------------------------------------------------------
# Train/test: single split, train on train set, predict on test set. Use when
# you have a fixed test set (e.g. temporal or external cohort). We save with
# run_name="train_test" so results go to output_dir / model_name / train_test /.


def run_train_test_random(
    y_test: np.ndarray,
    patient_ids_test: np.ndarray | None,
    random_state: int,
    stratum_test: np.ndarray | None = None,
) -> TrainTestResults:
    """Train/test evaluation with random model (no training, random predictions on test set)."""
    n = len(y_test)
    y_pred, y_prob = predict_random(n, random_state=random_state)
    if patient_ids_test is None:
        patient_ids_test = np.array([f"test_{i}" for i in range(n)])
    pred_df = pd.DataFrame(
        {
            "patient_id": patient_ids_test,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    )
    if stratum_test is not None:
        pred_df["stratum"] = stratum_test
    # Compute metrics via evaluator's method (we need an evaluator instance for save_results later;
    # for TrainTestResults we pass metrics from compute_metrics)
    from evaluation.metrics import compute_binary_metrics

    metrics = compute_binary_metrics(y_test, y_pred, y_prob)
    return TrainTestResults(
        metrics=metrics,
        predictions=pred_df,
        model_name="baseline_example",  # overwritten when saving
        run_name=None,
    )


def run_train_test_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    patient_ids_test: np.ndarray | None,
    random_state: int,
    stratum_test: np.ndarray | None = None,
) -> TrainTestResults:
    """Train/test evaluation with LogisticRegression."""
    from sklearn.linear_model import LogisticRegression

    from evaluation.metrics import compute_binary_metrics

    model = LogisticRegression(random_state=random_state, max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    if patient_ids_test is None:
        patient_ids_test = np.array([f"test_{i}" for i in range(len(y_test))])
    pred_df = pd.DataFrame(
        {
            "patient_id": patient_ids_test,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    )
    if stratum_test is not None:
        pred_df["stratum"] = stratum_test
    metrics = compute_binary_metrics(y_test, y_pred, y_prob)
    return TrainTestResults(
        metrics=metrics,
        predictions=pred_df,
        model_name="baseline_example",
        run_name=None,
    )


# ---------------------------------------------------------------------------
# Section 5: Main entry and CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline model example."""
    parser = argparse.ArgumentParser(
        description="Baseline model example for the evaluation system (Phase 2). "
        "Demonstrates random model and optional logistic baseline with k-fold and train/test."
    )
    parser.add_argument(
        "--model",
        choices=["random", "logistic"],
        default="random",
        help="Model to run: 'random' (no training, random predictions, AUC ~0.5) or 'logistic'",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Path to features CSV (optional). If not set, use synthetic data.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to labels CSV (optional). Required if --features is set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline_example"),
        help="Output directory for metrics, predictions, and plots.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for k-fold CV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=0.2,
        help="Fraction of data for test set when doing train/test evaluation (only with synthetic data).",
    )
    parser.add_argument(
        "--excel-metadata",
        type=Path,
        default=None,
        help="Path to Excel metadata file. If provided, k-fold splits are from Excel (site-exclusive, stratified). Can be used alone (synthetic data is generated for Excel cohort) or with --features/--labels.",
    )
    args = parser.parse_args()
    if (args.features is None) != (args.labels is None):
        parser.error(
            "Either provide both --features and --labels or neither (synthetic data)."
        )
    return args


def main() -> None:
    """Run the baseline model example (random or logistic) with k-fold and optional train/test."""
    args = parse_args()

    # --- Data preparation ---
    # When only --excel-metadata is set, define cohort from Excel and generate synthetic data for it.
    if (
        args.excel_metadata is not None
        and args.features is None
        and args.labels is None
    ):
        excel_patient_ids = get_patient_ids_from_excel(args.excel_metadata)
        X, y, patient_ids = generate_synthetic_data(
            random_state=args.random_state,
            patient_ids=excel_patient_ids,
        )
        do_train_test = False
        X_train, X_test = X, None
        y_train, y_test = y, None
        pid_test = None
        stratum = None
    elif args.features is not None and args.labels is not None:
        X, y, patient_ids, stratum = load_data_from_csv(args.features, args.labels)
        # For train/test we'd need a fixed split; with CSV we only do k-fold here for simplicity
        do_train_test = False
        X_train, X_test = X, None
        y_train, y_test = y, None
        pid_test = None
    else:
        X, y, patient_ids = generate_synthetic_data(random_state=args.random_state)
        n = len(y)
        from sklearn.model_selection import train_test_split

        idx = np.arange(n)
        train_idx, test_idx = train_test_split(
            idx,
            test_size=args.train_test_split,
            stratify=y,
            random_state=args.random_state,
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pid_test = patient_ids[test_idx]
        stratum = None  # no stratum for synthetic data unless added
        do_train_test = True

    model_name = "random_baseline" if args.model == "random" else "logistic_baseline"

    # --- Evaluator (same API for any model) ---
    evaluator = Evaluator(
        X=X_train,
        y=y_train,
        patient_ids=patient_ids if not do_train_test else patient_ids[train_idx],
        model_name=model_name,
        random_state=args.random_state,
    )

    # --- K-fold evaluation ---
    train_pids = patient_ids if not do_train_test else patient_ids[train_idx]
    if args.excel_metadata is not None:
        splits = create_splits_from_excel(
            excel_path=args.excel_metadata,
            patient_ids=train_pids,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        if args.model == "random":
            fold_results = run_kfold_random_with_splits(
                evaluator,
                X_train,
                y_train,
                splits,
                random_state=args.random_state,
                stratum=stratum,
            )
        else:
            fold_results = run_kfold_logistic_with_splits(
                evaluator,
                X_train,
                y_train,
                splits,
                random_state=args.random_state,
                stratum=stratum,
            )
    elif args.model == "random":
        fold_results = run_kfold_random(
            evaluator,
            X_train,
            y_train,
            train_pids,
            n_splits=args.n_splits,
            random_state=args.random_state,
            stratum=stratum,
        )
    else:
        fold_results = run_kfold_logistic(
            evaluator,
            X_train,
            y_train,
            train_pids,
            n_splits=args.n_splits,
            random_state=args.random_state,
            stratum=stratum,
        )

    kfold_results = evaluator.aggregate_kfold_results(fold_results)

    # Random baseline AUC distribution: compare model AUC to null distribution
    # Default n_runs=1000 generates a robust distribution for statistical comparison
    distribution = evaluator.compute_random_baseline_distribution(n_runs=1000)
    observed_auc = kfold_results.aggregated_metrics.get("auc", {}).get("mean")
    report_random_baseline(distribution, observed_auc=observed_auc)
    out_model_dir = args.output / model_name
    out_model_dir.mkdir(parents=True, exist_ok=True)
    save_random_baseline_distribution(
        distribution, out_model_dir / "random_baseline_distribution.json"
    )
    plots_dir = out_model_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_random_auc_distribution(
        distribution["auc_values"],
        plots_dir / "random_auc_distribution.png",
        observed_auc=observed_auc,
    )

    evaluator.save_results(
        kfold_results, args.output, random_baseline_distribution=distribution
    )

    print(f"K-fold results saved for {model_name}")
    print(f"  Aggregated AUC: {kfold_results.aggregated_metrics.get('auc', {})}")

    # --- Train/test evaluation (only when we have a test set) ---
    stratum_test = (
        stratum[test_idx] if (do_train_test and stratum is not None) else None
    )
    if do_train_test and X_test is not None:
        if args.model == "random":
            tt_results = run_train_test_random(
                y_test,
                pid_test,
                random_state=args.random_state,
                stratum_test=stratum_test,
            )
        else:
            tt_results = run_train_test_logistic(
                X_train,
                y_train,
                X_test,
                y_test,
                pid_test,
                args.random_state,
                stratum_test=stratum_test,
            )
        tt_results.model_name = model_name
        evaluator.save_results(tt_results, args.output, run_name="train_test")
        print("Train/test results saved (run_name=train_test)")
        print(f"  AUC: {tt_results.metrics.get('auc', 'N/A')}")

    print(f"Output directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
