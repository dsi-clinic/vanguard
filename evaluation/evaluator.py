"""Main Evaluator class for model evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.kfold import (
    FoldSplit,
    create_group_stratified_kfold_splits,
    create_kfold_splits,
)
from evaluation.logging_config import get_logger
from evaluation.metrics import (
    aggregate_fold_metrics,
    compute_binary_metrics,
    compute_metrics_by_group,
)
from evaluation.random_baseline import (
    compute_random_auc_distribution,
    empirical_p_value,
    z_score,
)
from evaluation.types import FoldResults, KFoldResults, TrainTestResults
from evaluation.utils import (
    align_data,
    validate_inputs,
)
from evaluation.visualizations import (
    VISUALIZATION_REGISTRY,
    plot_auc_distribution,
    plot_pr_per_split,
    plot_roc_per_split,
)

# Column name(s) used for stratum reporting (first present in predictions is used)
STRATUM_COLUMN_ALIASES = ("stratum", "subtype")


def _stratum_column(predictions: pd.DataFrame) -> str | None:
    """Return the first stratum column name present in predictions, or None."""
    for col in STRATUM_COLUMN_ALIASES:
        if col in predictions.columns:
            return col
    return None


def _print_validation_summary(
    validation_summary: dict,
    stratum_col: str,
) -> None:
    """Log overall and per-stratum metrics (e.g. AUC)."""
    logger = get_logger()
    overall = validation_summary.get("overall", {})
    by_group = validation_summary.get("by_group", {})
    logger.info("Validation summary:")
    if "auc" in overall:
        logger.info("  AUC (overall): %.3f", overall["auc"])
    for i, (stratum_name, metrics) in enumerate(by_group.items(), start=1):
        auc_val = metrics.get("auc", float("nan"))
        auc_str = (
            f"{auc_val:.3f}"
            if not (isinstance(auc_val, float) and np.isnan(auc_val))
            else "nan"
        )
        logger.info("  AUC (Strata %d / %s): %s", i, stratum_name, auc_str)


class Evaluator:
    """Main evaluator class for model evaluation with k-fold cross-validation.

    The evaluator generates splits and aggregates results, but does not train models.
    Models handle their own training and return predictions/metrics to the evaluator.
    """

    def __init__(
        self: Evaluator,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        patient_ids: np.ndarray | pd.Series | None = None,
        model_name: str = "model",
        random_state: int = 42,
    ) -> None:
        """Initialize evaluator with data.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Feature matrix
        y : np.ndarray | pd.Series
            Target labels
        patient_ids : np.ndarray | pd.Series, optional
            Patient IDs for tracking (recommended)
        model_name : str, default="model"
            Name of the model (e.g., "radiomics_baseline", "non_imaging_baseline").
            Used for organizing outputs and comparing multiple models.
        random_state : int, default=42
            Random seed for reproducibility

        Note: This accepts data directly (arrays/DataFrames), not file paths.
        Model systems are responsible for loading data from their configuration
        (CLI args, config files, config objects, etc.) and passing it here.

        Note: Model is NOT passed here - models handle their own training.
        """
        # Validate inputs
        validate_inputs(X, y, patient_ids)

        # Align and store data
        self.X, self.y, self.patient_ids = align_data(X, y, patient_ids)
        self.model_name = model_name
        self.random_state = random_state

    def create_kfold_splits(
        self: Evaluator,
        n_splits: int = 5,
        stratify: bool = True,
        shuffle: bool = True,
        groups: np.ndarray | None = None,
        stratify_labels: np.ndarray | None = None,
        validate_exclusivity: bool = True,
        return_report: bool = False,
    ) -> list[FoldSplit] | tuple[list[FoldSplit], dict]:
        """Create k-fold splits and return them to the model.

        Supports both standard stratified k-fold and group-stratified k-fold
        (group-exclusive folds with stratification by stratify_labels).

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds.
        stratify : bool, default=True
            Whether to use stratified k-fold (maintains class distribution).
            Ignored if groups/stratify_labels are provided (uses group-stratified).
        shuffle : bool, default=True
            Whether to shuffle data before splitting.
        groups : np.ndarray | None, default=None
            Group assignments for each sample (e.g. site). If provided along with
            stratify_labels, uses group-stratified k-fold so groups do not cross folds.
        stratify_labels : np.ndarray | None, default=None
            Stratification labels (e.g. subtype/dataset). If provided along with
            groups, used for group-stratified splitting. If provided alone,
            used for standard stratified splitting instead of y.
        validate_exclusivity : bool, default=True
            Whether to validate and warn if groups cross folds (only used when
            groups are provided).
        return_report : bool, default=False
            If True, also return a report dictionary with distribution statistics
            (only used when groups are provided).

        Returns:
        -------
        list[FoldSplit] | tuple[list[FoldSplit], dict]
            List of FoldSplit objects, one per fold, containing:
            - train_indices: indices for training
            - val_indices: indices for validation
            - train_patient_ids: patient IDs for training (if available)
            - val_patient_ids: patient IDs for validation (if available)
            If return_report=True and groups provided, returns tuple of (splits, report_dict).

        Examples:
        --------
        >>> # Standard stratified k-fold (existing behavior)
        >>> splits = evaluator.create_kfold_splits(n_splits=5)
        >>>
        >>> # Group-stratified k-fold (site-exclusive, subtype-stratified)
        >>> splits = evaluator.create_kfold_splits(
        ...     n_splits=5,
        ...     groups=site_array,
        ...     stratify_labels=subtype_array
        ... )
        """
        # Use group-stratified splitting if both groups and stratify_labels provided
        if groups is not None and stratify_labels is not None:
            result = create_group_stratified_kfold_splits(
                X=self.X,
                y=self.y,
                groups=groups,
                stratify_labels=stratify_labels,
                patient_ids=self.patient_ids,
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=self.random_state,
                validate_exclusivity=validate_exclusivity,
                return_report=return_report,
            )
            if return_report:
                split_dicts, report_dict = result
            else:
                split_dicts = result
                report_dict = None
        # Use standard stratified splitting with custom stratify_labels if provided alone
        elif stratify_labels is not None:
            # Use stratify_labels instead of y for stratification
            split_dicts = create_kfold_splits(
                X=self.X,
                y=stratify_labels,  # Use stratify_labels for stratification
                patient_ids=self.patient_ids,
                n_splits=n_splits,
                stratify=True,  # Always stratify when stratify_labels provided
                shuffle=shuffle,
                random_state=self.random_state,
            )
            report_dict = None
        # Standard behavior: use existing create_kfold_splits
        else:
            split_dicts = create_kfold_splits(
                X=self.X,
                y=self.y,
                patient_ids=self.patient_ids,
                n_splits=n_splits,
                stratify=stratify,
                shuffle=shuffle,
                random_state=self.random_state,
            )
            report_dict = None

        # Convert to FoldSplit objects
        splits = []
        for split_dict in split_dicts:
            splits.append(
                FoldSplit(
                    fold_idx=split_dict["fold_idx"],
                    train_indices=split_dict["train_indices"],
                    val_indices=split_dict["val_indices"],
                    train_patient_ids=split_dict["train_patient_ids"],
                    val_patient_ids=split_dict["val_patient_ids"],
                )
            )

        if return_report and report_dict is not None:
            return splits, report_dict

        return splits

    def aggregate_kfold_results(
        self: Evaluator,
        fold_results: list[FoldResults],
    ) -> KFoldResults:
        """Aggregate predictions and metrics from model across all folds.

        Parameters
        ----------
        fold_results : list[FoldResults]
            List of results from model, one per fold. Each FoldResults contains:
            - fold_idx: fold number
            - predictions: DataFrame with patient_id, y_true, y_pred, y_prob
            - metrics: dict of metrics for this fold (optional, can compute from predictions)

        Returns:
        -------
        KFoldResults
            Aggregated results with mean ± std metrics across folds
        """
        if not fold_results:
            raise ValueError("fold_results cannot be empty")

        # Extract metrics for each fold (compute if not provided)
        fold_metrics_list = []
        all_predictions = []

        for fold_result in fold_results:
            # Compute metrics if not provided
            if fold_result.metrics is None:
                y_true = fold_result.predictions["y_true"].to_numpy()
                y_pred = fold_result.predictions["y_pred"].to_numpy()
                y_prob = fold_result.predictions["y_prob"].to_numpy()
                metrics = compute_binary_metrics(y_true, y_pred, y_prob)
            else:
                metrics = fold_result.metrics

            fold_metrics_list.append(metrics)

            # Add fold column to predictions
            preds_with_fold = fold_result.predictions.copy()
            preds_with_fold["fold"] = fold_result.fold_idx
            all_predictions.append(preds_with_fold)

        # Aggregate metrics
        aggregated_metrics = aggregate_fold_metrics(fold_metrics_list)

        # Combine all predictions
        combined_predictions = pd.concat(all_predictions, ignore_index=True)

        return KFoldResults(
            fold_metrics=fold_metrics_list,
            aggregated_metrics=aggregated_metrics,
            predictions=combined_predictions,
            n_splits=len(fold_results),
            model_name=self.model_name,
        )

    def compute_metrics(
        self: Evaluator,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, float]:
        """Compute metrics from predictions.

        Can be called by models if they want metrics for a single fold,
        or used internally during aggregation.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        y_pred : np.ndarray
            Predicted binary labels
        y_prob : np.ndarray
            Predicted probabilities for positive class

        Returns:
        -------
        dict[str, float]
            Dictionary of computed metrics
        """
        return compute_binary_metrics(y_true, y_pred, y_prob)

    def compute_random_baseline_distribution(
        self: Evaluator,
        n_runs: int = 1000,
        random_state: int | None = None,
    ) -> dict:
        """Compute the distribution of AUCs from random predictors on evaluator labels.

        Uses the same labels (self.y) as used for splits. Returns dict with
        auc_values, mean, std, n_runs.

        Parameters
        ----------
        n_runs : int, default=1000
            Number of random predictor runs.
        random_state : int, optional
            Base random seed. If None, uses self.random_state.

        Returns:
        -------
        dict
            From compute_random_auc_distribution: auc_values, mean, std, n_runs.
        """
        rs = self.random_state if random_state is None else random_state
        return compute_random_auc_distribution(self.y, n_runs=n_runs, random_state=rs)

    def save_results(
        self: Evaluator,
        results: KFoldResults | TrainTestResults,
        output_dir: Path,
        run_name: str | None = None,
        random_baseline_distribution: dict | None = None,
        run_aucs: list[float] | None = None,
    ) -> None:
        """Save results to output directory, organized by model name and run name.

        Writes predictions CSV, metrics JSON (with per-fold metrics for k-fold),
        and registered visualizations. If results.predictions has a stratum
        column (e.g. "stratum" or "subtype"), per-stratum metrics are computed,
        added to metrics, and printed.

        Parameters
        ----------
        results : KFoldResults | TrainTestResults
            Evaluation results to save.
        output_dir : Path
            Base output directory. Results are saved to
            output_dir / model_name / run_name (if run_name provided), else
            output_dir / model_name.
        run_name : str, optional
            Name of this run (e.g. "run_001", "experiment_1"). If None, results
            are saved directly under model_name.
        random_baseline_distribution : dict, optional
            Precomputed null distribution from compute_random_baseline_distribution.
            If provided, adds random_baseline to metrics and, when results
            contain AUC, adds z_score_vs_random and p_value_vs_random.
        run_aucs : list[float], optional
            AUC values from multiple k-fold runs. If provided (e.g. when n_runs > 1),
            adds run_aucs, run_auc_mean, run_auc_std to metrics.

        Notes:
        -----
        output_dir is a Path. Model systems pass the path from their config.
        """
        output_dir = Path(output_dir)

        # Determine final output directory
        if run_name:
            final_output_dir = output_dir / self.model_name / run_name
            results.run_name = run_name
        else:
            final_output_dir = output_dir / self.model_name

        final_output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        predictions_path = final_output_dir / "predictions.csv"
        results.predictions.to_csv(predictions_path, index=False)

        # Prepare metrics dictionary
        if isinstance(results, KFoldResults):
            metrics_dict = {
                "evaluation_type": "kfold",
                "model_name": results.model_name,
                "n_splits": results.n_splits,
                "aggregated_metrics": results.aggregated_metrics,
                "per_fold_metrics": [
                    {"fold": i, **metrics}
                    for i, metrics in enumerate(results.fold_metrics)
                ],
                "n_samples": len(results.predictions),
                "n_features": self.X.shape[1] if len(self.X.shape) > 1 else 1,
            }
            if results.run_name:
                metrics_dict["run_name"] = results.run_name

            # Save per-fold metrics separately
            per_fold_path = final_output_dir / "metrics_per_fold.json"
            import json

            with per_fold_path.open("w") as f:
                json.dump(metrics_dict["per_fold_metrics"], f, indent=2)
        else:
            metrics_dict = {
                "evaluation_type": "train_test",
                "model_name": results.model_name,
                "metrics": results.metrics,
                "n_samples": len(results.predictions),
                "n_features": self.X.shape[1] if len(self.X.shape) > 1 else 1,
            }
            if results.run_name:
                metrics_dict["run_name"] = results.run_name

        # Optional: random baseline comparison
        if random_baseline_distribution is not None:
            metrics_dict["random_baseline"] = {
                "mean": random_baseline_distribution["mean"],
                "std": random_baseline_distribution["std"],
                "min": random_baseline_distribution.get("min", float("nan")),
                "max": random_baseline_distribution.get("max", float("nan")),
                "n_runs": random_baseline_distribution["n_runs"],
            }
            observed_auc = None
            if (
                isinstance(results, KFoldResults)
                and "auc" in results.aggregated_metrics
            ):
                observed_auc = results.aggregated_metrics["auc"].get("mean")
            elif isinstance(results, TrainTestResults) and "auc" in results.metrics:
                observed_auc = results.metrics["auc"]
            if observed_auc is not None and not np.isnan(observed_auc):
                mean_r = random_baseline_distribution["mean"]
                std_r = random_baseline_distribution["std"]
                metrics_dict["z_score_vs_random"] = z_score(observed_auc, mean_r, std_r)
                auc_values = random_baseline_distribution.get("auc_values", [])
                metrics_dict["p_value_vs_random"] = (
                    empirical_p_value(auc_values, observed_auc)
                    if auc_values
                    else float("nan")
                )

        # Multi-run AUC summary
        if run_aucs is not None and len(run_aucs) > 0:
            arr = np.asarray(run_aucs)
            arr = arr[~np.isnan(arr)]
            if arr.size > 0:
                metrics_dict["run_aucs"] = [float(x) for x in run_aucs]
                metrics_dict["run_auc_mean"] = float(np.mean(arr))
                metrics_dict["run_auc_std"] = float(np.std(arr))

        # Subgroup (stratum) metrics: compute, add to dict, and print summary
        stratum_col = _stratum_column(results.predictions)
        if stratum_col is not None:
            validation_summary = compute_metrics_by_group(
                results.predictions, stratum_col
            )
            metrics_dict["validation_summary"] = validation_summary

        # Save metrics
        metrics_path = final_output_dir / "metrics.json"
        import json

        with metrics_path.open("w") as f:
            json.dump(metrics_dict, f, indent=2)

        # Print validation summary (full set + stratum-specific)
        if stratum_col is not None and "validation_summary" in metrics_dict:
            _print_validation_summary(metrics_dict["validation_summary"], stratum_col)

        # Generate and save visualizations
        plots_dir = final_output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        y_true = results.predictions["y_true"].to_numpy()
        y_prob = results.predictions["y_prob"].to_numpy()

        # Generate all registered visualizations
        for viz_name, viz_func in VISUALIZATION_REGISTRY.items():
            output_path = plots_dir / f"{viz_name}.png"
            try:
                viz_func(y_true, y_prob, output_path)
            except Exception as e:
                # If visualization fails, log but continue
                get_logger().warning("Failed to generate %s: %s", viz_name, e)

        # K-fold: per-split ROC and PR curves
        if isinstance(results, KFoldResults) and "fold" in results.predictions.columns:
            try:
                plot_roc_per_split(
                    results.predictions,
                    plots_dir / "roc_per_split.png",
                    title="ROC Per Split",
                    fold_col="fold",
                )
            except Exception as e:
                get_logger().warning("Failed to generate roc_per_split: %s", e)
            try:
                plot_pr_per_split(
                    results.predictions,
                    plots_dir / "pr_per_split.png",
                    title="PR Per Split",
                    fold_col="fold",
                )
            except Exception as e:
                get_logger().warning("Failed to generate pr_per_split: %s", e)

        # Multi-run AUC distribution
        if run_aucs is not None and len(run_aucs) > 0:
            try:
                plot_auc_distribution(
                    run_aucs,
                    plots_dir / "auc_distribution.png",
                    title="AUC distribution across runs",
                )
            except Exception as e:
                get_logger().warning("Failed to generate auc_distribution: %s", e)
