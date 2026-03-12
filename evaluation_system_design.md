# Centralized Model Evaluation System Design

## Overview

This document describes the design for a centralized evaluation system that supports k-fold cross-validation and standardized metric computation for model systems across the vanguard repository. The system will be used by individual model systems (e.g., `radiomics_baseline/radiomics_train.py`, `non_imaging_baseline/baseline_pcr_simple.py`) to perform consistent, reproducible evaluation.

## Goals

1. **Centralization**: Single source of truth for evaluation logic, metrics, and visualization
2. **K-fold Support**: Robust k-fold cross-validation with proper data handling
3. **Standardization**: Consistent metric computation and output formats across all model systems
4. **Flexibility**: Support different model types (sklearn-compatible), data formats, and evaluation scenarios
5. **Reproducibility**: Deterministic splits and metrics with seed control
6. **Extensibility**: Easy to add new metrics or evaluation strategies
7. **Configuration Agnostic**: Work with any input method (CLI args, config files, config objects) - data comes in, evaluation happens, results go out
8. **Model Comparison**: Support comparing multiple models and multiple runs in parallel (Phase 1: model names, Phase 3: run names)

## Architecture

### Core Components

```
evaluation/
├── __init__.py
├── evaluator.py          # Main Evaluator class
├── types.py              # Result types (FoldResults, KFoldResults, TrainTestResults)
├── metrics.py            # Metric computation functions
├── kfold.py              # K-fold cross-validation logic
├── random_baseline.py    # Null distribution of random-model AUCs
├── visualizations.py    # Plotting functions and default plot theme
├── utils.py              # Helper functions (validation, alignment, prepare_predictions_df)
examples/
└── baseline_model_example.py  # Example with generic run_kfold / run_train_test
```

### Terminology

- **k-fold cross-validation**: Splitting scheme; spell "k-fold" with a hyphen. Subgroups for reporting are **strata** (singular **stratum**). Entities that must not cross folds (e.g. site) are **groups**; group-stratified k-fold keeps group exclusivity. **stratify_labels** are the labels used to balance folds.
- See `evaluation/__init__.py` for the full terminology and naming-convention docstring.

### Plot theme

All evaluation plots use a default theme defined in `evaluation/visualizations.py`: `PLOT_THEME` (figure size, DPI, style, font sizes) and `setup_figure()` / `save_figure()`. New plots should use these so appearance stays consistent.

### Key Classes and Functions

#### 1. `Evaluator` (main interface)

The primary class that generates splits and aggregates evaluation results. Supports both k-fold CV and train/test evaluation.

**Location**: `evaluation/evaluator.py`

**Key Methods**:
- `create_kfold_splits()`: Generate and return k-fold splits to the model
- `aggregate_kfold_results()`: Aggregate predictions/metrics from model across folds
- `compute_metrics()`: Calculate metrics from predictions
- `save_results()`: Write metrics, predictions, and plots to disk

**Interface**:
```python
class Evaluator:
    def __init__(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        patient_ids: np.ndarray | pd.Series | None = None,
        model_name: str = "model",
        random_state: int = 42,
    ):
        """
        Initialize evaluator with data.
        
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
        ...
    
    def create_kfold_splits(
        self,
        n_splits: int = 5,
        stratify: bool = True,
        shuffle: bool = True,
    ) -> list[FoldSplit]:
        """
        Create k-fold splits and return them to the model.
        
        Returns
        -------
        list[FoldSplit]
            List of FoldSplit objects, one per fold, containing:
            - train_indices: indices for training
            - val_indices: indices for validation
            - train_patient_ids: patient IDs for training (if available)
            - val_patient_ids: patient IDs for validation (if available)
        """
        ...
    
    def aggregate_kfold_results(
        self,
        fold_results: list[FoldResults],
    ) -> KFoldResults:
        """
        Aggregate predictions and metrics from model across all folds.
        
        Parameters
        ----------
        fold_results : list[FoldResults]
            List of results from model, one per fold. Each FoldResults contains:
            - fold_idx: fold number
            - predictions: DataFrame with patient_id, y_true, y_pred, y_prob
            - metrics: dict of metrics for this fold (optional, can compute from predictions)
        
        Returns
        -------
        KFoldResults
            Aggregated results with mean ± std metrics across folds
        """
        ...
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute metrics from predictions.
        
        Can be called by models if they want metrics for a single fold,
        or used internally during aggregation.
        """
        ...
    
    def save_results(
        self,
        results: KFoldResults | TrainTestResults,
        output_dir: Path,
        run_name: str | None = None,
    ) -> None:
        """
        Save results to output directory, organized by model name and run name.
        
        Parameters
        ----------
        results : KFoldResults | TrainTestResults
            Evaluation results to save
        output_dir : Path
            Base output directory. Results will be saved to:
            output_dir / model_name / run_name / (if run_name provided)
            output_dir / model_name / (if no run_name)
        run_name : str, optional
            Name of this run (e.g., "run_001", "experiment_1", timestamp).
            Used for tracking multiple runs of the same model.
            If None, results saved directly under model_name.
        
        Note: output_dir is a Path object. Model systems determine this path
        from their configuration (CLI args, config files, etc.) and pass it here.
        """
        ...
```

**New Data Classes**:
```python
@dataclass
class FoldSplit:
    """Represents a single fold split."""
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_patient_ids: np.ndarray | None = None
    val_patient_ids: np.ndarray | None = None

@dataclass
class FoldResults:
    """Results from a single fold."""
    fold_idx: int
    predictions: pd.DataFrame  # columns: patient_id, y_true, y_pred, y_prob
    metrics: dict[str, Any] | None = None  # Optional pre-computed metrics
```

#### 2. `KFoldResults` (data class)

Stores aggregated results from k-fold cross-validation.

**Location**: `evaluation/evaluator.py`

**Attributes**:
- `fold_metrics`: List of metric dictionaries (one per fold)
- `aggregated_metrics`: Dictionary with mean/std across folds
- `predictions`: DataFrame with patient_id, fold, y_true, y_pred, y_prob (all folds combined)
- `n_splits`: Number of folds
- `model_name`: Name of the model (from Evaluator)
- `run_name`: Name of the run (if provided)

#### 3. `TrainTestResults` (data class)

Stores results from train/test evaluation.

**Location**: `evaluation/evaluator.py`

**Attributes**:
- `metrics`: Dictionary of computed metrics
- `predictions`: DataFrame with patient_id, y_true, y_pred, y_prob
- `model_name`: Name of the model (from Evaluator)
- `run_name`: Name of the run (if provided)

#### 4. Metric Computation (`metrics.py`)

Standardized metric computation functions with extensible registry pattern.

**Location**: `evaluation/metrics.py`

**Core Structure**:
- `METRIC_REGISTRY`: Dictionary mapping metric names to computation functions
- `compute_binary_metrics()`: Main function that computes all registered metrics
- `aggregate_metrics()`: Aggregates metrics across k-fold splits (mean ± std)

**Phase 1 Functions** (minimal implementation):
- `compute_auc()`: AUC (ROC-AUC) computation
- `compute_binary_metrics()`: Computes AUC (extensible to add more metrics)

**Phase 3 Functions** (extended implementation):
- `compute_precision_recall()`: Precision and recall
- `compute_f1()`: F1 score
- `compute_sensitivity_specificity()`: Sensitivity (recall) and specificity
- `compute_confusion_matrix()`: Confusion matrix (TN, FP, FN, TP)
- `compute_average_precision()`: Average Precision (AP)
- `compute_roc_curve()`: FPR, TPR, thresholds (for visualization)
- `compute_pr_curve()`: Precision, recall, thresholds (for visualization)
- `compute_calibration()`: Calibration curve data
- `compute_metrics_by_group()`: Metrics stratified by subgroup (e.g., subtype)

**Supported Metrics** (after Phase 3):
- AUC (ROC-AUC) ✓ Phase 1
- Average Precision (AP) - Phase 3
- Precision, Recall, F1 - Phase 3
- Sensitivity (Recall), Specificity - Phase 3
- Confusion Matrix (TN, FP, FN, TP) - Phase 3
- Optimal threshold (Youden's J statistic) - Phase 3
- Calibration metrics (Brier score, ECE) - Phase 5

#### 5. K-fold Logic (`kfold.py`)

Handles k-fold split generation and metric aggregation.

**Location**: `evaluation/kfold.py`

**Functions**:
- `create_kfold_splits()`: Generate stratified or non-stratified splits (returns to model)
- `aggregate_fold_metrics()`: Combine metrics across folds from model results

**Features**:
- Stratified k-fold by default (maintains class distribution)
- Support for custom split strategies (e.g., GroupKFold for site-based splits)
- Proper handling of patient IDs to avoid data leakage
- Returns splits to model - model handles training
- Aggregates results returned by model

#### 6. Visualizations (`visualizations.py`)

Standardized plotting functions using seaborn for enhanced aesthetics with extensible registry pattern.

**Location**: `evaluation/visualizations.py`

**Core Structure**:
- `VISUALIZATION_REGISTRY`: Dictionary mapping visualization names to plotting functions
- All plotting functions follow consistent signature: `(y_true, y_prob, output_path, **kwargs)`

**Phase 1 Functions** (minimal implementation):
- `plot_roc_curve()`: ROC curve with AUC (seaborn lineplot)

**Per-split plots** (k-fold):
- `plot_roc_per_split()`: One ROC curve per fold/split plus overall curve; legend shows AUC and n per fold
- `plot_pr_per_split()`: One PR curve per fold/split plus overall curve; legend shows AP and n per fold

**Multi-run plots**:
- `plot_auc_distribution()`: Histogram of AUC values across runs (when `--n-runs` > 1)

**Phase 3 Functions** (extended implementation):
- `plot_pr_curve()`: Precision-recall curve with AP (seaborn lineplot)
- `plot_calibration_curve()`: Calibration plot (seaborn scatterplot with regression line)
- `plot_confusion_matrix()`: Confusion matrix heatmap (seaborn heatmap with annotations)

**Future Functions** (Phase 5+):
- `plot_metrics_comparison()`: Compare metrics across folds or models (seaborn barplot/boxplot)

**Styling**:
- Uses seaborn's default style and color palettes
- Consistent figure sizing (8x6 inches) and DPI (200)
- Professional appearance suitable for publications
- Clear labels, legends, and titles
- All plots saved with `bbox_inches="tight"` to avoid clipping

**Output Format**: All plots saved as PNG files with consistent styling

#### 7. Logging (`logging_config.py`)

Structured progress logging for the evaluation package.

**Location**: `evaluation/logging_config.py`

**Functions**:
- `setup_logging(level, format_string, stream)`: Configure the evaluation logger
- `get_logger()`: Return the evaluation package logger

**Usage**: Call `setup_logging(level=args.log_level)` at startup. The baseline example supports `--log-level` (DEBUG, INFO, WARNING, ERROR). Log messages include run/fold progress (e.g. "Run 1/5", "Fold 2/5 done").

#### 8. Utilities (`utils.py`)

Helper functions for data handling and validation.

**Location**: `evaluation/utils.py`

**Functions**:
- `align_data()`: Align features, labels, and patient IDs
- `validate_inputs()`: Check data consistency
- `prepare_predictions_df()`: Optional helper to build prediction DataFrames (patient_id, y_true, y_pred, y_prob; optional fold column) for use with the evaluator

#### 9. Baseline Model Example (`baseline_model_example.py`)

A well-commented example model that demonstrates how to integrate with the evaluation system. This serves as:
- A reference implementation for developers
- A testbed for the evaluation framework
- A baseline model for comparison

**Location**: `examples/baseline_model_example.py`

**Purpose**:
- Demonstrates complete integration workflow
- Shows best practices for using the evaluation system
- Provides a working example that can be run immediately
- Serves as documentation through code

**Features**:
- Generates synthetic binary classification data (or loads real data)
- Trains a simple sklearn model (e.g., LogisticRegression or RandomForest)
- Demonstrates both k-fold CV and train/test evaluation
- Supports multiple k-fold runs (`--n-runs`) to estimate AUC stability across seeds
- Structured progress logging (`--log-level`)
- Includes extensive comments explaining each step
- Shows how to interpret and use results
- Includes error handling and validation examples

**Structure**:
```python
"""
Baseline Model Example for Evaluation System

This script demonstrates how to use the centralized evaluation system
for model evaluation with k-fold cross-validation.

Key sections:
1. Data preparation
2. Model definition
3. K-fold evaluation
4. Train/test evaluation
5. Results interpretation
"""

# Extensive inline comments explaining:
# - Why each step is necessary
# - How the evaluation system works
# - What each parameter means
# - How to interpret outputs
# - Common pitfalls and how to avoid them
```

**Usage**:
```bash
# Run with default synthetic data
python examples/baseline_model_example.py

# Run with custom data
python examples/baseline_model_example.py \
    --features path/to/features.csv \
    --labels path/to/labels.csv \
    --output results/baseline_example

# Multiple k-fold runs (outputs run AUC distribution)
python examples/baseline_model_example.py --model random --n-runs 5 --n-splits 5 --output results/multirun

# Progress logging (DEBUG, INFO, WARNING, ERROR)
python examples/baseline_model_example.py --log-level DEBUG
```

## Data Flow

### K-fold Evaluation Flow

```
1. Model loads data (from config, CLI, etc.) and prepares X, y, patient_ids
2. Model creates Evaluator with data (no model passed)
3. Model calls evaluator.create_kfold_splits() to get splits
4. For each fold split returned:
   a. Model extracts train/val data using split indices
   b. Model trains on train fold (full control over training process)
   c. Model predicts on val fold
   d. Model creates FoldResults with predictions (and optionally metrics)
5. Model passes all FoldResults to evaluator.aggregate_kfold_results()
6. Evaluator aggregates metrics across folds (mean ± std)
7. Model calls evaluator.save_results() to save metrics, predictions, plots
8. Output: Metrics, predictions, plots
```

### Train/Test Evaluation Flow

```
1. Model loads data and prepares train/test splits
2. Model trains on training data (full control)
3. Model predicts on test set
4. Model creates TrainTestResults with predictions
5. Model calls evaluator.save_results() to save metrics, predictions, plots
6. Output: Metrics, predictions, plots
```

## Baseline Model Example

### Purpose and Design

The baseline model example (`examples/baseline_model_example.py`) serves multiple critical functions:

1. **Reference Implementation**: Provides a complete, working example that developers can study and adapt
2. **Testbed**: Validates the evaluation system works correctly before integration
3. **Documentation**: Extensive comments explain the evaluation workflow in detail
4. **Baseline Comparison**: Can be used as a simple baseline for model performance comparison

### Example Structure

The baseline model will include:

```python
"""
Baseline Model Example for Centralized Evaluation System

This script demonstrates the complete workflow for using the evaluation system:
1. Data preparation and loading
2. Model definition and configuration
3. K-fold cross-validation evaluation
4. Train/test split evaluation
5. Results interpretation and saving

This example uses synthetic data by default, but can be adapted to use real data.
"""

# Section 1: Imports and Setup
# - Import evaluation system components
# - Set up paths and configuration
# - Define constants

# Section 2: Data Preparation
# - Generate or load features and labels
# - Handle patient IDs
# - Split into train/test if needed
# - Validate data format

# Section 3: Model Definition
# - Create sklearn-compatible model
# - Configure hyperparameters
# - Wrap in pipeline if needed (preprocessing, etc.)

# Section 4: K-fold Cross-Validation
# - Initialize Evaluator
# - Run k-fold CV
# - Explain what happens in each fold
# - Show how to access fold-level results

# Section 5: Train/Test Evaluation
# - Train final model on full training set
# - Evaluate on held-out test set
# - Compare with k-fold results

# Section 6: Results Interpretation
# - Load and examine metrics
# - Interpret aggregated k-fold metrics (mean ± std)
# - Understand prediction outputs
# - View generated plots

# Section 7: Advanced Usage
# - Custom metrics
# - Subgroup analysis
# - Model persistence
# - Error handling examples
```

### Detailed Baseline Model Features

The baseline model example will demonstrate:

1. **Simple Model**: Uses a straightforward sklearn model (e.g., `LogisticRegression` or `RandomForestClassifier`) to keep focus on evaluation, not model complexity

2. **Synthetic Data Generation**: 
   - Creates realistic binary classification dataset
   - Configurable class imbalance
   - Includes patient IDs
   - Can be replaced with real data via command-line arguments

3. **Complete Workflow**:
   - Data loading/preparation
   - Model instantiation
   - K-fold CV with detailed comments on each step
   - Train/test evaluation
   - Results saving and loading
   - Plot generation and viewing

4. **Error Handling Examples**:
   - What happens with missing data
   - Handling edge cases (all one class, very small datasets)
   - Validation of inputs

5. **Best Practices**:
   - Proper random seed setting
   - Patient ID handling to avoid data leakage
   - Interpreting k-fold metrics (mean ± std)
   - When to use k-fold vs. train/test

6. **Extensive Documentation**:
   - Every major code block has explanatory comments
   - Docstrings for all functions
   - Inline comments explaining non-obvious choices
   - Comments explaining evaluation system internals

7. **CLI Interface**:
   - Command-line arguments for flexibility
   - Default values for quick testing
   - Help text explaining all options

### Key Commenting Strategy

Comments will explain:
- **What**: What each code block does
- **Why**: Why this step is necessary in the evaluation workflow
- **How**: How the evaluation system processes the data
- **When**: When to use k-fold vs. train/test evaluation
- **Common Issues**: What can go wrong and how to fix it
- **Best Practices**: Recommended patterns and approaches

### Example Output

The baseline model will produce the same standardized output as any model using the system:
- `metrics.json`: Comprehensive metrics
- `predictions.csv`: All predictions with patient IDs
- `plots/`: All visualization files
- Clear console output showing progress and key results

## Integration with Existing Model Systems

### Example: Integration with `radiomics_train.py`

**Current approach** (lines 548-590):
- Manual k-fold CV with `StratifiedKFold` and `cross_val_predict`
- Custom metric computation
- Manual plotting

**New approach** (split-based, configuration-agnostic):
```python
from evaluation import Evaluator, FoldResults

# Model system loads data however it wants (CLI args, config file, etc.)
Xtr, Xte, ytr, yte = load_and_prepare_data(args)  # or config, or whatever

# Create evaluator with data (no model - model handles its own training)
# Model name is required for organizing outputs and comparing models
evaluator = Evaluator(
    X=Xtr,
    y=ytr,
    patient_ids=Xtr.index,  # patient_id from DataFrame index
    model_name="radiomics_baseline",  # Phase 1: Required for model comparison
    random_state=42,
)

# K-fold CV: Get splits from evaluator
splits = evaluator.create_kfold_splits(
    n_splits=args.cv_folds,  # or config.cv_folds, or whatever
    stratify=True,
    shuffle=True,
)

# Model trains on each split and collects results
fold_results = []
for split in splits:
    # Extract train/val data using split indices
    X_train_fold = Xtr.iloc[split.train_indices] if isinstance(Xtr, pd.DataFrame) else Xtr[split.train_indices]
    y_train_fold = ytr[split.train_indices]
    X_val_fold = Xtr.iloc[split.val_indices] if isinstance(Xtr, pd.DataFrame) else Xtr[split.val_indices]
    y_val_fold = ytr[split.val_indices]
    
    # Model trains on this fold (full control)
    pipe.fit(X_train_fold, y_train_fold)
    
    # Model predicts on validation fold
    y_pred = pipe.predict(X_val_fold)
    y_prob = pipe.predict_proba(X_val_fold)[:, 1]  # positive class
    
    # Create FoldResults
    predictions_df = pd.DataFrame({
        "patient_id": split.val_patient_ids if split.val_patient_ids is not None else X_val_fold.index,
        "y_true": y_val_fold,
        "y_pred": y_pred,
        "y_prob": y_prob,
    })
    
    fold_results.append(FoldResults(
        fold_idx=split.fold_idx,
        predictions=predictions_df,
    ))

# Aggregate results
kfold_results = evaluator.aggregate_kfold_results(fold_results)

# Train/test evaluation (similar pattern)
# Model trains on full training set
pipe.fit(Xtr, ytr)
y_pred_test = pipe.predict(Xte)
y_prob_test = pipe.predict_proba(Xte)[:, 1]

# Create TrainTestResults
from evaluation import TrainTestResults
test_predictions = pd.DataFrame({
    "patient_id": Xte.index,
    "y_true": yte,
    "y_pred": y_pred_test,
    "y_prob": y_prob_test,
})
train_test_results = TrainTestResults(
    metrics=evaluator.compute_metrics(yte, y_pred_test, y_prob_test),
    predictions=test_predictions,
)

# Save all results
# Output organized by model_name (Phase 1) and run_name (Phase 3)
output_dir = Path(args.output)  # or config.output_dir, or whatever
run_name = getattr(args, "run_name", None)  # Phase 3: Optional run name for tracking multiple runs
evaluator.save_results(kfold_results, output_dir, run_name=run_name)
evaluator.save_results(train_test_results, output_dir, run_name=run_name)
```

**Key Points**:
- Evaluator creates splits and returns them to model
- Model has full control over training process
- Model returns predictions/metrics to evaluator for aggregation
- The evaluation system doesn't know or care about configuration method

### Example: Integration with `baseline_pcr_simple.py`

**Current approach**:
- Simple train/test evaluation
- Manual AUC computation
- Basic ROC plot

**New approach** (split-based, configuration-agnostic):
```python
from evaluation import Evaluator, TrainTestResults

# Model system handles data loading from its configuration
df_train, df_test = load_dataset(args.json_dir, args.split_csv)  # or config, etc.
X_train, y_train = prepare_features(df_train)
X_test, y_test = prepare_features(df_test)

# Model trains (full control)
pipe.fit(X_train, y_train)

# Create evaluator (no model passed)
# Model name required for organizing outputs
evaluator = Evaluator(
    X=X_train,  # For potential k-fold splits later
    y=y_train,
    patient_ids=df_train["patient_id"],
    model_name="non_imaging_baseline",  # Phase 1: Required
    random_state=42,
)

# Model predicts on test set
y_pred_test = pipe.predict(X_test)
y_prob_test = pipe.predict_proba(X_test)[:, 1]

# Create TrainTestResults
test_predictions = pd.DataFrame({
    "patient_id": df_test["patient_id"],
    "y_true": y_test,
    "y_pred": y_pred_test,
    "y_prob": y_prob_test,
})
train_test_results = TrainTestResults(
    metrics=evaluator.compute_metrics(y_test, y_pred_test, y_prob_test),
    predictions=test_predictions,
)

# Save results
# Output organized by model_name (Phase 1) and run_name (Phase 3)
output_dir = Path(args.output)  # or config.output_dir, etc.
run_name = getattr(args, "run_name", None)  # Phase 3: Optional
evaluator.save_results(train_test_results, output_dir, run_name=run_name)
```

**Key Points**:
- Model has full control over training
- Evaluator handles metric computation and result saving
- Works with any configuration method (CLI, config files, config objects)

## Output Structure

### Directory Layout

**Phase 1** (minimal, with model names):
```
output_dir/
├── radiomics_baseline/       # Model name directory
│   ├── metrics.json          # Aggregated metrics (k-fold: mean ± std) - only AUC
│   ├── metrics_per_fold.json # Per-fold metrics (k-fold only) - only AUC
│   ├── predictions.csv       # All predictions with patient_id, y_true, y_pred, y_prob
│   └── plots/
│       ├── roc_curve.png     # Overall ROC
│       ├── roc_per_split.png # One ROC line per fold (k-fold only)
│       ├── pr_per_split.png  # One PR line per fold (k-fold only)
│       └── auc_distribution.png # AUC histogram across runs (when n_runs > 1)
├── non_imaging_baseline/     # Another model
│   ├── metrics.json
│   ├── predictions.csv
│   └── plots/
│       └── roc_curve.png
└── ...
```

**Phase 3+** (extended, with model names and run names):
```
output_dir/
├── radiomics_baseline/       # Model name directory
│   ├── run_001/              # Run name directory (Phase 3)
│   │   ├── metrics.json      # Aggregated metrics (k-fold: mean ± std) - all metrics
│   │   ├── metrics_per_fold.json
│   │   ├── predictions.csv
│   │   └── plots/
│   │       ├── roc_curve.png
│   │       ├── pr_curve.png
│   │       ├── calibration_curve.png
│   │       └── confusion_matrix.png
│   ├── run_002/              # Another run of the same model
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   └── plots/
│   │       └── ...
│   └── run_003/
│       └── ...
├── non_imaging_baseline/      # Another model
│   ├── run_001/
│   │   └── ...
│   └── run_002/
│       └── ...
└── ...
```

**Use Case**: Run multiple models and multiple runs in parallel, then compare results:
```bash
# Run multiple models/runs in parallel
python radiomics_train.py --model-name radiomics_baseline --run-name run_001 &
python radiomics_train.py --model-name radiomics_baseline --run-name run_002 &
python non_imaging_train.py --model-name non_imaging_baseline --run-name run_001 &
# ... etc

# Later: Compare results across models and runs
# All results organized in output_dir/ for easy comparison
# Structure enables:
# - Comparing different models (radiomics_baseline vs non_imaging_baseline)
# - Comparing different runs of same model (run_001 vs run_002)
# - Picking best model/run combination based on metrics
# - Easy navigation: output_dir/model_name/run_name/
```

**Benefits**:
- **Parallel Execution**: Multiple models/runs can execute simultaneously without conflicts
- **Organized Output**: Hierarchical structure makes it easy to find and compare results
- **Model Comparison**: Compare performance across different model architectures
- **Run Tracking**: Track multiple experiments/hyperparameter sweeps of the same model
- **Best Model Selection**: Easily identify best performing model/run combination

### Metrics JSON Format

**Phase 1 Format** (minimal - only AUC, with model name):
```json
{
  "evaluation_type": "kfold",
  "model_name": "radiomics_baseline",
  "n_splits": 5,
  "aggregated_metrics": {
    "auc": {"mean": 0.75, "std": 0.03}
  },
  "per_fold_metrics": [
    {"fold": 0, "auc": 0.73},
    {"fold": 1, "auc": 0.76},
    ...
  ],
  "n_samples": 200,
  "n_features": 50,
  "run_aucs": [0.74, 0.75, 0.76],
  "run_auc_mean": 0.75,
  "run_auc_std": 0.01
}
```

When `--n-runs` > 1, `run_aucs`, `run_auc_mean`, and `run_auc_std` are added to metrics.json.

**Phase 3+ Format** (extended - all metrics, with model name and run name):
```json
{
  "evaluation_type": "kfold",
  "model_name": "radiomics_baseline",
  "run_name": "run_001",
  "n_splits": 5,
  "aggregated_metrics": {
    "auc": {"mean": 0.75, "std": 0.03},
    "precision": {"mean": 0.68, "std": 0.05},
    "recall": {"mean": 0.72, "std": 0.04},
    "f1": {"mean": 0.70, "std": 0.04},
    "sensitivity": {"mean": 0.72, "std": 0.04},
    "specificity": {"mean": 0.78, "std": 0.03}
  },
  "per_fold_metrics": [
    {"fold": 0, "auc": 0.73, "precision": 0.65, ...},
    {"fold": 1, "auc": 0.76, "precision": 0.70, ...},
    ...
  ],
  "n_samples": 200,
  "n_features": 50
}
```

**Train/test results** (Phase 1 - minimal, with model name):
```json
{
  "evaluation_type": "train_test",
  "model_name": "radiomics_baseline",
  "metrics": {
    "auc": 0.75
  },
  "n_samples": 100,
  "n_features": 50
}
```

**Train/test results** (Phase 3+ - extended, with model name and run name):
```json
{
  "evaluation_type": "train_test",
  "model_name": "radiomics_baseline",
  "run_name": "run_001",
  "metrics": {
    "auc": 0.75,
    "precision": 0.68,
    "recall": 0.72,
    "f1": 0.70,
    "sensitivity": 0.72,
    "specificity": 0.78,
    "confusion_matrix": {"tn": 45, "fp": 12, "fn": 8, "tp": 35},
    "optimal_threshold": 0.42
  },
  "n_samples": 100,
  "n_features": 50
}
```

### Random baseline distribution

The evaluation package can compute a **null distribution** of AUCs from random predictors on the same labels (many runs with different seeds). This answers: at what AUC is a model inconsistent with random guessing?

- **Run it**: Call `evaluator.compute_random_baseline_distribution(n_runs=1000)` (or use `compute_random_auc_distribution(y_true, n_runs=1000)` from `evaluation.random_baseline`). Pass the returned dict to `save_results(..., random_baseline_distribution=distribution)` to add `random_baseline` (mean, std, n_runs), `z_score_vs_random`, and `p_value_vs_random` to `metrics.json`.
- **Interpretation**: **z-score** = (observed AUC − mean) / std; use the [standard normal table](https://en.wikipedia.org/wiki/Standard_normal_table) for tail probabilities when the null distribution is roughly normal and well within [0, 1]. **Empirical p-value** = fraction of random runs with AUC ≥ observed; prefer this when std is large or the distribution is skewed (AUC is bounded in [0, 1]).
- **Example**: See `examples/baseline_model_example.py` (random baseline distribution, report, save JSON, and histogram plot).

### Predictions CSV Format

```csv
patient_id,fold,y_true,y_pred,y_prob
patient_001,0,1,1,0.85
patient_001,1,1,1,0.82
patient_002,0,0,0,0.23
...
```

For train/test evaluation, `fold` column is omitted or set to "test".

## Design Decisions

### 1. Model Interface

**Decision**: Evaluator does NOT take model as input - models handle their own training

**Rationale**: 
- Models have full control over training process
- Evaluator focuses on split generation and result aggregation
- More flexible - models can use any training approach
- Separation of concerns: models train, evaluator evaluates

**Previous approach considered**: Evaluator takes model and trains it
- Less flexible, models have less control
- Harder to support custom training logic

### 2. Data Format Flexibility

**Decision**: Accept both numpy arrays and pandas DataFrames

**Rationale**:
- Different model systems use different formats
- DataFrames preserve patient IDs naturally
- Easy conversion between formats

### 3. Patient ID Handling

**Decision**: Make patient_ids optional but recommended

**Rationale**:
- Some use cases may not need patient-level tracking
- Required for proper k-fold CV (avoid data leakage)
- Enables subgroup analysis

### 4. K-fold vs. Train/Test

**Decision**: Support both evaluation modes in single interface

**Rationale**:
- Different model systems have different needs
- K-fold for model selection/validation
- Train/test for final evaluation
- Can be used together (k-fold on train, then test evaluation)

### 5. Metric Aggregation

**Decision**: For k-fold, report mean ± std across folds

**Rationale**:
- Standard practice in ML
- Provides uncertainty estimate
- Easy to interpret

**Alternative considered**: Report median and IQR
- Less common, harder to compare with literature

### 6. Model Persistence

**Decision**: Optionally save models from each fold

**Rationale**:
- Useful for ensemble methods
- Enables inspection of fold-specific models
- Optional to avoid storage overhead

### 7. Visualization Standardization

**Decision**: Generate all standard plots automatically using seaborn

**Rationale**:
- Ensures consistency across model systems
- Reduces code duplication
- Easy to extend with new plot types
- Seaborn provides professional, publication-ready visualizations
- Better default aesthetics than matplotlib alone

### 8. Baseline Model Example

**Decision**: Create a well-commented example model before integrating with existing systems

**Rationale**:
- Provides clear reference implementation for developers
- Serves as testbed to validate evaluation system works correctly
- Demonstrates best practices and common patterns
- Reduces learning curve for new users
- Can be used to test framework changes before integration

### 9. Configuration and Input Handling

**Decision**: Keep evaluation system agnostic to how models handle configuration and input

**Rationale**:
- Models will transition from CLI args to config files/objects for data paths
- CLI args will be reserved for performance configs (GPUs, RAM, CPUs)
- Evaluation system should work with data regardless of how it's loaded
- Focus on data-in, evaluation, results-out pattern
- Adapt to model configuration system once it's finalized (Phase 4+)

**Design Approach**:
- `Evaluator` accepts data directly (numpy arrays, DataFrames) - not file paths
- Model systems handle data loading/configuration themselves
- Evaluation system is a library, not a CLI tool
- Future: Can add convenience functions to load from config objects if needed

## Extensibility Points

The evaluation system is designed to be easily extensible. The architecture uses a plugin-like pattern where new metrics and visualizations can be added without modifying core evaluation logic.

### Architecture for Extensibility

**Key Design Principles**:
1. **Separation of Concerns**: Metrics, visualizations, and evaluation logic are in separate modules
2. **Registry Pattern**: Metrics and visualizations are registered in a central location
3. **Interface Consistency**: All metrics follow the same function signature; all visualizations follow the same pattern
4. **Optional Dependencies**: New metrics/visualizations don't break existing functionality if dependencies are missing

### Adding New Metrics

**Step-by-Step Process**:

1. **Define the metric computation function** in `evaluation/metrics.py`:
```python
def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute precision and recall metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels
    y_prob : np.ndarray, optional
        Predicted probabilities (not used for precision/recall, but kept for consistency)
    
    Returns
    -------
    dict[str, float]
        Dictionary with 'precision' and 'recall' keys
    """
    from sklearn.metrics import precision_score, recall_score
    
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    
    return {
        "precision": precision,
        "recall": recall,
    }
```

2. **Register the metric** in `evaluation/metrics.py`:
```python
# Metric registry - maps metric names to computation functions
METRIC_REGISTRY = {
    "auc": compute_auc,
    "precision_recall": compute_precision_recall,  # New metric
    # ... other metrics
}
```

3. **Add to aggregation logic** (if needed for k-fold):
```python
def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """
    Aggregate metrics across folds.
    Handles both scalar metrics and metrics that return dicts.
    """
    aggregated = {}
    for metric_name, metric_func in METRIC_REGISTRY.items():
        # Extract metric values from each fold
        fold_values = [m.get(metric_name, {}) for m in metrics_list]
        # Aggregate (mean ± std for scalars, handle dicts appropriately)
        aggregated[metric_name] = _aggregate_fold_values(fold_values)
    return aggregated
```

4. **Update `compute_binary_metrics()`** to include the new metric:
```python
def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics_to_compute: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute all binary classification metrics.
    
    Parameters
    ----------
    metrics_to_compute : list[str], optional
        List of metric names to compute. If None, computes all registered metrics.
    """
    if metrics_to_compute is None:
        metrics_to_compute = list(METRIC_REGISTRY.keys())
    
    results = {}
    for metric_name in metrics_to_compute:
        if metric_name in METRIC_REGISTRY:
            metric_func = METRIC_REGISTRY[metric_name]
            metric_result = metric_func(y_true, y_pred, y_prob)
            results.update(metric_result)  # Handles both scalar and dict returns
    return results
```

5. **Add unit tests** in `tests/test_metrics.py`:
```python
def test_precision_recall():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    
    result = compute_precision_recall(y_true, y_pred, y_prob)
    assert "precision" in result
    assert "recall" in result
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
```

6. **Update JSON schema documentation** in the design doc or README

**Best Practices**:
- Keep metric functions pure (no side effects)
- Handle edge cases (all one class, empty arrays, etc.)
- Return NaN or appropriate defaults for invalid cases
- Document expected input/output formats
- Make metrics independent (don't rely on other metrics being computed first)

### Adding New Visualization Types

**Step-by-Step Process**:

1. **Define the visualization function** in `evaluation/visualizations.py`:
```python
def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "Precision-Recall Curve",
) -> None:
    """
    Plot precision-recall curve using seaborn.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities for positive class
    output_path : Path
        Path to save the plot
    title : str
        Plot title
    """
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Set seaborn style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    # Plot using seaborn
    sns.lineplot(x=recall, y=precision, ax=ax, label=f"AP = {ap:.3f}")
    
    # Styling
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
```

2. **Register the visualization** in `evaluation/visualizations.py`:
```python
# Visualization registry
VISUALIZATION_REGISTRY = {
    "roc_curve": plot_roc_curve,
    "pr_curve": plot_precision_recall_curve,  # New visualization
    # ... other visualizations
}
```

3. **Update `Evaluator.save_results()`** to call registered visualizations:
```python
def save_results(
    self,
    results: KFoldResults | TrainTestResults,
    output_dir: Path,
    visualizations_to_generate: list[str] | None = None,
) -> None:
    """
    Save evaluation results including metrics, predictions, and plots.
    
    Parameters
    ----------
    visualizations_to_generate : list[str], optional
        List of visualization names to generate. If None, generates all registered visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Save metrics and predictions (existing code)
    ...
    
    # Generate visualizations
    if visualizations_to_generate is None:
        visualizations_to_generate = list(VISUALIZATION_REGISTRY.keys())
    
    y_true = results.predictions["y_true"].values
    y_prob = results.predictions["y_prob"].values
    
    for viz_name in visualizations_to_generate:
        if viz_name in VISUALIZATION_REGISTRY:
            viz_func = VISUALIZATION_REGISTRY[viz_name]
            output_path = plots_dir / f"{viz_name}.png"
            viz_func(y_true, y_prob, output_path)
```

4. **Add unit/integration tests** (optional, visual inspection often sufficient):
```python
def test_pr_curve_plot(tmp_path):
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    output_path = tmp_path / "pr_curve.png"
    
    plot_precision_recall_curve(y_true, y_prob, output_path)
    assert output_path.exists()
```

5. **Update output structure documentation**

**Best Practices**:
- Use seaborn for consistent styling
- Follow existing plot dimensions (typically 8x6 inches)
- Use DPI 200 for publication quality
- Include meaningful titles and labels
- Add legends where appropriate
- Handle edge cases (all one class, etc.)
- Save plots with `bbox_inches="tight"` to avoid clipping

### Custom Split Strategies

1. Implement custom splitter (e.g., `GroupKFold` for site-based splits)
2. Pass to `evaluate_kfold(splitter=custom_splitter)`
3. Ensure splitter follows sklearn `BaseCrossValidator` interface

### Multi-class Support

Future extension: Add `MultiClassEvaluator` class with:
- Per-class metrics
- Macro/micro/weighted averages
- Confusion matrix visualization
- One-vs-rest ROC curves

### Custom Split Strategies

1. Implement custom splitter (e.g., `GroupKFold` for site-based splits)
2. Pass to `evaluate_kfold(splitter=custom_splitter)`
3. Ensure splitter follows sklearn `BaseCrossValidator` interface

### Multi-class Support

Future extension: Add `MultiClassEvaluator` class with:
- Per-class metrics
- Macro/micro/weighted averages
- Confusion matrix visualization
- One-vs-rest ROC curves

## Testing Strategy

### Unit Tests

- `test_metrics.py`: Test metric computation functions
- `test_kfold.py`: Test k-fold splitting and aggregation
- `test_evaluator.py`: Test main Evaluator class
- `test_visualizations.py`: Test plotting functions (optional, can use visual inspection)

### Integration Tests

- `test_integration_radiomics.py`: Test with radiomics_train.py workflow
- `test_integration_baseline.py`: Test with baseline_pcr_simple.py workflow

### Test Data

- Synthetic binary classification datasets
- Known ground truth for metric validation
- Edge cases: imbalanced classes, small datasets, missing values

## Migration Path

### Phase 1: Minimal Core Implementation

**Goal**: Create a working evaluation system with minimal but complete functionality.

**Deliverables**:
1. **Evaluator class** (`evaluation/evaluator.py`):
   - `__init__()`: Initialize with data, patient IDs, model_name (no model)
   - `create_kfold_splits()`: Generate and return k-fold splits to model
   - `aggregate_kfold_results()`: Aggregate predictions/metrics from model
   - `compute_metrics()`: Compute metrics from predictions
   - `save_results()`: Save metrics, predictions, and plots (organized by model_name)
   - Basic error handling and validation

2. **Data classes** (`evaluation/evaluator.py`):
   - `FoldSplit`: Represents a single fold split (indices, patient IDs)
   - `FoldResults`: Results from a single fold (predictions, optional metrics)
   - `KFoldResults`: Aggregated results across all folds (includes model_name)
   - `TrainTestResults`: Results from train/test evaluation (includes model_name)

3. **Model Name Support** (Phase 1 requirement):
   - `model_name` parameter in `Evaluator.__init__()` (required, default="model")
   - Output directory structure: `output_dir / model_name /`
   - Model name included in results data classes and JSON output
   - Enables comparison of multiple models in same output directory

4. **Core metric** (`evaluation/metrics.py`):
   - `compute_auc()`: AUC (ROC-AUC) computation
   - `compute_binary_metrics()`: Main function that computes AUC (extensible structure)
   - Metric registry pattern for future extensions
   - Aggregation logic for k-fold (mean ± std)

5. **Core visualization** (`evaluation/visualizations.py`):
   - `plot_roc_curve()`: ROC curve using seaborn
   - Visualization registry pattern for future extensions
   - Consistent styling and output format

6. **K-fold logic** (`evaluation/kfold.py`):
   - `create_kfold_splits()`: Stratified k-fold splitting (returns splits to model)
   - `aggregate_fold_metrics()`: Combine metrics across folds from model results

7. **Utilities** (`evaluation/utils.py`):
   - `validate_inputs()`: Data validation
   - `prepare_predictions_df()`: Optional helper to build prediction DataFrames
   - `align_data()`: Align features, labels, and patient IDs

8. **Basic unit tests**:
   - Test AUC computation
   - Test k-fold splitting
   - Test evaluator split creation and aggregation

**Success Criteria**:
- Can create k-fold splits and return them to model
- Can aggregate results from model across folds
- Can compute AUC metrics from predictions
- ROC curve is generated and saved
- Results are saved in standardized JSON/CSV format
- Results organized by model_name in output directory
- Multiple models can be run and compared in same output directory
- Code is well-structured for extension

### Phase 2: Baseline Model Example

**Goal**: Create a well-commented example that demonstrates the evaluation system and serves as a testbed.

**Deliverables**:
1. Create `examples/baseline_model_example.py` with extensive comments
2. Demonstrate complete workflow: data prep → model training → k-fold CV → train/test eval
3. Include synthetic data generation for immediate testing
4. Add comprehensive inline documentation explaining:
   - Each step of the evaluation process
   - How to interpret results
   - Common patterns and best practices
   - Error handling and validation
5. Test evaluation system thoroughly using this example
6. Validate that AUC metric and ROC visualization work correctly
7. Document the extensibility pattern for adding new metrics/visualizations

**Success Criteria**:
- Example runs end-to-end without errors
- Comments clearly explain the evaluation workflow
- Example demonstrates both k-fold and train/test evaluation
- Example can be used as reference by other developers

### Phase 3: Extend Metrics and Visualizations

**Goal**: Add additional commonly-used metrics and visualizations using the extensibility pattern, and add run name support for tracking multiple runs.

**Deliverables**:
1. **Additional Metrics** (`evaluation/metrics.py`):
   - `compute_precision_recall()`: Precision and recall
   - `compute_f1()`: F1 score
   - `compute_sensitivity_specificity()`: Sensitivity (recall) and specificity
   - `compute_confusion_matrix()`: Confusion matrix (TN, FP, FN, TP)
   - `compute_average_precision()`: Average precision (AP)
   - Update `compute_binary_metrics()` to include all new metrics
   - Update aggregation logic for k-fold

2. **Additional Visualizations** (`evaluation/visualizations.py`):
   - `plot_pr_curve()`: Precision-recall curve
   - `plot_calibration_curve()`: Calibration plot
   - `plot_confusion_matrix()`: Confusion matrix heatmap
   - Update `save_results()` to generate all visualizations

3. **Run Name Support**:
   - Add `run_name` parameter to `save_results()` method
   - Organize output directory structure: `output_dir / model_name / run_name /`
   - Add `run_name` to results data classes and JSON output
   - Enable tracking multiple runs of the same model for comparison

4. **Enhanced Output**:
   - Update metrics JSON format to include all new metrics and run_name
   - Ensure backward compatibility with Phase 1 format
   - Directory structure supports parallel runs of multiple models

5. **Tests**:
   - Unit tests for all new metrics
   - Integration tests with baseline example
   - Validate metric aggregation across folds
   - Test directory structure with model names and run names

**Success Criteria**:
- All metrics compute correctly
- All visualizations generate correctly
- Metrics aggregate properly in k-fold CV
- Run names organize outputs correctly
- Multiple models and runs can be compared easily
- No breaking changes to existing API
- Extensibility pattern is validated and documented

### Phase 4: Integration (by other developers)

**Goal**: Integrate evaluation system with existing model systems.

**Deliverables**:
1. Other developers use baseline example as reference
2. Integrate with `radiomics_train.py` as proof of concept
3. Refactor to use centralized system
4. Validate results match previous implementation
5. Gather feedback on API and usability
6. **Note**: Model systems may be transitioning to config files/objects for data paths, but evaluation system works with data regardless of how it's loaded

**Success Criteria**:
- Integration requires minimal changes to existing code
- Results match previous implementations (validation)
- API is intuitive and well-documented
- Performance is acceptable
- Works with both CLI-based and config-based model systems

### Phase 5: Expansion and Advanced Features

**Goal**: Add advanced features and integrate with remaining model systems.

**Deliverables**:
1. Integrate with other model systems (`baseline_pcr_simple.py`, `ml_pipeline/pcr_prediction.py`)
2. Add advanced metrics (calibration metrics, Brier score, ECE)
3. Add subgroup analysis (metrics by subtype, site, etc.)
4. Add more visualization types as needed
5. **Optional**: Add comparison utilities to aggregate and compare results across models/runs
   - Function to load all metrics from output directory
   - Function to create comparison table/plot across models
   - Function to identify best model/run based on metrics
6. Gather feedback from users and iterate
7. **Optional**: Add convenience functions for config object integration if model systems have finalized their config approach

**Success Criteria**:
- All model systems can use the centralized evaluator
- Advanced metrics work correctly
- Subgroup analysis is useful and well-documented
- System remains flexible to work with any configuration method
- Results from multiple models/runs can be easily compared

### Phase 6: Documentation and Polish

**Goal**: Complete documentation and finalize the system.

**Deliverables**:
1. Add comprehensive docstrings to all functions
2. Create additional usage examples if needed
3. Update README with evaluation system documentation
4. Create tutorial/guide based on baseline example
5. Document extensibility patterns clearly
6. Performance optimization if needed

**Success Criteria**:
- Documentation is complete and clear
- System is production-ready
- Extensibility is well-documented for future developers

## Dependencies

### Required
- `numpy >= 1.20.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `matplotlib >= 3.3.0`
- `seaborn >= 0.11.0` (for enhanced visualizations)
- `joblib >= 1.0.0` (for model persistence)

## Future Configuration Integration

### Current Approach (Phases 1-2)

The evaluation system is designed to be **configuration-agnostic**. It accepts data directly (numpy arrays, pandas DataFrames) rather than file paths or configuration objects. This means:

- **Model systems handle their own data loading**: Each model system (e.g., `radiomics_train.py`) is responsible for:
  - Loading data from files
  - Parsing configuration (CLI args, config files, or config objects)
  - Preparing features and labels
  - Passing prepared data to the evaluation system

- **Evaluation system is a library**: The `Evaluator` class is imported and used programmatically:
  ```python
  # Model system loads data however it wants
  X_train, y_train, patient_ids = load_data_from_config(config)
  
  # Evaluation system just needs the data
  evaluator = Evaluator(model, X_train, y_train, patient_ids=patient_ids)
  results = evaluator.evaluate_kfold(n_splits=5)
  ```

- **No assumptions about input method**: The evaluation system doesn't care if data came from:
  - CLI arguments
  - Configuration files (YAML, JSON, TOML)
  - Configuration objects
  - Hardcoded paths
  - Database queries
  - Any other method

### Future Direction (Phase 4+)

**Planned Changes to Model Systems**:
- **CLI arguments**: Reserved for performance-related configurations:
  - Number of GPUs
  - RAM allocation
  - CPU cores
  - Batch size
  - Other compute resources

- **Configuration files/objects**: Handle data-related configurations:
  - Input data paths (features, labels)
  - Output directory paths
  - Data preprocessing options
  - Model hyperparameters
  - Evaluation settings

**Evaluation System Adaptation**:
- Once the model configuration system is finalized, we can add:
  - Convenience functions to extract evaluation settings from config objects
  - Helper functions to load data from config-specified paths
  - Integration with common config formats (if needed)
  
- **Key Principle**: These will be **optional convenience functions**, not requirements. The core `Evaluator` API will remain unchanged and continue to accept data directly.

**Example Future Integration** (conceptual, to be implemented in Phase 4+):
```python
# Future: Optional convenience function
from evaluation import Evaluator, load_from_config

# If model uses config objects
config = load_model_config("config.yaml")
evaluator = load_from_config(config, model)  # Optional convenience
results = evaluator.evaluate_kfold(n_splits=config.evaluation.n_splits)

# Or continue using direct data (always supported)
X, y = config.get_data()  # Model handles this
evaluator = Evaluator(model, X, y)
results = evaluator.evaluate_kfold(n_splits=5)
```

**Design Philosophy**:
- **Flexibility First**: Current design works with any input method
- **Future-Proof**: Can add config integration without breaking changes
- **Separation of Concerns**: Data loading is model's responsibility, evaluation is system's responsibility
- **Progressive Enhancement**: Add convenience features later without changing core API

## Open Questions

1. **Subgroup Analysis**: How should we handle metrics by subtype/site? Should this be built-in or a separate function?
   - **Proposal**: Built-in `compute_metrics_by_group()` with optional grouping column

2. **Model Persistence Format**: Use joblib (sklearn standard) or pickle?
   - **Proposal**: joblib for sklearn compatibility

3. **Parallelization**: Should k-fold training be parallelized by default?
   - **Proposal**: Use sklearn's `n_jobs` parameter, default to -1 (all cores)

4. **Early Stopping**: Support for models with early stopping (e.g., XGBoost)?
   - **Proposal**: Defer to Phase 2, models handle this internally

5. **Custom Metrics**: How to allow users to add custom metrics?
   - **Proposal**: Pass custom metric functions to `Evaluator.__init__()` or `evaluate_*()` methods

6. **Configuration Integration**: When should we add config object support?
   - **Proposal**: Phase 4+, after model systems have finalized their config approach

## Success Criteria

1. ✅ All model systems can use the centralized evaluator
2. ✅ K-fold CV produces consistent, reproducible results
3. ✅ Metrics match previous implementations (validation)
4. ✅ Output format is standardized and easy to parse
5. ✅ Integration requires minimal changes to existing code
6. ✅ System is extensible for future metrics/visualizations
7. ✅ Baseline model example serves as clear reference for developers
8. ✅ Baseline model example validates evaluation system functionality
9. ✅ Visualizations use seaborn for professional, consistent appearance
10. ✅ Multiple models can be compared in same output directory (Phase 1: model names)
11. ✅ Multiple runs of same model can be tracked and compared (Phase 3: run names)
12. ✅ Results organized hierarchically for easy comparison and selection

## Next Steps

1. Review and approve this design document
2. Clarify any open questions
3. **Phase 1**: Implement minimal core (Evaluator, AUC metric, ROC visualization, utilities)
4. **Phase 2**: Create baseline model example with extensive comments
5. Test Phase 1 functionality using baseline model
6. **Phase 3**: Extend with additional metrics and visualizations using extensibility pattern
7. Validate extensibility pattern works correctly
8. Create PR with Phases 1-3 complete
9. **Phase 4+**: Integration with existing models (by other developers)
