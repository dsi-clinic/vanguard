"""Train a minimal Deep Sets model on tumor-local vessel point sets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from deepsets_data import (
    REQUIRED_DEEPSETS_MANIFEST_COLUMNS,
    SavedSetLookup,
    apply_feature_standardizer,
    collate_case_sets,
    fit_feature_standardizer,
)
from deepsets_model import DeepSetsClassifier
from evaluation import FoldResults
from evaluation.build_splits import create_splits_for_dataframe
from evaluation.kfold import FoldSplit
from evaluation.utils import prepare_predictions_df
from load_cohort import load_config, resolve_run_output_dir, write_config_snapshot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPTIONAL_SPLIT_COLUMNS = (
    "site",
    "tumor_subtype",
    "dataset",
    "bilateral",
)
DEFAULT_PROBABILITY_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ispy2.yaml"),
        help="Path to YAML config. Add deepsets_manifest_csv under data_paths.",
    )
    parser.add_argument("--outdir", type=Path, help="Override output directory")
    return parser.parse_args()


def describe_required_deepsets_config() -> dict[str, str]:
    """Return the config keys expected for the Deep Sets path."""
    return {
        "data_paths.deepsets_manifest_csv": (
            "CSV produced by build_deepsets_dataset.py with one row per case and set_path."
        ),
        "model_params.batch_size": "Mini-batch size for case sets.",
        "model_params.epochs": "Number of training epochs per outer fold.",
        "model_params.learning_rate": "Adam learning rate.",
        "model_params.hidden_dim": "Hidden width used by the phi and rho MLPs.",
        "model_params.num_layers": "Number of layers in the phi and rho MLPs.",
        "model_params.pooling": "Pooling variant: mean, max, sum, mean_max, or mean_max_logcount.",
        "model_params.loss": "Loss function: weighted_bce, unweighted_bce, or focal.",
    }


def validate_deepsets_manifest(
    manifest_df: pd.DataFrame, config: dict[str, Any]
) -> None:
    """Validate that the Deep Sets manifest contains the minimum required columns."""
    label_col = config.data_paths.deepsets_label_column
    required_columns = set(REQUIRED_DEEPSETS_MANIFEST_COLUMNS) | {label_col}
    missing = sorted(required_columns.difference(manifest_df.columns))
    if missing:
        raise ValueError(
            "Deep Sets manifest is missing required columns: "
            f"{missing}. Required columns are {sorted(required_columns)}."
        )
    for column in REQUIRED_DEEPSETS_MANIFEST_COLUMNS:
        if manifest_df[column].isna().any():
            raise ValueError(f"Deep Sets manifest has missing {column} values.")
    missing_sets = [
        str(path)
        for path in manifest_df["set_path"].astype(str)
        if not Path(path).exists()
    ]
    if missing_sets:
        raise FileNotFoundError(
            "Deep Sets manifest points to missing case-set files. First few: "
            f"{missing_sets[:5]}"
        )


def load_deepsets_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Load the case-level manifest built by build_deepsets_dataset.py."""
    _ = describe_required_deepsets_config()
    manifest_path = config.data_paths.deepsets_manifest_csv
    if not manifest_path:
        raise ValueError(
            "Missing data_paths.deepsets_manifest_csv. Run build_deepsets_dataset.py first."
        )
    manifest_df = pd.read_csv(Path(manifest_path))
    manifest_df["case_id"] = manifest_df["case_id"].astype(str)
    return manifest_df


def build_deepsets_dataset(
    manifest_df: pd.DataFrame, config: dict[str, Any]
) -> SavedSetLookup:
    """Create a lazy lookup over serialized case point sets."""
    _ = config
    return SavedSetLookup(manifest_df)


def infer_deepsets_feature_names(dataset: SavedSetLookup, case_id: str) -> list[str]:
    """Read feature names from one serialized case set."""
    payload = dataset.get(str(case_id))
    return [str(name) for name in payload.get("feature_names", [])]


def build_fold_prediction_table(
    *,
    fold_case_ids: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    fold_idx: int,
    manifest_df: pd.DataFrame,
    stratum_col: str | None,
) -> pd.DataFrame:
    """Build the evaluator-ready prediction table for one fold."""
    pred_df = prepare_predictions_df(
        case_ids=fold_case_ids,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fold=fold_idx,
    )
    if stratum_col and stratum_col in manifest_df.columns:
        pred_df["stratum"] = (
            manifest_df.set_index("case_id")
            .loc[pred_df["case_id"].astype(str), stratum_col]
            .astype(str)
            .to_numpy()
        )
    return pred_df


def _resolve_device(config: dict[str, Any]) -> torch.device:
    """Choose compute device for the Deep Sets run."""
    requested = str(config.model_params.device).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loaders(
    case_sets: SavedSetLookup,
    manifest_df: pd.DataFrame,
    split: FoldSplit,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    """Build loaders for one fold."""
    train_df = manifest_df.iloc[split.train_indices].copy().reset_index(drop=True)
    val_df = manifest_df.iloc[split.val_indices].copy().reset_index(drop=True)
    train_sets = case_sets.subset(train_df["case_id"].astype(str).tolist())
    val_sets = case_sets.subset(val_df["case_id"].astype(str).tolist())
    feature_mean, feature_std = fit_feature_standardizer(train_sets)
    train_sets = apply_feature_standardizer(
        train_sets,
        mean=feature_mean,
        std=feature_std,
    )
    val_sets = apply_feature_standardizer(
        val_sets,
        mean=feature_mean,
        std=feature_std,
    )
    train_loader = DataLoader(
        train_sets,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_case_sets,
    )
    val_loader = DataLoader(
        val_sets,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_case_sets,
    )
    return train_loader, val_loader, train_df, val_df


class FocalWithLogitsLoss(nn.Module):
    """Sigmoid focal loss for class-imbalanced binary classification.

    Applies a modulating factor (1 - p_t)^gamma to down-weight easy examples,
    with an optional alpha balancing term.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


LOSS_CHOICES = ("weighted_bce", "unweighted_bce", "focal")


def build_loss_fn(
    config: dict[str, Any],
    positive_count: int,
    negative_count: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate the training loss function from config."""
    loss_name = str(config.model_params.get("loss", "weighted_bce"))
    if loss_name not in LOSS_CHOICES:
        raise ValueError(
            f"Unknown loss {loss_name!r}. Choose from {LOSS_CHOICES}."
        )
    if loss_name == "weighted_bce":
        pos_weight_value = (
            float(negative_count) / float(positive_count)
            if positive_count > 0
            else 1.0
        )
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [pos_weight_value], dtype=torch.float32, device=device
            )
        )
    if loss_name == "unweighted_bce":
        return nn.BCEWithLogitsLoss()
    # focal
    alpha = float(config.model_params.get("focal_alpha", 0.25))
    gamma = float(config.model_params.get("focal_gamma", 2.0))
    return FocalWithLogitsLoss(alpha=alpha, gamma=gamma)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_cases = 0
    for batch in loader:
        x = batch["x"].to(device)
        batch_index = batch["batch_index"].to(device)
        targets = batch["y"].to(device)
        optimizer.zero_grad()
        logits = model(x, batch_index)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = int(targets.numel())
        total_loss += float(loss.item()) * batch_size
        total_cases += batch_size
    return total_loss / max(total_cases, 1)


def _predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run inference and return labels, predictions, probabilities, and case IDs."""
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []
    case_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            batch_index = batch["batch_index"].to(device)
            logits = model(x, batch_index)
            probs = torch.sigmoid(logits)
            preds = (probs >= DEFAULT_PROBABILITY_THRESHOLD).long()
            y_true_chunks.append(batch["y"].cpu().numpy().astype(int))
            y_pred_chunks.append(preds.cpu().numpy().astype(int))
            y_prob_chunks.append(probs.cpu().numpy().astype(float))
            case_ids.extend([str(case_id) for case_id in batch["case_ids"]])
    if not y_true_chunks:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
            [],
        )
    return (
        np.concatenate(y_true_chunks),
        np.concatenate(y_pred_chunks),
        np.concatenate(y_prob_chunks),
        case_ids,
    )


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Compute average loss over a loader without updating model weights."""
    model.eval()
    total_loss = 0.0
    total_cases = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            batch_index = batch["batch_index"].to(device)
            targets = batch["y"].to(device)
            logits = model(x, batch_index)
            loss = loss_fn(logits, targets)
            batch_size = int(targets.numel())
            total_loss += float(loss.item()) * batch_size
            total_cases += batch_size
    return total_loss / max(total_cases, 1)


def _plot_probability_distribution(
    predictions: pd.DataFrame, output_path: Path
) -> None:
    """Plot predicted-probability distributions split by true label."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    negative_probs = predictions.loc[predictions["y_true"] == 0, "y_prob"].to_numpy()
    positive_probs = predictions.loc[predictions["y_true"] == 1, "y_prob"].to_numpy()
    bins = np.linspace(0.0, 1.0, 31)
    ax.hist(negative_probs, bins=bins, alpha=0.6, label="y_true = 0", density=True)
    ax.hist(positive_probs, bins=bins, alpha=0.6, label="y_true = 1", density=True)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_loss_history(loss_history_df: pd.DataFrame, output_path: Path) -> None:
    """Plot train and validation loss by epoch for each fold."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for fold_idx, fold_df in loss_history_df.groupby("fold"):
        label = f"fold {int(fold_idx)}"
        axes[0].plot(fold_df["epoch"], fold_df["train_loss"], label=label)
        axes[1].plot(fold_df["epoch"], fold_df["val_loss"], label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation loss")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def fit_predict_one_fold(
    *,
    dataset: SavedSetLookup,
    manifest_df: pd.DataFrame,
    split: FoldSplit,
    config: dict[str, Any],
) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Train the starter Deep Sets model for one fold and return validation predictions."""
    params = config.model_params
    batch_size = int(params.batch_size)
    epochs = int(params.epochs)
    hidden_dim = int(params.hidden_dim)
    num_layers = int(params.num_layers)
    dropout = float(params.dropout)
    learning_rate = float(params.learning_rate)
    weight_decay = float(params.weight_decay)
    pooling = str(params.get("pooling", "mean_max_logcount"))
    device = _resolve_device(config)
    random_state = int(params.random_state)
    torch.manual_seed(random_state + int(split.fold_idx))
    np.random.seed(random_state + int(split.fold_idx))

    train_loader, val_loader, train_df, _ = _build_loaders(
        case_sets=dataset,
        manifest_df=manifest_df,
        split=split,
        batch_size=batch_size,
    )
    first_case = dataset.get(str(manifest_df.iloc[split.val_indices[0]]["case_id"]))
    model = DeepSetsClassifier(
        input_dim=int(first_case["x"].shape[1]),
        hidden_dim=hidden_dim,
        phi_layers=num_layers,
        rho_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    positive_count = int(train_df[config.data_paths.deepsets_label_column].sum())
    negative_count = int(len(train_df) - positive_count)
    loss_name = str(config.model_params.get("loss", "weighted_bce"))
    logging.info(
        "fold %d class balance: positives=%d negatives=%d loss=%s",
        split.fold_idx,
        positive_count,
        negative_count,
        loss_name,
    )
    loss_fn = build_loss_fn(
        config=config,
        positive_count=positive_count,
        negative_count=negative_count,
        device=device,
    )
    loss_history: list[dict[str, float]] = []

    for epoch_idx in range(epochs):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_loss = _evaluate_loss(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        loss_history.append(
            {
                "fold": float(split.fold_idx),
                "epoch": float(epoch_idx + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        logging.info(
            "fold %d epoch %d/%d train_loss=%.4f val_loss=%.4f",
            split.fold_idx,
            epoch_idx + 1,
            epochs,
            train_loss,
            val_loss,
        )

    y_true, y_pred, y_prob, case_ids = _predict_loader(
        model=model, loader=val_loader, device=device
    )
    return pd.Series(case_ids, dtype=str), y_true, y_pred, y_prob, loss_history


def run_deepsets_pipeline(config: dict[str, Any], outdir: Path) -> None:
    """Run the minimal Deep Sets training/evaluation pipeline."""
    manifest_df = load_deepsets_manifest(config)
    validate_deepsets_manifest(manifest_df, config)
    label_col = config.data_paths.deepsets_label_column
    manifest_df[label_col] = manifest_df[label_col].astype(int)

    dataset = build_deepsets_dataset(manifest_df, config)
    y = manifest_df[label_col].astype(int)
    case_ids = manifest_df["case_id"].astype(str)
    split_frame = manifest_df.copy()
    feature_names = infer_deepsets_feature_names(dataset, case_ids.iloc[0])
    split_X = pd.DataFrame(
        np.zeros((len(split_frame), len(feature_names)), dtype=np.float32),
        columns=feature_names,
        index=split_frame.index,
    )

    evaluator, splits, stratum_col = create_splits_for_dataframe(
        X=split_X,
        y=y,
        case_ids=case_ids,
        cohort_df=split_frame,
        config=config,
        model_name=config.experiment_setup.name,
    )

    fold_results: list[FoldResults] = []
    all_loss_rows: list[dict[str, float]] = []
    for split in splits:
        fold_case_ids, y_true, y_pred, y_prob, loss_history = fit_predict_one_fold(
            dataset=dataset,
            manifest_df=manifest_df,
            split=split,
            config=config,
        )
        all_loss_rows.extend(loss_history)
        pred_df = build_fold_prediction_table(
            fold_case_ids=fold_case_ids,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            fold_idx=split.fold_idx,
            manifest_df=manifest_df,
            stratum_col=stratum_col,
        )
        fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))

    kfold_results = evaluator.aggregate_kfold_results(fold_results)
    evaluator.save_results(kfold_results, outdir)
    model_dir = outdir / config.experiment_setup.name
    model_dir.mkdir(parents=True, exist_ok=True)
    loss_history_df = pd.DataFrame(all_loss_rows)
    if not loss_history_df.empty:
        loss_history_df.to_csv(model_dir / "loss_history.csv", index=False)
        _plot_loss_history(loss_history_df, model_dir / "loss_by_epoch.png")
    _plot_probability_distribution(
        kfold_results.predictions,
        model_dir / "probability_distribution.png",
    )
    probability_summary = {
        "mean": float(kfold_results.predictions["y_prob"].mean()),
        "std": float(kfold_results.predictions["y_prob"].std()),
        "min": float(kfold_results.predictions["y_prob"].min()),
        "median": float(kfold_results.predictions["y_prob"].median()),
        "max": float(kfold_results.predictions["y_prob"].max()),
        "positive_rate_at_0_5": float(
            (kfold_results.predictions["y_pred"] == 1).mean()
        ),
    }
    (model_dir / "probability_summary.json").write_text(
        json.dumps(probability_summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Entry point for the starter Deep Sets pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    outdir = resolve_run_output_dir(config=config, outdir_override=args.outdir)
    write_config_snapshot(config=config, outdir=outdir, config_source=args.config)
    run_deepsets_pipeline(config, outdir)


if __name__ == "__main__":
    main()
