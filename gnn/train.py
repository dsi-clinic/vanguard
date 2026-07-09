"""Training script for the GNN centerline classifier.

Builds one row per graph (case_id, label, dataset, site, graph_index, and an
optional predefined fold column) from a cached ``VanguardCenterlineDataset``,
then hands that cohort table to the shared ``evaluation`` framework
(``evaluation.build_splits.create_splits_for_dataframe``) for splitting,
metric aggregation, and output saving. See ``gnn/README.md``.

Usage::

    python -m gnn.train --config configs/gnn.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from evaluation import FoldResults
from evaluation.build_splits import build_split_manifest, create_splits_for_dataframe
from evaluation.kfold import FoldSplit
from evaluation.metrics import compute_binary_metrics
from evaluation.utils import prepare_predictions_df
from gnn.data_loader import VanguardCenterlineDataset
from gnn.graph_qc_plots import GRAPH_QC_PLOTS_DIRNAME, write_prediction_plot
from gnn.model import GCNClassifier
from load_cohort import load_config, resolve_run_output_dir, write_config_snapshot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MIN_STD = 1e-6
_DECISION_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gnn.yaml"))
    parser.add_argument("--outdir", type=Path, help="Override output directory")
    parser.add_argument(
        "--pcr-dummy-class1-mean",
        type=float,
        default=None,
        help=(
            "Override model_params.gnn_pcr_dummy_class1_mean "
            "(pcr_dummy Gaussian-noise sweep; see _apply_pcr_dummy_noise)"
        ),
    )
    parser.add_argument(
        "--pcr-dummy-noise-std",
        type=float,
        default=None,
        help=(
            "Override model_params.gnn_pcr_dummy_noise_std "
            "(pcr_dummy Gaussian-noise sweep; see _apply_pcr_dummy_noise)"
        ),
    )
    parser.add_argument(
        "--pcr-dummy-noise-seed",
        type=int,
        default=None,
        help="Override model_params.gnn_pcr_dummy_noise_seed",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    """Choose compute device, failing loudly if CUDA was requested but absent."""
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "model_params.device='cuda' requested but no GPU is available."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_graph_cohort(
    dataset: VanguardCenterlineDataset,
    *,
    labels_path: Path,
    id_column: str,
    fold_column: str,
) -> pd.DataFrame:
    """One row per built graph: case_id, y, dataset, site, graph_index, num_nodes, num_edges.

    ``graph_index`` is the position of the graph in ``dataset`` -- splits
    computed over this dataframe's row order are translated back to actual
    graphs via that column, since restricting the cohort (e.g. by dataset)
    changes row order/count but not each graph's underlying dataset position.

    If the labels file has a ``fold_column`` column (e.g. "fold"), it is
    merged in so ``model_params.split_mode: "predefined"`` can use it. Every
    built graph must then resolve to a fold value -- a labels file with a
    fold column that doesn't cover every case is a stale/mismatched labels
    file, not something to silently drop rows for.
    """
    rows = []
    for i in range(len(dataset)):
        graph = dataset[i]
        rows.append(
            {
                "case_id": str(graph.case_id),
                "y": int(graph.y.item()),
                "dataset": graph.dataset,
                "site": graph.site,
                "graph_index": i,
                "num_nodes": int(graph.num_nodes),
                "num_edges": int(graph.num_edges),
            }
        )
    cohort_df = pd.DataFrame(rows)

    if Path(labels_path).suffix.lower() == ".csv":
        labels_df = pd.read_csv(labels_path)
        if fold_column in labels_df.columns:
            fold_map = labels_df.set_index(labels_df[id_column].astype(str))[
                fold_column
            ]
            cohort_df[fold_column] = cohort_df["case_id"].map(fold_map)
            if cohort_df[fold_column].isna().any():
                missing = cohort_df.loc[
                    cohort_df[fold_column].isna(), "case_id"
                ].tolist()
                raise ValueError(
                    f"labels file has a {fold_column!r} column but is missing "
                    f"values for {len(missing)} built graph(s): {missing[:10]}"
                )

    return cohort_df


def fit_node_standardizer(graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std over training-split node features.

    Takes already-cloned graphs (post noise-injection, if configured -- see
    ``_apply_pcr_dummy_noise``) rather than indexing into the cached dataset
    directly, so standardization reflects whatever the model actually trains
    on.
    """
    feature_matrix = torch.cat([graph.x for graph in graphs], dim=0)
    mean = feature_matrix.mean(dim=0)
    std = feature_matrix.std(dim=0, unbiased=False)
    std = torch.where(std < _MIN_STD, torch.ones_like(std), std)
    return mean, std


def _apply_pcr_dummy_noise(
    graphs: list[Data],
    graph_indices: list[int],
    node_features: tuple[str, ...],
    params: Any,
) -> None:
    """In place: redraw the ``pcr_dummy`` column as class-conditional Gaussian noise.

    ``pcr_dummy`` is cached (``gnn.data_loader``) as the graph's own label
    broadcast onto every node -- a deterministic 0/1 leakage canary. This
    layers a *train-time* Gaussian noise model on top of that cached column,
    without ever rebuilding the (expensive, raw-DCE-derived) dataset cache:
    label 0 -> ``class0_mean``, label 1 -> ``class1_mean``, both plus
    ``Normal(0, noise_std)``. Defaults (0.0, 1.0, 0.0) exactly reproduce the
    original deterministic broadcast, so this is a no-op unless at least one
    of the three params is explicitly overridden away from its default.

    Note ``gnn/train.py`` standardizes every node feature per-fold as
    ``(x - train_mean) / train_std`` (see ``fit_node_standardizer``), which
    exactly cancels any constant offset added to this column -- so
    ``class0_mean``/``class1_mean`` only matter through their *difference*
    relative to ``noise_std`` (a Cohen's-d / SNR quantity), not their
    absolute values.

    Noise is drawn once per *graph* (not per node) and broadcast across all
    of that graph's nodes, matching how the clean dummy is already broadcast
    -- this experiment tests whether the model can extract a graded per-case
    signal, not whether GCNConv's mean-pool averages out per-node iid noise
    on its own. The draw is keyed by ``graph_index`` (the graph's position in
    the cached dataset) rather than fold/loader order, so the same case gets
    the same noisy value across folds, seeds, and reruns that share
    ``gnn_pcr_dummy_noise_seed``.
    """
    class0_mean = float(params.gnn_pcr_dummy_class0_mean)
    class1_mean = float(params.gnn_pcr_dummy_class1_mean)
    noise_std = float(params.gnn_pcr_dummy_noise_std)
    if class0_mean == 0.0 and class1_mean == 1.0 and noise_std == 0.0:
        return
    if "pcr_dummy" not in node_features:
        raise ValueError(
            "gnn_pcr_dummy_{class0_mean,class1_mean,noise_std} were set away "
            "from their defaults, but 'pcr_dummy' is not in gnn_node_features "
            "-- there is no column for this noise to apply to."
        )
    if noise_std < 0.0:
        raise ValueError(f"gnn_pcr_dummy_noise_std must be >= 0, got {noise_std}")
    col = node_features.index("pcr_dummy")
    generator = torch.Generator().manual_seed(int(params.gnn_pcr_dummy_noise_seed))
    noise_by_index = (
        torch.randn(max(graph_indices) + 1, generator=generator) * noise_std
    )
    for graph, index in zip(graphs, graph_indices, strict=True):
        class_mean = class1_mean if int(graph.y.item()) == 1 else class0_mean
        graph.x[:, col] = class_mean + noise_by_index[index]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; return the mean per-graph loss."""
    model.train()
    total_loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        num_graphs += batch.num_graphs
    return total_loss / num_graphs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run inference over a loader; return loss + compute_binary_metrics."""
    model.eval()
    total_loss = 0.0
    num_graphs = 0
    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.float())
        total_loss += loss.item() * batch.num_graphs
        num_graphs += batch.num_graphs
        y_true_parts.append(batch.y.cpu().numpy())
        y_prob_parts.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(y_true_parts)
    y_prob = np.concatenate(y_prob_parts)
    y_pred = (y_prob >= _DECISION_THRESHOLD).astype(int)
    metrics = compute_binary_metrics(y_true, y_pred.astype(int), y_prob)
    metrics["loss"] = total_loss / num_graphs
    metrics["error_rate"] = float(np.mean(y_pred != y_true))
    return metrics


@torch.no_grad()
def _predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a shuffle=False loader; return concatenated (y_true, y_prob)."""
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        y_true_parts.append(batch.y.cpu().numpy())
        y_prob_parts.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(y_true_parts), np.concatenate(y_prob_parts)


def build_fold_prediction_table(
    *,
    val_case_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    fold_idx: int,
    cohort_df: pd.DataFrame,
    stratum_col: str | None,
) -> pd.DataFrame:
    """Build the evaluator-ready prediction table for one fold.

    Adds dataset/site/pcr alongside the evaluator's standard case_id/y_true/
    y_pred/y_prob/fold columns, so predictions.csv is enough on its own to
    check whether the model is tracking biology (pcr), site effects, or
    dataset effects without cross-referencing the cohort table. ``pcr`` is a
    duplicate of ``y_true`` under the label's own name -- kept because
    downstream evaluator code (metrics, plots) requires the "y_true" column
    name, so it can't just be renamed.
    """
    pred_df = prepare_predictions_df(
        case_ids=val_case_ids,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fold=fold_idx,
    )
    cohort_by_case = cohort_df.set_index("case_id")
    case_ids_str = pred_df["case_id"].astype(str)
    pred_df["dataset"] = cohort_by_case.loc[case_ids_str, "dataset"].to_numpy()
    pred_df["site"] = cohort_by_case.loc[case_ids_str, "site"].to_numpy()
    pred_df["pcr"] = pred_df["y_true"]
    if stratum_col and stratum_col in cohort_df.columns:
        pred_df["stratum"] = (
            cohort_by_case.loc[case_ids_str, stratum_col].astype(str).to_numpy()
        )
    return pred_df


def fit_predict_one_fold(
    *,
    dataset: VanguardCenterlineDataset,
    cohort_df: pd.DataFrame,
    split: FoldSplit,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Train a fresh GCNClassifier for one fold and return validation predictions."""
    params = config.model_params
    device = _resolve_device(str(params.device))
    torch.manual_seed(int(params.random_state) + int(split.fold_idx))

    train_graph_idx = cohort_df.iloc[split.train_indices]["graph_index"].tolist()
    val_graph_idx = cohort_df.iloc[split.val_indices]["graph_index"].tolist()

    train_graphs = [dataset[i].clone() for i in train_graph_idx]
    val_graphs = [dataset[i].clone() for i in val_graph_idx]
    _apply_pcr_dummy_noise(
        train_graphs + val_graphs,
        train_graph_idx + val_graph_idx,
        node_features=tuple(params.gnn_node_features),
        params=params,
    )
    mean, std = fit_node_standardizer(train_graphs)
    for graph in train_graphs + val_graphs:
        graph.x = (graph.x - mean) / std

    batch_size = int(params.batch_size)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    model = GCNClassifier(
        input_dim=train_graphs[0].x.shape[1],
        hidden_dim=int(params.hidden_dim),
        num_layers=int(params.num_layers),
        dropout=float(params.dropout),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params.learning_rate))
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    epochs = int(params.epochs)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_eval_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "fold": float(split.fold_idx),
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_error_rate": train_metrics["error_rate"],
                "train_auc": train_metrics.get("auc", float("nan")),
                "val_loss": val_metrics["loss"],
                "val_error_rate": val_metrics["error_rate"],
                "val_auc": val_metrics.get("auc", float("nan")),
            }
        )
        logging.info(
            "fold %d epoch %d/%d train_loss=%.4f train_auc=%.4f "
            "val_loss=%.4f val_auc=%.4f",
            split.fold_idx,
            epoch,
            epochs,
            train_loss,
            train_metrics.get("auc", float("nan")),
            val_metrics["loss"],
            val_metrics.get("auc", float("nan")),
        )

    y_true, y_prob = _predict_loader(model, val_loader, device)
    y_pred = (y_prob >= _DECISION_THRESHOLD).astype(int)
    val_case_ids = cohort_df.iloc[split.val_indices]["case_id"].astype(str).to_numpy()
    return val_case_ids, y_true, y_pred, y_prob, history


def _plot_metric_history(
    history_df: pd.DataFrame, metric: str, output_path: Path
) -> None:
    """Plot train/val curves for one metric (loss or auc), one line per fold."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for fold_idx, fold_df in history_df.groupby("fold"):
        label = f"fold {int(fold_idx)}"
        axes[0].plot(fold_df["epoch"], fold_df[f"train_{metric}"], label=label)
        axes[1].plot(fold_df["epoch"], fold_df[f"val_{metric}"], label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(f"train {metric}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(f"validation {metric}")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_gnn_pipeline(config: Any, outdir: Path) -> None:
    """Build the dataset cohort table, run k-fold CV, and save evaluator outputs."""
    params = config.model_params
    data_paths = config.data_paths

    dataset = VanguardCenterlineDataset(
        root=data_paths.gnn_centerline_root,
        labels_path=data_paths.gnn_labels_path,
        dce_root=data_paths.gnn_dce_root,
        cache_dir=data_paths.gnn_cache_dir,
        cases=list(data_paths.gnn_cases) if data_paths.gnn_cases else None,
        node_features=tuple(params.gnn_node_features),
        id_column=data_paths.gnn_id_column,
        label_column=data_paths.gnn_label_column,
        allow_manifest_mismatch=bool(data_paths.gnn_allow_manifest_mismatch),
    )
    logging.info(
        "Loaded %d graph(s) from cache %s", len(dataset), data_paths.gnn_cache_dir
    )

    cohort_df = build_graph_cohort(
        dataset,
        labels_path=Path(data_paths.gnn_labels_path),
        id_column=data_paths.gnn_id_column,
        fold_column=str(params.split_col),
    )

    if data_paths.gnn_dataset_include:
        wanted = set(data_paths.gnn_dataset_include)
        cohort_df = cohort_df[cohort_df["dataset"].isin(wanted)]
        if cohort_df.empty:
            raise ValueError(
                f"gnn_dataset_include={data_paths.gnn_dataset_include!r} "
                "matched no graphs in the cache"
            )
    cohort_df = cohort_df.reset_index(drop=True)

    y = cohort_df["y"].astype(int)
    case_ids = cohort_df["case_id"].astype(str)
    feature_names = list(params.gnn_node_features)
    split_X = pd.DataFrame(
        np.zeros((len(cohort_df), len(feature_names)), dtype=np.float32),
        columns=feature_names,
    )

    evaluator, splits, stratum_col = create_splits_for_dataframe(
        X=split_X,
        y=y,
        case_ids=case_ids,
        cohort_df=cohort_df,
        config=config,
        model_name=config.experiment_setup.name,
    )

    model_dir = outdir / config.experiment_setup.name
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_columns = ["graph_index", "dataset", "site", "y", "num_nodes", "num_edges"]
    split_manifest_df = build_split_manifest(
        cohort_df, splits, columns=manifest_columns
    )
    split_manifest_df = split_manifest_df.rename(columns={"y": "pcr"})[
        [
            "case_id",
            "graph_index",
            "dataset",
            "site",
            "pcr",
            "fold",
            "train_or_val",
            "num_nodes",
            "num_edges",
        ]
    ]
    split_manifest_df.to_csv(model_dir / "split_manifest.csv", index=False)

    fold_results: list[FoldResults] = []
    all_history_rows: list[dict[str, float]] = []
    for split in splits:
        val_case_ids, y_true, y_pred, y_prob, history = fit_predict_one_fold(
            dataset=dataset, cohort_df=cohort_df, split=split, config=config
        )
        all_history_rows.extend(history)
        pred_df = build_fold_prediction_table(
            val_case_ids=val_case_ids,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            fold_idx=split.fold_idx,
            cohort_df=cohort_df,
            stratum_col=stratum_col,
        )
        fold_results.append(FoldResults(fold_idx=split.fold_idx, predictions=pred_df))

    kfold_results = evaluator.aggregate_kfold_results(fold_results)
    evaluator.save_results(kfold_results, outdir)

    # prediction_vs_num_nodes.png needs a trained model's predictions, which
    # don't exist at build time (see gnn/data_loader.py::_write_graph_qc for
    # the other 4 graph_qc plots). Written into this run's own output dir
    # (alongside predictions.csv) and back into the cache's graph_qc_plots/,
    # so the cache always reflects whichever training run against it is most
    # recent -- it is overwritten on every run, not accumulated per-run.
    case_num_nodes = cohort_df[["case_id", "num_nodes"]]
    write_prediction_plot(kfold_results.predictions, case_num_nodes, model_dir)
    write_prediction_plot(
        kfold_results.predictions,
        case_num_nodes,
        Path(dataset.processed_dir) / GRAPH_QC_PLOTS_DIRNAME,
    )

    history_df = pd.DataFrame(all_history_rows)
    history_df.to_csv(model_dir / "loss_history.csv", index=False)
    _plot_metric_history(history_df, "loss", model_dir / "loss_by_epoch.png")
    _plot_metric_history(history_df, "auc", model_dir / "auc_by_epoch.png")

    logging.info("Wrote GNN run outputs to %s", outdir)


def main() -> None:
    """Build the dataset, train, evaluate, and write auditing outputs."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    if args.pcr_dummy_class1_mean is not None:
        config.model_params.gnn_pcr_dummy_class1_mean = args.pcr_dummy_class1_mean
    if args.pcr_dummy_noise_std is not None:
        config.model_params.gnn_pcr_dummy_noise_std = args.pcr_dummy_noise_std
    if args.pcr_dummy_noise_seed is not None:
        config.model_params.gnn_pcr_dummy_noise_seed = args.pcr_dummy_noise_seed
    outdir = resolve_run_output_dir(config=config, outdir_override=args.outdir)
    write_config_snapshot(config=config, outdir=outdir, config_source=args.config)
    run_gnn_pipeline(config, outdir)


if __name__ == "__main__":
    main()
