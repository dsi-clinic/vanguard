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
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from evaluation import FoldResults, KFoldResults
from evaluation.build_splits import build_split_manifest, create_splits_for_dataframe
from evaluation.kfold import FoldSplit
from evaluation.metrics import compute_binary_metrics
from evaluation.utils import prepare_predictions_df
from gnn.clinical import fit_clinical_transformer, transform_clinical_frame
from gnn.data_loader import VanguardCenterlineDataset
from gnn.graph_qc_plots import GRAPH_QC_PLOTS_DIRNAME, write_prediction_plot
from gnn.model import EdgeGNNClassifier, GCNClassifier
from gnn.morphometry import fit_morphometry_imputer, transform_morphometry_frame
from load_cohort import load_config, resolve_run_output_dir, write_config_snapshot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MIN_STD = 1e-6
_DECISION_THRESHOLD = 0.5
# Smallest usable StratifiedKFold: fewer than 2 folds (or 2 of either class)
# can't hold out a stratified inner-validation split -- see _stratified_inner_split.
_MIN_INNER_SPLIT_FOLDS = 2


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
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Override model_params.random_state (per-fold torch.manual_seed "
        "offset; does not affect split_mode='predefined' fold assignment, "
        "which is frozen by the labels file's fold column regardless).",
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


def _fit_standardizer(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column mean/std, with near-constant columns left unscaled (std->1)."""
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, unbiased=False)
    std = torch.where(std < _MIN_STD, torch.ones_like(std), std)
    return mean, std


def fit_node_standardizer(graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std over training-split node features.

    Takes already-cloned graphs (post noise-injection, if configured -- see
    ``_apply_pcr_dummy_noise``) rather than indexing into the cached dataset
    directly, so standardization reflects whatever the model actually trains
    on.
    """
    return _fit_standardizer(torch.cat([graph.x for graph in graphs], dim=0))


def fit_edge_standardizer(graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std over training-split edge features.

    Junction mode (segment-as-edge) only: the segment summary lives on
    ``edge_attr``, so it needs the same per-fold standardization ``data.x`` gets.
    """
    return _fit_standardizer(torch.cat([graph.edge_attr for graph in graphs], dim=0))


def fit_graph_standardizer(graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std over training-split graph-level (clinical) features.

    Applied after :func:`_fit_transform_graph_features` has already fit the
    clinical impute+one-hot and morphometry imputer on the training split and
    attached ``graph_features`` to each graph. The one-hot-encoded categorical
    columns get standardized the same as the numeric ones (a
    constant-across-the-fold indicator column is caught by
    ``_fit_standardizer``'s near-constant guard, same as any other feature).
    """
    return _fit_standardizer(
        torch.cat([graph.graph_features for graph in graphs], dim=0)
    )


def _fit_transform_graph_features(
    inputs: pd.DataFrame,
    train_case_ids: list[str],
    val_case_ids: list[str],
    groups: tuple[list[str], list[str], list[str]],
) -> dict[str, np.ndarray]:
    """Fit clinical/morphometry preprocessing on the TRAIN split, transform both.

    ``inputs`` is the raw graph-feature-input frame
    (``VanguardCenterlineDataset.load_graph_feature_inputs``), indexed by
    ``case_id``. The clinical imputer + one-hot encoder and the morphometry
    mean-imputer are fit on the fold's **training** cases only; the validation
    cases are transformed with those fitted objects, and any category unseen in
    training encodes as all-zero (``handle_unknown="ignore"``, i.e. treated as
    unknown). Graph-derived columns are pure per-graph reads with no cross-case
    fit, so they pass through unchanged.

    This is what makes cross-validation reflect a genuinely unseen cohort:
    validation cases never contribute to the imputer means/modes or the one-hot
    vocabulary. Blocks are concatenated in the fixed source order
    ``(clinical, graph-derived, morphometry)`` -- the same order the sidecar
    columns are laid out in. Returns a ``{case_id: feature_vector}`` mapping for
    every train and validation case.
    """
    clinical_cols, graph_derived_cols, morphometry_cols = groups
    fold_case_ids = list(train_case_ids) + list(val_case_ids)
    fold = inputs.loc[fold_case_ids]
    train = inputs.loc[list(train_case_ids)]

    blocks: list[np.ndarray] = []
    if clinical_cols:
        transformer = fit_clinical_transformer(train[clinical_cols], clinical_cols)
        clinical_matrix, _ = transform_clinical_frame(transformer, fold[clinical_cols])
        blocks.append(clinical_matrix)
    if graph_derived_cols:
        blocks.append(fold[graph_derived_cols].to_numpy(dtype=np.float64))
    if morphometry_cols:
        imputer = fit_morphometry_imputer(train[morphometry_cols])
        blocks.append(transform_morphometry_frame(imputer, fold[morphometry_cols]))

    combined = np.concatenate(blocks, axis=1)
    return {case_id: combined[row] for row, case_id in enumerate(fold_case_ids)}


def _model_forward(model: nn.Module, batch: Data) -> torch.Tensor:
    """Call ``model`` with the right signature for its representation.

    ``EdgeGNNClassifier`` (junction mode) consumes ``edge_attr``; the plain
    ``GCNClassifier`` (voxel/segment) does not. Either passes ``graph_features``
    through when the batch carries it (``gnn_graph_features`` was set at
    build time), regardless of node mode.
    """
    graph_features = getattr(batch, "graph_features", None)
    if isinstance(model, EdgeGNNClassifier):
        return model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, graph_features
        )
    return model(batch.x, batch.edge_index, batch.batch, graph_features)


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


class FocalWithLogitsLoss(nn.Module):
    """Sigmoid focal loss for class-imbalanced binary classification.

    Applies a modulating factor (1 - p_t)^gamma to down-weight easy examples,
    with an optional alpha balancing term. Mirrors
    ``deepsets.train.FocalWithLogitsLoss`` (reimplemented rather than imported
    to avoid coupling the GNN track to the Deep Sets one -- the two are kept
    independent everywhere else too).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the mean focal loss over the batch."""
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
    params: Any,
    positive_count: int,
    negative_count: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate the training loss from ``model_params``.

    ``weighted_bce`` (the config default) sets ``BCEWithLogitsLoss``'s
    ``pos_weight`` to ``negative_count / positive_count`` computed on the
    fold's *train* split only, so the minority (pCR-positive) class is
    up-weighted; ``unweighted_bce`` reproduces the historical GNN loss
    exactly; ``focal`` uses ``focal_alpha`` / ``focal_gamma``. Parallels
    ``deepsets.train.build_loss_fn`` (same three choices, same pos_weight
    convention) -- kept as a small local copy rather than a cross-track import.
    A fold with zero training positives is a broken split, not something to
    paper over, so pos_weight falls back to 1.0 only in that degenerate case
    (matching Deep Sets) and the class balance is logged by the caller.
    """
    loss_name = str(params.get("loss", "weighted_bce"))
    if loss_name not in LOSS_CHOICES:
        raise ValueError(f"Unknown loss {loss_name!r}. Choose from {LOSS_CHOICES}.")
    if loss_name == "weighted_bce":
        pos_weight_value = (
            float(negative_count) / float(positive_count) if positive_count > 0 else 1.0
        )
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [pos_weight_value], dtype=torch.float32, device=device
            )
        )
    if loss_name == "unweighted_bce":
        return nn.BCEWithLogitsLoss()
    alpha = float(params.get("focal_alpha", 0.25))
    gamma = float(params.get("focal_gamma", 2.0))
    return FocalWithLogitsLoss(alpha=alpha, gamma=gamma)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 0.0,
) -> float:
    """Run one training epoch; return the mean per-graph loss.

    ``max_grad_norm > 0`` clips the global gradient norm before each optimizer
    step (``0.0`` disables clipping, reproducing the historical behavior).
    """
    model.train()
    total_loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = _model_forward(model, batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        if max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
        logits = _model_forward(model, batch)
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
        logits = _model_forward(model, batch)
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


def _stratified_inner_split(
    graphs: list[Data], *, random_state: int, val_fraction: float
) -> tuple[list[Data], list[Data]]:
    """Split a fold's training graphs into stratified inner-train / inner-val.

    Stratified on the binary pCR label and used to drive early stopping, the LR
    scheduler, and best-epoch selection off data that is never the outer
    held-out fold, so the reported OOF AUC is not biased by the held-out labels
    influencing which checkpoint is kept. Fails loudly on a fold too small or
    too class-skewed to hold out a stratified split -- that is a broken fold,
    not something to paper over with a silent fallback.
    """
    labels = np.array([int(graph.y.item()) for graph in graphs])
    positives = int(labels.sum())
    negatives = len(labels) - positives
    n_splits = min(round(1.0 / val_fraction), positives, negatives)
    if n_splits < _MIN_INNER_SPLIT_FOLDS:
        raise ValueError(
            "Cannot form a stratified inner early-stopping split: this fold has "
            f"{positives} positive / {negatives} negative training cases, too few "
            "to hold out a stratified inner-validation split (need >=2 of each). "
            "Disable early stopping / the LR scheduler for this run or use a "
            "larger training split."
        )
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    inner_train_idx, inner_val_idx = next(splitter.split(labels, labels))
    inner_train = [graphs[i] for i in inner_train_idx]
    inner_val = [graphs[i] for i in inner_val_idx]
    return inner_train, inner_val


def fit_predict_one_fold(
    *,
    dataset: VanguardCenterlineDataset,
    cohort_df: pd.DataFrame,
    split: FoldSplit,
    config: Any,
    graph_feature_inputs: pd.DataFrame | None = None,
    graph_feature_groups: tuple[list[str], list[str], list[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Train a fresh GCNClassifier for one fold and return validation predictions.

    When any model-selection control is active (LR scheduler, early stopping, or
    best-epoch restore), a stratified inner-validation split is carved out of
    this fold's OWN training cases and drives every selection signal, so the
    outer held-out fold (``split.val_indices``) is untouched until the final
    OOF prediction. Without that split the held-out labels would choose the
    reported checkpoint and bias the CV AUC upward. When no control is active
    (the frozen baseline) there is nothing to select on, so the model trains on
    the full fold and reports its final epoch, reproducing the historical
    behavior exactly.

    ``graph_feature_inputs`` (the raw graph-feature-input frame from
    ``dataset.load_graph_feature_inputs``) and ``graph_feature_groups`` (from
    ``dataset.graph_feature_groups``) are passed through when
    ``gnn_graph_features`` was set at build time. The clinical/morphometry
    preprocessing is fit on this fold's training cases only and applied to its
    validation cases (see :func:`_fit_transform_graph_features`), so the
    imputer means/modes and one-hot vocabulary never see validation data.
    """
    params = config.model_params
    device = _resolve_device(str(params.device))
    max_grad_norm = float(params.get("max_grad_norm", 0.0))
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

    # Junction mode (segment-as-edge) carries the segment summary on edge_attr,
    # which needs the same per-fold standardization -- fit on the train split
    # only, then apply to both splits. Detected off the graphs themselves so
    # this branch is inert for voxel/segment mode (no edge_attr).
    is_edge_mode = getattr(train_graphs[0], "edge_attr", None) is not None
    if is_edge_mode:
        edge_mean, edge_std = fit_edge_standardizer(train_graphs)
        for graph in train_graphs + val_graphs:
            graph.edge_attr = (graph.edge_attr - edge_mean) / edge_std

    # graph_features (clinical covariates, gnn.clinical) are opt-in and, unlike
    # x/edge_attr, don't depend on node_mode. The clinical imputer/one-hot
    # encoder and morphometry imputer are fit on THIS fold's training cases only
    # (never the whole cohort at build time), then applied to both splits --
    # this is the cross-validation-leakage fix. The per-fold standardizer runs
    # afterwards on the resulting vectors, exactly as it does for x/edge_attr.
    if graph_feature_inputs is not None:
        if graph_feature_groups is None:
            raise ValueError(
                "graph_feature_inputs was provided without graph_feature_groups; "
                "pass dataset.graph_feature_groups() alongside the inputs."
            )
        train_case_ids = [str(graph.case_id) for graph in train_graphs]
        val_case_ids = [str(graph.case_id) for graph in val_graphs]
        features_by_case = _fit_transform_graph_features(
            graph_feature_inputs, train_case_ids, val_case_ids, graph_feature_groups
        )
        for graph in train_graphs + val_graphs:
            graph.graph_features = torch.tensor(
                features_by_case[str(graph.case_id)], dtype=torch.float
            ).unsqueeze(0)
        graph_mean, graph_std = fit_graph_standardizer(train_graphs)
        for graph in train_graphs + val_graphs:
            graph.graph_features = (graph.graph_features - graph_mean) / graph_std
        graph_dim = train_graphs[0].graph_features.shape[1]
    else:
        graph_dim = 0

    batch_size = int(params.batch_size)

    # Model-selection controls, resolved up front because they decide whether we
    # train on the full fold or on an inner-train split (see below).
    scheduler_name = str(params.get("lr_scheduler", "none")).lower()
    epochs = int(params.epochs)
    patience = int(params.get("early_stopping_patience", 0))
    restore_best_epoch = bool(params.get("restore_best_epoch", False))
    selection_active = scheduler_name != "none" or patience > 0 or restore_best_epoch

    # When any selection control is active, hold out a stratified inner-validation
    # split from this fold's OWN training cases and drive all selection signals
    # (scheduler / early stopping / best epoch) off it; the outer fold stays
    # untouched until the final prediction. When nothing selects, train on the
    # full fold and report the final epoch (historical frozen-baseline behavior).
    if selection_active:
        fit_graphs, selection_graphs = _stratified_inner_split(
            train_graphs,
            random_state=int(params.random_state) + int(split.fold_idx),
            val_fraction=float(params.get("inner_val_fraction", 0.2)),
        )
    else:
        fit_graphs, selection_graphs = train_graphs, val_graphs

    train_loader = DataLoader(fit_graphs, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(fit_graphs, batch_size=batch_size, shuffle=False)
    # Selection signal (val_loss for scheduler / early stopping / best epoch):
    # the inner split when active, else the outer fold (logging-only, never
    # selected on in the frozen baseline).
    selection_loader = DataLoader(
        selection_graphs, batch_size=batch_size, shuffle=False
    )
    # Outer held-out fold -- touched only for the final OOF prediction below.
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    if is_edge_mode:
        model: nn.Module = EdgeGNNClassifier(
            input_dim=train_graphs[0].x.shape[1],
            edge_dim=train_graphs[0].edge_attr.shape[1],
            hidden_dim=int(params.hidden_dim),
            num_layers=int(params.num_layers),
            dropout=float(params.dropout),
            graph_dim=graph_dim,
        ).to(device)
    else:
        model = GCNClassifier(
            input_dim=train_graphs[0].x.shape[1],
            hidden_dim=int(params.hidden_dim),
            num_layers=int(params.num_layers),
            dropout=float(params.dropout),
            graph_dim=graph_dim,
            pooling=str(params.get("gnn_pooling", "mean")),
        ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.learning_rate),
        weight_decay=float(params.weight_decay),
    )

    # Loss: weighted_bce (default) up-weights the minority pCR class using the
    # training split's own class balance; unweighted_bce reproduces the historical
    # plain BCEWithLogitsLoss; focal uses focal_alpha/gamma. See build_loss_fn.
    # Uses fit_graphs (the inner-train split when selection is active) so the
    # class weighting matches the data the model actually optimizes over.
    positive_count = sum(int(graph.y.item()) for graph in fit_graphs)
    negative_count = len(fit_graphs) - positive_count
    logging.info(
        "fold %d class balance: positives=%d negatives=%d loss=%s",
        split.fold_idx,
        positive_count,
        negative_count,
        str(params.get("loss", "weighted_bce")),
    )
    criterion = build_loss_fn(params, positive_count, negative_count, device)

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(params.get("lr_scheduler_factor", 0.5)),
            patience=int(params.get("lr_scheduler_patience", 5)),
        )
    else:
        raise ValueError(
            f"Unrecognized lr_scheduler {scheduler_name!r}; expected one of "
            "'none', 'cosine', 'plateau'. A typo must fail loudly, not silently "
            "run with no scheduler."
        )

    # Best-epoch selection + early stopping. The signal is validation *loss*
    # (not val AUC), matching deepsets.train.fit_predict_one_fold -- loss is
    # smoother than AUC on these small, imbalanced folds, and keeping the two
    # tracks consistent matters more than the plan's original "best val AUC"
    # wording. restore_best_epoch=False + patience=0 reproduces the historical
    # behavior of reporting the final-epoch model. patience / restore_best_epoch
    # / scheduler_name were resolved above (they gate the inner-split decision).
    track_best_state = patience > 0 or restore_best_epoch
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm
        )
        train_metrics = evaluate(model, train_eval_loader, criterion, device)
        # val_* columns track the SELECTION set (inner split when active), never
        # the outer held-out fold -- that is what keeps the outer fold clean.
        val_metrics = evaluate(model, selection_loader, criterion, device)
        val_loss = val_metrics["loss"]
        history.append(
            {
                "fold": float(split.fold_idx),
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_error_rate": train_metrics["error_rate"],
                "train_auc": train_metrics.get("auc", float("nan")),
                "val_loss": val_loss,
                "val_error_rate": val_metrics["error_rate"],
                "val_auc": val_metrics.get("auc", float("nan")),
                "lr": float(optimizer.param_groups[0]["lr"]),
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
            val_loss,
            val_metrics.get("auc", float("nan")),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if track_best_state:
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if patience > 0 and epochs_without_improvement >= patience:
            logging.info(
                "fold %d early stopping at epoch %d (best epoch %d, val_loss %.4f)",
                split.fold_idx,
                epoch,
                best_epoch,
                best_val_loss,
            )
            break

    if restore_best_epoch and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logging.info(
            "fold %d restored best model from epoch %d", split.fold_idx, best_epoch
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


def run_gnn_pipeline(config: Any, outdir: Path) -> KFoldResults:
    """Build the dataset cohort table, run k-fold CV, and save evaluator outputs."""
    params = config.model_params
    data_paths = config.data_paths

    dataset = VanguardCenterlineDataset(
        root=data_paths.gnn_centerline_root,
        labels_path=data_paths.gnn_labels_path,
        dce_root=data_paths.gnn_dce_root,
        cache_dir=data_paths.gnn_cache_dir,
        cases=list(data_paths.gnn_cases) if data_paths.gnn_cases else None,
        node_mode=str(params.gnn_node_mode),
        node_features=tuple(params.gnn_node_features),
        edge_features=(
            tuple(params.gnn_edge_features) if params.gnn_edge_features else None
        ),
        graph_features=(
            tuple(params.gnn_graph_features) if params.gnn_graph_features else None
        ),
        patient_info_dir=data_paths.gnn_patient_info_dir or None,
        clinical_excel=data_paths.gnn_clinical_excel or None,
        max_missing_clinical_frac=float(params.gnn_max_missing_clinical_frac),
        id_column=data_paths.gnn_id_column,
        label_column=data_paths.gnn_label_column,
        kinetic_baseline_floor_frac=float(params.gnn_kinetic_baseline_floor_frac),
        kinetic_denominator=str(params.gnn_kinetic_denominator),
        allow_manifest_mismatch=bool(data_paths.gnn_allow_manifest_mismatch),
        breast_split_mode=params.gnn_breast_split_mode or None,
        breast_split_skeleton_root=data_paths.gnn_breast_split_skeleton_root or None,
        max_missing_breast_split_frac=float(params.gnn_max_missing_breast_split_frac),
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

    # Raw graph-feature inputs are loaded once; the per-fold clinical/morphometry
    # preprocessing is fit on each fold's training split inside fit_predict_one_fold.
    graph_feature_inputs = dataset.load_graph_feature_inputs()
    graph_feature_groups = (
        dataset.graph_feature_groups() if graph_feature_inputs is not None else None
    )

    fold_results: list[FoldResults] = []
    all_history_rows: list[dict[str, float]] = []
    for split in splits:
        val_case_ids, y_true, y_pred, y_prob, history = fit_predict_one_fold(
            dataset=dataset,
            cohort_df=cohort_df,
            split=split,
            config=config,
            graph_feature_inputs=graph_feature_inputs,
            graph_feature_groups=graph_feature_groups,
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
    return kfold_results


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
    if args.random_state is not None:
        config.model_params.random_state = args.random_state
    outdir = resolve_run_output_dir(config=config, outdir_override=args.outdir)
    write_config_snapshot(config=config, outdir=outdir, config_source=args.config)
    run_gnn_pipeline(config, outdir)


if __name__ == "__main__":
    main()
