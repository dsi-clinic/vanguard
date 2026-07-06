"""MVP training script for the GNN centerline classifier.

Deliberately minimal: a single stratified train/val split (no k-fold CV) and
plain CLI args (no YAML config integration) so the loader -> model -> training
loop can be validated end-to-end before this grows into a full pipeline that
mirrors ``deepsets/train.py`` (k-fold CV via ``evaluation/build_splits.py``,
a YAML config, results-dir conventions). It does reuse the shared
``evaluation.metrics`` module so those numbers stay comparable once the two
tracks are wired together.

Usage::

    python -m gnn.train --cache-dir /path/to/gnn_cache_smoke --cases NACT_01,NACT_02

Every run writes a ``README.md`` + ``metrics.json`` + ``config_used.json`` to
its output directory per the project's auditing convention -- see the
generated README for the exact command and inputs used.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.loader import DataLoader

from evaluation.metrics import compute_binary_metrics
from gnn.data_loader import VanguardCenterlineDataset
from gnn.model import GCNClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEFAULT_ROOT = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies"
)
_DEFAULT_LABELS_PATH = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/pcr_labels.csv"
)
_DEFAULT_CACHE_DIR = Path(
    "/gpfs/data/karczmar-lab/workspaces/spencervenancio/gnn_cache_smoke"
)
_DEFAULT_OUTDIR_BASE = Path(__file__).resolve().parent.parent / "experiments"
_MIN_STD = 1e-6
_DECISION_THRESHOLD = 0.5


@dataclass
class RunConfig:
    """Every knob that affects the run, captured verbatim for the README."""

    root: str
    labels_path: str
    cache_dir: str
    id_column: str
    label_column: str
    node_features: str
    cases: str | None
    dataset_filter: str | None
    val_frac: float
    seed: int
    hidden_dim: int
    num_layers: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    device: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--labels-path", type=Path, default=_DEFAULT_LABELS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--id-column", type=str, default="case_id")
    parser.add_argument("--label-column", type=str, default="pcr")
    parser.add_argument("--node-features", type=str, default="peak_time,radius")
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Optional comma-separated case-ID whitelist.",
    )
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help="Optional comma-separated dataset-name whitelist (matches "
        "data.dataset, e.g. DUKE) applied in-memory after loading the cache "
        "-- restricts the train/val split without rebuilding.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.3,
        help="Fraction of cases held out for validation (single split, not k-fold).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Override the output directory (default: experiments/gnn_mvp_<timestamp>).",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    """Choose compute device, failing loudly if CUDA was requested but absent."""
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but no GPU is available.")
    return torch.device("cuda")


def split_dataset(
    dataset: VanguardCenterlineDataset,
    *,
    val_frac: float,
    seed: int,
    candidate_indices: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Single stratified train/val split, optionally restricted to a candidate pool."""
    indices = (
        list(range(len(dataset))) if candidate_indices is None else candidate_indices
    )
    labels = [int(dataset[i].y.item()) for i in indices]
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_frac,
        random_state=seed,
        stratify=labels,
    )
    return train_idx, val_idx


def fit_node_standardizer(
    dataset: VanguardCenterlineDataset, indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std over training-split node features."""
    feature_matrix = torch.cat([dataset[i].x for i in indices], dim=0)
    mean = feature_matrix.mean(dim=0)
    std = feature_matrix.std(dim=0, unbiased=False)
    std = torch.where(std < _MIN_STD, torch.ones_like(std), std)
    return mean, std


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
    return metrics


def _plot_loss_history(loss_history_df: pd.DataFrame, output_path: Path) -> None:
    """Plot train and validation loss by epoch."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(loss_history_df["epoch"], loss_history_df["train_loss"], label="train")
    ax.plot(loss_history_df["epoch"], loss_history_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if git is unavailable."""
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_readme(
    *,
    outdir: Path,
    run_config: RunConfig,
    command: str,
    num_train: int,
    num_val: int,
    final_val_metrics: dict[str, float],
) -> None:
    """Write the auditing README required for every new results directory."""
    lines = [
        "# GNN MVP training run",
        "",
        "Single stratified train/val split of `VanguardCenterlineDataset` graphs "
        "through a minimal `GCNClassifier` (GCNConv stack + global mean pool + "
        "linear head). This is a wiring-correctness smoke test, not a tuned "
        "model -- no k-fold CV, no YAML config integration yet (see `gnn/README.md`).",
        "",
        "## Command",
        "",
        "```",
        command,
        "```",
        "",
        f"## Git commit\n\n{_git_commit()}",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(asdict(run_config), indent=2),
        "```",
        "",
        "## Input data",
        "",
        f"- Centerline root: `{run_config.root}`",
        f"- Labels: `{run_config.labels_path}`",
        f"- Dataset cache: `{run_config.cache_dir}`",
        f"- Cases: `{run_config.cases or 'all cases in cache'}`",
        f"- Dataset filter: `{run_config.dataset_filter or 'none (all datasets in cache)'}`",
        f"- Train / val split: {num_train} / {num_val} graphs "
        f"(val_frac={run_config.val_frac}, seed={run_config.seed})",
        "",
        "## Final validation metrics",
        "",
        "```json",
        json.dumps(final_val_metrics, indent=2),
        "```",
        "",
        "## Train / val loss curve",
        "",
        "![loss curve](loss_by_epoch.png)",
        "",
        "## Regenerate",
        "",
        "Rerun the exact command above from the repo root with the "
        "`vanguard` conda env active (`micromamba activate vanguard`). "
        "Full metric history per epoch is in `metrics.json` in this directory; "
        "the loss curve is `loss_by_epoch.png`.",
        "",
    ]
    outdir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Build the dataset, train, evaluate, and write auditing outputs."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    node_features = tuple(args.node_features.split(","))
    cases = args.cases.split(",") if args.cases else None
    device = _resolve_device(args.device)

    run_config = RunConfig(
        root=str(args.root),
        labels_path=str(args.labels_path),
        cache_dir=str(args.cache_dir),
        id_column=args.id_column,
        label_column=args.label_column,
        node_features=args.node_features,
        cases=args.cases,
        dataset_filter=args.dataset_filter,
        val_frac=args.val_frac,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=str(device),
    )

    torch.manual_seed(args.seed)

    dataset = VanguardCenterlineDataset(
        root=args.root,
        labels_path=args.labels_path,
        cache_dir=args.cache_dir,
        cases=cases,
        node_features=node_features,
        id_column=args.id_column,
        label_column=args.label_column,
    )
    logging.info("Loaded %d graph(s) from cache %s", len(dataset), args.cache_dir)

    candidate_indices: list[int] | None = None
    if args.dataset_filter:
        wanted = set(args.dataset_filter.split(","))
        candidate_indices = [
            i for i in range(len(dataset)) if dataset[i].dataset in wanted
        ]
        if not candidate_indices:
            raise ValueError(
                f"--dataset-filter={args.dataset_filter!r} matched no graphs "
                f"in {args.cache_dir}"
            )
        logging.info(
            "Restricted to %d/%d graph(s) matching dataset_filter=%s",
            len(candidate_indices),
            len(dataset),
            args.dataset_filter,
        )

    train_idx, val_idx = split_dataset(
        dataset,
        val_frac=args.val_frac,
        seed=args.seed,
        candidate_indices=candidate_indices,
    )
    mean, std = fit_node_standardizer(dataset, train_idx)

    train_graphs = [dataset[i].clone() for i in train_idx]
    val_graphs = [dataset[i].clone() for i in val_idx]
    for graph in train_graphs + val_graphs:
        graph.x = (graph.x - mean) / std

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    input_dim = dataset[0].x.shape[1]
    model = GCNClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                **{k: v for k, v in val_metrics.items() if k != "loss"},
            }
        )
        logging.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_auc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics.get("auc", float("nan")),
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or (_DEFAULT_OUTDIR_BASE / f"gnn_mvp_{timestamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    outdir.joinpath("metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    loss_history_df = pd.DataFrame(history)[["epoch", "train_loss", "val_loss"]]
    _plot_loss_history(loss_history_df, outdir / "loss_by_epoch.png")
    outdir.joinpath("config_used.json").write_text(
        json.dumps(asdict(run_config), indent=2), encoding="utf-8"
    )
    command = "python -m gnn.train " + " ".join(
        f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items() if v is not None
    )
    write_readme(
        outdir=outdir,
        run_config=run_config,
        command=command,
        num_train=len(train_idx),
        num_val=len(val_idx),
        final_val_metrics=history[-1] if history else {},
    )
    logging.info("Wrote run outputs to %s", outdir)


if __name__ == "__main__":
    main()
