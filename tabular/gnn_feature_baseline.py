"""Tabular baseline: logistic regression / XGBoost on hand-summarized GNN inputs.

Per the code review's outcome-signature framework: if this tabular baseline
also stays near chance, the six inputs (peak_time, peak_enhancement,
time_to_enhancement, washin_slope, auc_positive, radius) likely lack
generalizable pCR signal, independent of architecture; if it clearly beats
the GNN, pooling/message-passing is the more likely bottleneck.

Reuses arm 1 (mixed baseline)'s existing cache
(gnn_cache_voxel_mixed_baseline) and frozen folds -- same 1332-case cohort,
same 6 features, same 5 predefined folds -- so results are directly
comparable to that arm's GNN AUC (0.509 mean, 3 seeds x 5 folds, see
experiments/harmonized_single_breast_v1/README.md). Per case, summarizes
each of the 6 node-level features to (mean, std, q10, q50, q90), adds the
TTE no-arrival fraction (fraction of nodes at TTE_NO_ARRIVAL_SENTINEL) and
the already-recorded node/edge counts -- 33 tabular columns total. No new
graph build: loads the existing cache read-only.

Writes the feature table and per-fold results to --out-dir (default
experiments/tabular_baseline_v1/, see that directory's README.md for the
full writeup), matching the results-directory convention used elsewhere in
this repo. Lives in tabular/, not analysis/: this fits real predictive
models (logistic regression, XGBoost), which is what tabular/models.py and
tabular/train.py are for -- analysis/ is for scripts that summarize or
diagnose already-computed results, not ones that fit their own models.

Usage::

    python -m tabular.gnn_feature_baseline --config configs/gnn_voxel_mixed_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from xgboost import XGBClassifier

from gnn.data_loader import TTE_NO_ARRIVAL_SENTINEL, VanguardCenterlineDataset
from load_cohort import load_config

_SUMMARY_QUANTILES = (0.10, 0.50, 0.90)
_SEEDS = (42, 142, 242)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gnn_voxel_mixed_baseline.yaml")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("experiments/tabular_baseline_v1")
    )
    return parser.parse_args()


def load_dataset_from_config(
    config_path: Path,
) -> tuple[VanguardCenterlineDataset, tuple[str, ...]]:
    """Load an existing cache read-only, exactly as gnn/train.py does for training."""
    config = load_config(config_path)
    mp, dp = config.model_params, config.data_paths
    node_features = tuple(mp.gnn_node_features)
    dataset = VanguardCenterlineDataset(
        root=dp.gnn_centerline_root,
        labels_path=dp.gnn_labels_path,
        dce_root=dp.gnn_dce_root,
        cache_dir=dp.gnn_cache_dir,
        node_mode=str(mp.gnn_node_mode),
        node_features=node_features,
    )
    return dataset, node_features


def summarize_graph(data: Data, node_features: tuple[str, ...]) -> dict[str, float]:
    """Collapse one graph's per-node feature matrix into tabular summary stats."""
    x = data.x.numpy()
    row: dict[str, float] = {}
    for i, name in enumerate(node_features):
        column = x[:, i]
        row[f"{name}_mean"] = float(np.mean(column))
        row[f"{name}_std"] = float(np.std(column))
        for q in _SUMMARY_QUANTILES:
            row[f"{name}_q{int(q * 100)}"] = float(np.quantile(column, q))
    tte_idx = node_features.index("time_to_enhancement")
    tte_column = x[:, tte_idx]
    row["tte_no_arrival_fraction"] = float(
        np.mean(tte_column == TTE_NO_ARRIVAL_SENTINEL)
    )
    row["num_nodes"] = float(data.num_nodes)
    row["num_edges"] = int(data.edge_index.shape[1])
    row["case_id"] = str(data.case_id)
    row["pcr"] = int(data.y.item())
    row["dataset"] = data.dataset
    return row


def build_feature_table(
    dataset: VanguardCenterlineDataset, node_features: tuple[str, ...]
) -> pd.DataFrame:
    """Summarize every graph in the dataset into one tabular row each."""
    rows = [summarize_graph(dataset[i], node_features) for i in range(len(dataset))]
    return pd.DataFrame(rows)


def add_folds(feature_table: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    """Join in each case's frozen fold assignment."""
    folds = pd.read_csv(labels_path)[["case_id", "fold"]]
    merged = feature_table.merge(folds, on="case_id", how="left")
    if merged["fold"].isna().any():
        missing = merged.loc[merged["fold"].isna(), "case_id"].tolist()
        raise ValueError(
            f"{len(missing)} case(s) have no fold assignment: {missing[:10]}..."
        )
    return merged


def run_cv(
    feature_table: pd.DataFrame, feature_cols: list[str], *, model_name: str, seed: int
) -> list[dict[str, object]]:
    """5-fold CV using the frozen fold column, one held-out AUC per fold."""
    rows = []
    for fold in sorted(feature_table["fold"].unique()):
        train = feature_table[feature_table["fold"] != fold]
        test = feature_table[feature_table["fold"] == fold]
        x_train, y_train = train[feature_cols].to_numpy(), train["pcr"].to_numpy()
        x_test, y_test = test[feature_cols].to_numpy(), test["pcr"].to_numpy()

        if model_name == "logistic_regression":
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
            )
        elif model_name == "xgboost":
            model = XGBClassifier(
                n_estimators=100, max_depth=3, random_state=seed, eval_metric="logloss"
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name!r}")

        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "fold": int(fold),
                "auc": auc,
                "n_test": len(test),
            }
        )
    return rows


def main() -> None:
    """Build the tabular feature table once, then run both models across seeds."""
    args = _parse_args()
    config = load_config(args.config)
    dataset, node_features = load_dataset_from_config(args.config)
    print(f"Loaded {len(dataset)} graphs from {config.data_paths.gnn_cache_dir}")

    feature_table = build_feature_table(dataset, node_features)
    feature_table = add_folds(feature_table, Path(config.data_paths.gnn_labels_path))
    feature_cols = [
        c
        for c in feature_table.columns
        if c not in ("case_id", "pcr", "dataset", "fold")
    ]
    print(f"{len(feature_table)} cases, {len(feature_cols)} tabular feature columns")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(args.out_dir / "tabular_baseline_features.csv", index=False)

    all_rows: list[dict[str, object]] = []
    for model_name in ("logistic_regression", "xgboost"):
        for seed in _SEEDS:
            all_rows.extend(
                run_cv(feature_table, feature_cols, model_name=model_name, seed=seed)
            )

    results = pd.DataFrame(all_rows)
    print()
    print("=== Per-model aggregated AUC (mean +/- std across seed x fold) ===")
    print(results.groupby("model")["auc"].agg(["mean", "std", "count"]))
    print()
    print(f"=== Per-fold AUC, one model at a time (seed={_SEEDS[0]}) ===")
    print(
        results[results["seed"] == _SEEDS[0]].pivot_table(
            index="fold", columns="model", values="auc"
        )
    )

    out_path = args.out_dir / "tabular_baseline_results.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
