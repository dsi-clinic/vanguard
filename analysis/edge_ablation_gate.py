"""Does the voxel GNN use the vessel topology at all? Paired read of the ablation.

Scores the three arms of ``gnn/slurm/submit_gnn_uchicago_edge_ablation.slurm`` on
pooled OOF AUC with bootstrap CIs, and runs the paired comparisons that matter:

- ``rewire - none`` -- randomizing *which* nodes are adjacent while preserving every
  degree. A model using the specific vessel topology should lose AUC here.
- ``drop - none`` -- removing message passing entirely, leaving a per-node map plus
  a permutation-invariant readout (a Deep Sets model). A model using edges at all
  should lose AUC here.

Both deltas being indistinguishable from zero means the graph contributes nothing,
which reframes every prior graph-mode null: the bottleneck is the operator (a
2-layer GCN plus mean-pool on a graph of diameter 290-618 hops), not the
vasculature. See ``docs/design/gnn_graph_representation_plan.md`` Stage 0.1.

Per-arm OOF predictions are averaged across the 5 seeds before scoring, matching
``analysis/gnn_tabular_gate.py``, so seed noise does not drive the comparison.

Reads only each run's ``predictions.csv`` -- head-node safe.

Usage:
    python -m analysis.edge_ablation_gate \
        --sweep-root /ess/.../experiments/gnn_edge_ablation \
        --out-dir experiments/uchicago_edge_ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.protocol_confound_check import (
    bootstrap_auc_ci,
    bootstrap_paired_delta,
)

_ARMS: tuple[str, ...] = ("none", "rewire", "drop")
_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def load_arm_predictions(sweep_root: Path, arm: str) -> pd.DataFrame:
    """Seed-averaged OOF probability per case for one ablation arm.

    Every seed scores all 179 cases out-of-fold, so the seeds are aligned on
    ``case_id`` and averaged. A missing or short seed run raises rather than being
    quietly averaged over fewer seeds.
    """
    frames = []
    for seed in _SEEDS:
        matches = sorted((sweep_root / f"{arm}_seed{seed}").glob("*/*/predictions.csv"))
        if not matches:
            raise FileNotFoundError(f"No predictions.csv for arm={arm} seed={seed}")
        frame = pd.read_csv(matches[0])[["case_id", "y_true", "y_prob"]]
        frames.append(frame.set_index("case_id"))
    sizes = {len(f) for f in frames}
    if len(sizes) != 1:
        raise ValueError(f"arm={arm}: seeds disagree on case count: {sizes}")
    y_true = frames[0]["y_true"]
    probs = pd.concat([f["y_prob"] for f in frames], axis=1)
    if probs.isna().any().any():
        raise ValueError(f"arm={arm}: seeds disagree on which cases were scored")
    return pd.DataFrame({"y_true": y_true, "y_prob": probs.mean(axis=1)})


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the edge-ablation gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Score each arm, run the paired deltas, and write the gate outputs."""
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arms = {arm: load_arm_predictions(args.sweep_root, arm) for arm in _ARMS}
    reference = arms["none"]
    for arm, frame in arms.items():
        if not frame.index.equals(reference.index):
            raise ValueError(f"arm={arm} scored a different case set than 'none'")
        if not np.array_equal(frame["y_true"], reference["y_true"]):
            raise ValueError(f"arm={arm} disagrees with 'none' on labels")

    y = reference["y_true"].to_numpy()
    auc_rows = []
    for arm, frame in arms.items():
        probs = frame["y_prob"].to_numpy()
        low, high = bootstrap_auc_ci(y, probs, rng)
        auc_rows.append(
            {
                "arm": arm,
                "n": len(y),
                "auc": roc_auc_score(y, probs),
                "ci_low": low,
                "ci_high": high,
            }
        )

    delta_rows = []
    for arm in ("rewire", "drop"):
        delta, low, high = bootstrap_paired_delta(
            y, arms[arm]["y_prob"].to_numpy(), reference["y_prob"].to_numpy(), rng
        )
        delta_rows.append(
            {
                "comparison": f"{arm} - none",
                "mean_delta": delta,
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
            }
        )

    pd.DataFrame(auc_rows).to_csv(args.out_dir / "auc_ci.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(args.out_dir / "paired_delta.csv", index=False)
    print(pd.DataFrame(auc_rows).round(3).to_string(index=False))
    print()
    print(pd.DataFrame(delta_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
