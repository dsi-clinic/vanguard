"""Gate G0 analysis for the GNN Tier-0 sweep (gnn/PLAN_advanced_modeling.md).

Reads each run's predictions.csv from the baseline-vs-tier0 sweep and reports:
  - OOF AUC per run + mean/std per arm,
  - paired ΔAUC (tier0 − baseline) by seed×fold (shared folds per seed),
  - per-arm fold-only AUC spread (stability),
  - per-dataset OOF AUC.

Summarizes an already-computed sweep (no training) -- hence analysis/, not
tabular/. Point --out-root at the sweep's output tree; default is the gpfs
location submit_gnn_tier0_sweep.slurm writes to.

Usage:
    python -m analysis.gnn_tier0_gate_g0 [--out-root <dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_OUT_ROOT = (
    "/gpfs/data/karczmar-lab/workspaces/spencervenancio/" "experiments/gnn_tier0_sweep"
)
ARMS = ("baseline", "tier0")
SEEDS = range(5)
_BOTH_CLASSES = 2  # both labels present -> AUC defined


def _auc(df: pd.DataFrame) -> float:
    """OOF AUC, NaN if a (sub)set has only one class present."""
    if df["y_true"].nunique() < _BOTH_CLASSES:
        return float("nan")
    return roc_auc_score(df["y_true"], df["y_prob"])


def load_predictions(out_root: str) -> dict[tuple[str, int], pd.DataFrame]:
    """One predictions.csv per (arm, seed); exactly one match required."""
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for arm in ARMS:
        for seed in SEEDS:
            matches = sorted(
                Path(out_root).glob(f"{arm}_seed{seed}/*/*/predictions.csv")
            )
            if len(matches) != 1:
                raise RuntimeError(f"{arm}_seed{seed}: expected 1 file, got {matches}")
            frames[(arm, seed)] = pd.read_csv(matches[0])
    return frames


def main() -> None:
    """Print the Gate G0 tables for the sweep under --out-root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, type=Path)
    args = parser.parse_args()
    frames = load_predictions(str(args.out_root))

    per_fold = pd.DataFrame(
        [
            {"arm": arm, "seed": seed, "fold": int(fold), "auc": _auc(fdf)}
            for (arm, seed), df in frames.items()
            for fold, fdf in df.groupby("fold")
        ]
    )
    oof = pd.DataFrame(
        [
            {"arm": arm, "seed": seed, "oof_auc": _auc(df)}
            for (arm, seed), df in frames.items()
        ]
    )

    print("=== OOF AUC per run ===")
    print(oof.pivot_table(index="seed", columns="arm", values="oof_auc").round(4))
    print("\n=== OOF AUC mean±std across seeds ===")
    for arm in ARMS:
        v = oof[oof.arm == arm]["oof_auc"]
        print(f"  {arm:9s}: {v.mean():.4f} ± {v.std(ddof=1):.4f}")

    b = per_fold[per_fold.arm == "baseline"].set_index(["seed", "fold"])["auc"]
    t = per_fold[per_fold.arm == "tier0"].set_index(["seed", "fold"])["auc"]
    delta = (t - b).dropna()
    tstat = delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)))
    print(f"\n=== Paired ΔAUC (tier0 − baseline) by seed×fold, n={len(delta)} ===")
    print(
        f"  mean {delta.mean():+.4f}  median {delta.median():+.4f}  "
        f"std {delta.std(ddof=1):.4f}"
    )
    print(
        f"  frac>0 {(delta > 0).mean():.2f} ({(delta > 0).sum()}/{len(delta)})  "
        f"paired t {tstat:+.3f}"
    )

    print("\n=== per-arm fold-only AUC spread ===")
    for arm in ARMS:
        a = per_fold[per_fold.arm == arm]["auc"]
        print(f"  {arm:9s}: mean {a.mean():.4f}, std {a.std(ddof=1):.4f}")

    print("\n=== Per-dataset OOF AUC (pooled across seeds) ===")
    for arm in ARMS:
        pooled = pd.concat([frames[(arm, s)] for s in SEEDS])
        cells = [f"{ds}={_auc(g):.3f}" for ds, g in pooled.groupby("dataset")]
        print(f"  {arm:9s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
