"""Gate G1 analysis: Tier-1 mean_max readout vs Tier-0 mean pool.

Pairs the Tier-1 sweep (gnn_pooling="mean_max") seed-for-seed against the
already-computed Tier-0 (mean-pool) runs -- same cache, folds, and training
knobs, only the readout differs -- and reports OOF AUC per arm, paired ΔAUC by
seed×fold, and per-dataset OOF AUC. See gnn/PLAN_advanced_modeling.md Gate G1.

Usage:
    python -m analysis.gnn_tier1_readout_gate_g1 [--tier0-root D] [--tier1-root D]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_WS = "/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments"
DEFAULT_T0 = f"{_WS}/gnn_tier0_sweep"
DEFAULT_T1 = f"{_WS}/gnn_tier1_sweep"
SEEDS = range(5)
_BOTH_CLASSES = 2
XGBOOST_TABULAR_REF = 0.540  # Gate G0 tabular baseline to clear


def _auc(df: pd.DataFrame) -> float:
    """OOF AUC, NaN if only one class is present."""
    if df["y_true"].nunique() < _BOTH_CLASSES:
        return float("nan")
    return roc_auc_score(df["y_true"], df["y_prob"])


def _load(root: str, tag: str) -> pd.DataFrame:
    """The single predictions.csv for one run tag."""
    matches = sorted(Path(root).glob(f"{tag}/*/*/predictions.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"{tag}: expected 1 file, got {matches}")
    return pd.read_csv(matches[0])


def main() -> None:
    """Print the Gate G1 tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier0-root", default=DEFAULT_T0)
    parser.add_argument("--tier1-root", default=DEFAULT_T1)
    args = parser.parse_args()

    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for s in SEEDS:
        frames[("tier0", s)] = _load(args.tier0_root, f"tier0_seed{s}")
        frames[("tier1", s)] = _load(args.tier1_root, f"tier1_seed{s}")

    oof = pd.DataFrame(
        [{"arm": a, "seed": s, "oof": _auc(df)} for (a, s), df in frames.items()]
    )
    piv = oof.pivot_table(index="seed", columns="arm", values="oof")
    print("=== OOF AUC per run ===")
    print(piv.round(4))
    print("\n=== OOF AUC mean±std ===")
    for a in ("tier0", "tier1"):
        v = oof[oof.arm == a]["oof"]
        print(f"  {a}: {v.mean():.4f} ± {v.std(ddof=1):.4f}")
    print(f"  tier1 > tier0 on {(piv['tier1'] > piv['tier0']).sum()}/5 seeds")
    print(
        f"  tier1 vs XGBoost tabular ref {XGBOOST_TABULAR_REF}: "
        f"{'clears' if oof[oof.arm=='tier1']['oof'].mean() > XGBOOST_TABULAR_REF else 'below'}"
    )

    rows = [
        {"arm": a, "seed": s, "fold": int(fold), "auc": _auc(fdf)}
        for (a, s), df in frames.items()
        for fold, fdf in df.groupby("fold")
    ]
    pf = pd.DataFrame(rows)
    b = pf[pf.arm == "tier0"].set_index(["seed", "fold"])["auc"]
    t = pf[pf.arm == "tier1"].set_index(["seed", "fold"])["auc"]
    d = (t - b).dropna()
    tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"\n=== Paired ΔAUC (tier1 − tier0) by seed×fold, n={len(d)} ===")
    print(f"  mean {d.mean():+.4f}  median {d.median():+.4f}  std {d.std(ddof=1):.4f}")
    print(
        f"  frac>0 {(d > 0).mean():.2f} ({(d > 0).sum()}/{len(d)})  "
        f"paired t {tstat:+.3f}"
    )

    print("\n=== Per-dataset OOF AUC (pooled across seeds) ===")
    for a in ("tier0", "tier1"):
        pooled = pd.concat([frames[(a, s)] for s in SEEDS])
        cells = [f"{ds}={_auc(g):.3f}" for ds, g in pooled.groupby("dataset")]
        print(f"  {a}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
