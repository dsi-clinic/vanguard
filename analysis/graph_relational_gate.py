"""Stage 2 gate: does the vessel GRAPH carry pCR signal the per-node means discard?

The cheapest decisive test of the graph question. Every gate so far compared a GNN
against a tabular model on per-case node-feature *means* and found them equal
(``experiments/uchicago_tabular_bar_floored``, ``..._graph_mode_floored``) -- but
both sides were built from node-intrinsic scalars, so neither could use topology.
Rather than building a better architecture first, this computes the missing
*relational* quantities explicitly (``gnn.relational_features``: transit slope,
arrival monotonicity, sibling asymmetry, radius tapering) and tests them with a
plain logistic regression.

Why that is the right first move at N=179: these scalars are joint properties of
(kinetics, tree position), so they are **not computable from per-case means**. If a
7-feature logreg on them clears the bar, the graph carries signal -- proven with a
model too simple to be winning on capacity, and no GNN required to find out. If it
does not, no architecture will save it, and that answer costs one CPU job instead
of another sweep.

Bars it is measured against, all pooled OOF AUC over the same 179 cases and the
same predefined folds:

- ``tabular_7means`` -- the published 0.613 bar.
- ``tabular_7means_protocol_residualized`` -- 0.539, the physiology-only bar after
  acquisition protocol is regressed out (``experiments/uchicago_protocol_confound``).
  **This is the gate's reference**, since the raw 0.613 is substantially protocol.
- ``protocol_only`` -- 0.593, reachable with no imaging at all. Any arm that fails
  to beat this is not doing imaging-based prediction.

Also checks the claim that motivates these features -- that being expressed in
differences of seconds over millimetres makes them protocol-invariant -- by
correlating each against the acquisition descriptors and by residualizing them the
same way. If they turn out protocol-coupled too, that is reported, not buried.

Reads the built voxel cache (no DCE reload, no rebuild) plus small CSVs.

Usage:
    python -m analysis.graph_relational_gate \
        --cache-dir /ess/.../gnn_cache_voxel_richfeat_floored \
        --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
        --dce-root /ess/.../preprocessing_out_v5/dce \
        --out-dir experiments/uchicago_graph_relational
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from analysis.protocol_confound_check import (
    _MEANS7,
    _PROTOCOL,
    bootstrap_auc_ci,
    bootstrap_paired_delta,
    load_protocol,
    oof_predictions,
    residualize_on_protocol,
)
from gnn.relational_features import (
    RELATIONAL_FEATURE_COLUMNS,
    case_relational_features,
)

_GRAPH_SUFFIX = "_graph.pt"


def extract_relational_features(cache_dir: Path) -> pd.DataFrame:
    """One row of relational features per cached graph, keyed by ``case_id``."""
    paths = sorted((cache_dir / "processed").glob(f"*{_GRAPH_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"No {_GRAPH_SUFFIX} files under {cache_dir}/processed")
    rows = []
    for path in paths:
        cached = torch.load(path)
        data = cached[0] if isinstance(cached, tuple | list) else cached
        rows.append({"case_id": str(data.case_id), **case_relational_features(data)})
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the Stage 2 relational-feature gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tabular-features", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Extract relational features, score every arm, and write the gate outputs."""
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    relational = extract_relational_features(args.cache_dir)
    relational.to_csv(args.out_dir / "relational_features.csv", index=False)
    print(f"extracted relational features for {len(relational)} cases")
    print(relational[list(RELATIONAL_FEATURE_COLUMNS)].describe().round(4).to_string())

    tabular = pd.read_csv(args.tabular_features)
    data = tabular.merge(relational, on="case_id", validate="one_to_one")
    data = data.merge(
        load_protocol(args.dce_root, data["case_id"].tolist()), on="case_id"
    )
    features = list(RELATIONAL_FEATURE_COLUMNS)
    # A relational feature is NaN when a case cannot support the measurement at all
    # (e.g. no bifurcation with two arrival-bearing downstream branches). Logistic
    # regression cannot consume NaN, and imputing would invent geometry, so a
    # feature missing for any case is dropped from the gate and reported.
    missing = {f: int(data[f].isna().sum()) for f in features}
    dropped = [f for f, n in missing.items() if n > 0]
    if dropped:
        print(f"\nDropped feature(s) with missing values: {missing}")
        features = [f for f in features if f not in dropped]
    if not features:
        raise ValueError(
            "Every relational feature had missing values; nothing to gate."
        )

    y = data["pcr"].to_numpy()
    folds = data["fold"].to_numpy()

    # Is the protocol-invariance claim true? Correlate each feature with acquisition.
    coupling = pd.DataFrame(
        [
            {
                "feature": name,
                **{
                    f"abs_corr_{p}": abs(np.corrcoef(data[name], data[p])[0, 1])
                    for p in _PROTOCOL
                },
                "univariate_auc": max(
                    roc_auc_score(y, data[name]), 1.0 - roc_auc_score(y, data[name])
                ),
            }
            for name in features
        ]
    )

    arms = {
        "relational_graph": oof_predictions(data[features].to_numpy(), y, folds),
        "relational_graph_protocol_residualized": oof_predictions(
            residualize_on_protocol(data, tuple(features), folds), y, folds
        ),
        "tabular_7means": oof_predictions(data[list(_MEANS7)].to_numpy(), y, folds),
        "tabular_7means_protocol_residualized": oof_predictions(
            residualize_on_protocol(data, _MEANS7, folds), y, folds
        ),
        "protocol_only": oof_predictions(data[list(_PROTOCOL)].to_numpy(), y, folds),
        "relational_plus_7means": oof_predictions(
            data[features + list(_MEANS7)].to_numpy(), y, folds
        ),
    }
    auc_rows = []
    for name, probs in arms.items():
        low, high = bootstrap_auc_ci(y, probs, rng)
        auc_rows.append(
            {
                "model": name,
                "n": len(y),
                "auc": roc_auc_score(y, probs),
                "ci_low": low,
                "ci_high": high,
            }
        )

    # The gate, plus the two comparisons needed to interpret it.
    gate_rows = []
    for label, arm_a, arm_b in (
        (
            "GATE: relational - tabular_7means_protocol_residualized",
            "relational_graph",
            "tabular_7means_protocol_residualized",
        ),
        ("relational - tabular_7means", "relational_graph", "tabular_7means"),
        ("relational - protocol_only", "relational_graph", "protocol_only"),
        (
            "relational_plus_7means - tabular_7means",
            "relational_plus_7means",
            "tabular_7means",
        ),
    ):
        delta, low, high = bootstrap_paired_delta(y, arms[arm_a], arms[arm_b], rng)
        gate_rows.append(
            {
                "comparison": label,
                "mean_delta": delta,
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
            }
        )

    coupling.to_csv(args.out_dir / "feature_protocol_coupling.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(args.out_dir / "auc_ci.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(args.out_dir / "gate_paired_delta.csv", index=False)

    print("\n=== relational feature vs acquisition protocol ===")
    print(coupling.round(3).to_string(index=False))
    print("\n=== pooled OOF AUC ===")
    print(pd.DataFrame(auc_rows).round(3).to_string(index=False))
    print("\n=== paired deltas ===")
    print(pd.DataFrame(gate_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
