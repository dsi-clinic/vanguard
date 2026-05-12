"""Append Phase 2a attention-sweep results to results/her2_deepsets_tracker.csv.

For each of the 6 attention variants:
- Reads the variant's predictions.csv from
  experiments/deepsets_sweep_her2_attention_full/<vid>/train/*/*/predictions.csv
- Restricts to the n=68 HER2 intersection cohort
  (data/her2_intersection_case_ids.csv x tumor_subtype=her2_enriched).
- Computes per-fold AUC/AP (using the Deep Sets native 5-fold assignment,
  seed=42) and the pooled AUC/AP across those 5 folds.
- Appends one summary row per variant, mirroring the schema of the
  existing phase3_prior summary rows.

The fold_map_path column is left empty because attention sweep uses the
DS-native 5-fold split (seed=42), not the canonical 3-fold
data/fold_map_her2.csv (which was built for the XGB-on-n=68 comparison).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
TRACKER = REPO / "results" / "her2_deepsets_tracker.csv"
MANIFEST = REPO / "experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/deepsets_manifest.csv"
INTERSECTION = REPO / "data/her2_intersection_case_ids.csv"
SWEEP_ROOT = REPO / "experiments/deepsets_sweep_her2_attention_full"

VARIANTS = [
    "attn_h16",
    "attn_h32",
    "attn_h64",
    "attn_logn_h16",
    "attn_logn_h32",
    "attn_logn_h64",
]


def _her2_intersection_ids() -> set[str]:
    inter = set(pd.read_csv(INTERSECTION)["case_id"].astype(str))
    manifest = pd.read_csv(MANIFEST)
    manifest["case_id"] = manifest["case_id"].astype(str)
    her2 = set(manifest.loc[manifest["tumor_subtype"] == "her2_enriched", "case_id"].astype(str))
    return inter & her2


def _row_for(variant: str, her2_ids: set[str]) -> dict:
    matches = sorted(glob.glob(str(SWEEP_ROOT / variant / "train" / "*" / "*" / "predictions.csv")))
    if not matches:
        raise FileNotFoundError(f"No predictions.csv for variant {variant}")
    pred_path = matches[0]
    df = pd.read_csv(pred_path)
    df["case_id"] = df["case_id"].astype(str)
    sub = df[df["case_id"].isin(her2_ids)].copy()
    if sub.empty:
        raise RuntimeError(f"No HER2-intersection cases found for {variant}")

    fold_aucs = []
    for fold in sorted(sub["fold"].unique()):
        f_df = sub[sub["fold"] == fold]
        if f_df["y_true"].nunique() < 2:
            continue
        fold_aucs.append(roc_auc_score(f_df["y_true"], f_df["y_prob"]))

    mean_fold_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    std_fold_auc = float(np.std(fold_aucs, ddof=0)) if fold_aucs else float("nan")
    pooled_auc = float(roc_auc_score(sub["y_true"], sub["y_prob"]))
    pooled_ap = float(average_precision_score(sub["y_true"], sub["y_prob"]))

    config_path = SWEEP_ROOT / variant / "runtime_config.yaml"
    return {
        "phase": "phase2a",
        "run_id": f"{variant}__seed42",
        "model_family": "deepsets",
        "subgroup": "her2_intersection_n68",
        "n_cases": int(len(sub["case_id"].unique())),
        "n_folds": int(sub["fold"].nunique()),
        "seed": 42,
        "mean_fold_auc": mean_fold_auc,
        "std_fold_auc": std_fold_auc,
        "pooled_auc": pooled_auc,
        "pooled_ap": pooled_ap,
        "cross_seed_mean_auc": "",
        "cross_seed_std_auc": "",
        "fold_map_path": "",
        "config_path": str(config_path.relative_to(REPO)),
        "predictions_path": str(Path(pred_path).relative_to(REPO)),
        "notes": (
            "Phase 2a attention sweep on full n=980 cohort (5-fold seed=42); "
            "HER2 cases restricted to n=68 intersection. Single seed; repeated-CV "
            "across additional seeds is the next step before paired-fold testing."
        ),
    }


def main() -> None:
    her2_ids = _her2_intersection_ids()
    print(f"HER2 intersection cohort: {len(her2_ids)} case_ids")

    existing = pd.read_csv(TRACKER)
    new_rows = [_row_for(v, her2_ids) for v in VARIANTS]
    new_df = pd.DataFrame(new_rows)
    print(new_df[["run_id", "n_cases", "n_folds", "mean_fold_auc", "std_fold_auc", "pooled_auc"]].to_string(index=False))
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(TRACKER, index=False)
    print(f"\nWrote {TRACKER} ({len(combined)} rows, +{len(new_df)})")


if __name__ == "__main__":
    main()
