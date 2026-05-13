"""Append Phase 2a repeated-CV rows (seeds 7 and 123) and cross-seed summaries
to results/her2_deepsets_tracker.csv.

For each of the top-2 attention variants (attn_logn_h16, attn_h64) this
script:

- Loads the predictions.csv produced by jobs 852330-852333 from
  experiments/deepsets_phase2a_repeated_cv/<vid>/seed{7,123}/train/.../
  and restricts to the n=68 HER2 intersection cohort.
- Computes per-fold and pooled AUC/AP on that subset, identical to the
  seed=42 rows already added by scripts/append_her2_attention_to_tracker.py.
- Appends two seed-specific rows per variant.
- Also computes a per-variant cross-seed summary (mean and std of
  per-seed pooled AUCs across all three seeds: 42, 7, 123) and appends a
  __cross_seed row mirroring the schema of the existing phase3_prior
  __cross_seed rows.

Output: 6 new rows total (2 seed × 2 variant + 2 cross_seed summary).

The seed=42 attention rows from the prior commit are left in place. The
cross-seed summary row reuses the seed=42 pooled-AUC numbers without
re-running anything.
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
REPCV_ROOT = REPO / "experiments/deepsets_phase2a_repeated_cv"

VARIANTS = ["attn_logn_h16", "attn_h64"]
NEW_SEEDS = ["7", "123"]


def _her2_intersection_ids() -> set[str]:
    inter = set(pd.read_csv(INTERSECTION)["case_id"].astype(str))
    manifest = pd.read_csv(MANIFEST)
    manifest["case_id"] = manifest["case_id"].astype(str)
    her2 = set(manifest.loc[manifest["tumor_subtype"] == "her2_enriched", "case_id"].astype(str))
    return inter & her2


def _pred_path(variant: str, seed: str) -> Path:
    if seed == "42":
        matches = sorted(glob.glob(str(SWEEP_ROOT / variant / "train" / "*" / "*" / "predictions.csv")))
    else:
        matches = sorted(glob.glob(str(REPCV_ROOT / variant / f"seed{seed}" / "train" / "*" / "*" / "predictions.csv")))
    if not matches:
        raise FileNotFoundError(f"No predictions.csv for {variant} seed={seed}")
    return Path(matches[0])


def _config_path(variant: str, seed: str) -> Path:
    if seed == "42":
        return SWEEP_ROOT / variant / "runtime_config.yaml"
    return REPCV_ROOT / variant / f"seed{seed}" / "runtime_config.yaml"


def _summary_row(variant: str, seed: str, her2_ids: set[str]) -> dict:
    pred_path = _pred_path(variant, seed)
    df = pd.read_csv(pred_path)
    df["case_id"] = df["case_id"].astype(str)
    sub = df[df["case_id"].isin(her2_ids)].copy()

    fold_aucs = []
    for fold in sorted(sub["fold"].unique()):
        f_df = sub[sub["fold"] == fold]
        if f_df["y_true"].nunique() < 2:
            continue
        fold_aucs.append(roc_auc_score(f_df["y_true"], f_df["y_prob"]))

    return {
        "phase": "phase2a_repcv",
        "run_id": f"{variant}__seed{seed}",
        "model_family": "deepsets",
        "subgroup": "her2_intersection_n68",
        "n_cases": int(len(sub["case_id"].unique())),
        "n_folds": int(sub["fold"].nunique()),
        "seed": int(seed),
        "mean_fold_auc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "std_fold_auc": float(np.std(fold_aucs, ddof=0)) if fold_aucs else float("nan"),
        "pooled_auc": float(roc_auc_score(sub["y_true"], sub["y_prob"])),
        "pooled_ap": float(average_precision_score(sub["y_true"], sub["y_prob"])),
        "cross_seed_mean_auc": "",
        "cross_seed_std_auc": "",
        "fold_map_path": "",
        "config_path": str(_config_path(variant, seed).relative_to(REPO)),
        "predictions_path": str(pred_path.relative_to(REPO)),
        "notes": (
            "Phase 2a repeated-CV (seed cloned from seed=42 sweep). "
            "HER2 intersection n=68 metrics; DS-native 5-fold split."
        ),
    }


def _cross_seed_row(variant: str, seed_rows: list[dict]) -> dict:
    pooled = [r["pooled_auc"] for r in seed_rows]
    return {
        "phase": "phase2a_cross_seed",
        "run_id": f"{variant}__cross_seed",
        "model_family": "deepsets",
        "subgroup": "her2_intersection_n68",
        "n_cases": seed_rows[0]["n_cases"],
        "n_folds": seed_rows[0]["n_folds"],
        "seed": "cross",
        "mean_fold_auc": "",
        "std_fold_auc": "",
        "pooled_auc": "",
        "pooled_ap": "",
        "cross_seed_mean_auc": float(np.mean(pooled)),
        "cross_seed_std_auc": float(np.std(pooled, ddof=0)),
        "fold_map_path": "",
        "config_path": str((SWEEP_ROOT / variant / "runtime_config.yaml").relative_to(REPO)),
        "predictions_path": "",
        "notes": "Cross-seed summary across seeds 42, 7, 123 (3 seeds, single-seed pooled AUCs averaged).",
    }


def main() -> None:
    her2_ids = _her2_intersection_ids()
    existing = pd.read_csv(TRACKER)

    seed_rows: list[dict] = []
    cross_rows: list[dict] = []
    all_seeds = ["42"] + NEW_SEEDS

    for variant in VARIANTS:
        rows_for_variant = [_summary_row(variant, s, her2_ids) for s in all_seeds]
        seed_rows.extend(_summary_row(variant, s, her2_ids) for s in NEW_SEEDS)
        cross_rows.append(_cross_seed_row(variant, rows_for_variant))

    new_df = pd.DataFrame(seed_rows + cross_rows)
    print(new_df[["run_id", "seed", "mean_fold_auc", "std_fold_auc", "pooled_auc", "cross_seed_mean_auc", "cross_seed_std_auc"]].to_string(index=False))

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(TRACKER, index=False)
    print(f"\nWrote {TRACKER} ({len(combined)} rows, +{len(new_df)})")


if __name__ == "__main__":
    main()
