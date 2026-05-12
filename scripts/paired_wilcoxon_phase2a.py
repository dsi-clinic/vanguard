"""Paired-fold Wilcoxon signed-rank test: Phase 2a attention vs Phase 3 winners.

For each (attention_variant, p3_winner) pair and each subgroup endpoint
(overall / her2_enriched / her2_intersection_n68 / triple_negative /
luminal_a / luminal_b), this script:

1. Loads predictions.csv for both models across all matching seeds
   (42, 7, 123). Seeds where either side is missing are skipped with
   a warning so the script also works mid-sweep with only seed 42.
2. For each (seed, fold) pair, restricts to the subgroup's case_ids
   and computes AUC for each model. Pairs are well-defined because
   identical seed -> identical fold assignment in the DS pipeline.
3. Runs scipy.stats.wilcoxon(zero_method="pratt") on the per-fold AUC
   deltas (attention - p3_winner).
4. Reports median delta, IQR, n_pairs, Wilcoxon W and p-value.

Run any time; will gracefully degrade when seed 7/123 attention jobs
have not yet completed.

Outputs results/phase3/her2_phase2a_paired_wilcoxon.csv
and a console summary table.
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/deepsets_manifest.csv"
INTERSECTION = REPO / "data/her2_intersection_case_ids.csv"
ATTN_SWEEP_ROOT = REPO / "experiments/deepsets_sweep_her2_attention_full"
ATTN_REPCV_ROOT = REPO / "experiments/deepsets_phase2a_repeated_cv"
P3_REPCV_ROOT = REPO / "experiments/deepsets_phase3_repeated_cv"

# Seed -> seed=42 predictions live in the old single-seed sweep dirs
P3_SEED42_PATHS = {
    "cos_T80": REPO / "experiments/deepsets_sweep_dynamic_fine/cos_T80/train",
    "h256_d02_lfocal": REPO / "experiments/deepsets_sweep_dynamic_coarse/h256_d02_lfocal/train",
    "h128_d02_lfocal": REPO / "experiments/deepsets_sweep_dynamic_coarse/h128_d02_lfocal/train",
}

ATTN_VARIANTS = ["attn_logn_h16", "attn_h64"]
P3_WINNERS = ["cos_T80", "h128_d02_lfocal", "h256_d02_lfocal"]
SEEDS = ["42", "7", "123"]


def _resolve_attn_pred(variant: str, seed: str) -> Path | None:
    if seed == "42":
        m = sorted(glob.glob(str(ATTN_SWEEP_ROOT / variant / "train" / "*" / "*" / "predictions.csv")))
    else:
        m = sorted(glob.glob(str(ATTN_REPCV_ROOT / variant / f"seed{seed}" / "train" / "*" / "*" / "predictions.csv")))
    return Path(m[0]) if m else None


def _resolve_p3_pred(cfg: str, seed: str) -> Path | None:
    if seed == "42":
        m = sorted(glob.glob(str(P3_SEED42_PATHS[cfg] / "*" / "*" / "predictions.csv")))
    else:
        m = sorted(glob.glob(str(P3_REPCV_ROOT / cfg / f"seed{seed}" / "train" / "*" / "*" / "predictions.csv")))
    return Path(m[0]) if m else None


def _build_subgroup_filters() -> dict[str, set[str]]:
    """Map subgroup name -> set of case_ids to restrict scoring to.

    'overall' uses None (i.e., no filter).
    """
    manifest = pd.read_csv(MANIFEST)
    manifest["case_id"] = manifest["case_id"].astype(str)
    inter = set(pd.read_csv(INTERSECTION)["case_id"].astype(str))
    out: dict[str, set[str]] = {"overall": None}  # type: ignore[dict-item]
    for sub in ["her2_enriched", "luminal_a", "luminal_b", "triple_negative"]:
        out[sub] = set(manifest.loc[manifest["tumor_subtype"] == sub, "case_id"].astype(str))
    out["her2_intersection_n68"] = inter & out["her2_enriched"]
    return out


def _per_fold_aucs(predictions_csv: Path, case_ids: set[str] | None) -> dict[int, float]:
    df = pd.read_csv(predictions_csv)
    df["case_id"] = df["case_id"].astype(str)
    score_col = "y_prob" if "y_prob" in df.columns else "y_pred"
    if case_ids is not None:
        df = df[df["case_id"].isin(case_ids)]
    aucs: dict[int, float] = {}
    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]
        if sub["y_true"].nunique() < 2 or len(sub) < 3:
            continue
        aucs[int(fold)] = float(roc_auc_score(sub["y_true"], sub[score_col]))
    return aucs


def compare(
    attn_variant: str,
    p3_cfg: str,
    subgroup: str,
    case_ids: set[str] | None,
) -> dict:
    deltas: list[float] = []
    detail: list[dict] = []
    for seed in SEEDS:
        a_path = _resolve_attn_pred(attn_variant, seed)
        b_path = _resolve_p3_pred(p3_cfg, seed)
        if a_path is None or b_path is None:
            warnings.warn(
                f"Missing predictions for seed {seed}: attn={a_path}, p3={b_path}; skipping seed",
                stacklevel=2,
            )
            continue
        a_aucs = _per_fold_aucs(a_path, case_ids)
        b_aucs = _per_fold_aucs(b_path, case_ids)
        common = sorted(set(a_aucs.keys()) & set(b_aucs.keys()))
        for fold in common:
            d = a_aucs[fold] - b_aucs[fold]
            deltas.append(d)
            detail.append({"seed": seed, "fold": fold, "attn_auc": a_aucs[fold], "p3_auc": b_aucs[fold], "delta": d})

    n_pairs = len(deltas)
    if n_pairs < 3:
        return {
            "attn": attn_variant, "p3": p3_cfg, "subgroup": subgroup,
            "n_pairs": n_pairs, "median_delta": float("nan"),
            "iqr_lo": float("nan"), "iqr_hi": float("nan"),
            "wilcoxon_W": float("nan"), "wilcoxon_p": float("nan"),
            "n_pos_delta": int(sum(1 for d in deltas if d > 0)),
            "detail": detail,
        }

    deltas_arr = np.array(deltas)
    iqr_lo, iqr_hi = np.percentile(deltas_arr, [25, 75])
    median_delta = float(np.median(deltas_arr))
    if not np.any(deltas_arr != 0):
        w_stat = 0.0
        p_val = 1.0
    else:
        res = wilcoxon(deltas_arr, zero_method="pratt", alternative="two-sided")
        w_stat = float(res.statistic)
        p_val = float(res.pvalue)
    return {
        "attn": attn_variant, "p3": p3_cfg, "subgroup": subgroup,
        "n_pairs": n_pairs,
        "median_delta": median_delta,
        "iqr_lo": float(iqr_lo), "iqr_hi": float(iqr_hi),
        "wilcoxon_W": w_stat, "wilcoxon_p": p_val,
        "n_pos_delta": int(sum(1 for d in deltas if d > 0)),
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Print per-fold detail rows.")
    args = parser.parse_args()

    subgroup_filters = _build_subgroup_filters()
    summary_rows: list[dict] = []

    print(f"{'attn variant':<18} {'p3 winner':<22} {'subgroup':<24} {'n':>4} {'median Δ':>10} {'IQR':>20} {'+':>3}/{'n':<3} {'p (Wilcoxon)':>14}")
    print("-" * 120)
    for sub in ["overall", "her2_enriched", "her2_intersection_n68", "triple_negative", "luminal_a", "luminal_b"]:
        ids = subgroup_filters[sub]
        for attn in ATTN_VARIANTS:
            for p3 in P3_WINNERS:
                r = compare(attn, p3, sub, ids)
                summary_rows.append({k: v for k, v in r.items() if k != "detail"})
                if r["n_pairs"] >= 3:
                    iqr_s = f"[{r['iqr_lo']:+.3f}, {r['iqr_hi']:+.3f}]"
                    print(f"{attn:<18} {p3:<22} {sub:<24} {r['n_pairs']:>4} {r['median_delta']:>+10.4f} {iqr_s:>20} {r['n_pos_delta']:>3}/{r['n_pairs']:<3} {r['wilcoxon_p']:>14.4f}")
                else:
                    print(f"{attn:<18} {p3:<22} {sub:<24} {r['n_pairs']:>4} {'(insufficient pairs — seeds 7/123 not yet returned)':>70}")
        print()

    out_path = REPO / "results" / "phase3" / "her2_phase2a_paired_wilcoxon.csv"
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\nWrote summary: {out_path}")

    if args.verbose:
        print("\nPer-fold detail tables (verbose mode):")
        for sub in ["overall", "triple_negative", "her2_enriched"]:
            ids = subgroup_filters[sub]
            for attn in ATTN_VARIANTS:
                for p3 in P3_WINNERS:
                    r = compare(attn, p3, sub, ids)
                    if r["n_pairs"] < 3: continue
                    print(f"\n--- {attn} vs {p3} on {sub} ---")
                    print(pd.DataFrame(r["detail"]).to_string(index=False))


if __name__ == "__main__":
    main()
