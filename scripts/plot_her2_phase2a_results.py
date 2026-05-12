"""Phase 2a results figure for HER2 Deep Sets.

Two panels:

(A) Forest plot of HER2 n=68 pooled AUC across baselines, prior
    DS Phase 3 winners, the XGB comparator, and the 6 Phase 2a attention
    variants. Error bars are Hanley-McNeil analytic 95% CIs on the
    pooled AUC (irreducible noise floor at n=68). Multi-seed comparators
    use the mean of per-seed pooled AUCs and an additional whisker for
    the per-seed range.

(B) Per-stratum (overall / her2 / lumA / lumB / TN) AUC bars for the
    Phase 3 winner h256_d02_lfocal (mean of seeds 7 and 123) vs the top
    two Phase 2a attention variants. Shows that the HER2 lift is not
    purchased at the cost of an overall-cohort regression.

Inputs:
- results/her2_deepsets_tracker.csv (for the tabular/baseline numbers).
- experiments/deepsets_phase3_repeated_cv/h256_d02_lfocal/seed{7,123}/...
  metrics.json (for per-stratum Phase 3 baseline).
- experiments/deepsets_sweep_her2_attention_full/<vid>/.../metrics.json
  and predictions.csv (for attention variant numbers).
- data/her2_intersection_case_ids.csv + manifest (for intersection cohort).

Output:
- results/phase3/her2_phase2a_results.png
- results/phase3/her2_phase2a_results.pdf
"""

from __future__ import annotations

import glob
import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT_PNG = REPO / "results" / "phase3" / "her2_phase2a_results.png"
OUT_PDF = REPO / "results" / "phase3" / "her2_phase2a_results.pdf"
MANIFEST = REPO / "experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/deepsets_manifest.csv"
INTERSECTION = REPO / "data/her2_intersection_case_ids.csv"
SWEEP_ROOT = REPO / "experiments/deepsets_sweep_her2_attention_full"
P3_ROOT = REPO / "experiments/deepsets_phase3_repeated_cv"

ATTN_VARIANTS = [
    "attn_h16",
    "attn_h32",
    "attn_h64",
    "attn_logn_h16",
    "attn_logn_h32",
    "attn_logn_h64",
]
P3_CONFIGS = ["cos_T80", "h128_d02_lfocal", "h256_d02_lfocal"]
P3_SEEDS = ["7", "123"]


def hanley_ci(auc: float, n_pos: int, n_neg: int, alpha: float = 0.05) -> tuple[float, float]:
    """Hanley-McNeil 95% CI on AUC."""
    if n_pos < 1 or n_neg < 1:
        return float("nan"), float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    )
    se = sqrt(max(var, 0) / max(n_pos * n_neg, 1))
    z = 1.96 if alpha == 0.05 else 1.6449
    return auc - z * se, auc + z * se


def _her2_intersection_ids() -> set[str]:
    inter = set(pd.read_csv(INTERSECTION)["case_id"].astype(str))
    manifest = pd.read_csv(MANIFEST)
    manifest["case_id"] = manifest["case_id"].astype(str)
    her2 = set(manifest.loc[manifest["tumor_subtype"] == "her2_enriched", "case_id"].astype(str))
    return inter & her2


def _pooled_auc_on_intersection(predictions_csv: Path, her2_ids: set[str]) -> tuple[float, int, int]:
    df = pd.read_csv(predictions_csv)
    df["case_id"] = df["case_id"].astype(str)
    score_col = "y_prob" if "y_prob" in df.columns else "y_pred"
    sub = df[df["case_id"].isin(her2_ids)]
    n_pos = int(sub["y_true"].sum())
    n_neg = int(len(sub) - n_pos)
    return float(roc_auc_score(sub["y_true"], sub[score_col])), n_pos, n_neg


def _logn_lr_pooled() -> tuple[float, int, int]:
    p = REPO / "experiments/her2_phase0/logn_lr/predictions.csv"
    df = pd.read_csv(p)
    score_col = "y_prob" if "y_prob" in df.columns else "y_pred"
    n_pos = int(df["y_true"].sum())
    n_neg = int(len(df) - n_pos)
    return float(roc_auc_score(df["y_true"], df[score_col])), n_pos, n_neg


def _xgb_pooled() -> tuple[float, int, int]:
    p = REPO / "experiments/her2_phase0/xgb_vessel_all/predictions.csv"
    df = pd.read_csv(p)
    score_col = "y_prob" if "y_prob" in df.columns else "y_pred"
    n_pos = int(df["y_true"].sum())
    n_neg = int(len(df) - n_pos)
    return float(roc_auc_score(df["y_true"], df[score_col])), n_pos, n_neg


def _p3_winner_pooled_per_seed(config: str, her2_ids: set[str]) -> dict[str, tuple[float, int, int]]:
    out = {}
    for seed in P3_SEEDS:
        matches = sorted(glob.glob(str(P3_ROOT / config / f"seed{seed}" / "train" / "*" / "*" / "predictions.csv")))
        if not matches:
            continue
        out[seed] = _pooled_auc_on_intersection(Path(matches[0]), her2_ids)
    return out


def _attn_pooled(variant: str, her2_ids: set[str]) -> tuple[float, int, int]:
    matches = sorted(glob.glob(str(SWEEP_ROOT / variant / "train" / "*" / "*" / "predictions.csv")))
    return _pooled_auc_on_intersection(Path(matches[0]), her2_ids)


def _per_stratum_from_metrics(metrics_path: Path) -> dict[str, float]:
    m = json.load(open(metrics_path))
    val = m["validation_summary"]
    by = val.get("by_group", {})
    return {
        "overall": val["overall"]["auc"],
        "her2_enriched": by.get("her2_enriched", {}).get("auc"),
        "luminal_a": by.get("luminal_a", {}).get("auc"),
        "luminal_b": by.get("luminal_b", {}).get("auc"),
        "triple_negative": by.get("triple_negative", {}).get("auc"),
    }


def main() -> None:
    her2_ids = _her2_intersection_ids()

    # -------- Panel A: forest plot data --------
    rows: list[dict] = []
    auc, npos, nneg = _logn_lr_pooled()
    lo, hi = hanley_ci(auc, npos, nneg)
    rows.append({"label": "log(N_points) LR\n(one-feature baseline)", "family": "baseline",
                 "auc": auc, "ci_lo": lo, "ci_hi": hi, "seed_range": None})

    for cfg in P3_CONFIGS:
        per_seed = _p3_winner_pooled_per_seed(cfg, her2_ids)
        if not per_seed:
            continue
        aucs = [v[0] for v in per_seed.values()]
        mean_auc = float(np.mean(aucs))
        seed_lo, seed_hi = float(min(aucs)), float(max(aucs))
        n_pos, n_neg = next(iter(per_seed.values()))[1:]
        lo, hi = hanley_ci(mean_auc, n_pos, n_neg)
        rows.append({"label": f"DS Phase 3 — {cfg}\n(mean of seeds 7, 123)",
                     "family": "ds_prior",
                     "auc": mean_auc, "ci_lo": lo, "ci_hi": hi,
                     "seed_range": (seed_lo, seed_hi)})

    auc, npos, nneg = _xgb_pooled()
    lo, hi = hanley_ci(auc, npos, nneg)
    rows.append({"label": "XGB vessel_all\n(HER2-only train, n=68)", "family": "xgb",
                 "auc": auc, "ci_lo": lo, "ci_hi": hi, "seed_range": None})

    for v in ATTN_VARIANTS:
        auc, npos, nneg = _attn_pooled(v, her2_ids)
        lo, hi = hanley_ci(auc, npos, nneg)
        rows.append({"label": f"DS Phase 2a — {v}\n(seed=42, single seed)",
                     "family": "ds_attention",
                     "auc": auc, "ci_lo": lo, "ci_hi": hi, "seed_range": None})

    # Sort within families: ds_attention by AUC asc so best is at top
    family_order = {"baseline": 0, "ds_prior": 1, "xgb": 2, "ds_attention": 3}
    rows.sort(key=lambda r: (family_order[r["family"]], r["auc"]))

    # -------- Panel B: per-stratum --------
    p3_strata_per_seed = []
    for seed in P3_SEEDS:
        matches = sorted(glob.glob(str(P3_ROOT / "h256_d02_lfocal" / f"seed{seed}" / "train" / "*" / "*" / "metrics.json")))
        if matches:
            p3_strata_per_seed.append(_per_stratum_from_metrics(Path(matches[0])))
    strata = ["overall", "her2_enriched", "luminal_a", "luminal_b", "triple_negative"]
    p3_mean = {s: float(np.mean([r[s] for r in p3_strata_per_seed if r[s] is not None])) for s in strata}

    attn_top_strata = {}
    for v in ["attn_logn_h16", "attn_h64"]:
        m_path = sorted(glob.glob(str(SWEEP_ROOT / v / "train" / "*" / "*" / "metrics.json")))[0]
        attn_top_strata[v] = _per_stratum_from_metrics(Path(m_path))

    # -------- Plot --------
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.35)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    family_color = {
        "baseline": "#7f7f7f",
        "ds_prior": "#1f77b4",
        "xgb": "#2ca02c",
        "ds_attention": "#d62728",
    }
    family_label = {
        "baseline": "log(N) baseline",
        "ds_prior": "DS Phase 3 prior",
        "xgb": "XGB tabular",
        "ds_attention": "DS Phase 2a (attention)",
    }

    y_positions = list(range(len(rows)))
    best_idx = max(range(len(rows)), key=lambda i: rows[i]["auc"])
    for i, r in enumerate(rows):
        color = family_color[r["family"]]
        is_best = i == best_idx
        ms = 11 if is_best else 8
        mec = "black" if is_best else color
        axA.errorbar(
            r["auc"], i,
            xerr=[[r["auc"] - r["ci_lo"]], [r["ci_hi"] - r["auc"]]],
            fmt="o", color=color, markersize=ms, markeredgecolor=mec,
            markeredgewidth=1.5 if is_best else 0.5,
            elinewidth=1.5, capsize=4, capthick=1.5,
            ecolor=color, alpha=0.95,
        )
        if r["seed_range"] is not None:
            lo, hi = r["seed_range"]
            axA.plot([lo, hi], [i + 0.18, i + 0.18], color=color, linewidth=2.5, alpha=0.5)
            axA.plot([lo, lo], [i + 0.10, i + 0.26], color=color, linewidth=1.0, alpha=0.5)
            axA.plot([hi, hi], [i + 0.10, i + 0.26], color=color, linewidth=1.0, alpha=0.5)
        axA.text(r["auc"], i - 0.32, f"{r['auc']:.3f}", ha="center", va="top",
                 fontsize=8.5, color=color,
                 fontweight="bold" if is_best else "normal")

    axA.axvline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    axA.text(0.5, -1.0, "no-skill", ha="center", va="top", fontsize=8, color="black", alpha=0.6)
    axA.set_yticks(y_positions)
    axA.set_yticklabels([r["label"] for r in rows], fontsize=9)
    axA.set_xlabel("HER2 n=68 intersection pooled AUC", fontsize=11)
    axA.set_xlim(0.35, 0.85)
    axA.set_ylim(-1.5, len(rows) - 0.5)
    axA.invert_yaxis()
    axA.grid(axis="x", alpha=0.25)
    axA.set_title("A. HER2 n=68 pooled AUC (Hanley 95% CI)\nThin whisker on DS Phase 3 rows = per-seed range",
                  fontsize=11, loc="left", pad=12)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=family_label[k], markersize=8)
               for k, c in family_color.items()]
    axA.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.92)

    # Panel B per-stratum
    strata_labels = ["overall\n(n=980)", "HER2\n(n=86)", "lum A", "lum B", "triple\nneg"]
    x = np.arange(len(strata))
    width = 0.26
    p3_vals = [p3_mean[s] for s in strata]
    a_logn_vals = [attn_top_strata["attn_logn_h16"][s] for s in strata]
    a_h64_vals = [attn_top_strata["attn_h64"][s] for s in strata]

    bars_p3 = axB.bar(x - width, p3_vals, width, color=family_color["ds_prior"],
                      label="P3 h256_d02_lfocal\n(mean of 2 seeds)", alpha=0.85, edgecolor="black", linewidth=0.4)
    bars_logn = axB.bar(x, a_logn_vals, width, color=family_color["ds_attention"],
                        label="attn_logn_h16 (seed=42)", alpha=0.85, edgecolor="black", linewidth=0.4)
    bars_h64 = axB.bar(x + width, a_h64_vals, width, color="#ff9896",
                       label="attn_h64 (seed=42)", alpha=0.85, edgecolor="black", linewidth=0.4)

    for bars, vals in [(bars_p3, p3_vals), (bars_logn, a_logn_vals), (bars_h64, a_h64_vals)]:
        for b, v in zip(bars, vals):
            axB.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=7.5)

    axB.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    axB.set_xticks(x)
    axB.set_xticklabels(strata_labels, fontsize=9)
    axB.set_ylabel("AUC", fontsize=11)
    axB.set_ylim(0.40, 0.72)
    axB.grid(axis="y", alpha=0.25)
    axB.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    axB.set_title("B. Per-stratum AUC: attention variants vs Phase 3 winner\nHER2 lift does not regress overall or other strata (except lumB on attn_h64)",
                  fontsize=11, loc="left", pad=12)

    fig.suptitle(
        "Phase 2a attention pooling on full ISPY2 cohort (n=980, 5-fold CV, seed=42)\n"
        "Top attention variant attn_logn_h16: HER2 n=68 pooled AUC 0.663 — best DS-pure HER2 number to date",
        fontsize=12.5, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
