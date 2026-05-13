"""Phase 2a closing figure for HER2 Deep Sets (cross-seed view).

Two panels, designed to close Phase 2a as a statistically robust
*negative* architectural finding on HER2 with an incidental small lift
on luminal A.

(A) Forest plot of HER2 n=68 pooled AUC across baselines and DS configs.
    For multi-seed entries (DS Phase 3 winners across seeds 7, 123 and
    DS Phase 2a attention top-2 across seeds 42, 7, 123) the marker is
    the mean of per-seed pooled AUCs and the whisker is the per-seed
    range. Single-seed entries (log(N) LR, XGB n=68) keep the analytic
    Hanley-McNeil 95% CI on the pooled AUC for context.

(B) Heatmap of paired-fold (Wilcoxon) median delta for each top-2
    attention variant vs each Phase 3 winner across 5 subgroups. Cell
    color encodes the median delta; cell annotation gives `Δ / p`. Cells
    with `p < 0.10` are outlined to call out the only consistent signal
    (luminal A) against the noise floor.

Inputs:
- results/her2_deepsets_tracker.csv (used implicitly via the helpers).
- experiments/deepsets_phase3_repeated_cv/<cfg>/seed{7,123}/train/.../
  predictions.csv (DS Phase 3 winners, used for per-seed pooled AUC).
- experiments/deepsets_sweep_her2_attention_full/<vid>/.../
  predictions.csv (DS Phase 2a, seed=42).
- experiments/deepsets_phase2a_repeated_cv/<vid>/seed{7,123}/train/.../
  predictions.csv (DS Phase 2a, seeds 7 and 123).
- experiments/her2_phase0/{logn_lr,xgb_vessel_all}/predictions.csv.
- results/phase3/her2_phase2a_paired_wilcoxon.csv (Panel B).
- data/her2_intersection_case_ids.csv + manifest (for intersection).

Output:
- results/phase3/her2_phase2a_results.png
- results/phase3/her2_phase2a_results.pdf
"""

from __future__ import annotations

import glob
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT_PNG = REPO / "results" / "phase3" / "her2_phase2a_results.png"
OUT_PDF = REPO / "results" / "phase3" / "her2_phase2a_results.pdf"
MANIFEST = REPO / "experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/deepsets_manifest.csv"
INTERSECTION = REPO / "data/her2_intersection_case_ids.csv"
SWEEP_ROOT = REPO / "experiments/deepsets_sweep_her2_attention_full"
P3_ROOT = REPO / "experiments/deepsets_phase3_repeated_cv"
REPCV_ROOT = REPO / "experiments/deepsets_phase2a_repeated_cv"
WILCOXON_CSV = REPO / "results" / "phase3" / "her2_phase2a_paired_wilcoxon.csv"

ATTN_TOP = ["attn_logn_h16", "attn_h64"]
ATTN_SEEDS = ["42", "7", "123"]
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


def _single_seed_pooled(path: Path) -> tuple[float, int, int]:
    df = pd.read_csv(path)
    score_col = "y_prob" if "y_prob" in df.columns else "y_pred"
    n_pos = int(df["y_true"].sum())
    n_neg = int(len(df) - n_pos)
    return float(roc_auc_score(df["y_true"], df[score_col])), n_pos, n_neg


def _p3_winner_pooled_per_seed(config: str, her2_ids: set[str]) -> dict[str, tuple[float, int, int]]:
    out: dict[str, tuple[float, int, int]] = {}
    for seed in P3_SEEDS:
        matches = sorted(glob.glob(str(P3_ROOT / config / f"seed{seed}" / "train" / "*" / "*" / "predictions.csv")))
        if not matches:
            continue
        out[seed] = _pooled_auc_on_intersection(Path(matches[0]), her2_ids)
    return out


def _attn_pooled_per_seed(variant: str, her2_ids: set[str]) -> dict[str, tuple[float, int, int]]:
    out: dict[str, tuple[float, int, int]] = {}
    for seed in ATTN_SEEDS:
        if seed == "42":
            matches = sorted(glob.glob(str(SWEEP_ROOT / variant / "train" / "*" / "*" / "predictions.csv")))
        else:
            matches = sorted(glob.glob(str(REPCV_ROOT / variant / f"seed{seed}" / "train" / "*" / "*" / "predictions.csv")))
        if not matches:
            continue
        out[seed] = _pooled_auc_on_intersection(Path(matches[0]), her2_ids)
    return out


def main() -> None:
    her2_ids = _her2_intersection_ids()

    # -------- Panel A: forest plot with cross-seed view --------
    rows: list[dict] = []

    auc, npos, nneg = _single_seed_pooled(REPO / "experiments/her2_phase0/logn_lr/predictions.csv")
    lo, hi = hanley_ci(auc, npos, nneg)
    rows.append({"label": "log(N_points) LR\n(one-feature baseline)", "family": "baseline",
                 "auc": auc, "ci_lo": lo, "ci_hi": hi, "seed_range": None, "n_seeds": 1})

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
                     "seed_range": (seed_lo, seed_hi), "n_seeds": len(aucs)})

    auc, npos, nneg = _single_seed_pooled(REPO / "experiments/her2_phase0/xgb_vessel_all/predictions.csv")
    lo, hi = hanley_ci(auc, npos, nneg)
    rows.append({"label": "XGB vessel_all\n(HER2-only train, n=68)", "family": "xgb",
                 "auc": auc, "ci_lo": lo, "ci_hi": hi, "seed_range": None, "n_seeds": 1})

    for v in ATTN_TOP:
        per_seed = _attn_pooled_per_seed(v, her2_ids)
        aucs = [per_seed[s][0] for s in ATTN_SEEDS if s in per_seed]
        mean_auc = float(np.mean(aucs))
        seed_lo, seed_hi = float(min(aucs)), float(max(aucs))
        n_pos, n_neg = next(iter(per_seed.values()))[1:]
        lo, hi = hanley_ci(mean_auc, n_pos, n_neg)
        rows.append({"label": f"DS Phase 2a — {v}\n(mean of seeds 42, 7, 123)",
                     "family": "ds_attention",
                     "auc": mean_auc, "ci_lo": lo, "ci_hi": hi,
                     "seed_range": (seed_lo, seed_hi), "n_seeds": len(aucs)})

    family_order = {"baseline": 0, "ds_prior": 1, "xgb": 2, "ds_attention": 3}
    rows.sort(key=lambda r: (family_order[r["family"]], r["auc"]))

    # -------- Panel B: paired-Wilcoxon heatmap --------
    wdf = pd.read_csv(WILCOXON_CSV)
    subgroups = ["overall", "her2_enriched", "her2_intersection_n68", "luminal_a", "luminal_b", "triple_negative"]
    sg_labels = ["overall\n(n=980)", "HER2\n(n=86)", "HER2 ∩\n(n=68)", "luminal A", "luminal B", "triple\nneg"]
    pair_rows = []
    for v in ATTN_TOP:
        for p in P3_CONFIGS:
            pair_rows.append((v, p))
    delta_mat = np.full((len(pair_rows), len(subgroups)), np.nan)
    p_mat = np.full((len(pair_rows), len(subgroups)), np.nan)
    for i, (v, p) in enumerate(pair_rows):
        for j, sg in enumerate(subgroups):
            r = wdf[(wdf["attn"] == v) & (wdf["p3"] == p) & (wdf["subgroup"] == sg)]
            if not r.empty:
                delta_mat[i, j] = float(r["median_delta"].iloc[0])
                p_mat[i, j] = float(r["wilcoxon_p"].iloc[0])

    # -------- Plot --------
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.30], wspace=0.45)
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
            axA.plot([lo, hi], [i + 0.20, i + 0.20], color=color, linewidth=2.5, alpha=0.55)
            axA.plot([lo, lo], [i + 0.12, i + 0.28], color=color, linewidth=1.0, alpha=0.55)
            axA.plot([hi, hi], [i + 0.12, i + 0.28], color=color, linewidth=1.0, alpha=0.55)
            axA.text(hi + 0.005, i + 0.20, f"n={r['n_seeds']} seeds",
                     fontsize=7, color=color, va="center", alpha=0.75)
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
    axA.set_title(
        "A. HER2 n=68 pooled AUC — cross-seed view\n"
        "Marker = mean of per-seed pooled AUCs; whisker = per-seed range (3 seeds for attention, 2 for P3)",
        fontsize=10.5, loc="left", pad=12,
    )
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=family_label[k], markersize=8)
               for k, c in family_color.items()]
    axA.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.92)

    # Panel B paired-Wilcoxon heatmap
    vmax = float(np.nanmax(np.abs(delta_mat)))
    im = axB.imshow(delta_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(delta_mat.shape[0]):
        for j in range(delta_mat.shape[1]):
            d = delta_mat[i, j]
            p = p_mat[i, j]
            if np.isnan(d):
                continue
            txt = f"Δ={d:+.03f}\np={p:.2f}"
            axB.text(j, i, txt, ha="center", va="center",
                     fontsize=7.5, color="black" if abs(d) < vmax * 0.55 else "white")
            if p < 0.10:
                axB.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                        fill=False, edgecolor="black", linewidth=2.0))

    axB.set_xticks(range(len(subgroups)))
    axB.set_xticklabels(sg_labels, fontsize=9)
    pair_labels = [f"{v}\n vs {p}" for v, p in pair_rows]
    axB.set_yticks(range(len(pair_rows)))
    axB.set_yticklabels(pair_labels, fontsize=8.5)
    axB.set_title(
        "B. Paired-fold Wilcoxon: attention variant vs Phase 3 winner\n"
        "15 paired folds (seeds 42, 7, 123). Black outline = p < 0.10",
        fontsize=10.5, loc="left", pad=12,
    )
    cbar = fig.colorbar(im, ax=axB, fraction=0.04, pad=0.02)
    cbar.set_label("Median per-fold AUC delta (attn − P3)", fontsize=9)

    fig.suptitle(
        "Phase 2a closes: attention pooling does NOT robustly lift HER2 across seeds (median Δ ≈ 0, p ≫ 0.10)\n"
        "Single consistent positive signal is on luminal A (Δ ≈ +0.02-0.04, p ≈ 0.07-0.15) — small, secondary",
        fontsize=12.5, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
