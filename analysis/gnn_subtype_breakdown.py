"""Subtype-stratified pCR performance, HER2 first, from saved OOF predictions.

Answers "how does the model do specifically on HER2+?" by *stratifying the
existing pooled out-of-fold predictions*, not by retraining within the subgroup.
That choice is deliberate and matters: the HER2-positive subgroup is n=31, far
too small to refit and cross-validate inside. Stratifying the pooled OOF instead
asks the well-posed question -- "of the predictions the cohort-wide model already
made, how well do the HER2+ ones rank?" -- and keeps the subgroup number tied to
the same fits as the headline number, since both read
``oof_predictions.csv`` written by ``analysis/gnn_tabular_gate.py``.

Subtype comes from the cleaned pretreatment cohort manifest, where it is
appended as **patient-level post-hoc provenance** (``subtype_role =
patient_level``, ``accession_specific_subtype_claim = False``). It was not used
to select the cohort or to fit anything, so this is a descriptive breakdown, not
a subgroup the model was tuned on.

Two documented edge cases are reported explicitly rather than silently dropped
(see the cohort README and AUDITING_RESULTS.md):

1. **Single-class subgroups have no defined AUC.** In this cohort all 8
   subtype-unavailable patients are non-pCR, so their AUC is undefined. Such a
   subgroup is emitted with ``auc = NaN`` and ``status =
   undefined_single_class`` -- it is never quietly skipped or imputed.
2. **Degenerate bootstrap draws.** In a small subgroup some resamples contain
   only one class and have no AUC. Those draws are excluded from the percentile
   CI and counted in ``n_boot_degenerate`` so the reader can see how much of the
   CI rests on usable draws.

Usage::

    python -m analysis.gnn_subtype_breakdown \
        --oof-predictions experiments/uchicago_pretreat83/oof_predictions.csv \
        --manifest /gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v1/dce2d_internal_ultrafast_manifest.csv \
        --out-dir experiments/uchicago_pretreat83
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_N_BOOTSTRAP = 2000
#: AUC needs both outcome classes present; below this a subgroup or a bootstrap
#: draw has no defined AUC.
_MIN_CLASSES_FOR_AUC = 2
_CI_PCT = (2.5, 97.5)

#: Subgroup column -> the value each subgroup is defined by. ``her2_consensus``
#: is the receptor axis the mentor asked for; ``subtype`` is the finer clinical
#: grouping reported alongside it for context.
_GROUPINGS = ("her2_consensus", "subtype")


def _subgroup_auc_ci(
    y: np.ndarray, prob: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """Point AUC + percentile bootstrap CI, resampling cases within the subgroup.

    Returns ``auc = NaN`` with a status when the subgroup holds a single outcome
    class, since AUC is undefined there (edge case 1 in the module docstring).
    """
    if len(np.unique(y)) < _MIN_CLASSES_FOR_AUC:
        return {
            "auc": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_boot_degenerate": 0,
            "status": "undefined_single_class",
        }
    point = float(roc_auc_score(y, prob))
    aucs = []
    degenerate = 0
    for _ in range(_N_BOOTSTRAP):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < _MIN_CLASSES_FOR_AUC:
            degenerate += 1
            continue
        aucs.append(roc_auc_score(y[idx], prob[idx]))
    lo, hi = np.percentile(aucs, _CI_PCT)
    return {
        "auc": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot_degenerate": degenerate,
        "status": "ok",
    }


def main() -> None:
    """Score every model within every subtype subgroup and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    oof = pd.read_csv(args.oof_predictions)
    manifest = pd.read_csv(args.manifest)

    meta_cols = ["exam_id", *_GROUPINGS]
    meta = manifest[meta_cols].rename(columns={"exam_id": "case_id"})
    merged = oof.merge(meta, on="case_id", how="left", validate="one_to_one")
    missing = merged[list(_GROUPINGS)].isna().all(axis=1).sum()
    if missing:
        raise ValueError(
            f"{missing} scored case(s) have no manifest row; the OOF predictions "
            "and the manifest describe different cohorts."
        )
    # Manifest leaves her2_consensus empty when receptors are unavailable; name
    # that state so it forms its own visible subgroup instead of vanishing.
    merged[list(_GROUPINGS)] = merged[list(_GROUPINGS)].fillna("unavailable")

    model_cols = [c for c in oof.columns if c not in ("case_id", "pcr", "fold")]
    rng = np.random.default_rng(0)

    rows = []
    for grouping in _GROUPINGS:
        for value, block in merged.groupby(grouping):
            y = block["pcr"].to_numpy()
            for model in model_cols:
                rows.append(
                    {
                        "grouping": grouping,
                        "subgroup": value,
                        "model": model,
                        "n": len(block),
                        "n_pcr": int((y == 1).sum()),
                        "n_non_pcr": int((y == 0).sum()),
                        **_subgroup_auc_ci(y, block[model].to_numpy(), rng),
                    }
                )
    table = pd.DataFrame(rows)

    for grouping in _GROUPINGS:
        print(f"\n=== {grouping} ===")
        block = table[table["grouping"] == grouping]
        for value, sub in block.groupby("subgroup"):
            head = sub.iloc[0]
            print(
                f"  {value}  (n={head['n']}, {head['n_pcr']} pCR / "
                f"{head['n_non_pcr']} non-pCR)"
            )
            for _, r in sub.iterrows():
                if r["status"] != "ok":
                    print(f"      {r['model']:26s} AUC undefined ({r['status']})")
                else:
                    print(
                        f"      {r['model']:26s} {r['auc']:.3f}  "
                        f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "subtype_breakdown.csv"
    table.to_csv(out_csv, index=False)
    _forest_plot(table, args.out_dir / "subtype_forest.png")
    print(f"\nWrote {out_csv}, {args.out_dir}/subtype_forest.png")


def _forest_plot(table: pd.DataFrame, out_path: Path) -> None:
    """Forest plot of per-subgroup AUC for the headline GNN and tabular arms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_models = [
        m for m in ("gnn_baseline", "tabular_7means_logreg") if m in set(table["model"])
    ]
    block = table[
        (table["grouping"] == "her2_consensus")
        & (table["model"].isin(plot_models))
        & (table["status"] == "ok")
    ]
    if block.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(block) + 1.5))
    labels, colors = [], {"gnn_baseline": "tab:blue", "tabular_7means_logreg": "grey"}
    for i, (_, r) in enumerate(block.iterrows()):
        ax.errorbar(
            r["auc"],
            i,
            xerr=[[r["auc"] - r["ci_lo"]], [r["ci_hi"] - r["auc"]]],
            fmt="o",
            color=colors.get(r["model"], "k"),
            ecolor=colors.get(r["model"], "k"),
            capsize=3,
        )
        labels.append(f"{r['subgroup']} / {r['model']} (n={r['n']})")
    ax.axvline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_yticks(range(len(block)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Subgroup OOF AUC (95% bootstrap CI)")
    ax.set_title("UChicago pCR by HER2 status (pooled-OOF stratification)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
