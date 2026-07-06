#!/usr/bin/env python3
"""Head-on native-vs-resampled pCR comparison on the MATCHED ISPY2 cohort.

The "resampled made models worse" observation may be an artifact of comparing a
980-case native run against an 808-case resampled run with different folds. This
script removes that confound: it evaluates pCR prediction on the *same* 808
matched cases with the *same* cross-validation folds, swapping only the feature
source (native vs resampled). If resampling truly hurts, the resampled AUC should
drop even under this fair, paired comparison.

We compare across the ablation's key feature arms:
  * vessel_all                     (morph + graph + kinematic)
  * tumor_size
  * clinical + tumor_size          (the standard baseline)
  * clinical + tumor_size + vessel_all   (the full model)

Notes on what changes between native and resampled:
  * clinical features are patient metadata -> identical in both (they are the
    controlled, unchanged block).
  * tumor_size comes from the tumor masks -> differs (resampled masks).
  * vessel arms come from the centerlines -> differ (resampled centerlines).

Models mirror the pipeline: elasticnet logistic regression and XGBoost. Same
5-fold stratified folds for native and resampled, so the comparison is paired.

Usage::

    python scripts/compare_native_resampled_pcr_ispy2.py \
        --native   .../tabular/all_datasets_run/features_full_labeled.csv \
        --resampled .../independent_signal_resampled_ispy2/features_full_labeled.csv \
        --outdir   /ess/scratch/scratch1/t-9sbose/ispy2_native_vs_resampled_pcr
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATASET = "ISPY2"
ID_COL = "case_id"
PCR_COL = "pcr"
N_SPLITS = 5
VESSEL_PREFIXES = ("morph_", "graph_", "kinematic_")
TUMOR_SIZE_PREFIX = "tumor_size_"
# Feature arms to evaluate; each is a list of block names understood by
# build_matrix(). "clinical" is source-independent; the others are per-source.
FEATURE_SETS = {
    "vessel_all": ["vessel"],
    "tumor_size": ["tumor_size"],
    "clinical+tumor_size": ["clinical", "tumor_size"],
    "clinical+tumor_size+vessel_all": ["clinical", "tumor_size", "vessel"],
}


def load_ispy2(path: Path) -> pd.DataFrame:
    """Load and return the ISPY2 rows of a feature file, indexed by case_id."""
    feats = pd.read_csv(path, low_memory=False)
    if "dataset" in feats.columns:
        feats = feats[feats["dataset"] == DATASET]
    return feats.set_index(ID_COL)


def build_clinical(meta: pd.DataFrame) -> pd.DataFrame:
    """Encode the clinical block (identical for native & resampled).

    age stays numeric; bilateral becomes 0/1; menopausal_status is collapsed to
    its leading word (pre/peri/post); tumor_subtype is one-hot encoded. This is a
    faithful-enough stand-in for the pipeline's clinical block for a relative
    native-vs-resampled comparison. breast_density is dropped (all empty).
    """
    out = pd.DataFrame(index=meta.index)
    out["age"] = pd.to_numeric(meta["age"], errors="coerce")
    out["bilateral"] = meta["bilateral"].astype("boolean").astype(float)
    meno = meta["menopausal_status"].astype(str).str.split().str[0].fillna("unknown")
    subtype = meta["tumor_subtype"].astype(str).fillna("unknown")
    dummies = pd.get_dummies(
        pd.DataFrame({"meno": meno, "subtype": subtype}), dtype=float
    )
    return pd.concat([out, dummies], axis=1)


def block_columns(feats: pd.DataFrame, block: str) -> list[str]:
    """Return the source-dependent columns for a block ('vessel'|'tumor_size')."""
    if block == "vessel":
        return sorted(c for c in feats.columns if c.startswith(VESSEL_PREFIXES))
    if block == "tumor_size":
        return sorted(c for c in feats.columns if c.startswith(TUMOR_SIZE_PREFIX))
    raise ValueError(block)


def build_matrix(
    blocks: list[str], source: pd.DataFrame, clinical: pd.DataFrame
) -> np.ndarray:
    """Assemble the numeric feature matrix for a feature set from one source."""
    parts = []
    for block in blocks:
        if block == "clinical":
            parts.append(clinical)
        else:
            parts.append(source[block_columns(source, block)])
    return pd.concat(parts, axis=1).to_numpy(dtype=float)


def make_model(name: str, y: np.ndarray, seed: int) -> Pipeline:
    """Elasticnet LR or XGBoost, wrapped with median-impute + standardize."""
    if name == "lr_enet":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
        )
    else:  # xgboost
        pos = float(y.sum())
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=4,
            random_state=seed,
            scale_pos_weight=(len(y) - pos) / pos if pos else 1.0,
        )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def cv_auc(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> float:
    """5-fold cross-validated pCR ROC AUC."""
    pipe = make_model(model, y, seed)
    proba = cross_val_predict(pipe, x, y, cv=folds, method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))


def main() -> None:
    """Run the matched native-vs-resampled pCR comparison and write outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--resampled", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    native = load_ispy2(args.native)
    resampled = load_ispy2(args.resampled)
    shared_ids = sorted(set(native.index) & set(resampled.index))
    native = native.loc[shared_ids]
    resampled = resampled.loc[shared_ids]
    y = native[PCR_COL].to_numpy().astype(int)
    log.info("Matched cases: %d | pCR rate %.3f", len(shared_ids), y.mean())

    # Clinical block is identical across sources; build once (from native).
    clinical = build_clinical(native)

    # Fixed folds -> identical splits for native and resampled (paired).
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(y)), y))

    rows = []
    for fs_name, blocks in FEATURE_SETS.items():
        x_nat = build_matrix(blocks, native, clinical)
        x_res = build_matrix(blocks, resampled, clinical)
        for model in ("lr_enet", "xgboost"):
            auc_nat = cv_auc(x_nat, y, model, folds, args.seed)
            auc_res = cv_auc(x_res, y, model, folds, args.seed)
            rows.append(
                {
                    "feature_set": fs_name,
                    "model": model,
                    "n_features": x_nat.shape[1],
                    "native_auc": auc_nat,
                    "resampled_auc": auc_res,
                    "auc_change_res_minus_nat": auc_res - auc_nat,
                }
            )
            log.info(
                "  %-32s %-8s native=%.3f resampled=%.3f (Δ=%+.3f)",
                fs_name,
                model,
                auc_nat,
                auc_res,
                auc_res - auc_nat,
            )

    results = pd.DataFrame(rows)
    results.to_csv(args.outdir / "native_vs_resampled_pcr.csv", index=False)
    _plot(results, args.outdir / "native_vs_resampled_pcr.png")
    _write_summary(results, len(shared_ids), y, args.outdir / "SUMMARY.txt")
    log.info("\n=== results ===\n%s", results.to_string(index=False))
    log.info("\nAll outputs written to: %s", args.outdir)


def _plot(results: pd.DataFrame, path: Path) -> None:
    """Grouped bars: native vs resampled pCR AUC per (feature_set, model)."""
    combos = results[["feature_set", "model"]].to_numpy()
    labels = [f"{fs}\n{m}" for fs, m in combos]
    x = np.arange(len(combos))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2, results["native_auc"], width, label="native", color="steelblue"
    )
    ax.bar(
        x + width / 2,
        results["resampled_auc"],
        width,
        label="resampled",
        color="darkorange",
    )
    ax.axhline(0.5, ls="--", color="0.5", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("5-fold CV pCR ROC AUC")
    ax.set_ylim(
        0.45,
        max(0.7, float(results[["native_auc", "resampled_auc"]].max().max()) + 0.05),
    )
    ax.set_title("Matched ISPY2 (808 cases, same folds): native vs resampled pCR AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_summary(
    results: pd.DataFrame, n_cases: int, y: np.ndarray, path: Path
) -> None:
    """Human-readable summary of the matched comparison."""
    with path.open("w") as fh:
        w = lambda s: fh.write(s + "\n")  # noqa: E731 - tiny local writer
        w("Matched ISPY2 native-vs-resampled pCR comparison")
        w("=" * 50)
        w(f"Matched cases: {n_cases} | pCR rate {y.mean():.3f} | {N_SPLITS}-fold CV")
        w("Same cases + same folds; only the feature source is swapped.")
        w("clinical block is identical across sources (patient metadata).")
        w("")
        w(results.to_string(index=False))
        w("")
        mean_delta = results["auc_change_res_minus_nat"].mean()
        n_res_worse = int((results["auc_change_res_minus_nat"] < 0).sum())
        w(
            f"resampled worse in {n_res_worse}/{len(results)} cells; "
            f"mean AUC change (resampled - native) = {mean_delta:+.4f}"
        )
    log.info("\n%s", path.read_text())


if __name__ == "__main__":
    main()
