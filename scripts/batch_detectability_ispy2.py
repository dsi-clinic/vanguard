#!/usr/bin/env python3
"""Batch-detectability test for ISPY2 graph features: native vs resampled.

Question: how strongly do the vessel *graph* features carry a scanner
"fingerprint"? We answer it operationally -- if a classifier can predict which
scanner acquired a case *from the features alone*, the features encode a batch
effect. The higher the classifier does above chance, the stronger the effect.

We run this on the NATIVE features and on the RESAMPLED features over the SAME
cases and SAME cross-validation folds, so the two are directly comparable. If
resampling to a common voxel grid removes the batch effect, the resampled AUC
should drop toward chance; if the effect survives resampling (e.g. because it
comes from contrast/sequence/field-strength, not voxel spacing), it will not.

Scope (fixed by request):
  * ISPY2 only.
  * GRAPH feature arm only (the arm available for both native and resampled).
  * Matched cases: the intersection present in both feature sets.

Batch labels (the thing we try to predict):
  * scanner manufacturer  (GE / Philips / SIEMENS) -- 3-class
  * field strength        (1.5T / 3.0T)            -- binary

For each (feature_set x batch_label x model) we report macro one-vs-rest ROC AUC
and balanced accuracy under 5-fold stratified CV, alongside a stratified-dummy
chance baseline. AUC ~ 0.5 (or balanced accuracy ~ 1/n_classes) means "not
detectable"; AUC well above 0.5 means "the batch is written into the features".

Usage::

    python scripts/batch_detectability_ispy2.py \
        --native   .../tabular/all_datasets_run/features_full_labeled.csv \
        --resampled .../independent_signal_resampled_ispy2/features_full_labeled.csv \
        --clinical .../MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx \
        --outdir   /ess/scratch/scratch1/t-9sbose/ispy2_batch_detectability
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATASET = "ISPY2"
ID_COL = "case_id"
N_SPLITS = 5
N_CLASSES_BINARY = 2
# The batch labels we try to predict, and where they come from in the xlsx.
BATCH_LABELS = ("manufacturer", "field_strength")


def load_ispy2_features(path: Path, prefixes: tuple[str, ...]) -> pd.DataFrame:
    """Load an ISPY2 feature file and keep case_id + the selected feature arms."""
    feats = pd.read_csv(path, low_memory=False)
    if "dataset" in feats.columns:
        feats = feats[feats["dataset"] == DATASET]
    feat_cols = sorted(c for c in feats.columns if c.startswith(prefixes))
    return feats[[ID_COL, *feat_cols]].copy()


def macro_auc(y_int: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Macro one-vs-rest ROC AUC that also handles the binary case.

    ``y_int`` is label-encoded 0..K-1 (sorted classes); ``proba`` columns are in
    that same class order (guaranteed because every fold sees every class).
    """
    if n_classes == N_CLASSES_BINARY:
        return float(roc_auc_score(y_int, proba[:, 1]))
    return float(
        roc_auc_score(
            y_int,
            proba,
            multi_class="ovr",
            average="macro",
            labels=list(range(n_classes)),
        )
    )


def make_model(name: str, seed: int) -> Pipeline:
    """Impute (median) -> standardize -> classifier, fit fresh in every fold."""
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(name)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def evaluate(
    X: np.ndarray,
    y_int: np.ndarray,
    n_classes: int,
    model: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[float, float, list[float]]:
    """Return (pooled macro AUC, pooled balanced accuracy, per-fold AUCs).

    We fit one model per fold and score that fold's held-out cases on its own, so
    each fold yields an independent AUC (this is what the error bars summarize).
    We also accumulate the out-of-fold predictions to report the pooled AUC/
    balanced-accuracy the bar plot used, so both views come from one pass.
    """
    per_fold: list[float] = []
    oof_proba = np.zeros((len(y_int), n_classes))
    oof_pred = np.zeros(len(y_int), dtype=int)
    for train_idx, test_idx in folds:
        pipe = make_model(model, seed)  # fresh, unfitted model each fold
        pipe.fit(X[train_idx], y_int[train_idx])
        proba = pipe.predict_proba(X[test_idx])
        oof_proba[test_idx] = proba
        oof_pred[test_idx] = pipe.predict(X[test_idx])
        per_fold.append(macro_auc(y_int[test_idx], proba, n_classes))
    pooled_auc = macro_auc(y_int, oof_proba, n_classes)
    pooled_bacc = float(balanced_accuracy_score(y_int, oof_pred))
    return pooled_auc, pooled_bacc, per_fold


def chance(
    y_int: np.ndarray,
    n_classes: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[float, float]:
    """Empirical chance level from a stratified dummy over the same folds."""
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    proba = cross_val_predict(
        dummy, np.zeros((len(y_int), 1)), y_int, cv=folds, method="predict_proba"
    )
    preds = cross_val_predict(
        dummy, np.zeros((len(y_int), 1)), y_int, cv=folds, method="predict"
    )
    return macro_auc(y_int, proba, n_classes), float(
        balanced_accuracy_score(y_int, preds)
    )


def main() -> None:
    """Run native-vs-resampled batch detectability and write outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--resampled", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--feature-prefixes",
        default="graph_",
        help="comma-separated feature-arm prefixes, e.g. 'morph_,kinematic_'",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(p for p in args.feature_prefixes.split(",") if p)
    log.info("Feature arms: %s", prefixes)

    # --- Assemble matched, aligned native & resampled feature matrices --------
    native = load_ispy2_features(args.native, prefixes)
    resampled = load_ispy2_features(args.resampled, prefixes)
    feat_cols = sorted(set(native.columns) & set(resampled.columns) - {ID_COL})
    shared_ids = sorted(set(native[ID_COL]) & set(resampled[ID_COL]))
    log.info(
        "Matched cases: %d | shared features: %d",
        len(shared_ids),
        len(feat_cols),
    )

    # Index by case_id and reindex both to the exact same case order.
    native = native.set_index(ID_COL).loc[shared_ids, feat_cols]
    resampled = resampled.set_index(ID_COL).loc[shared_ids, feat_cols]

    # --- Batch labels for those cases (from the imaging metadata) -------------
    xl = pd.read_excel(args.clinical, sheet_name="dataset_info")
    xl = xl.set_index("patient_id").loc[shared_ids]
    feature_sets = {"native": native.to_numpy(), "resampled": resampled.to_numpy()}

    rows = []
    fold_rows = []  # long-format per-fold AUCs (for error bars / poster plot)
    for label in BATCH_LABELS:
        # Label-encode the batch variable to 0..K-1 by sorted class value.
        classes = sorted(xl[label].dropna().unique(), key=str)
        code = {c: i for i, c in enumerate(classes)}
        y_int = xl[label].map(code).to_numpy()
        n_classes = len(classes)
        # One fixed fold assignment per label -> identical across feature sets.
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
        folds = list(skf.split(np.zeros(len(y_int)), y_int))
        counts = xl[label].value_counts().to_dict()
        log.info("\n== batch label '%s' (%d classes): %s ==", label, n_classes, counts)

        ch_auc, ch_bacc = chance(y_int, n_classes, folds, args.seed)
        for model in ("logreg", "rf"):
            for fs_name, X in feature_sets.items():
                auc, bacc, fold_aucs = evaluate(
                    X, y_int, n_classes, model, folds, args.seed
                )
                for k, fa in enumerate(fold_aucs):
                    fold_rows.append(
                        {
                            "batch_label": label,
                            "model": model,
                            "feature_set": fs_name,
                            "fold": k,
                            "macro_auc": fa,
                        }
                    )
                rows.append(
                    {
                        "batch_label": label,
                        "model": model,
                        "feature_set": fs_name,
                        "n_classes": n_classes,
                        "macro_auc": auc,  # pooled out-of-fold AUC
                        "auc_mean_folds": float(np.mean(fold_aucs)),
                        "auc_std_folds": float(np.std(fold_aucs, ddof=1)),
                        "balanced_acc": bacc,
                        "chance_auc": ch_auc,
                        "chance_balanced_acc": ch_bacc,
                    }
                )
                log.info(
                    "  %-6s %-9s AUC=%.3f  bacc=%.3f  (fold mean=%.3f ± %.3f)",
                    model,
                    fs_name,
                    auc,
                    bacc,
                    np.mean(fold_aucs),
                    np.std(fold_aucs, ddof=1),
                )

    results = pd.DataFrame(rows)
    results.to_csv(args.outdir / "batch_detectability_results.csv", index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(args.outdir / "batch_detectability_per_fold.csv", index=False)

    # --- native-vs-resampled delta per (label, model) -------------------------
    wide = results.pivot_table(
        index=["batch_label", "model"], columns="feature_set", values="macro_auc"
    ).reset_index()
    wide["auc_drop_native_minus_resampled"] = wide["native"] - wide["resampled"]
    wide.to_csv(args.outdir / "batch_detectability_delta.csv", index=False)

    arm = "+".join(p.rstrip("_") for p in prefixes)
    _plot(results, args.outdir / "batch_detectability.png", arm)
    _plot_folds(results, args.outdir / "batch_detectability_folds.png", arm)
    _write_summary(
        results,
        wide,
        len(shared_ids),
        len(feat_cols),
        arm,
        args.outdir / "SUMMARY.txt",
    )
    log.info("\n=== results ===\n%s", results.to_string(index=False))
    log.info("\n=== native - resampled AUC ===\n%s", wide.to_string(index=False))
    log.info("\nAll outputs written to: %s", args.outdir)


def _plot(results: pd.DataFrame, path: Path, arm: str) -> None:
    """Grouped bars: native vs resampled macro AUC per (label, model)."""
    combos = results[["batch_label", "model"]].drop_duplicates().to_numpy()
    labels = [f"{bl}\n{m}" for bl, m in combos]
    x = np.arange(len(combos))
    width = 0.38
    nat = [
        results.query("batch_label==@bl and model==@m and feature_set=='native'")[
            "macro_auc"
        ].iloc[0]
        for bl, m in combos
    ]
    res = [
        results.query("batch_label==@bl and model==@m and feature_set=='resampled'")[
            "macro_auc"
        ].iloc[0]
        for bl, m in combos
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, nat, width, label="native", color="steelblue")
    ax.bar(x + width / 2, res, width, label="resampled", color="darkorange")
    ax.axhline(0.5, ls="--", color="0.5", label="chance (AUC 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("macro one-vs-rest ROC AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        f"ISPY2 batch detectability from {arm} features\n"
        "(higher = stronger scanner fingerprint)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Human-readable names for the batch labels, used in axis grouping.
_BATCH_PRETTY = {
    "manufacturer": "Scanner manufacturer\n(3-class)",
    "field_strength": "Field strength\n(2-class)",
}
_MODEL_PRETTY = {"logreg": "logistic reg.", "rf": "random forest"}
_NATIVE_COLOR = "#2c6fbb"
_RESAMPLED_COLOR = "#e8802a"


def _plot_folds(results: pd.DataFrame, path: Path, arm: str) -> None:
    """Poster-quality per-fold detectability figure.

    Points = per-fold mean macro AUC, error bars = +/-1 SD across CV folds.
    Native and resampled are drawn as a connected pair per condition so the
    resampling drop (and its magnitude) reads at a glance; conditions are grouped
    by batch variable with a light divider.
    """
    plt.rcParams.update({"font.size": 12})
    combos = results[["batch_label", "model"]].drop_duplicates().to_numpy()
    x = np.arange(len(combos))
    offset = 0.16

    def pick(bl: str, m: str, fs: str, col: str) -> float:
        """Look up one scalar via boolean mask (avoids .query scope issues)."""
        mask = (
            (results["batch_label"] == bl)
            & (results["model"] == m)
            & (results["feature_set"] == fs)
        )
        return float(results.loc[mask, col].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Shade the "chance" zone and draw a light horizontal grid for readability.
    ax.axhspan(0.4, 0.5, color="0.92", zorder=0)
    ax.axhline(0.5, ls="--", color="0.55", lw=1.3, zorder=1)
    ax.text(
        len(combos) - 0.55,
        0.505,
        "chance (AUC 0.5)",
        color="0.4",
        fontsize=10,
        va="bottom",
        ha="right",
    )
    ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    for i, (bl, m) in enumerate(combos):
        n_mean, n_std = (
            pick(bl, m, "native", "auc_mean_folds"),
            pick(bl, m, "native", "auc_std_folds"),
        )
        r_mean, r_std = (
            pick(bl, m, "resampled", "auc_mean_folds"),
            pick(bl, m, "resampled", "auc_std_folds"),
        )
        # Faint connector showing the native -> resampled drop, with delta label.
        ax.plot(
            [x[i] - offset, x[i] + offset],
            [n_mean, r_mean],
            color="0.6",
            lw=1.2,
            zorder=2,
        )
        ax.annotate(
            f"Δ {r_mean - n_mean:+.02f}",
            xy=(x[i], (n_mean + r_mean) / 2),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color="0.35",
            va="center",
        )
        ax.errorbar(
            x[i] - offset,
            n_mean,
            yerr=n_std,
            fmt="o",
            capsize=5,
            markersize=9,
            color=_NATIVE_COLOR,
            zorder=3,
            label="native" if i == 0 else None,
        )
        ax.errorbar(
            x[i] + offset,
            r_mean,
            yerr=r_std,
            fmt="s",
            capsize=5,
            markersize=9,
            color=_RESAMPLED_COLOR,
            zorder=3,
            label="resampled" if i == 0 else None,
        )
        ax.annotate(
            f"{n_mean:.02f}",
            (x[i] - offset, n_mean + n_std),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
            color=_NATIVE_COLOR,
        )
        ax.annotate(
            f"{r_mean:.02f}",
            (x[i] + offset, r_mean - r_std),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color=_RESAMPLED_COLOR,
        )

    # Vertical dividers + a group header wherever the batch variable changes.
    batch_seq = [bl for bl, _ in combos]
    for i in range(1, len(batch_seq)):
        if batch_seq[i] != batch_seq[i - 1]:
            ax.axvline(i - 0.5, color="0.85", lw=1.0, zorder=0)
    for bl in dict.fromkeys(batch_seq):
        idx = [i for i, b in enumerate(batch_seq) if b == bl]
        ax.text(
            np.mean(idx),
            0.965,
            _BATCH_PRETTY.get(bl, bl),
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_PRETTY.get(m, m) for _, m in combos])
    ax.set_xlim(-0.5, len(combos) - 0.5)
    ax.set_ylabel("Scanner detectability (macro one-vs-rest ROC AUC)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        f"Vessel {arm} features carry a scanner fingerprint that resampling only "
        f"partly removes\nISPY2, matched cases; per-fold mean ± 1 SD over "
        f"{N_SPLITS} CV folds  (AUC 0.5 = undetectable, 1.0 = perfectly identifiable)",
        fontsize=12,
    )
    ax.legend(loc="lower left", frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    plt.rcParams.update({"font.size": 10})


def _write_summary(
    results: pd.DataFrame,
    wide: pd.DataFrame,
    n_cases: int,
    n_feats: int,
    arm: str,
    path: Path,
) -> None:
    """Human-readable summary of the detectability comparison."""
    with path.open("w") as fh:
        w = lambda s: fh.write(s + "\n")  # noqa: E731 - tiny local writer
        w(f"ISPY2 batch detectability — native vs resampled {arm} features")
        w("=" * 62)
        w(f"Matched cases: {n_cases} | {arm} features: {n_feats} | {N_SPLITS}-fold CV")
        w("")
        w("Macro one-vs-rest ROC AUC (chance = 0.5):")
        w(results.to_string(index=False))
        w("")
        w("Native minus resampled AUC (positive = resampling reduced detectability):")
        w(wide.to_string(index=False))
    log.info("\n%s", path.read_text())


if __name__ == "__main__":
    main()
