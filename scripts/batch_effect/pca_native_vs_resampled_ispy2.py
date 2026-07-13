#!/usr/bin/env python3
"""Side-by-side native vs resampled PCA of ISPY2 vessel features.

Runs PCA separately on the NATIVE and on the RESAMPLED vessel features (same
matched ISPY2 cases, all arms: morph + graph + kinematic), then draws PC1-vs-PC2
scatter plots colored by each metadata variable, native panel next to resampled
panel, so you can see whether resampling changes how cases cluster.

Coloring variables (one side-by-side figure each):
  * pCR outcome
  * scanner manufacturer   (the batch variable of interest)
  * site
  * z-spacing (voxel through-plane spacing)

Notes:
  * PCA is fit independently on each source, so the two panels live in their own
    PC spaces. PCA has an arbitrary sign per component, so we flip the resampled
    components to line up with the native ones (same cases) purely for readability.
  * Each panel is clipped to its own 1st-99th percentile window so a few extreme
    cases do not squash the view (every case still contributes to the fit).

Usage::

    python scripts/batch_effect/pca_native_vs_resampled_ispy2.py \
        --native   .../tabular/all_datasets_run/features_full_labeled.csv \
        --resampled .../independent_signal_resampled_ispy2/features_full_labeled.csv \
        --spacing  .../data_viz/spacing_by_hospital.csv \
        --clinical .../MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx \
        --outdir   /ess/scratch/scratch1/t-9sbose/ispy2_pca_native_vs_resampled
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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATASET = "ISPY2"
ID_COL = "case_id"
FEATURE_PREFIXES = ("morph_", "graph_", "kinematic_")
# (metadata column, pretty label, kind) for each side-by-side figure.
COLOR_VARS = [
    ("pcr", "pCR", "binary"),
    ("manufacturer", "scanner manufacturer", "categorical"),
    ("site", "site", "categorical"),
    ("z_spacing_mm", "z-spacing (mm)", "continuous"),
]


def load_ispy2(path: Path) -> pd.DataFrame:
    """Load and return ISPY2 rows of a feature file, indexed by case_id."""
    feats = pd.read_csv(path, low_memory=False)
    if "dataset" in feats.columns:
        feats = feats[feats["dataset"] == DATASET]
    return feats.set_index(ID_COL)


def pca_scores(feats: pd.DataFrame, cols: list[str], seed: int) -> np.ndarray:
    """Median-impute, z-score, then return the first two PCA components."""
    x = SimpleImputer(strategy="median").fit_transform(feats[cols].to_numpy())
    x = StandardScaler().fit_transform(x)
    return PCA(n_components=2, random_state=seed).fit_transform(x)


_MAX_TAB10 = 10


def _robust_lim(v: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    """(min, max) from the 1st-99th percentiles, padded, for a clean view."""
    lo, hi = np.percentile(v, [1, 99])
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _plot_panel(
    ax: plt.Axes, coords: np.ndarray, values: pd.Series, kind: str, title: str
) -> object:
    """Draw one PC1-vs-PC2 panel colored by `values`; return a mappable/None."""
    pc1, pc2 = coords[:, 0], coords[:, 1]
    mappable = None
    if kind == "continuous":
        mappable = ax.scatter(
            pc1, pc2, c=values.to_numpy(dtype=float), s=16, alpha=0.8, cmap="viridis"
        )
    else:
        cats = sorted(pd.Series(values).dropna().unique(), key=str)
        cmap = plt.get_cmap(
            "tab10" if len(cats) <= _MAX_TAB10 else "tab20", max(len(cats), 1)
        )
        for i, c in enumerate(cats):
            m = (values == c).to_numpy()
            ax.scatter(pc1[m], pc2[m], s=16, alpha=0.7, color=cmap(i), label=str(c))
    ax.set_xlim(*_robust_lim(pc1))
    ax.set_ylim(*_robust_lim(pc2))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    return mappable


def main() -> None:
    """Build native-vs-resampled PCA figures colored by each metadata variable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--resampled", type=Path, required=True)
    parser.add_argument("--spacing", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    native = load_ispy2(args.native)
    resampled = load_ispy2(args.resampled)
    cols = sorted(
        {c for c in native.columns if c.startswith(FEATURE_PREFIXES)}
        & {c for c in resampled.columns if c.startswith(FEATURE_PREFIXES)}
    )
    ids = sorted(set(native.index) & set(resampled.index))
    native, resampled = native.loc[ids], resampled.loc[ids]
    log.info("Matched cases: %d | vessel features: %d", len(ids), len(cols))

    # --- Metadata for coloring (aligned to the matched case order) ------------
    meta = pd.DataFrame(index=ids)
    meta["pcr"] = native["pcr"].to_numpy()
    meta["site"] = native["site"].to_numpy()
    xl = pd.read_excel(args.clinical, sheet_name="dataset_info").set_index("patient_id")
    meta["manufacturer"] = xl["manufacturer"].reindex(ids).to_numpy()
    sp = pd.read_csv(args.spacing).set_index("patient_id")
    meta["z_spacing_mm"] = sp["z_spacing_mm"].reindex(ids).to_numpy()

    # --- PCA per source, with sign alignment for readability ------------------
    nat_pc = pca_scores(native, cols, args.seed)
    res_pc = pca_scores(resampled, cols, args.seed)
    for j in range(2):  # flip resampled component if it anti-correlates with native
        if stats.pearsonr(nat_pc[:, j], res_pc[:, j])[0] < 0:
            res_pc[:, j] *= -1

    # --- One side-by-side figure per coloring variable ------------------------
    for col, label, kind in COLOR_VARS:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=False, sharey=False)
        _plot_panel(axes[0], nat_pc, meta[col], kind, "Native")
        m_res = _plot_panel(axes[1], res_pc, meta[col], kind, "Resampled")
        if kind == "continuous":
            fig.colorbar(m_res, ax=axes, label=label, fraction=0.046, pad=0.02)
        else:
            handles, lbls = axes[0].get_legend_handles_labels()
            ncol = 2 if len(lbls) > _MAX_TAB10 else 1
            fig.legend(
                handles,
                lbls,
                title=label,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=8,
                ncol=ncol,
            )
        fig.suptitle(
            f"ISPY2 vessel-feature PCA colored by {label} — native vs resampled "
            f"(matched {len(ids)} cases)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        out = args.outdir / f"pca_native_vs_resampled_by_{col}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("wrote %s", out)

    log.info("All outputs written to: %s", args.outdir)


if __name__ == "__main__":
    main()
