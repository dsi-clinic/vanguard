#!/usr/bin/env python3
"""PCA batch-effect investigation for the ISPY2 vessel-feature cohort.

Investigates whether a suspected acquisition batch effect (site / through-plane
z-spacing) dominates the ISPY2 native-spacing vessel features more than the
biological outcome of interest (pCR). The idea is simple: if the first couple of
principal components separate cases by *which hospital scanned them* (or by slice
thickness) rather than by *whether they responded to treatment*, that is evidence
of a batch effect that modeling needs to account for.

Scope (fixed by request):
  * ISPY2 cases only (DUKE / ISPY1 / NACT excluded).
  * NATIVE (non-resampled) feature outputs only.
  * Feature set = all vessel arms combined: morph + graph + kinematic
    (this union is what the pipeline calls "vessel_all").

Pipeline:
  1. Assemble the ISPY2 feature matrix and merge per-case z-spacing.
  2. Median-impute missing feature values, then z-score standardize.
  3. Run PCA; scree plot of variance explained (first ~10 components).
  4. Three PC1-vs-PC2 scatter plots (same coords): site, z-spacing, pCR.
  5. Top-15 loadings for PC1 and PC2.
  6. One-way ANOVA of PC1/PC2 ~ site and ~ pCR (R^2 = eta^2, and p-value).
  7. Alignment check: do site clusters track the z-spacing gradient?

Usage::

    python scripts/batch_effect/pca_ispy2_batch_effect.py \
        --features /gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/features_full_labeled_native.csv \
        --spacing  /gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/spacing_by_hospital.csv \
        --outdir   /ess/scratch/scratch1/t-9sbose/ispy2_pca_batch_effect
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: we only write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# --- fixed analysis choices (confirmed with the user before building) ---------
DATASET = "ISPY2"
# The three vessel feature arms. Their union is "vessel_all".
FEATURE_PREFIXES = ("morph_", "graph_", "kinematic_")
# Metadata columns already merged into the feature file by the tabular run.
SITE_COL = "site"
PCR_COL = "pcr"
ID_COL = "case_id"
# z-spacing source: actual spacing read from the NATIVE NIfTI headers.
SPACING_ID_COL = "patient_id"
SPACING_COL = "z_spacing_mm"
N_SCREE = 10  # number of components to show in the scree plot
N_TOP_LOADINGS = 15
MIN_GROUPS_FOR_ANOVA = 2  # need >=2 groups for a between-group comparison


def select_feature_columns(feats: pd.DataFrame) -> list[str]:
    """Return the vessel-feature columns (morph + graph + kinematic)."""
    return [c for c in feats.columns if c.startswith(FEATURE_PREFIXES)]


def eta_squared_and_p(values: np.ndarray, groups: pd.Series) -> tuple[float, float]:
    """One-way ANOVA of a continuous score across categorical groups.

    Returns (R^2, p_value) where R^2 is eta-squared: the fraction of the score's
    total variance that lies *between* groups. eta^2 near 1 means the grouping
    (e.g. hospital) explains almost all the spread in that PC; near 0 means it
    explains almost none. This is the standard ANOVA effect size.
    """
    # Build one array of scores per group, dropping any NaN group labels.
    frame = pd.DataFrame({"y": values, "g": groups.to_numpy()}).dropna()
    per_group = [g["y"].to_numpy() for _, g in frame.groupby("g")]
    per_group = [a for a in per_group if len(a) > 0]
    if len(per_group) < MIN_GROUPS_FOR_ANOVA:
        return float("nan"), float("nan")

    grand_mean = frame["y"].mean()
    ss_total = float(((frame["y"] - grand_mean) ** 2).sum())
    ss_between = float(sum(len(a) * (a.mean() - grand_mean) ** 2 for a in per_group))
    r2 = ss_between / ss_total if ss_total > 0 else float("nan")
    # scipy's f_oneway gives the F-test p-value for "all group means equal".
    _, p_value = stats.f_oneway(*per_group)
    return r2, float(p_value)


def main() -> None:
    """Run the full ISPY2 PCA batch-effect analysis and write all outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--spacing", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- 1. Assemble the ISPY2 feature matrix ---------------------------------
    feats = pd.read_csv(args.features, low_memory=False)
    feats = feats[feats["dataset"] == DATASET].copy()
    feat_cols = select_feature_columns(feats)
    log.info(
        "ISPY2 cases: %d | vessel features (morph+graph+kinematic): %d",
        len(feats),
        len(feat_cols),
    )
    arm_counts = {
        p.rstrip("_"): sum(c.startswith(p) for c in feat_cols) for p in FEATURE_PREFIXES
    }
    log.info("  feature-arm breakdown: %s", arm_counts)

    # Merge per-case z-spacing (header-derived, native NIfTIs).
    spacing = pd.read_csv(args.spacing)[[SPACING_ID_COL, SPACING_COL]]
    feats = feats.merge(spacing, left_on=ID_COL, right_on=SPACING_ID_COL, how="left")
    n_missing_spacing = int(feats[SPACING_COL].isna().sum())
    log.info("  cases missing z-spacing after merge: %d", n_missing_spacing)

    # --- Missing-value report (no silent dropping) ----------------------------
    X_raw = feats[feat_cols]
    n_cases_any_missing = int(X_raw.isna().any(axis=1).sum())
    total_cells = X_raw.size
    n_missing_cells = int(X_raw.isna().sum().sum())
    log.info(
        "Missing values: %d/%d cases have >=1 missing feature; "
        "%d/%d cells missing (%.3f%%). Handling: per-feature median imputation.",
        n_cases_any_missing,
        len(feats),
        n_missing_cells,
        total_cells,
        100 * n_missing_cells / total_cells,
    )

    # --- 2. Median-impute, then z-score standardize ---------------------------
    # Impute first (fill each column's NaNs with that column's median), then
    # standardize so every feature has mean 0 / std 1. Standardizing matters for
    # PCA: without it, features with large numeric ranges would dominate purely
    # because of their units, not because they carry more signal.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X_raw.to_numpy()))

    # --- 3. PCA ---------------------------------------------------------------
    pca = PCA(random_state=args.seed)
    scores = pca.fit_transform(X)  # rows = cases, cols = principal components
    feats["PC1"] = scores[:, 0]
    feats["PC2"] = scores[:, 1]
    evr = pca.explained_variance_ratio_

    # Scree plot: how much variance each of the first N components explains.
    k = min(N_SCREE, len(evr))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(np.arange(1, k + 1), evr[:k] * 100, color="steelblue", label="Per component")
    ax.plot(
        np.arange(1, k + 1),
        np.cumsum(evr[:k]) * 100,
        "o-",
        color="darkorange",
        label="Cumulative",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"ISPY2 vessel-feature PCA — scree (first {k} PCs)")
    ax.set_xticks(np.arange(1, k + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "scree_plot.png", dpi=150)
    plt.close(fig)

    # --- 4. Three PC1-vs-PC2 scatter plots (identical coordinates) ------------
    # A few cases have extreme PC2 values (heavy-tailed kinematic ratios) that
    # would otherwise squash the bulk of the cohort into an unreadable band. We
    # clip the *view* to a robust 1st-99th percentile window so the plots stay
    # legible. This does NOT change the PCA or any statistic -- every case still
    # contributed to the fit; a handful simply fall outside the plotted frame.
    lims = _robust_limits(feats, pad=0.05)
    n_offview = int(
        (
            (feats["PC1"] < lims[0][0])
            | (feats["PC1"] > lims[0][1])
            | (feats["PC2"] < lims[1][0])
            | (feats["PC2"] > lims[1][1])
        ).sum()
    )
    log.info(
        "Scatter view clipped to 1-99pct window; %d/%d cases off-view.",
        n_offview,
        len(feats),
    )
    _scatter_by_site(feats, args.outdir / "pc1_pc2_by_site.png", evr, lims, n_offview)
    _scatter_by_spacing(
        feats, args.outdir / "pc1_pc2_by_zspacing.png", evr, lims, n_offview
    )
    _scatter_by_pcr(feats, args.outdir / "pc1_pc2_by_pcr.png", evr, lims, n_offview)

    # --- 5. Top loadings for PC1 and PC2 --------------------------------------
    # A "loading" is a feature's weight in a principal component. Large absolute
    # loading = that feature strongly drives that PC. We report the 15 biggest.
    loadings = pd.DataFrame(
        pca.components_[:2].T, index=feat_cols, columns=["PC1", "PC2"]
    )
    top_rows = []
    for pc in ("PC1", "PC2"):
        top = (
            loadings[pc]
            .reindex(loadings[pc].abs().sort_values(ascending=False).index)
            .head(N_TOP_LOADINGS)
        )
        for rank, (feat, val) in enumerate(top.items(), start=1):
            top_rows.append(
                {"component": pc, "rank": rank, "feature": feat, "loading": val}
            )
    top_loadings = pd.DataFrame(top_rows)
    top_loadings.to_csv(args.outdir / "top_loadings.csv", index=False)

    # --- 6. ANOVA: PC1/PC2 ~ site and ~ pCR -----------------------------------
    anova_rows = []
    for pc in ("PC1", "PC2"):
        for factor, label in ((SITE_COL, "site"), (PCR_COL, "pCR")):
            r2, p = eta_squared_and_p(feats[pc].to_numpy(), feats[factor])
            anova_rows.append(
                {"component": pc, "factor": label, "R2": r2, "p_value": p}
            )
    anova = pd.DataFrame(anova_rows)
    anova.to_csv(args.outdir / "anova_summary.csv", index=False)

    # --- 7. Alignment check: do site clusters track the z-spacing gradient? ---
    # Three quantities tell the story:
    #   a) how strongly PC1/PC2 correlate with z-spacing case-by-case (Pearson r);
    #   b) how much of z-spacing's variance is explained by site (eta^2) --
    #      i.e. is spacing itself basically a site property?
    #   c) at the cluster level, do site-mean PC coords track site-mean spacing?
    valid = feats[["PC1", "PC2", SPACING_COL, SITE_COL]].dropna()
    r_pc1, p_pc1 = stats.pearsonr(valid["PC1"], valid[SPACING_COL])
    r_pc2, p_pc2 = stats.pearsonr(valid["PC2"], valid[SPACING_COL])
    spacing_by_site_r2, _ = eta_squared_and_p(
        valid[SPACING_COL].to_numpy(), valid[SITE_COL]
    )
    site_means = valid.groupby(SITE_COL).agg(
        PC1=("PC1", "mean"),
        PC2=("PC2", "mean"),
        z=(SPACING_COL, "mean"),
        n=("PC1", "size"),
    )
    # Correlate site-mean PC location with site-mean spacing (weighted view via
    # simple Pearson across sites; sites are the unit here).
    cr_pc1 = stats.pearsonr(site_means["PC1"], site_means["z"])[0]
    cr_pc2 = stats.pearsonr(site_means["PC2"], site_means["z"])[0]
    site_means.to_csv(args.outdir / "site_means.csv")

    # Persist the merged per-case table used for the plots (reproducibility).
    keep = [ID_COL, SITE_COL, PCR_COL, SPACING_COL, "PC1", "PC2"]
    feats[keep].to_csv(args.outdir / "ispy2_pca_scores.csv", index=False)

    # --- Written summary ------------------------------------------------------
    summary_path = args.outdir / "SUMMARY.txt"
    with summary_path.open("w") as fh:
        w = lambda s: fh.write(s + "\n")  # noqa: E731 - tiny local writer
        w("ISPY2 PCA batch-effect investigation")
        w("=" * 42)
        w(f"Cases: {len(feats)} ISPY2 (native, non-resampled)")
        w(f"Features: {len(feat_cols)} = {arm_counts}")
        w(
            f"Missing: {n_cases_any_missing}/{len(feats)} cases had >=1 missing value; "
            f"{100 * n_missing_cells / total_cells:.3f}% of cells median-imputed."
        )
        w("")
        w("Variance explained (first components):")
        for i in range(min(N_SCREE, len(evr))):
            w(
                f"  PC{i + 1}: {evr[i] * 100:5.2f}%  (cumulative "
                f"{np.cumsum(evr)[i] * 100:5.2f}%)"
            )
        w("")
        w("ANOVA  (R^2 = fraction of PC variance explained by the factor):")
        w(f"  {'component':>9} {'factor':>5} {'R^2':>8} {'p_value':>12}")
        for _, r in anova.iterrows():
            w(
                f"  {r['component']:>9} {r['factor']:>5} {r['R2']:>8.4f} "
                f"{r['p_value']:>12.3e}"
            )
        w("")
        w("Site-vs-spacing alignment:")
        w(f"  Pearson r(PC1, z-spacing) = {r_pc1:+.3f} (p={p_pc1:.2e})")
        w(f"  Pearson r(PC2, z-spacing) = {r_pc2:+.3f} (p={p_pc2:.2e})")
        w(f"  z-spacing variance explained by site (eta^2) = {spacing_by_site_r2:.3f}")
        w(
            f"  site-mean r(PC1, z-spacing) = {cr_pc1:+.3f} | "
            f"r(PC2, z-spacing) = {cr_pc2:+.3f}"
        )
        w("")
        w("Top loadings written to top_loadings.csv; ANOVA to anova_summary.csv.")
    log.info("\n%s", summary_path.read_text())

    # Console tables for quick reading in the Slurm log.
    log.info("\n=== ANOVA summary ===\n%s", anova.to_string(index=False))
    log.info(
        "\n=== Top %d loadings (PC1) ===\n%s",
        N_TOP_LOADINGS,
        top_loadings[top_loadings["component"] == "PC1"].to_string(index=False),
    )
    log.info(
        "\n=== Top %d loadings (PC2) ===\n%s",
        N_TOP_LOADINGS,
        top_loadings[top_loadings["component"] == "PC2"].to_string(index=False),
    )
    log.info("\nAll outputs written to: %s", args.outdir)


def _pc_axis_labels(evr: np.ndarray) -> tuple[str, str]:
    """Axis labels annotated with each PC's variance explained."""
    return (f"PC1 ({evr[0] * 100:.1f}% var)", f"PC2 ({evr[1] * 100:.1f}% var)")


Limits = tuple[tuple[float, float], tuple[float, float]]


def _robust_limits(feats: pd.DataFrame, pad: float = 0.05) -> Limits:
    """Return ((xmin, xmax), (ymin, ymax)) from the 1st-99th PC percentiles."""
    out = []
    for col in ("PC1", "PC2"):
        lo, hi = np.percentile(feats[col], [1, 99])
        span = hi - lo
        out.append((lo - pad * span, hi + pad * span))
    return out[0], out[1]


def _apply_view(ax: plt.Axes, lims: Limits, n_offview: int) -> None:
    """Apply the shared clipped view and note how many cases fall outside it."""
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    if n_offview:
        ax.text(
            0.99,
            0.01,
            f"{n_offview} outlier case(s) off-view",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.4",
        )


def _scatter_by_site(
    feats: pd.DataFrame, path: Path, evr: np.ndarray, lims: Limits, n_offview: int
) -> None:
    """PC1 vs PC2 colored by hospital/site (categorical)."""
    xlab, ylab = _pc_axis_labels(evr)
    sites = sorted(feats[SITE_COL].dropna().unique())
    cmap = plt.get_cmap("tab20", max(len(sites), 1))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i, s in enumerate(sites):
        m = feats[SITE_COL] == s
        ax.scatter(
            feats.loc[m, "PC1"],
            feats.loc[m, "PC2"],
            s=14,
            alpha=0.7,
            color=cmap(i),
            label=str(s),
        )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("ISPY2 PCA — colored by site")
    _apply_view(ax, lims, n_offview)
    ax.legend(
        title="site",
        fontsize=7,
        ncol=2,
        markerscale=1.5,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter_by_spacing(
    feats: pd.DataFrame, path: Path, evr: np.ndarray, lims: Limits, n_offview: int
) -> None:
    """PC1 vs PC2 colored by z-spacing (continuous colormap)."""
    xlab, ylab = _pc_axis_labels(evr)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(
        feats["PC1"],
        feats["PC2"],
        c=feats[SPACING_COL],
        s=16,
        alpha=0.8,
        cmap="viridis",
    )
    fig.colorbar(sc, ax=ax, label="z-spacing (mm)")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("ISPY2 PCA — colored by z-spacing")
    _apply_view(ax, lims, n_offview)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter_by_pcr(
    feats: pd.DataFrame, path: Path, evr: np.ndarray, lims: Limits, n_offview: int
) -> None:
    """PC1 vs PC2 colored by pCR outcome (0 vs 1)."""
    xlab, ylab = _pc_axis_labels(evr)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = {0: "tab:blue", 1: "tab:red"}
    labels = {0: "non-pCR (0)", 1: "pCR (1)"}
    for val, color in colors.items():
        m = feats[PCR_COL] == val
        ax.scatter(
            feats.loc[m, "PC1"],
            feats.loc[m, "PC2"],
            s=16,
            alpha=0.6,
            color=color,
            label=labels[val],
        )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("ISPY2 PCA — colored by pCR")
    _apply_view(ax, lims, n_offview)
    ax.legend(title="outcome")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
