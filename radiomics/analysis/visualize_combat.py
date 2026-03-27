#!/usr/bin/env python3
"""ComBat harmonization diagnostic visualizations.

Generates five diagnostic plots comparing feature distributions before and
after ComBat harmonization to assess whether site effects are reduced while
pCR signal is preserved.

Plots
-----
1. PCA scatter (before vs after) — colour by site, shape by pCR.
2. Feature distribution violins — top site-affected features.
3. Site-mean heatmap (before vs after).
4. Pairwise site-centroid distance heatmap (before vs after).
5. Signal-retained bar chart — fold-safe site-AUC vs pCR-AUC.

Usage
-----
    python visualize_combat.py \
        --features-dir outputs/shared_extraction/rerun_bin100_kinsubonly \
        --labels labels.csv \
        --output-dir outputs/combat_viz \
        --harmonization-mode combat_param \
        --batch-col site \
        --cv-folds 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Import harmonization helpers from radiomics_train.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from radiomics_train import (  # noqa: E402
    FeatureHarmonizer,
    load_features,
    load_labels,
    sanitize_numeric,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIN_GROUP_SIZE = 2
TOP_FEATURES_DEFAULT = 8
FOLD_COUNT_DEFAULT = 5
PLOT_DPI = 200
HEATMAP_EPS = 1e-8


def extract_site(pid: str) -> str:
    """Extract site prefix from patient ID (e.g. ISPY2_0042 → ISPY2)."""
    m = re.match(r"^([A-Za-z]+\d*)", pid)
    return m.group(1) if m else "UNKNOWN"


def load_and_align(
    features_dir: Path, labels_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training features and labels, align by patient_id index."""
    feat_path = features_dir / "features_train_final.csv"
    X = load_features(str(feat_path))
    labels = load_labels(str(labels_path))

    # Ensure site column exists
    if "site" not in labels.columns:
        labels["site"] = labels.index.map(extract_site)

    common = X.index.intersection(labels.index)
    X = X.loc[common]
    labels = labels.loc[common]

    # Drop non-numeric columns and sanitize
    X = X.select_dtypes(include=[np.number])
    X, _ = sanitize_numeric(X, "viz")

    return X, labels


def apply_combat_full(
    X: pd.DataFrame, labels: pd.DataFrame, mode: str, batch_col: str
) -> pd.DataFrame:
    """Fit ComBat on the full matrix (for qualitative visualisations only)."""
    medians = X.median(axis=0)
    X_imp = X.fillna(medians)

    harmonizer = FeatureHarmonizer(
        mode=mode,
        batch_col=batch_col,
        covariate_cols=[],
    ).fit(X_imp, labels)

    X_h, _ = harmonizer.transform(X_imp, labels)
    return X_h


def select_site_affected_features(
    X: pd.DataFrame, labels: pd.DataFrame, batch_col: str, n: int = 8
) -> list[str]:
    """Select top-n features with strongest site effect (ANOVA F-stat)."""
    sites = labels.loc[X.index, batch_col].astype(str)
    unique_sites = sites.unique()

    f_scores = {}
    for col in X.columns:
        groups = [X.loc[sites == s, col].dropna().to_numpy() for s in unique_sites]
        groups = [g for g in groups if len(g) >= MIN_GROUP_SIZE]
        if len(groups) >= MIN_GROUP_SIZE:
            try:
                f_stat, _ = f_oneway(*groups)
                if np.isfinite(f_stat):
                    f_scores[col] = f_stat
            except ValueError:
                pass

    ranked = sorted(f_scores, key=f_scores.get, reverse=True)
    return ranked[:n]


# ---------------------------------------------------------------------------
# Plot 1: PCA before vs after
# ---------------------------------------------------------------------------


def plot_pca_before_after(
    X_raw: pd.DataFrame,
    X_combat: pd.DataFrame,
    labels: pd.DataFrame,
    batch_col: str,
    output_path: Path,
) -> None:
    """Two-panel PCA scatter: colour = site, shape = pCR."""
    # Impute for PCA
    medians = X_raw.median(axis=0)
    Xr = X_raw.fillna(medians).to_numpy()
    Xc = X_combat.fillna(medians).to_numpy()

    # Standardize using raw stats (same basis for both panels)
    scaler = StandardScaler().fit(Xr)
    Xr_s = scaler.transform(Xr)
    Xc_s = scaler.transform(Xc)

    pca = PCA(n_components=2).fit(Xr_s)
    Zr = pca.transform(Xr_s)
    Zc = pca.transform(Xc_s)

    sites = labels.loc[X_raw.index, batch_col].astype(str).to_numpy()
    pcr = labels.loc[X_raw.index, "pcr"].astype(int).to_numpy()
    unique_sites = sorted(set(sites))
    colors = plt.cm.Set1.colors

    # Shared axis limits
    all_z = np.vstack([Zr, Zc])
    margin = 0.05
    xlim = (all_z[:, 0].min() * (1 + margin), all_z[:, 0].max() * (1 + margin))
    ylim = (all_z[:, 1].min() * (1 + margin), all_z[:, 1].max() * (1 + margin))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, Z, title in [(ax1, Zr, "Before ComBat"), (ax2, Zc, "After ComBat")]:
        for si, site in enumerate(unique_sites):
            c = colors[si % len(colors)]
            mask_site = sites == site

            # pCR = 0: circles, pCR = 1: triangles
            m0 = mask_site & (pcr == 0)
            m1 = mask_site & (pcr == 1)

            ax.scatter(
                Z[m0, 0],
                Z[m0, 1],
                c=[c],
                marker="o",
                alpha=0.5,
                s=25,
                label=f"{site} (pCR=0)",
            )
            ax.scatter(
                Z[m1, 0],
                Z[m1, 1],
                c=[c],
                marker="^",
                alpha=0.5,
                s=25,
                label=f"{site} (pCR=1)",
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(title)

    # Shared legend
    handles = []
    for si, site in enumerate(unique_sites):
        c = colors[si % len(colors)]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=c,
                markersize=8,
                label=site,
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            linestyle="None",
            markersize=8,
            label="pCR=0",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="grey",
            linestyle="None",
            markersize=8,
            label="pCR=1",
        )
    )
    fig.legend(
        handles=handles, loc="center right", fontsize=8, bbox_to_anchor=(1.0, 0.5)
    )

    fig.suptitle("PCA — site clustering before vs after ComBat", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved PCA plot → {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Feature distribution violins
# ---------------------------------------------------------------------------


def plot_feature_distributions(
    X_raw: pd.DataFrame,
    X_combat: pd.DataFrame,
    labels: pd.DataFrame,
    batch_col: str,
    top_features: list[str],
    output_path: Path,
) -> None:
    """Per-site violin plots for top site-affected features, before vs after."""
    n_feat = len(top_features)
    fig, axes = plt.subplots(n_feat, 2, figsize=(12, 2.5 * n_feat))
    if n_feat == 1:
        axes = axes.reshape(1, 2)

    sites = labels.loc[X_raw.index, batch_col].astype(str)

    for i, feat in enumerate(top_features):
        for j, (X, panel_title) in enumerate([(X_raw, "Before"), (X_combat, "After")]):
            ax = axes[i, j]
            if feat not in X.columns:
                ax.set_visible(False)
                continue

            df_plot = pd.DataFrame(
                {
                    "value": X[feat].to_numpy(),
                    "site": sites.to_numpy(),
                }
            )
            sns.violinplot(
                data=df_plot,
                x="site",
                y="value",
                ax=ax,
                cut=0,
                inner="quartile",
                density_norm="width",
                palette="Set1",
            )
            ax.set_title(f"{panel_title}: {feat[:40]}", fontsize=9)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45, labelsize=7)

    axes[0, 0].set_title("Before ComBat", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("After ComBat", fontsize=11, fontweight="bold")
    fig.suptitle(
        "Feature distributions by site (top site-affected features)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved feature distributions → {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Site-mean heatmap
# ---------------------------------------------------------------------------


def plot_site_mean_heatmap(
    X_raw: pd.DataFrame,
    X_combat: pd.DataFrame,
    labels: pd.DataFrame,
    batch_col: str,
    top_features: list[str],
    output_path: Path,
) -> None:
    """Heatmap of standardised per-site means for selected features."""
    sites = labels.loc[X_raw.index, batch_col].astype(str)
    unique_sites = sorted(sites.unique())

    def site_means(X: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for s in unique_sites:
            mask = (sites == s).to_numpy()
            rows.append(X.loc[mask, top_features].mean(axis=0))
        site_mean_df = pd.DataFrame(rows, index=unique_sites, columns=top_features)
        # Standardize columns for comparable heatmap
        return (site_mean_df - site_mean_df.mean()) / (site_mean_df.std() + HEATMAP_EPS)

    Z_raw = site_means(X_raw)
    Z_combat = site_means(X_combat)

    vmax = max(Z_raw.abs().max().max(), Z_combat.abs().max().max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4 + 0.3 * len(unique_sites)))

    # Shorten feature names for display
    short_names = [f[:25] for f in top_features]

    sns.heatmap(
        Z_raw.rename(columns=dict(zip(top_features, short_names))),
        ax=ax1,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    ax1.set_title("Before ComBat")
    ax1.set_xlabel("")

    sns.heatmap(
        Z_combat.rename(columns=dict(zip(top_features, short_names))),
        ax=ax2,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    ax2.set_title("After ComBat")
    ax2.set_xlabel("")

    fig.suptitle("Standardised site-mean feature values", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved site-mean heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: Pairwise site-centroid distance heatmap
# ---------------------------------------------------------------------------


def plot_site_distance_heatmap(
    X_raw: pd.DataFrame,
    X_combat: pd.DataFrame,
    labels: pd.DataFrame,
    batch_col: str,
    output_path: Path,
) -> None:
    """Pairwise Euclidean distance between site centroids, before vs after."""
    sites = labels.loc[X_raw.index, batch_col].astype(str)
    unique_sites = sorted(sites.unique())

    def centroid_distances(X: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler().fit(X)
        Xs = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        centroids = []
        for s in unique_sites:
            mask = (sites == s).to_numpy()
            centroids.append(Xs.loc[mask].mean(axis=0).to_numpy())
        centroids = np.array(centroids)

        return pd.DataFrame(
            pairwise_distances(centroids, metric="euclidean"),
            index=unique_sites,
            columns=unique_sites,
        )

    D_raw = centroid_distances(X_raw)
    D_combat = centroid_distances(X_combat)

    vmax = max(D_raw.max().max(), D_combat.max().max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        D_raw,
        ax=ax1,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax1.set_title("Before ComBat")

    sns.heatmap(
        D_combat,
        ax=ax2,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax2.set_title("After ComBat")

    fig.suptitle("Pairwise Euclidean distance between site centroids", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved site-distance heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Plot 5: Signal-retained bar chart (fold-safe)
# ---------------------------------------------------------------------------


def plot_signal_retained(
    X_raw: pd.DataFrame,
    labels: pd.DataFrame,
    output_path: Path,
    harmonization_mode: str,
    batch_col: str,
    cv_folds: int,
) -> None:
    """Bar chart comparing site-AUC and pCR-AUC before vs after ComBat.

    Uses fold-safe ComBat: fit on train fold only, transform both folds.
    """
    sites = labels.loc[X_raw.index, batch_col].astype(str)
    pcr = labels.loc[X_raw.index, "pcr"].astype(int).to_numpy()
    site_enc = LabelEncoder().fit(sites)
    y_site = site_enc.transform(sites)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = {
        "site_raw": [],
        "site_combat": [],
        "pcr_raw": [],
        "pcr_combat": [],
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_raw, pcr)):
        tr_ids = X_raw.index[tr_idx]
        val_ids = X_raw.index[val_idx]

        X_tr_raw = X_raw.loc[tr_ids]
        X_val_raw = X_raw.loc[val_ids]

        # Impute with train medians
        medians = X_tr_raw.median(axis=0)
        X_tr_imp = X_tr_raw.fillna(medians)
        X_val_imp = X_val_raw.fillna(medians)

        # Scale raw features
        scaler = StandardScaler().fit(X_tr_imp)
        X_tr_s = scaler.transform(X_tr_imp)
        X_val_s = scaler.transform(X_val_imp)

        # --- Raw features ---
        # Site classifier
        try:
            clf_site = LogisticRegression(max_iter=2000, random_state=42)
            clf_site.fit(X_tr_s, y_site[tr_idx])
            if len(np.unique(y_site[val_idx])) >= MIN_GROUP_SIZE:
                proba = clf_site.predict_proba(X_val_s)
                auc = roc_auc_score(y_site[val_idx], proba, multi_class="ovr")
                results["site_raw"].append(auc)
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"Site classifier failed on fold: {exc}", RuntimeWarning, stacklevel=1
            )

        # pCR classifier
        try:
            clf_pcr = LogisticRegression(max_iter=2000, random_state=42)
            clf_pcr.fit(X_tr_s, pcr[tr_idx])
            proba = clf_pcr.predict_proba(X_val_s)[:, 1]
            auc = roc_auc_score(pcr[val_idx], proba)
            results["pcr_raw"].append(auc)
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"pCR classifier failed on fold: {exc}", RuntimeWarning, stacklevel=1
            )

        # --- ComBat-harmonized features ---
        harmonizer = FeatureHarmonizer(
            mode=harmonization_mode,
            batch_col=batch_col,
            covariate_cols=[],
        )
        labels_tr = labels.loc[tr_ids]
        labels_val = labels.loc[val_ids]

        try:
            harmonizer.fit(X_tr_imp, labels_tr)
            X_tr_h, _ = harmonizer.transform(X_tr_imp, labels_tr)
            X_val_h, _ = harmonizer.transform(X_val_imp, labels_val)

            scaler_h = StandardScaler().fit(X_tr_h)
            X_tr_hs = scaler_h.transform(X_tr_h)
            X_val_hs = scaler_h.transform(X_val_h)

            # Site classifier
            clf_site_h = LogisticRegression(max_iter=2000, random_state=42)
            clf_site_h.fit(X_tr_hs, y_site[tr_idx])
            if len(np.unique(y_site[val_idx])) >= MIN_GROUP_SIZE:
                proba = clf_site_h.predict_proba(X_val_hs)
                auc = roc_auc_score(y_site[val_idx], proba, multi_class="ovr")
                results["site_combat"].append(auc)

            # pCR classifier
            clf_pcr_h = LogisticRegression(max_iter=2000, random_state=42)
            clf_pcr_h.fit(X_tr_hs, pcr[tr_idx])
            proba = clf_pcr_h.predict_proba(X_val_hs)[:, 1]
            auc = roc_auc_score(pcr[val_idx], proba)
            results["pcr_combat"].append(auc)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(
                f"Fold {fold_idx}: ComBat failed — {e}",
                RuntimeWarning,
                stacklevel=1,
            )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Site AUC", "pCR AUC"]
    x = np.arange(len(categories))
    width = 0.3

    means_before = [np.mean(results["site_raw"]), np.mean(results["pcr_raw"])]
    stds_before = [np.std(results["site_raw"]), np.std(results["pcr_raw"])]
    means_after = [np.mean(results["site_combat"]), np.mean(results["pcr_combat"])]
    stds_after = [np.std(results["site_combat"]), np.std(results["pcr_combat"])]

    bars1 = ax.bar(
        x - width / 2,
        means_before,
        width,
        yerr=stds_before,
        label="Before ComBat",
        color="#4C72B0",
        capsize=5,
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2,
        means_after,
        width,
        yerr=stds_after,
        label="After ComBat",
        color="#DD8452",
        capsize=5,
        alpha=0.85,
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("CV AUC (mean ± std)", fontsize=11)
    ax.set_title(
        f"Signal Retained — {harmonization_mode}\n" f"(fold-safe, {cv_folds}-fold CV)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved signal-retained chart → {output_path}")

    # Print numeric summary
    summary = {
        "harmonization_mode": harmonization_mode,
        "cv_folds": cv_folds,
        "site_auc_before": (
            f"{np.mean(results['site_raw']):.4f}"
            f" ± {np.std(results['site_raw']):.4f}"
        ),
        "site_auc_after": (
            f"{np.mean(results['site_combat']):.4f}"
            f" ± {np.std(results['site_combat']):.4f}"
        ),
        "pcr_auc_before": (
            f"{np.mean(results['pcr_raw']):.4f}" f" ± {np.std(results['pcr_raw']):.4f}"
        ),
        "pcr_auc_after": (
            f"{np.mean(results['pcr_combat']):.4f}"
            f" ± {np.std(results['pcr_combat']):.4f}"
        ),
    }
    summary_path = output_path.parent / "signal_retained_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[VIZ] Saved numeric summary → {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build ComBat diagnostic plots and numeric summaries."""
    parser = argparse.ArgumentParser(
        description="ComBat harmonization diagnostic visualizations.",
    )
    parser.add_argument(
        "--features-dir", required=True, help="Directory with features_train_final.csv"
    )
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output PNG files"
    )
    parser.add_argument(
        "--harmonization-mode",
        default="combat_param",
        choices=["zscore_site", "combat_param", "combat_nonparam"],
        help="Harmonization method (default: combat_param)",
    )
    parser.add_argument(
        "--batch-col",
        default="site",
        help="Column in labels for batch/site (default: site)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=FOLD_COUNT_DEFAULT,
        help="Number of CV folds for signal-retained plot (default: 5)",
    )
    parser.add_argument(
        "--n-top-features",
        type=int,
        default=TOP_FEATURES_DEFAULT,
        help="Number of top site-affected features to show (default: 8)",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[VIZ] Loading features from {features_dir}")
    X_raw, labels = load_and_align(features_dir, Path(args.labels))
    print(f"[VIZ] Loaded {X_raw.shape[0]} patients × {X_raw.shape[1]} features")
    print(f"[VIZ] Sites: {sorted(labels[args.batch_col].unique())}")

    # Apply ComBat on full training set (for qualitative plots 1-4)
    print(f"[VIZ] Applying {args.harmonization_mode} harmonization (full-set fit)...")
    X_combat = apply_combat_full(X_raw, labels, args.harmonization_mode, args.batch_col)

    # Select top site-affected features
    top_features = select_site_affected_features(
        X_raw,
        labels,
        args.batch_col,
        n=args.n_top_features,
    )
    print(f"[VIZ] Top {len(top_features)} site-affected features selected")

    # --- Generate all 5 plots ---

    print("\n[VIZ] === Plot 1: PCA before vs after ===")
    plot_pca_before_after(
        X_raw,
        X_combat,
        labels,
        args.batch_col,
        output_dir / "combat_pca_before_after.png",
    )

    print("\n[VIZ] === Plot 2: Feature distributions ===")
    plot_feature_distributions(
        X_raw,
        X_combat,
        labels,
        args.batch_col,
        top_features,
        output_dir / "combat_feature_distributions.png",
    )

    print("\n[VIZ] === Plot 3: Site-mean heatmap ===")
    plot_site_mean_heatmap(
        X_raw,
        X_combat,
        labels,
        args.batch_col,
        top_features,
        output_dir / "combat_site_mean_heatmap.png",
    )

    print("\n[VIZ] === Plot 4: Site-distance heatmap ===")
    plot_site_distance_heatmap(
        X_raw,
        X_combat,
        labels,
        args.batch_col,
        output_dir / "combat_site_distance_heatmap.png",
    )

    print("\n[VIZ] === Plot 5: Signal retained (fold-safe) ===")
    plot_signal_retained(
        X_raw,
        labels,
        output_dir / "combat_signal_retained.png",
        args.harmonization_mode,
        args.batch_col,
        args.cv_folds,
    )

    print(f"\n[VIZ] All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
