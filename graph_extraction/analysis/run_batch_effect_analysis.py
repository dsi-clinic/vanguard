"""Phase 3: Batch effect and nuisance analysis.

Dimensionality reduction (PCA, UMAP, t-SNE) colored by label/site/dataset/manufacturer,
and site-prediction classifier as batch-effect indicator.

Requires features_with_metadata.csv (from build_features_with_metadata.py).

Usage:
    python graph_extraction/analysis/run_batch_effect_analysis.py \
        --features-csv analysis/graph_extraction/features_with_metadata.csv \
        --output-dir analysis/graph_extraction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Metadata and label columns (not features)
META_COLS = {
    "case_id",
    "patient_id",
    "site",
    "dataset",
    "manufacturer",
    "model",
    "field_strength",
}
LABEL_COL = "pcr"
MAX_MISSING_PCT = 30
MAX_ROW_MISSING_PCT = 20
RANDOM_STATE = 42
PCA_N_COMPONENTS = 50
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITY = 30
N_FOLDS_SITE = 5
MIN_SITE_CLASSES = 2
AUC_STRONG_THRESHOLD = 0.8
AUC_MODERATE_THRESHOLD = 0.6


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding metadata and label."""
    exclude = META_COLS | {LABEL_COL}
    numeric = df.select_dtypes(include=[np.number])
    return [c for c in numeric.columns if c not in exclude]


def _prepare_matrix(
    df: pd.DataFrame,
    qc_csv: Path | None = None,
    max_feature_missing_pct: float = MAX_MISSING_PCT,
    max_row_missing_pct: float = MAX_ROW_MISSING_PCT,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Prepare case-level feature matrix after QC.

    - Drop constant columns
    - Optionally drop features with > max_feature_missing_pct missing (from qc_csv)
    - Impute remaining NaN with median
    - Drop rows with > max_row_missing_pct missing

    Returns:
        X: (n_samples, n_features) float array
        meta: DataFrame with case_id and metadata (site, dataset, etc.)
        feature_names: list of feature column names
    """
    feature_cols = _get_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns found")

    X_df = df[feature_cols].copy()

    # Drop constant columns (use len(unique) to avoid PD101: nunique is inefficient for this)
    const_mask = X_df.apply(lambda s: len(s.dropna().unique()) <= 1)
    non_const = [c for c in feature_cols if not const_mask.get(c, False)]
    X_df = X_df[non_const]

    # Optionally drop high-missing features from qc_per_feature.csv
    if qc_csv and qc_csv.exists():
        qc = pd.read_csv(qc_csv)
        qc_map = dict(zip(qc["feature"], qc["missing_pct"]))
        low_missing = [
            c for c in non_const if qc_map.get(c, 0) <= max_feature_missing_pct
        ]
        if low_missing:
            X_df = X_df[low_missing]
            non_const = low_missing

    # Replace Inf with NaN, then impute
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    row_missing_pct = 100.0 * X_df.isna().mean(axis=1)
    keep_rows = row_missing_pct <= max_row_missing_pct
    X_df = X_df.loc[keep_rows]
    for col in X_df.columns:
        X_df[col] = X_df[col].fillna(X_df[col].median())

    X = X_df.to_numpy().astype(np.float64)
    meta = df.loc[
        keep_rows,
        [
            c
            for c in ["case_id", "site", "dataset", "manufacturer", LABEL_COL]
            if c in df.columns
        ],
    ].copy()
    return X, meta, list(X_df.columns)


def _run_pca(X: np.ndarray, n_components: int = PCA_N_COMPONENTS) -> np.ndarray:
    """Run PCA and return 2D embedding (first 2 components)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    return coords[:, :2]


def _run_umap(X: np.ndarray) -> np.ndarray | None:
    """Run UMAP 2D embedding. Returns None if umap not installed."""
    try:
        import umap
    except ImportError:
        # Intentional: umap is optional; return None and skip UMAP.
        return None
    n_neighbors = min(UMAP_N_NEIGHBORS, X.shape[0] // 4, X.shape[0] - 1)
    n_neighbors = max(2, n_neighbors)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
        metric="euclidean",
    )
    return reducer.fit_transform(X)


def _run_tsne(X: np.ndarray) -> np.ndarray:
    """Run t-SNE 2D embedding."""
    from sklearn.manifold import TSNE

    perplexity = min(TSNE_PERPLEXITY, X.shape[0] // 4, X.shape[0] - 1)
    perplexity = max(5, perplexity)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE)
    return tsne.fit_transform(X)


def _plot_embeddings(
    coords: np.ndarray,
    meta: pd.DataFrame,
    method: str,
    plots_dir: Path,
) -> None:
    """Create 2D scatter plots colored by label, site, dataset, manufacturer."""
    import matplotlib.pyplot as plt

    x, y = coords[:, 0], coords[:, 1]
    color_cols = [LABEL_COL, "site", "dataset", "manufacturer"]
    color_cols = [c for c in color_cols if c in meta.columns]

    for col in color_cols:
        hue = meta[col].astype(str).fillna("NA")
        fig, ax = plt.subplots(figsize=(6, 5))
        unique = hue.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 2)))
        for i, u in enumerate(unique):
            mask = hue == u
            ax.scatter(
                x[mask],
                y[mask],
                c=[colors[i % len(colors)]],
                label=str(u)[:30],
                alpha=0.6,
                s=20,
            )
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        ax.set_title(f"{method} colored by {col} (n={len(meta)})")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        out_path = plots_dir / f"{method.lower()}_colored_by_{col}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {out_path.name}")


def _predict_site_and_report(
    X: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> dict:
    """Train RF to predict site from features; report AUC and confusion matrix."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    site_col = "site"
    if site_col not in meta.columns:
        print("[batch_effect] No 'site' column; skipping site prediction")
        return {}

    y_site = meta[site_col].astype(str).fillna("UNKNOWN")
    valid = y_site != "UNKNOWN"
    n_unique = y_site.nunique()
    if n_unique < MIN_SITE_CLASSES:
        print(
            f"[batch_effect] Site has only {n_unique} class(es); skipping site prediction"
        )
        return {}

    X_clean = X[valid]
    y_clean = y_site[valid]
    le = LabelEncoder()
    y_enc = le.fit_transform(y_clean)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=N_FOLDS_SITE, shuffle=True, random_state=RANDOM_STATE)

    y_pred_proba = cross_val_predict(
        clf, X_scaled, y_enc, cv=cv, method="predict_proba"
    )
    y_pred = np.argmax(y_pred_proba, axis=1)

    from sklearn.metrics import roc_auc_score

    try:
        auc_macro = roc_auc_score(
            y_enc,
            y_pred_proba,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        # Intentional: edge case (e.g. too few classes); record NaN.
        auc_macro = float("nan")

    report = {
        "target": site_col,
        "n_samples": int(len(y_clean)),
        "n_classes": int(n_unique),
        "auc_macro": float(auc_macro),
        "interpretation": (
            "strong batch encoding (confounding risk)"
            if auc_macro > AUC_STRONG_THRESHOLD
            else "moderate batch signal"
            if auc_macro > AUC_MODERATE_THRESHOLD
            else "weak batch encoding"
        ),
    }

    # Confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_enc, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=le.classes_,
        )
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"Site prediction (AUC macro={auc_macro:.3f})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            output_dir / "plots" / "site_prediction_confusion.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print("  site_prediction_confusion.png")
    except Exception as e:
        # Intentional: optional plot; log and continue so report still completes.
        print(f"  [batch_effect] Confusion plot failed: {e}")

    # Feature importances (fit on full data)
    clf.fit(X_scaled, y_enc)
    fi = pd.DataFrame(
        {
            "feature": feature_names[: len(clf.feature_importances_)],
            "importance": clf.feature_importances_,
        }
    )
    fi = fi.sort_values("importance", ascending=False).head(20)
    fi.to_csv(output_dir / "site_top_features.csv", index=False)
    print("  site_top_features.csv")

    return report


def main() -> None:
    """Entry point: run batch effect analysis and site prediction."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Batch effect and nuisance analysis (PCA/UMAP/t-SNE + site prediction)."
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="features_with_metadata.csv (must have site, dataset, manufacturer for full analysis)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/graph_extraction"),
        help="Output directory for plots and site_prediction_metrics.json",
    )
    parser.add_argument(
        "--qc-csv",
        type=Path,
        default=None,
        help="Optional qc_per_feature.csv to drop high-missing features",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip UMAP (faster; UMAP requires umap-learn)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    qc_csv = args.qc_csv or output_dir / "qc_per_feature.csv"

    print("[batch_effect] Loading features...")
    features_df = pd.read_csv(args.features_csv)
    print(f"  {len(features_df)} rows, {len(features_df.columns)} columns")

    X, meta, feat_names = _prepare_matrix(features_df, qc_csv=qc_csv)
    print(f"  Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")

    # Standardize for PCA and site prediction
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3.1 Dimensionality reduction
    print("[batch_effect] PCA...")
    coords_pca = _run_pca(
        X_scaled, n_components=min(PCA_N_COMPONENTS, X.shape[1], X.shape[0] - 1)
    )
    _plot_embeddings(coords_pca, meta, "PCA", plots_dir)

    if not args.no_umap:
        print("[batch_effect] UMAP...")
        coords_umap = _run_umap(X_scaled)
        if coords_umap is not None:
            _plot_embeddings(coords_umap, meta, "UMAP", plots_dir)
        else:
            print("  (umap-learn not installed; pip install umap-learn to enable)")

    print("[batch_effect] t-SNE...")
    coords_tsne = _run_tsne(X_scaled)
    _plot_embeddings(coords_tsne, meta, "t-SNE", plots_dir)

    # Save embedding coords for report notebook
    emb_df = meta.copy()
    emb_df["pca_1"] = coords_pca[:, 0]
    emb_df["pca_2"] = coords_pca[:, 1]
    emb_df["tsne_1"] = coords_tsne[:, 0]
    emb_df["tsne_2"] = coords_tsne[:, 1]
    emb_df.to_csv(output_dir / "embedding_coords.csv", index=False)
    print(f"  embedding_coords.csv -> {output_dir / 'embedding_coords.csv'}")

    # 3.2 Site prediction
    print("[batch_effect] Site prediction...")
    report = _predict_site_and_report(X_scaled, meta, feat_names, output_dir)
    if report:
        metrics_path = output_dir / "site_prediction_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"  site_prediction_metrics.json -> {metrics_path}")
        print(f"  AUC macro = {report['auc_macro']:.3f} ({report['interpretation']})")

    print("[batch_effect] Done")


if __name__ == "__main__":
    main()
