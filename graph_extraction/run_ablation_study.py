"""Phase 4: Ablation study with fixed train/val/test split.

Compare Full, No-coordinate-like, Geometry-only, and Count-only feature subsets.
Reports AUC and AP for each ablation.

Requires features_with_metadata.csv.

Usage:
    python graph_extraction/run_ablation_study.py \
        --features-csv report/features_with_metadata.csv \
        --output-dir report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_extraction.run_batch_effect_analysis import (  # noqa: E402
    LABEL_COL,
    MAX_MISSING_PCT,
    MAX_ROW_MISSING_PCT,
    RANDOM_STATE,
    _get_feature_cols,
)

TEST_SIZE = 0.15
VAL_SIZE = 0.15
MIN_FEATURES = 2
# Train = 1 - 0.15 - 0.15 = 0.70


def _is_coordinate_like(col: str) -> bool:
    """True if column encodes voxel coordinates (segment start/end, bifurcation midpoint)."""
    c = col.lower()
    if "__segment__" in col and ("start" in c or "end" in c):
        return True
    if "midpoint" in c:
        return True
    if "points_angle" in c:
        return True
    return False


def _filter_ablation(cols: list[str], ablation: str) -> list[str]:
    """Return feature columns for the given ablation."""

    def is_geometry(col: str) -> bool:
        c = col.lower()
        return any(
            k in c for k in ["radius", "length", "tortuosity", "curvature", "volume"]
        )

    def is_count(col: str) -> bool:
        return "__count" in col

    if ablation == "full":
        return list(cols)
    if ablation == "no_coordinate_like":
        return [c for c in cols if not _is_coordinate_like(c)]
    if ablation == "geometry_only":
        return [c for c in cols if not _is_coordinate_like(c) and is_geometry(c)]
    if ablation == "count_only":
        return [c for c in cols if is_count(c)]
    raise ValueError(f"Unknown ablation: {ablation}")


def _prepare_data(
    df: pd.DataFrame,
    qc_csv: Path | None,
) -> tuple[pd.DataFrame, pd.Series, list[str], np.ndarray]:
    """Prepare base data: X_df (imputed), y, all_feature_cols, keep_mask."""
    feature_cols = _get_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns found")

    X_df = df[feature_cols].copy()
    const_mask = X_df.apply(lambda s: len(s.dropna().unique()) <= 1)
    non_const = [c for c in feature_cols if not const_mask.get(c, False)]
    X_df = X_df[non_const]

    if qc_csv and qc_csv.exists():
        qc = pd.read_csv(qc_csv)
        qc_map = dict(zip(qc["feature"], qc["missing_pct"]))
        low_missing = [c for c in non_const if qc_map.get(c, 0) <= MAX_MISSING_PCT]
        if low_missing:
            X_df = X_df[low_missing]
            non_const = low_missing

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    row_missing_pct = 100.0 * X_df.isna().mean(axis=1)
    keep_mask = row_missing_pct <= MAX_ROW_MISSING_PCT
    X_df = X_df.loc[keep_mask]
    for col in X_df.columns:
        X_df[col] = X_df[col].fillna(X_df[col].median())

    y = df.loc[keep_mask, LABEL_COL]
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    return X_df, y, list(X_df.columns), keep_mask


def main() -> None:
    """Run ablation study: Full, No-coord, Geometry-only, Count-only."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Ablation study (Full, No-coord, Geometry-only, Count-only)."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("report"))
    parser.add_argument("--qc-csv", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    qc_csv = args.qc_csv or output_dir / "qc_per_feature.csv"

    print("[ablation] Loading features...")
    features_df = pd.read_csv(args.features_csv)
    if LABEL_COL not in features_df.columns:
        print(f"[ablation] ERROR: no '{LABEL_COL}' column")
        sys.exit(1)

    X_df, y, all_cols, _ = _prepare_data(features_df, qc_csv)
    print(f"  {len(X_df)} samples, {len(all_cols)} base features")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Fixed split: 70/15/15
    X_mat = X_df.to_numpy().astype(np.float64)
    X_trval, X_te, y_trval, y_te = train_test_split(
        X_mat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rel_val = VAL_SIZE / (1.0 - TEST_SIZE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=rel_val, random_state=RANDOM_STATE, stratify=y_trval
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    ablations = ["full", "no_coordinate_like", "geometry_only", "count_only"]
    results = []

    for abl in ablations:
        cols = _filter_ablation(all_cols, abl)
        if not cols:
            print(f"  [{abl}] 0 features, skipping")
            results.append(
                {"ablation": abl, "n_features": 0, "auc": np.nan, "ap": np.nan}
            )
            continue

        idx = [all_cols.index(c) for c in cols if c in all_cols]
        if len(idx) != len(cols):
            idx = [i for i, c in enumerate(all_cols) if c in cols]
        X_tr_a = X_tr_s[:, idx]
        X_te_a = X_te_s[:, idx]

        if X_tr_a.shape[1] < MIN_FEATURES:
            print(f"  [{abl}] too few features ({X_tr_a.shape[1]}), skipping")
            results.append(
                {
                    "ablation": abl,
                    "n_features": X_tr_a.shape[1],
                    "auc": np.nan,
                    "ap": np.nan,
                }
            )
            continue

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE
        )
        clf.fit(X_tr_a, y_tr)
        proba = clf.predict_proba(X_te_a)[:, 1]
        auc = roc_auc_score(y_te, proba)
        ap = average_precision_score(y_te, proba)
        results.append(
            {
                "ablation": abl,
                "n_features": len(cols),
                "auc": float(auc),
                "ap": float(ap),
            }
        )
        print(f"  [{abl}] n={len(cols)}, AUC={auc:.3f}, AP={ap:.3f}")

    res_df = pd.DataFrame(results)
    res_path = output_dir / "ablation_results.csv"
    res_df.to_csv(res_path, index=False)
    print(f"  -> {res_path}")

    # Bar chart
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        valid = res_df["n_features"] > 0
        r = res_df[valid]
        x = np.arange(len(r))
        axes[0].bar(x, r["auc"], color="steelblue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(r["ablation"], rotation=30, ha="right")
        axes[0].set_ylabel("AUC")
        axes[0].set_title("AUC by ablation")
        axes[1].bar(x, r["ap"], color="coral")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(r["ablation"], rotation=30, ha="right")
        axes[1].set_ylabel("AP")
        axes[1].set_title("Average Precision by ablation")
        plt.tight_layout()
        plt.savefig(
            plots_dir / "ablation_auc_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"  -> {plots_dir / 'ablation_auc_comparison.png'}")
    except Exception as e:
        # Intentional: optional plot; log and continue so ablation results still saved.
        print(f"  [ablation] Plot failed: {e}")

    print("[ablation] Done")


if __name__ == "__main__":
    main()
