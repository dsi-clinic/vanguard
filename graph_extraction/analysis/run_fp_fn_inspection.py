r"""Phase 5: FP/FN case inspection.

Identify top FP and FN, feature outlier check, and skeleton visualization.
Requires features_with_metadata.csv, morphometry dir (for skeleton masks), and same split as Phase 4.

Usage:
    python graph_extraction/analysis/run_fp_fn_inspection.py \\
        --features-csv analysis/graph_extraction/features_with_metadata.csv \\
        --morphometry-dir analysis/graph_extraction/4d_morphometry \\
        --output-dir analysis/graph_extraction
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

from graph_extraction.analysis.run_batch_effect_analysis import (  # noqa: E402
    LABEL_COL,
    MAX_MISSING_PCT,
    MAX_ROW_MISSING_PCT,
    RANDOM_STATE,
    _get_feature_cols,
)

TEST_SIZE = 0.15
VAL_SIZE = 0.15
PROBA_THRESHOLD = 0.5
TOP_N = 10
OUTLIER_Z_THRESHOLD = 3.0


def _prepare_data(
    df: pd.DataFrame, qc_csv: Path | None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare X_df, y, feature_cols (same logic as ablation)."""
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
    return X_df, y, list(X_df.columns)


def _plot_skeleton_static(
    mask_zyx: np.ndarray, output_path: Path, case_id: str
) -> bool:
    """Save static 3-view (axial, coronal, sagittal) PNG of skeleton."""
    try:
        import matplotlib.pyplot as plt

        z, y_ax, x = np.nonzero(mask_zyx)
        if len(z) == 0:
            return False

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(mask_zyx[z.mean().astype(int)], cmap="gray")
        axes[0].set_title("Axial")
        axes[0].axis("off")
        axes[1].imshow(mask_zyx[:, y_ax.mean().astype(int), :], cmap="gray")
        axes[1].set_title("Coronal")
        axes[1].axis("off")
        axes[2].imshow(mask_zyx[:, :, x.mean().astype(int)], cmap="gray")
        axes[2].set_title("Sagittal")
        axes[2].axis("off")
        fig.suptitle(f"Skeleton: {case_id}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        # Intentional: optional viz; report and return False so batch continues.
        print(f"    [viz] {e}")
        return False


def main() -> None:
    """Run FP/FN case inspection: top cases, outlier check, skeleton viz."""
    parser = argparse.ArgumentParser(
        description="Phase 5: FP/FN case inspection (top FP/FN, outlier check, skeleton viz)."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--morphometry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/graph_extraction"))
    parser.add_argument("--qc-csv", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    output_dir = args.output_dir
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    qc_csv = args.qc_csv or output_dir / "qc_per_feature.csv"

    print("[fp_fn] Loading features...")
    df_full = pd.read_csv(args.features_csv)
    if LABEL_COL not in df_full.columns:
        print(f"[fp_fn] ERROR: no '{LABEL_COL}' column")
        sys.exit(1)

    X_df, y, feat_cols = _prepare_data(df_full, qc_csv)
    case_ids = df_full.loc[X_df.index, "case_id"].to_numpy()
    if case_ids is None:
        print("[fp_fn] ERROR: no case_id column")
        sys.exit(1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_mat = X_df.to_numpy().astype(np.float64)
    n = len(X_mat)
    idx_all = np.arange(n)
    idx_trval, idx_te = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rel_val = VAL_SIZE / (1.0 - TEST_SIZE)
    y_trval = y.iloc[idx_trval]
    idx_tr, _idx_val = train_test_split(
        idx_trval, test_size=rel_val, random_state=RANDOM_STATE, stratify=y_trval
    )

    X_tr = X_mat[idx_tr]
    X_te = X_mat[idx_te]
    y_tr = y.iloc[idx_tr]
    y_te = y.iloc[idx_te]
    te_case_ids = case_ids[idx_te]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE
    )
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]

    # 5.1 Top FP and FN
    y_te_arr = y_te.to_numpy()
    fp_mask = (y_te_arr == 0) & (proba > PROBA_THRESHOLD)
    fn_mask = (y_te_arr == 1) & (proba < PROBA_THRESHOLD)
    fp_proba = proba[fp_mask]
    fn_proba = proba[fn_mask]
    fp_idx = np.where(fp_mask)[0]
    fn_idx = np.where(fn_mask)[0]
    # Sort FP by proba desc, FN by proba asc
    if len(fp_idx) > 0:
        order = np.argsort(-fp_proba)[: args.top_n]
        top_fp_idx = fp_idx[order]
    else:
        top_fp_idx = np.array([], dtype=int)
    if len(fn_idx) > 0:
        order = np.argsort(fn_proba)[: args.top_n]
        top_fn_idx = fn_idx[order]
    else:
        top_fn_idx = np.array([], dtype=int)

    rows = []
    for i in top_fp_idx:
        cid = te_case_ids[i]
        rows.append(
            {
                "case_id": cid,
                "y_true": int(y_te_arr[i]),
                "y_pred_proba": float(proba[i]),
                "fp_or_fn": "fp",
            }
        )
    for i in top_fn_idx:
        cid = te_case_ids[i]
        rows.append(
            {
                "case_id": cid,
                "y_true": int(y_te_arr[i]),
                "y_pred_proba": float(proba[i]),
                "fp_or_fn": "fn",
            }
        )

    fp_fn_df = pd.DataFrame(rows)
    fp_fn_path = output_dir / "fp_fn_cases.csv"
    fp_fn_df.to_csv(fp_fn_path, index=False)
    print(f"[fp_fn] fp_fn_cases.csv -> {fp_fn_path} ({len(rows)} cases)")

    # 5.2 Feature outlier check (z-score vs train; X_te_s is already scaled)
    outlier_rows = []
    for _, row in fp_fn_df.iterrows():
        cid = row["case_id"]
        try:
            j = list(te_case_ids).index(cid)
        except ValueError:
            continue
        z_scores = np.abs(X_te_s[j])
        out_feats = np.where(z_scores > OUTLIER_Z_THRESHOLD)[0]
        for k in out_feats:
            outlier_rows.append(
                {
                    "case_id": cid,
                    "feature": feat_cols[k],
                    "value": float(X_te[j, k]),
                    "z_score": float(z_scores[k]),
                    "is_outlier": True,
                }
            )
        if len(out_feats) > 0:
            outlier_rows.append(
                {
                    "case_id": cid,
                    "feature": "_n_outliers",
                    "value": float(len(out_feats)),
                    "z_score": np.nan,
                    "is_outlier": True,
                }
            )

    if outlier_rows:
        out_df = pd.DataFrame(outlier_rows)
        out_path = output_dir / "fp_fn_outliers.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[fp_fn] fp_fn_outliers.csv -> {out_path}")
    else:
        print("[fp_fn] No extreme outliers (|z|>3) in selected FP/FN cases")

    # 5.3 Skeleton visualization
    morpho_dir = args.morphometry_dir
    viz_status = []
    for _, r in fp_fn_df.iterrows():
        cid = str(r["case_id"])
        tag = "fp" if r["fp_or_fn"] == "fp" else "fn"
        # case_id = study_id (e.g. DUKE_001, ISPY2_202539)
        skel_path = morpho_dir / f"{cid}_skeleton_4d_exam_mask.npy"
        if not skel_path.exists():
            viz_status.append(
                {"case_id": cid, "fp_or_fn": tag, "status": "skeleton_missing"}
            )
            continue
        try:
            mask = np.load(skel_path).astype(bool)
        except Exception as e:
            # Intentional: record load error in status and continue with other cases.
            viz_status.append(
                {"case_id": cid, "fp_or_fn": tag, "status": f"load_error:{e}"}
            )
            continue
        out_path = viz_dir / f"{tag}_{cid}.png"
        ok = _plot_skeleton_static(mask, out_path, cid)
        viz_status.append(
            {"case_id": cid, "fp_or_fn": tag, "status": "ok" if ok else "plot_failed"}
        )
        if ok:
            print(f"  viz/{out_path.name}")

    status_df = pd.DataFrame(viz_status)
    status_df.to_csv(output_dir / "fp_fn_viz_status.csv", index=False)
    print(f"[fp_fn] fp_fn_viz_status.csv -> {output_dir / 'fp_fn_viz_status.csv'}")
    print("[fp_fn] Done")


if __name__ == "__main__":
    main()
