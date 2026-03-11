"""Phase 2: Per-feature QC, sanity checks, and distribution plots.

Requires features_with_metadata.csv (or features_raw.csv) and morphometry JSONs.

Usage:
    python graph_extraction/run_feature_qc.py \
        --features-csv report/features_with_metadata.csv \
        --morphometry-dir report/4d_morphometry \
        --output-dir report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TITLE_MAX_LEN = 40
HIGH_MISSING_PCT = 50
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def compute_qc_per_feature(
    df: pd.DataFrame, id_cols: list[str] | None = None
) -> pd.DataFrame:
    """Compute per-feature QC metrics: missingness, constant, NaN, Inf, outliers."""
    if id_cols is None:
        id_cols = ["case_id", "patient_id"]
    feature_cols = [
        c
        for c in df.columns
        if c not in id_cols and df[c].dtype in ("float64", "int64", "float32")
    ]
    if not feature_cols:
        feature_cols = [
            c
            for c in df.columns
            if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

    rows = []
    for col in feature_cols:
        s = df[col]
        n = len(s)
        missing = s.isna().sum()
        n_valid = n - missing
        inf_count = int(np.isinf(s.astype(float)).sum())
        is_constant = n_valid > 0 and s.std(skipna=True) == 0
        # Outliers: beyond 5 IQR from median
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_pct = 0.0
        else:
            lo = q1 - 5 * iqr
            hi = q3 + 5 * iqr
            outlier_count = ((s < lo) | (s > hi)).sum()
            outlier_pct = 100.0 * outlier_count / n if n > 0 else 0.0

        rows.append(
            {
                "feature": col,
                "missing_pct": 100.0 * missing / n if n > 0 else 0.0,
                "is_constant": is_constant,
                "nan_count": int(missing),
                "inf_count": inf_count,
                "outlier_pct": outlier_pct,
                "n_valid": int(n_valid),
            }
        )
    return pd.DataFrame(rows)


def plot_core_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot distributions for core morphometry features."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        # Intentional: optional plots; skip and return so QC still runs.
        print("[qc] matplotlib/seaborn not installed; skipping distribution plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Identify core feature groups
    keywords = {
        "radius": ["radius"],
        "length": ["length"],
        "tortuosity": ["tortuosity"],
        "curvature": ["curvature"],
        "angle": ["angle"],
        "volume": ["volume"],
        "count": ["count"],
    }

    id_cols = {
        "case_id",
        "patient_id",
        "site",
        "dataset",
        "manufacturer",
        "model",
        "pcr",
        "field_strength",
    }
    numeric = df.select_dtypes(include=[np.number])
    numeric = numeric.drop(
        columns=[c for c in numeric.columns if c in id_cols], errors="ignore"
    )

    # Sample of key features for distribution plots (avoid 100+ panels)
    core_cols = []
    for kw, patterns in keywords.items():
        for c in numeric.columns:
            if any(p in c.lower() for p in patterns):
                # Prefer __mean or aggregate stats
                if (
                    "__mean" in c
                    or "mean" in c.lower()
                    or "__median" in c
                    or kw in ["tortuosity", "length", "volume", "count"]
                ):
                    core_cols.append(c)
                    break

    if not core_cols:
        core_cols = numeric.columns[:12].tolist()  # fallback: first 12 numeric

    n_plots = len(core_cols)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, col in enumerate(core_cols):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]
        vals = df[col].dropna()
        if len(vals) > 0:
            sns.histplot(vals, ax=ax, kde=True, bins=min(50, len(vals)))
        ax.set_title(col[:TITLE_MAX_LEN] + "..." if len(col) > TITLE_MAX_LEN else col)
        ax.set_xlabel("")

    for idx in range(len(core_cols), n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "distributions_core.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[qc] distributions_core.png -> {plots_dir / 'distributions_core.png'}")


def main() -> None:
    """Entry point for per-feature QC, sanity checks, and distribution plots."""
    parser = argparse.ArgumentParser(
        description="Per-feature QC, sanity checks, and distribution plots."
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="features_with_metadata.csv or features_raw.csv",
    )
    parser.add_argument(
        "--morphometry-dir",
        type=Path,
        help="Directory with morphometry JSONs for sanity checks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report"),
        help="Output directory for qc_per_feature.csv, sanity_violations.csv, plots",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    features_df = pd.read_csv(args.features_csv)
    print(
        f"[qc] Loaded {len(features_df)} rows, {len(features_df.columns)} columns from {args.features_csv}"
    )

    # Phase 2.1: Per-feature QC
    qc_df = compute_qc_per_feature(features_df)
    qc_path = output_dir / "qc_per_feature.csv"
    qc_df.to_csv(qc_path, index=False)
    print(f"[qc] qc_per_feature.csv -> {qc_path}")

    n_const = qc_df["is_constant"].sum()
    n_high_missing = (qc_df["missing_pct"] > HIGH_MISSING_PCT).sum()
    print(
        f"[qc] Constant columns: {n_const}, High missing (>{HIGH_MISSING_PCT}%): {n_high_missing}"
    )

    # Phase 2.2: Sanity checks (if morphometry dir provided)
    if args.morphometry_dir and args.morphometry_dir.exists():
        from graph_extraction.feature_sanity import check_morphometry_dir

        violations = check_morphometry_dir(args.morphometry_dir)
        if violations:
            vio_df = pd.DataFrame(violations)
            vio_path = output_dir / "sanity_violations.csv"
            vio_df.to_csv(vio_path, index=False)
            print(
                f"[qc] sanity_violations.csv -> {vio_path} ({len(violations)} violations)"
            )
        else:
            print("[qc] No sanity violations found")
    else:
        print("[qc] Skipping sanity checks (no --morphometry-dir or dir not found)")

    # Phase 2.3: Distribution plots
    plot_core_distributions(features_df, output_dir)
    print("[qc] Done")


if __name__ == "__main__":
    main()
