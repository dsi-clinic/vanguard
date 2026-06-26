#!/usr/bin/env python3
"""Scattergram of per-patient voxel dimensions across MAMA-MIA cohorts.

Each patient is one point: in-plane voxel size (xy) on the x-axis, slice
thickness (z) on the y-axis, colored by cohort. Complements the existing
voxel_spacing_boxplot.png, which collapses the two dimensions into separate
panels, by showing their joint distribution.

Data source: the MAMA-MIA clinical_and_imaging_info.xlsx 'dataset_info' sheet,
which provides 'pixel_spacing' (e.g. "[0.78125, 0.78125]") and 'slice_thickness'
per patient along with the 'dataset' (cohort) label.

Usage examples:
    python scripts/plot_voxel_spacing_scatter.py
    python scripts/plot_voxel_spacing_scatter.py --out /tmp/scatter.png --jitter
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from clinic_metadata import load_clinic_metadata_excel  # noqa: E402

_DEFAULT_EXCEL = Path(
    "/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx"
)
_DEFAULT_OUT = REPO_ROOT / "voxel_spacing_scatter.png"

# Colors chosen to match voxel_spacing_boxplot.png (matplotlib tab10).
COHORT_COLORS = {
    "DUKE": "tab:blue",
    "ISPY1": "tab:orange",
    "ISPY2": "tab:green",
    "NACT": "tab:purple",
}


def _parse_in_plane(spacing: object) -> float | None:
    """Return the in-plane voxel size (mm) from a 'pixel_spacing' cell.

    The cell is a string like "[0.78125, 0.78125]" (or already a sequence).
    Returns the first element as a float, or None if unparseable.
    """
    if spacing is None or (isinstance(spacing, float) and np.isnan(spacing)):
        return None
    try:
        value = spacing if isinstance(spacing, (list, tuple)) else ast.literal_eval(str(spacing))
        return float(value[0])
    except (ValueError, SyntaxError, TypeError, IndexError):
        return None


def main() -> None:
    """Build and save the voxel-spacing scattergram."""
    parser = argparse.ArgumentParser(
        description="Scattergram of in-plane voxel size vs slice thickness, colored by cohort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=_DEFAULT_EXCEL,
        help="MAMA-MIA clinical_and_imaging_info.xlsx with the 'dataset_info' sheet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--jitter",
        action="store_true",
        help="Add small horizontal jitter to spread overlapping discrete voxel sizes.",
    )
    args = parser.parse_args()

    if not args.excel.exists():
        print(f"ERROR: Excel not found: {args.excel}", file=sys.stderr)
        sys.exit(1)

    df = load_clinic_metadata_excel(args.excel, sheet_name="dataset_info")
    for col in ("dataset", "pixel_spacing", "slice_thickness"):
        if col not in df.columns:
            print(f"ERROR: expected column '{col}' missing from {args.excel}", file=sys.stderr)
            sys.exit(1)

    df = df.copy()
    df["voxel_xy"] = df["pixel_spacing"].map(_parse_in_plane)
    df["slice_thickness"] = df["slice_thickness"].astype(float)

    n_total = len(df)
    df = df.dropna(subset=["voxel_xy", "slice_thickness"])
    n_dropped = n_total - len(df)
    if n_dropped:
        print(f"WARNING: dropped {n_dropped} rows with unparseable/missing spacing")

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", color="0.85")

    # Cohorts first by known order, then any extras for forward-compatibility.
    cohorts = [c for c in COHORT_COLORS if c in set(df["dataset"])]
    cohorts += [c for c in sorted(df["dataset"].unique()) if c not in COHORT_COLORS]

    for cohort in cohorts:
        sub = df[df["dataset"] == cohort]
        x = sub["voxel_xy"].to_numpy()
        if args.jitter:
            x = x + rng.uniform(-0.01, 0.01, size=len(x))
        ax.scatter(
            x,
            sub["slice_thickness"].to_numpy(),
            s=22,
            alpha=0.5,
            edgecolors="none",
            color=COHORT_COLORS.get(cohort, "tab:gray"),
            label=f"{cohort} (n={len(sub)})",
        )
        print(f"  {cohort}: {len(sub)} points")

    # Reference marker: DUKE mode / resampled target spacing.
    ref_x, ref_y = 0.7, 1.0
    ax.scatter(
        [ref_x],
        [ref_y],
        s=220,
        marker="*",
        color="black",
        zorder=5,
        label="DUKE mode/Resampled Size",
    )
    ax.annotate(
        "DUKE mode/Resampled Size",
        xy=(ref_x, 0.9),
        ha="center",
        va="top",
        fontsize=9,
        color="black",
    )

    ax.set_xlabel("In-plane voxel size  (mm)")
    ax.set_ylabel("Slice thickness  (mm)")
    ax.set_title("Voxel spacing across MAMA-MIA cohorts  —  all subjects")
    ax.legend(title="Cohort", frameon=True)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nSaved scattergram ({len(df)} subjects) -> {args.out}")


if __name__ == "__main__":
    main()
