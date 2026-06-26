#!/usr/bin/env python3
"""Check voxel-spacing distribution shift between DUKE and the other cohorts.

DUKE is the training cohort. We treat DUKE's central 5th-95th percentile band of
in-plane (xy) and slice (z) spacing as the "in-distribution" range, then ask how
many ISPY1 / ISPY2 / NACT-Pilot scans fall outside that range. This is a quick
domain-shift QC check before resampling everything to the DUKE target spacing.

Why headers only: a NIfTI header stores the voxel spacing (and shape) in a tiny
fixed-size block at the top of the file. `nibabel.load()` is lazy -- it reads the
header but does NOT pull the (large) image array into memory until you ask for it.
So reading spacing for ~1500 scans is fast and light enough to run on the login
node without a GPU.

Outputs:
  - Per-scan CSV : ~/vanguard/scripts/spacing_range_check_results.csv
  - Scatter plot : $VANGUARD_QC_OUTPUT/spacing_distribution.png  (falls back to ~/vanguard_qc_pngs)

Usage:
    micromamba run -n vanguard python scripts/spacing_range_check.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: render to file, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# /net/projects2/vanguard/ is NOT mounted on the randi login/compute nodes; the
# MAMA-MIA NIfTI files actually live under this scratch path (see lab_notebook.md).
_DEFAULT_DATA_ROOT = Path(
    "/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images"
)
_DEFAULT_CSV = Path.home() / "vanguard" / "scripts" / "spacing_range_check_results.csv"
_DEFAULT_FIG = Path(os.environ.get("VANGUARD_QC_OUTPUT", str(Path.home() / "vanguard_qc_pngs"))) / "spacing_distribution.png"

# Directory prefix -> cohort label used in outputs. The pilot NACT cohort is
# stored on disk under the "NACT" prefix but the user calls it "NACT-Pilot".
_COHORT_PREFIXES = {
    "DUKE": "DUKE",
    "ISPY1": "ISPY1",
    "ISPY2": "ISPY2",
    "NACT": "NACT-Pilot",
}
_TRAIN_COHORT = "DUKE"

# Distinct color AND marker per cohort so the plot stays readable in grayscale.
_COHORT_STYLE = {
    "DUKE": {"color": "tab:blue", "marker": "o"},      # circle
    "ISPY1": {"color": "tab:orange", "marker": "^"},   # triangle
    "ISPY2": {"color": "tab:green", "marker": "s"},    # square
    "NACT-Pilot": {"color": "tab:purple", "marker": "D"},  # diamond
}


def _cohort_label(case_id: str) -> str | None:
    """Map a case_id (e.g. 'NACT_01') to its cohort label, or None if unknown."""
    for prefix, label in _COHORT_PREFIXES.items():
        if case_id.startswith(prefix):
            return label
    return None


def _find_precontrast_nii(case_dir: Path) -> Path | None:
    """Return the pre-contrast (timepoint _0000) NIfTI for a case, or None.

    Each case has several DCE timepoints (_0000, _0001, ...) that share the same
    geometry, so one scan per case is enough for a spacing check. We use _0000.
    Accepts both .nii.gz and .nii.
    """
    for ext in (".nii.gz", ".nii"):
        hits = sorted(case_dir.glob(f"*_0000{ext}"))
        if hits:
            return hits[0]
    return None


def _read_spacing(nii_path: Path) -> tuple[float, float]:
    """Return (xy_spacing, z_spacing) in mm from a NIfTI header (no array load).

    nibabel's get_zooms() gives the voxel size along each array axis. For these
    files the axis order is (x, y, z), so zooms[0] is the in-plane spacing
    (x == y here) and zooms[2] is the slice thickness.
    """
    import nibabel as nib  # noqa: PLC0415  (import here to keep startup light)

    zooms = nib.load(str(nii_path)).header.get_zooms()
    return float(zooms[0]), float(zooms[2])


def collect_spacings(data_root: Path) -> list[dict]:
    """Walk every case dir, read header spacing, return one record per scan.

    Each record: {scan_id, cohort, xy_spacing, z_spacing}. Unreadable or missing
    files are skipped with a warning rather than crashing the whole run.
    """
    records: list[dict] = []
    n_skipped = 0
    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue
        cohort = _cohort_label(case_dir.name)
        if cohort is None:
            continue  # not one of our four cohorts

        nii_path = _find_precontrast_nii(case_dir)
        if nii_path is None:
            print(f"WARNING: no _0000 NIfTI found for {case_dir.name}, skipping")
            n_skipped += 1
            continue

        try:
            xy, z = _read_spacing(nii_path)
        except Exception as exc:  # noqa: BLE001 - any read error -> skip + warn
            print(f"WARNING: could not read header for {nii_path.name}: {exc}")
            n_skipped += 1
            continue

        records.append(
            {
                "scan_id": case_dir.name,
                "cohort": cohort,
                "xy_spacing": xy,
                "z_spacing": z,
            }
        )

    print(f"\nRead spacing for {len(records)} scans ({n_skipped} skipped).")
    return records


def _print_stats(name: str, values: np.ndarray) -> None:
    """Print min / max / median / mean for one spacing axis."""
    print(
        f"  {name}: min={values.min():.4f}  max={values.max():.4f}  "
        f"median={np.median(values):.4f}  mean={values.mean():.4f}  mm"
    )


def main() -> None:
    """Run the full DUKE-range domain-shift check end to end."""
    parser = argparse.ArgumentParser(
        description="Flag non-DUKE scans whose voxel spacing falls outside the "
        "DUKE 5th-95th percentile range; write a CSV and a scatter plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--fig", type=Path, default=_DEFAULT_FIG)
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"ERROR: data root does not exist: {args.data_root}", file=sys.stderr)
        sys.exit(1)

    # ----- Read spacing for every scan -----------------------------------------
    records = collect_spacings(args.data_root)
    if not records:
        print("ERROR: no scans read; nothing to do.", file=sys.stderr)
        sys.exit(1)

    cohorts = [r["cohort"] for r in records]
    xy_all = np.array([r["xy_spacing"] for r in records])
    z_all = np.array([r["z_spacing"] for r in records])

    # ----- STEP 1: DUKE training distribution ----------------------------------
    is_duke = np.array([c == _TRAIN_COHORT for c in cohorts])
    duke_xy, duke_z = xy_all[is_duke], z_all[is_duke]

    # np.percentile returns the value below which that % of the data falls.
    xy_lo, xy_hi = np.percentile(duke_xy, [5, 95])
    z_lo, z_hi = np.percentile(duke_z, [5, 95])

    print(f"\n=== DUKE training distribution (n={is_duke.sum()}) ===")
    print("Full statistics:")
    _print_stats("xy spacing", duke_xy)
    _print_stats("z  spacing", duke_z)
    print("5th-95th percentile (in-distribution) range:")
    print(f"  xy: [{xy_lo:.4f}, {xy_hi:.4f}] mm")
    print(f"  z : [{z_lo:.4f}, {z_hi:.4f}] mm")

    # ----- STEP 2: flag every scan against the DUKE range -----------------------
    # We compute the flags for ALL scans (DUKE included) so the CSV is complete;
    # by construction ~10% of DUKE itself sits outside its own 5-95 band, which
    # is a useful sanity check. The summary below focuses on the non-DUKE cohorts.
    for r in records:
        xy_out = bool(r["xy_spacing"] < xy_lo or r["xy_spacing"] > xy_hi)
        z_out = bool(r["z_spacing"] < z_lo or r["z_spacing"] > z_hi)
        r["xy_out_of_range"] = xy_out
        r["z_out_of_range"] = z_out
        r["either_out_of_range"] = xy_out or z_out

    # ----- Write per-scan CSV ---------------------------------------------------
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan_id",
        "cohort",
        "xy_spacing",
        "z_spacing",
        "xy_out_of_range",
        "z_out_of_range",
        "either_out_of_range",
    ]
    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWrote per-scan results -> {args.csv}")

    # ----- STEP 3: per-cohort summary -------------------------------------------
    print("\n=== Out-of-range summary (vs DUKE 5th-95th range) ===")
    non_duke = [lbl for lbl in _COHORT_PREFIXES.values() if lbl != _TRAIN_COHORT]
    # Count how many non-DUKE scans fall outside the box, per cohort. Stored for
    # reuse as the on-plot annotations in STEP 4.
    outside_counts: dict[str, int] = {}
    for cohort in non_duke:
        rows = [r for r in records if r["cohort"] == cohort]
        total = len(rows)
        n_xy = sum(r["xy_out_of_range"] for r in rows)
        n_z = sum(r["z_out_of_range"] for r in rows)
        n_either = sum(r["either_out_of_range"] for r in rows)
        outside_counts[cohort] = n_either
        print(
            f"  {cohort:>10}: n={total:>4} | xy_out={n_xy:>4} | "
            f"z_out={n_z:>4} | either_out={n_either:>4}"
        )

    # ----- STEP 4: scatter plot -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", color="0.85")

    for cohort in _COHORT_PREFIXES.values():
        mask = np.array([c == cohort for c in cohorts])
        if not mask.any():
            continue
        style = _COHORT_STYLE[cohort]
        ax.scatter(
            xy_all[mask],
            z_all[mask],
            s=28,
            alpha=0.55,
            edgecolors="none",
            color=style["color"],
            marker=style["marker"],
            label=f"{cohort} (n={int(mask.sum())})",
        )

    # Dashed rectangle marking the DUKE 5th-95th percentile box.
    rect = plt.Rectangle(
        (xy_lo, z_lo),
        xy_hi - xy_lo,
        z_hi - z_lo,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        zorder=5,
        label="Duke 5th-95th percentile range",
    )
    ax.add_patch(rect)

    # One annotation per non-DUKE cohort, near that cohort's cluster (its median
    # point), showing how many scans fall outside the box. The non-DUKE clusters
    # overlap around z~2 mm, so we stagger the label offsets to avoid collisions.
    _label_offset = {
        "ISPY1": (10, 8),
        "ISPY2": (12, 16),
        "NACT-Pilot": (12, -24),
    }
    for cohort in non_duke:
        mask = np.array([c == cohort for c in cohorts])
        if not mask.any():
            continue
        med_xy = float(np.median(xy_all[mask]))
        med_z = float(np.median(z_all[mask]))
        total = int(mask.sum())
        ax.annotate(
            f"{cohort}: {outside_counts[cohort]}/{total} outside range",
            xy=(med_xy, med_z),
            xytext=_label_offset.get(cohort, (10, 8)),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=_COHORT_STYLE[cohort]["color"],
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.85},
            zorder=6,
        )

    ax.set_xlabel("In-plane (xy) voxel spacing  (mm)")
    ax.set_ylabel("Slice (z) spacing  (mm)")
    ax.set_title("Voxel Spacing Distribution by Cohort vs. Duke Training Range")
    ax.legend(loc="upper right", frameon=True)

    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=150, bbox_inches="tight")
    print(f"\nSaved scatter plot -> {args.fig}")

    # ----- How to run -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Done. To re-run this check:")
    print("  micromamba run -n vanguard python scripts/spacing_range_check.py")
    print("Outputs:")
    print(f"  CSV : {args.csv}")
    print(f"  PNG : {args.fig}")
    print("=" * 70)


if __name__ == "__main__":
    main()
