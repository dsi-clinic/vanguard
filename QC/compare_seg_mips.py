#!/usr/bin/env python3
"""Compare axial MIPs of original vs. resampled vessel segmentations.

Randomly selects --n cases from each of --cohorts, loads the vessel
segmentation from both --orig-root and --resampled-root, computes
an axial maximum-intensity projection (collapsing the z-axis), and
saves a side-by-side PNG so you can see at a glance how resampling
changes the segmentation.

The random selection is seeded so the same cases are always chosen —
run it twice with the same --seed and you get the same 6 cases.

Usage
-----
    # All defaults (2 cases per cohort, seed 42):
    python scripts/compare_seg_mips.py

    # Different seed or more cases:
    python scripts/compare_seg_mips.py --seed 7 --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write straight to a file
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_ORIG = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/vessel_segmentations"
)
_DEFAULT_RESAMPLED = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations"
)
_DEFAULT_OUT = Path.home() / "vanguard_qc_pngs" / "seg_mip_comparison.png"

# Background colours for the two rows of the figure.
_COLOR_ORIG = "white"
_COLOR_RESAMPLED = "#e8f4fd"  # light blue tint for resampled row


def _npz_path(root: Path, cohort: str, case_id: str, timepoint: int) -> Path:
    return (
        root / cohort / case_id / "images"
        / f"{case_id}_{timepoint:04d}_vessel_segmentation.npz"
    )


def _cases_in_both(orig_root: Path, resampled_root: Path, cohort: str) -> list[str]:
    """Return case IDs present in both roots for a given cohort."""
    orig_cases = {p.name for p in (orig_root / cohort).iterdir() if p.is_dir()}
    resampled_cases = {p.name for p in (resampled_root / cohort).iterdir() if p.is_dir()}
    return sorted(orig_cases & resampled_cases)


def _load_axial_mip(
    root: Path, cohort: str, case_id: str, timepoint: int, threshold: float
) -> np.ndarray:
    """Load vessel npz and return the axial MIP (max over z-axis).

    The array in the npz is shaped (256, 256, Z) — the last axis is
    depth (z / axial). Projecting along axis=2 with np.max collapses
    all z-slices into a single (256, 256) image by keeping the
    highest probability value at each (x, y) position.  Thresholding
    first (> threshold) makes the result binary so the MIP shows
    'was there *any* vessel detected at this x-y location across all
    depths?'
    """
    path = _npz_path(root, cohort, case_id, timepoint)
    arr = np.load(path)["vessel"].astype(np.float32)
    binary = arr > threshold                          # (256, 256, Z) bool
    return np.max(binary, axis=2).astype(np.float32)  # (256, 256)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side axial MIP comparison: original vs. resampled segmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--orig-root", type=Path, default=_DEFAULT_ORIG)
    parser.add_argument("--resampled-root", type=Path, default=_DEFAULT_RESAMPLED)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for case selection.")
    parser.add_argument("--timepoint", type=int, default=2,
                        help="Timepoint index to load (2 = first post-contrast).")
    parser.add_argument("--n", type=int, default=2, help="Cases per cohort.")
    parser.add_argument(
        "--cohorts", nargs="+", default=["ISPY1", "ISPY2", "NACT"],
        help="Cohort names to include.",
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary mask.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- 1. Select cases ---------------------------------------------------
    selected: list[tuple[str, str]] = []  # (cohort, case_id) pairs
    for cohort in args.cohorts:
        cases = _cases_in_both(args.orig_root, args.resampled_root, cohort)
        if len(cases) < args.n:
            raise RuntimeError(
                f"Only {len(cases)} cases found for cohort {cohort} in both roots; "
                f"need at least {args.n}."
            )
        chosen = rng.choice(cases, size=args.n, replace=False).tolist()
        for c in chosen:
            selected.append((cohort, c))
        print(f"{cohort}: selected {chosen}")

    n_cases = len(selected)  # should be args.n * len(args.cohorts) = 6

    # --- 2. Load axial MIPs ------------------------------------------------
    orig_mips: list[np.ndarray] = []
    resampled_mips: list[np.ndarray] = []

    for cohort, case_id in selected:
        orig_mip = _load_axial_mip(
            args.orig_root, cohort, case_id, args.timepoint, args.threshold
        )
        resampled_mip = _load_axial_mip(
            args.resampled_root, cohort, case_id, args.timepoint, args.threshold
        )
        orig_mips.append(orig_mip)
        resampled_mips.append(resampled_mip)
        print(
            f"  {case_id}: orig nonzero={int(orig_mip.sum())}, "
            f"resampled nonzero={int(resampled_mip.sum())}"
        )

    # --- 3. Build figure ---------------------------------------------------
    # Layout: 2 rows (original / resampled) × n_cases columns.
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_cases,
        figsize=(3.0 * n_cases, 7.0),
        gridspec_kw={"hspace": 0.10, "wspace": 0.04},
    )
    fig.patch.set_facecolor("white")

    row_labels = ["original", "resampled"]
    row_mips = [orig_mips, resampled_mips]
    row_bg_colors = [_COLOR_ORIG, _COLOR_RESAMPLED]

    for row_idx, (label, mips, bg) in enumerate(
        zip(row_labels, row_mips, row_bg_colors)
    ):
        for col_idx, ((cohort, case_id), mip) in enumerate(zip(selected, mips)):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(bg)

            # Display the MIP.  vmin=0 / vmax=1 because the values are binary
            # (0.0 = no vessel detected, 1.0 = vessel detected).
            ax.imshow(mip, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            # Column title on the top row only.
            if row_idx == 0:
                ax.set_title(f"{case_id}\n({cohort})", fontsize=9, pad=4)

        # Row label on the leftmost column.
        axes[row_idx, 0].set_ylabel(label, fontsize=11, labelpad=6)

    fig.suptitle(
        "Axial MIP: original vs. resampled vessel segmentation\n"
        f"(timepoint={args.timepoint:04d}, threshold={args.threshold}, seed={args.seed})",
        fontsize=12,
        y=1.01,
    )

    # --- 4. Save -----------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
