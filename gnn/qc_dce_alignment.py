#!/usr/bin/env python3
r"""Visual QC: confirm raw DCE volumes align with the GNN skeleton/support masks.

Graph/skeleton coordinates are saved in the orientation of the segmentation
pipeline's outputs, not necessarily the raw SimpleITK array orientation, so
before sampling kinetic node features from the raw DCE series (``gnn.raw_dce``),
this overlays a raw DCE slice with the skeleton voxels used for those features
(colored by raw peak-contrast time) and the support-mask contour, for a
handful of cases -- the same alignment path ``gnn.data_loader`` uses.

Run from the repo root with the vanguard environment activated:

    PYTHONPATH=. python gnn/qc_dce_alignment.py \\
        --centerline-root /gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies \\
        --dce-root /gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images \\
        --case-ids DUKE_001,DUKE_002 \\
        --out-dir analysis/gnn_dce_qc
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT_CONFIG
from gnn.raw_dce import discover_raw_dce_paths, load_raw_dce_series

_CENTERLINE_PATTERN: str = DEFAULT_CONFIG["feature_toggles"]["centerline_file_pattern"]
_SUPPORT_PATTERN: str = DEFAULT_CONFIG["feature_toggles"][
    "deepsets_support_mask_pattern"
]
CONTOUR_LEVEL = 0.5


def _find_study_dir(centerline_root: Path, case_id: str) -> Path:
    """Resolve ``<centerline_root>/<dataset>/<case_id>`` without hardcoding the dataset name."""
    matches = sorted(centerline_root.glob(f"*/{case_id}"))
    if not matches:
        raise FileNotFoundError(
            f"No study directory found for case_id={case_id} under {centerline_root}"
        )
    if len(matches) > 1:
        raise ValueError(f"Ambiguous study directory for case_id={case_id}: {matches}")
    return matches[0]


def _choose_axial_slice_z(skeleton_zyx: np.ndarray) -> int:
    """Pick the axial slice with the most skeleton voxels, for a representative overlay."""
    counts = skeleton_zyx.sum(axis=(1, 2))
    return int(np.argmax(counts))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the alignment QC script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument(
        "--case-ids", type=str, required=True, help="Comma-separated case IDs."
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=1,
        help="Position (not raw phase number) of the DCE phase shown as the "
        "background slice; clamped to the last available phase.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/gnn_dce_qc"))
    return parser.parse_args()


def main() -> None:
    """Load each case's skeleton/support masks + raw DCE series and write an overlay PNG."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]

    for case_id in case_ids:
        study_dir = _find_study_dir(args.centerline_root, case_id)
        skeleton = np.load(
            study_dir / _CENTERLINE_PATTERN.format(case_id=case_id)
        ).astype(bool, copy=False)
        support = np.load(study_dir / _SUPPORT_PATTERN.format(case_id=case_id)).astype(
            bool, copy=False
        )
        shape_zyx = tuple(int(v) for v in skeleton.shape)

        summary = json.loads((study_dir / "run_summary.json").read_text())
        study_timepoints = summary["study_timepoints"]

        dce_paths = discover_raw_dce_paths(args.dce_root, case_id, study_timepoints)
        dce_4d = load_raw_dce_series(dce_paths, expected_shape_zyx=shape_zyx)
        assert dce_4d.shape[1:] == shape_zyx, (
            f"case={case_id}: aligned raw DCE shape {dce_4d.shape[1:]} != "
            f"skeleton/support shape {shape_zyx}"
        )

        time_idx = int(min(args.time_index, dce_4d.shape[0] - 1))
        slice_z = _choose_axial_slice_z(skeleton)

        coords_zyx = np.argwhere(skeleton)
        on_slice = coords_zyx[coords_zyx[:, 0] == slice_z]
        xs = on_slice[:, 2].astype(float)
        ys = on_slice[:, 1].astype(float)
        colors = [
            float(np.argmax(dce_4d[:, z, y, x] - dce_4d[0, z, y, x]))
            for z, y, x in on_slice
        ]

        slice_img = dce_4d[time_idx, slice_z, :, :]
        support_slice = support[slice_z, :, :].astype(float)
        finite = slice_img[np.isfinite(slice_img)]
        vmin, vmax = (
            float(np.percentile(finite, 1.0)),
            float(np.percentile(finite, 99.0)),
        )
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max()) + 1.0

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(slice_img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if np.any(support_slice > CONTOUR_LEVEL):
            ax.contour(
                support_slice, levels=[CONTOUR_LEVEL], colors="cyan", linewidths=1.0
            )
        if xs.size:
            sc = ax.scatter(xs, ys, c=colors, cmap="plasma", s=12, edgecolors="none")
            fig.colorbar(
                sc, ax=ax, fraction=0.046, pad=0.04, label="peak_time (raw idx)"
            )
        ax.set_title(f"{case_id} — z={slice_z} — raw DCE t={time_idx:04d}")
        ax.set_xlabel("x (voxel)")
        ax.set_ylabel("y (voxel)")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
        out_path = args.out_dir / f"gnn_dce_alignment_{safe}_z{slice_z}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
