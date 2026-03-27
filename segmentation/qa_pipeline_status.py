#!/usr/bin/env python3
"""Quick QA script to report how much data is processed at each pipeline stage.

This script is intended for interactive use from the command line. It reports,
for each stage of the pipeline:

- how many input files exist
- how many outputs have been produced
- the completion percentage
- approximate output disk usage

Defaults are set for the vanguard cluster paths but can be overridden with
command-line arguments.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


BYTES_PER_KIB = 1024.0


@dataclass
class StageConfig:
    """Configuration for a single processing stage."""

    name: str
    root: Path
    pattern: str  # glob pattern, e.g. "*.nii.gz" or "**/*_vessel_segmentation.npz"


@dataclass
class StageStatus:
    """Computed status for a processing stage."""

    name: str
    count: int
    size_bytes: int


class StudyTimepoint(NamedTuple):
    """Simple (case_id, timepoint) identifier."""

    case_id: str
    timepoint: str


def human_bytes(n: int) -> str:
    """Convert bytes to a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    value = float(n)
    while value >= BYTES_PER_KIB and i < len(units) - 1:
        value /= BYTES_PER_KIB
        i += 1
    return f"{value:.1f} {units[i]}"


def iter_files(root: Path, pattern: str) -> Iterable[Path]:
    """Yield files under root matching pattern, if root exists."""
    if not root.exists():
        return []
    return root.rglob(pattern)


def compute_stage_status(cfg: StageConfig) -> StageStatus:
    """Compute simple counts and total size for a stage."""
    count = 0
    size_bytes = 0

    for path in iter_files(cfg.root, cfg.pattern):
        try:
            stat = path.stat()
        except OSError:
            # Ignore files that disappear or cannot be stat'ed
            continue
        count += 1
        size_bytes += stat.st_size

    return StageStatus(cfg.name, count, size_bytes)


def print_stage_report(
    name: str,
    total_inputs: int,
    produced_count: int,
    produced_size_bytes: int,
) -> None:
    """Print a one-block summary for a pipeline stage."""
    pct = (produced_count / total_inputs * 100.0) if total_inputs > 0 else 0.0
    print(f"=== {name} ===")
    print(f"  Input files:        {total_inputs}")
    print(f"  Output files:       {produced_count}")
    print(f"  Completion:         {pct:5.1f}%")
    print(f"  Output disk usage:  {human_bytes(produced_size_bytes)}")
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Report how much data has been processed at each stage of the "
            "vessel-segmentation and centerline pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/net/projects2/vanguard/MAMA-MIA-syn60868042/images"),
        help="Root directory of raw images (*.nii.gz).",
    )
    parser.add_argument(
        "--vessel-dir",
        type=Path,
        default=Path("/net/projects2/vanguard/vessel_segmentations"),
        help="Root directory of vessel segmentations (*.npz).",
    )
    parser.add_argument(
        "--centerlines-3d-dir",
        type=Path,
        default=Path("/net/projects2/vanguard/centerlines"),
        help="Root directory of 3D centerline outputs.",
    )
    parser.add_argument(
        "--centerlines-4d-dir",
        type=Path,
        default=Path("/net/projects2/vanguard/centerlines_4d"),
        help="Root directory of 4D centerline / graph outputs.",
    )

    parser.add_argument(
        "--id-table",
        type=Path,
        default=Path(
            "/net/projects2/vanguard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx"
        ),
        help=(
            "Optional CSV/Excel file listing expected case IDs and timepoints "
            "(e.g. one row per (case_id, timepoint))."
        ),
    )
    parser.add_argument(
        "--id-column",
        default="case_id",
        help="Column name in id-table for case IDs.",
    )
    parser.add_argument(
        "--timepoint-column",
        default=None,
        help=(
            "Optional column name in id-table for timepoints. If omitted or "
                "missing, QA is performed at the case level only."
        ),
    )
    parser.add_argument(
        "--id-table-sheet",
        default=None,
        help="Optional sheet name when id-table is an Excel file.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point: compute and print QA summaries."""
    args = parse_args()

    # Stage 0: raw images
    images_cfg = StageConfig(
        name="Raw images (Stage 0)",
        root=args.images_dir,
        pattern="*.nii.gz",
    )
    images_status = compute_stage_status(images_cfg)

    # Stage 1: vessel segmentations
    vessel_cfg = StageConfig(
        name="Vessel segmentations (Stage 1)",
        root=args.vessel_dir,
        pattern="*_vessel_segmentation.npz",
    )
    vessel_status = compute_stage_status(vessel_cfg)

    # Stage 2: 3D centerlines
    cl3_cfg = StageConfig(
        name="Centerlines 3D (Stage 2)",
        root=args.centerlines_3d_dir,
        pattern="*",
    )
    cl3_status = compute_stage_status(cl3_cfg)

    # Stage 3: 4D centerlines / graph outputs
    cl4_cfg = StageConfig(
        name="Centerlines 4D / graphs (Stage 3)",
        root=args.centerlines_4d_dir,
        pattern="*",
    )
    cl4_status = compute_stage_status(cl4_cfg)

    print()
    print("Pipeline QA status\n")

    # Stage 1 vs raw images
    print_stage_report(
        name="Vessel segmentations vs raw images",
        total_inputs=images_status.count,
        produced_count=vessel_status.count,
        produced_size_bytes=vessel_status.size_bytes,
    )

    # Stage 2 vs vessel segmentations
    print_stage_report(
        name="Centerlines 3D vs vessel segmentations",
        total_inputs=vessel_status.count,
        produced_count=cl3_status.count,
        produced_size_bytes=cl3_status.size_bytes,
    )

    # Stage 3 vs vessel segmentations
    print_stage_report(
        name="Centerlines 4D / graphs vs vessel segmentations",
        total_inputs=vessel_status.count,
        produced_count=cl4_status.count,
        produced_size_bytes=cl4_status.size_bytes,
    )

    print("Raw images total disk usage:", human_bytes(images_status.size_bytes))

    # Optional: compare file tree against an ID/timepoint table.
    if args.id_table is not None:
        print("\nID table QA\n")

        if pd is None:
            raise SystemExit(
                "pandas is required for --id-table support; "
                "install it in your environment to use this feature."
            )

        id_table_path: Path = args.id_table
        if not id_table_path.exists():
            raise SystemExit(f"id-table not found: {id_table_path}")

        if id_table_path.suffix.lower() in {".xls", ".xlsx"}:
            # Use the first sheet by default if none is specified.
            sheet = args.id_table_sheet or 0
            id_table_df = pd.read_excel(id_table_path, sheet_name=sheet)
        else:
            id_table_df = pd.read_csv(id_table_path)

        if args.id_column not in id_table_df.columns:
            raise SystemExit(
                "id-column "
                f"'{args.id_column}' not in id-table columns: "
                f"{list(id_table_df.columns)}"
            )
        tp_col = args.timepoint_column

        if tp_col is None or tp_col not in id_table_df.columns:
            # Case-level QA only.
            expected_ids = {
                str(v) for v in id_table_df[args.id_column].dropna().astype(str)
            }

            # Discover case IDs from raw images (directory names).
            image_ids: set[str] = set()
            if args.images_dir.exists():
                for patient_dir in args.images_dir.iterdir():
                    if patient_dir.is_dir():
                        image_ids.add(patient_dir.name)

            # Discover case IDs from vessel segmentations (source/case/images layout).
            vessel_ids: set[str] = set()
            for seg_path in iter_files(args.vessel_dir, "*_vessel_segmentation.npz"):
                # Parent layout: .../<source>/<case_id>/images/file.npz
                parent = seg_path.parent
                if parent.name == "images" and parent.parent is not None:
                    vessel_ids.add(parent.parent.name)

            # Discover case IDs from 4D centerline / graph outputs.
            cl4_ids: set[str] = set()
            if args.centerlines_4d_dir.exists():
                for patient_dir in args.centerlines_4d_dir.iterdir():
                    if patient_dir.is_dir():
                        cl4_ids.add(patient_dir.name)

            missing_in_images_ids = sorted(expected_ids - image_ids)
            missing_in_vessels_ids = sorted(expected_ids - vessel_ids)
            missing_in_cl4_ids = sorted(expected_ids - cl4_ids)

            print(
                f"Total expected case IDs:                  {len(expected_ids)}",
            )
            print(f"Found in raw images:                      {len(image_ids)}")
            print(
                f"Found in vessel segmentations:            {len(vessel_ids)}",
            )
            print(
                "Found in 4D centerline / graph outputs:   " f"{len(cl4_ids)}",
            )
            print(
                f"Missing case IDs from raw images:         {len(missing_in_images_ids)}",
            )
            print(
                "Missing case IDs from vessel "
                f"segmentations:             {len(missing_in_vessels_ids)}",
            )
            print(
                "Missing case IDs from 4D centerline "
                f"outputs:                   {len(missing_in_cl4_ids)}",
            )

            max_show = 10
            if missing_in_images_ids:
                print("\nFirst few missing case IDs in raw images:")
                for pid in missing_in_images_ids[:max_show]:
                    print(f"  {pid}")
                if len(missing_in_images_ids) > max_show:
                    print(
                        f"  ... and {len(missing_in_images_ids) - max_show} more",
                    )

            if missing_in_vessels_ids:
                print("\nFirst few missing case IDs in vessel segmentations:")
                for pid in missing_in_vessels_ids[:max_show]:
                    print(f"  {pid}")
                if len(missing_in_vessels_ids) > max_show:
                    print(
                        f"  ... and {len(missing_in_vessels_ids) - max_show} more",
                    )
        else:
            # Full (case_id, timepoint) QA when timepoint column is available.
            expected_pairs = {
                StudyTimepoint(str(row[args.id_column]), str(row[tp_col]))
                for _, row in id_table_df.iterrows()
            }

            # Discover (case_id, timepoint) pairs from raw images.
            image_pairs: set[StudyTimepoint] = set()
            patt_img = re.compile(
                r"^(?P<study>.+)_(?P<tp>\d{4})\.nii\.gz$",
                re.IGNORECASE,
            )
            for img_path in iter_files(args.images_dir, "*.nii.gz"):
                case_id = img_path.parent.name
                m = patt_img.match(img_path.name)
                if not m:
                    continue
                tp = m.group("tp")
                image_pairs.add(StudyTimepoint(case_id, tp))

            # Discover (case_id, timepoint) pairs from vessel segmentations.
            vessel_pairs: set[StudyTimepoint] = set()
            patt_seg = re.compile(
                r"^(?P<study>.+)_(?P<tp>\d{4})_vessel_segmentation\.npz$",
                re.IGNORECASE,
            )
            for seg_path in iter_files(
                args.vessel_dir,
                "*_vessel_segmentation.npz",
            ):
                m = patt_seg.match(seg_path.name)
                if not m:
                    continue
                case_id = m.group("study")
                tp = m.group("tp")
                vessel_pairs.add(StudyTimepoint(case_id, tp))

            missing_in_images = sorted(expected_pairs - image_pairs)
            missing_in_vessels = sorted(expected_pairs - vessel_pairs)

            print(
                "Total expected (case_id, timepoint) pairs: " f"{len(expected_pairs)}",
            )
            print(
                f"Found in raw images:                      {len(image_pairs)}",
            )
            print(
                "Found in vessel segmentations:            " f"{len(vessel_pairs)}",
            )
            print(
                f"Missing from raw images:                  {len(missing_in_images)}",
            )
            print(
                "Missing from vessel segmentations:        "
                f"{len(missing_in_vessels)}",
            )

            max_show = 10
            if missing_in_images:
                print("\nFirst few missing in raw images:")
                for pair in missing_in_images[:max_show]:
                    print(f"  {pair.case_id}, timepoint {pair.timepoint}")
                if len(missing_in_images) > max_show:
                    print(
                        f"  ... and {len(missing_in_images) - max_show} more",
                    )

            if missing_in_vessels:
                print("\nFirst few missing in vessel segmentations:")
                for pair in missing_in_vessels[:max_show]:
                    print(f"  {pair.case_id}, timepoint {pair.timepoint}")
                if len(missing_in_vessels) > max_show:
                    print(
                        f"  ... and {len(missing_in_vessels) - max_show} more",
                    )


if __name__ == "__main__":
    main()
