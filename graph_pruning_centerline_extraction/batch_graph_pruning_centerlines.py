#!/usr/bin/env python3
"""Batch graph pruning centerline extraction for vessel segmentations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Return segmentation files matching pattern under input_dir."""
    parser = argparse.ArgumentParser(
        description=(
            "Run graph pruning centerline extraction on vessel segmentation .npy files."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/net/projects2/vanguard/vessel_segmentations",
        help="Directory containing vessel segmentation .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/net/projects2/vanguard/graph_pruning_outdir",
        help="Directory to write graph pruning JSON outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="*_vessel_segmentation.npy",
        help="Glob pattern to match vessel segmentation files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for files recursively under input-dir.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for skeletonization (voxels below are ignored).",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        help="Process a single file by index (0-based) within the sorted list.",
    )
    parser.add_argument(
        "--file-start",
        type=int,
        help="Start index (0-based, inclusive) for batch processing.",
    )
    parser.add_argument(
        "--file-end",
        type=int,
        help="End index (0-based, inclusive) for batch processing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files with existing non-empty JSON outputs.",
    )
    return parser


def list_segmentation_files(
    input_dir: Path, pattern: str, recursive: bool
) -> list[Path]:
    """Return segmentation files matching pattern under input_dir."""
    if recursive:
        files = sorted(input_dir.rglob(pattern))
    else:
        files = sorted(input_dir.glob(pattern))
    return [path for path in files if path.is_file()]


def select_files(
    files: list[Path], file_index: int | None, file_start: int | None, file_end: int | None
) -> list[Path]:
    """Select a subset of files based on index/range arguments."""
    if not files:
        return []

    if file_index is not None:
        if file_index < 0 or file_index >= len(files):
            raise IndexError(
                f"file-index {file_index} is out of range (0-{len(files) - 1})"
            )
        return [files[file_index]]

    if file_start is not None or file_end is not None:
        start = 0 if file_start is None else file_start
        end = len(files) - 1 if file_end is None else file_end
        if start < 0 or end < 0 or start > end:
            raise ValueError("Invalid file-start/file-end range")
        if start >= len(files):
            return []
        end = min(end, len(files) - 1)
        return files[start : end + 1]

    return files


def output_json_path(output_dir: Path, input_file: Path) -> Path:
    """Compute the output JSON path for a given input file."""
    return output_dir / f"{input_file.stem}_morphometry.json"


def main() -> int:
    """Main entry point for batch graph pruning centerline extraction."""
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root / "graph_pruning_centerline_extraction"))
    sys.path.append(
        str(project_root / "graph_pruning_centerline_extraction" / "skeleton3d_utils")
    )

    from skeleton3d_utils.pipeline import process_vessel_image

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = list_segmentation_files(input_dir, args.pattern, args.recursive)
    if not all_files:
        print(f"No files found in {input_dir} matching pattern: {args.pattern}")
        return 0

    try:
        files_to_process = select_files(
            all_files, args.file_index, args.file_start, args.file_end
        )
    except (IndexError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    if not files_to_process:
        print("No files selected for processing.")
        return 0

    print(f"Processing {len(files_to_process)} file(s) with threshold {args.threshold}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    successes = 0
    failures = 0

    for input_file in files_to_process:
        output_path = output_json_path(output_dir, input_file)
        if args.resume and output_path.exists() and output_path.stat().st_size > 0:
            print(f"[SKIP] {input_file.name} already processed: {output_path}")
            successes += 1
            continue

        try:
            process_vessel_image(str(input_file), args.threshold, str(output_dir))
            if output_path.exists() and output_path.stat().st_size > 0:
                successes += 1
            else:
                print(f"[WARN] Missing output for {input_file.name}")
                failures += 1
        except Exception as exc:
            print(f"[ERROR] Failed for {input_file.name}: {exc}")
            failures += 1

    print("Batch graph pruning complete.")
    print(f"  Successful: {successes}")
    print(f"  Failed: {failures}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
