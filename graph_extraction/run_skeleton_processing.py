"""Single entrypoint for skeleton extraction and morphometry generation.

This script supports both 3D and 4D extraction and always follows this flow:
1) Ensure skeleton output exists (compute if missing or forced).
2) Ensure morphometry output exists (compute if missing or forced).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

DEFAULT_SEGMENTATION_DIR = Path("/net/projects2/vanguard/vessel_segmentations")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run 3D/4D skeleton extraction and morphometry in one pipeline."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_3d = subparsers.add_parser("3d", help="Process one 3D segmentation file.")
    parser_3d.add_argument("--input-file", type=Path, required=True, help="Input .npy file.")
    parser_3d.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser_3d.add_argument("--threshold-low", type=float, default=0.5)
    parser_3d.add_argument("--npy-channel", type=int, default=1)
    parser_3d.add_argument("--force-skeleton", action="store_true")
    parser_3d.add_argument("--force-features", action="store_true")

    parser_4d = subparsers.add_parser("4d", help="Process one exam across all timepoints.")
    parser_4d.add_argument("--study-id", type=str, required=True)
    parser_4d.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="Directory containing per-timepoint .npy segmentation files.",
    )
    parser_4d.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser_4d.add_argument("--npy-channel", type=int, default=1)
    parser_4d.add_argument("--threshold-low", type=float, default=0.5)
    parser_4d.add_argument("--threshold-high", type=float, default=0.85)
    parser_4d.add_argument("--max-temporal-radius", type=int, default=1)
    parser_4d.add_argument("--min-voxels-per-timepoint", type=int, default=64)
    parser_4d.add_argument("--min-anchor-fraction", type=float, default=0.005)
    parser_4d.add_argument("--min-anchor-voxels", type=int, default=128)
    parser_4d.add_argument("--max-candidates", type=int, default=None)
    parser_4d.add_argument("--min-temporal-support", type=int, default=2)
    parser_4d.add_argument("--force-skeleton", action="store_true")
    parser_4d.add_argument("--force-features", action="store_true")
    parser_4d.add_argument(
        "--save-center-manifold-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the full 4D manifold mask as an output .npy.",
    )

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    from graph_extraction.processing import (
        process_3d_case,
        process_4d_study,
    )

    start = time.perf_counter()

    if args.mode == "3d":
        result = process_3d_case(
            input_file=args.input_file,
            output_dir=args.output_dir,
            threshold_low=args.threshold_low,
            npy_channel=args.npy_channel,
            force_skeleton=args.force_skeleton,
            force_features=args.force_features,
        )
    else:
        result = process_4d_study(
            input_dir=args.input_dir,
            study_id=args.study_id,
            output_dir=args.output_dir,
            npy_channel=args.npy_channel,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
            max_temporal_radius=args.max_temporal_radius,
            min_voxels_per_timepoint=args.min_voxels_per_timepoint,
            min_anchor_fraction=args.min_anchor_fraction,
            min_anchor_voxels=args.min_anchor_voxels,
            max_candidates=args.max_candidates,
            min_temporal_support=args.min_temporal_support,
            force_skeleton=args.force_skeleton,
            force_features=args.force_features,
            save_center_manifold_mask=args.save_center_manifold_mask,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "processing_summary.json"
    result["elapsed_seconds"] = float(time.perf_counter() - start)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[done] Pipeline finished")
    print(f"  mode: {result['mode']}")
    print(f"  skeleton: {result['skeleton_path']}")
    print(f"  morphometry: {result['morphometry_path']}")
    if result.get("support_path") is not None:
        print(f"  support: {result['support_path']}")
    if result.get("manifold_path") is not None:
        print(f"  manifold_4d: {result['manifold_path']}")
    print(f"  summary: {summary_path}")
    print(f"[done] Total time: {result['elapsed_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
