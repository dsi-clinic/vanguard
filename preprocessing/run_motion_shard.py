"""Run one restartable exam-level motion-correction shard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from preprocessing.motion import motion_correct_exam_to_shard
from preprocessing.spgr import read_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--row-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
        help="Zero-based manifest row; defaults to SLURM_ARRAY_TASK_ID.",
    )
    return parser.parse_args()


def main() -> None:
    """Load one row, run motion correction, and print its shard metadata."""
    args = _parse_args()
    records = read_manifest(args.manifest)
    if not 0 <= int(args.row_index) < len(records):
        message = f"row-index must be in [0, {len(records) - 1}]"
        raise ValueError(message)
    metadata = motion_correct_exam_to_shard(
        records[int(args.row_index)],
        output_root=args.output_root,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
