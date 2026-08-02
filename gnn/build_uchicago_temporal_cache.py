"""CLI for building labeled or unlabeled UChicago temporal graph caches."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gnn.uchicago_temporal_data import build_temporal_cache


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse manifest and output paths for one temporal cache build."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-labeled-manifest",
        type=Path,
        help="Remove every patient_key in this labeled manifest before building.",
    )
    parser.add_argument(
        "--cases",
        help="Optional comma-separated exam_id whitelist for a Slurm smoke build.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse existing per-exam .pt files in the cache directory.",
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument(
        "--defer-manifest",
        action="store_true",
        help="Build graph files without writing the shared final cache manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build the requested UChicago temporal cache."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    records = build_temporal_cache(
        manifest_path=args.manifest,
        centerline_root=args.centerline_root,
        dce_root=args.dce_root,
        cache_dir=args.cache_dir,
        exclude_labeled_manifest=args.exclude_labeled_manifest,
        cases=(
            [case.strip() for case in args.cases.split(",") if case.strip()]
            if args.cases
            else None
        ),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        resume=args.resume,
        write_manifest=not args.defer_manifest,
    )
    logging.info("Built %d UChicago temporal exam graphs", len(records))


if __name__ == "__main__":
    main()
