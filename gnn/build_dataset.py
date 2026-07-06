"""Command-line entrypoint that builds the full GNN centerline dataset cache.

Thin wrapper around :class:`gnn.data_loader.VanguardCenterlineDataset`: simply
constructing the dataset triggers ``process()`` (build + collate + cache) when
no cache is present at ``--cache-dir``. Defaults point at the real cluster
paths from the project ``CLAUDE.md`` (not the stale ``/net/projects2/...``
paths baked into some older scripts).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from gnn.data_loader import VanguardCenterlineDataset

_DEFAULT_ROOT = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies"
)
_DEFAULT_LABELS_PATH = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/pcr_labels.csv"
)
_DEFAULT_CACHE_DIR = Path(
    "/gpfs/data/karczmar-lab/workspaces/spencervenancio/gnn_cache"
)
_DEFAULT_NODE_FEATURES = "peak_time,radius"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--labels-path", type=Path, default=_DEFAULT_LABELS_PATH)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help="Where the collated cache is written. Kept out of the shared "
        "centerline tree by default so a full build doesn't write into "
        "another user's workspace.",
    )
    parser.add_argument("--id-column", type=str, default="case_id")
    parser.add_argument("--label-column", type=str, default="pcr")
    parser.add_argument(
        "--node-features",
        type=str,
        default=_DEFAULT_NODE_FEATURES,
        help="Comma-separated node feature names (see VanguardCenterlineDataset).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Optional comma-separated case-ID whitelist, e.g. for a smoke run.",
    )
    parser.add_argument(
        "--max-missing-label-frac",
        type=float,
        default=0.1,
        help="Max fraction of discovered cases allowed to be dropped for "
        "lacking a label before the build raises (see VanguardCenterlineDataset).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Always rebuild from source instead of reading an existing cache.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Archive any existing cache under --cache-dir (renames "
        "processed/ to processed_archive_<timestamp>/) before building a "
        "fresh one, so old cache versions are kept for the record instead "
        "of being overwritten. Incompatible with --no-cache, which never "
        "persists a cache to archive in the first place.",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable per-stage timing logs (on by default for full builds).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("GNN_BUILD_WORKERS", "1")),
        help="Process-pool workers for building cases in parallel (each case "
        "is built independently from its own files). Default 1 (sequential "
        "-- safe on a login node); set to match --cpus-per-task for a full "
        "Slurm build.",
    )
    return parser


def _archive_existing_cache(cache_dir: Path) -> Path | None:
    """Rename ``<cache_dir>/processed`` aside so a rebuild doesn't destroy it.

    Returns the archive path, or ``None`` if there was no existing cache.
    """
    processed_dir = cache_dir / "processed"
    if not processed_dir.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = cache_dir / f"processed_archive_{timestamp}"
    shutil.move(str(processed_dir), str(archive_dir))
    return archive_dir


def main() -> None:
    """Build (or load) the dataset cache and log the resulting graph count."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = build_parser().parse_args()
    if args.force_rebuild and args.no_cache:
        raise ValueError(
            "--force-rebuild and --no-cache are incompatible: --force-rebuild "
            "archives the on-disk cache before building a fresh persisted one, "
            "but --no-cache never writes a cache to disk, so there would be "
            "nothing new to replace the archived cache with."
        )
    node_features = tuple(args.node_features.split(","))
    cases = args.cases.split(",") if args.cases else None

    if args.force_rebuild:
        archived = _archive_existing_cache(args.cache_dir)
        if archived is not None:
            logging.info("Archived existing cache to %s", archived)
        else:
            logging.info(
                "--force-rebuild set but no existing cache found at %s",
                args.cache_dir,
            )

    dataset = VanguardCenterlineDataset(
        root=args.root,
        labels_path=args.labels_path,
        cache_dir=args.cache_dir,
        cases=cases,
        no_cache=args.no_cache,
        node_features=node_features,
        id_column=args.id_column,
        label_column=args.label_column,
        max_missing_label_frac=args.max_missing_label_frac,
        profile=not args.no_profile,
        num_workers=args.num_workers,
    )
    logging.info(
        "Dataset ready: %d graph(s) cached under %s (%d dropped for missing label)",
        len(dataset),
        args.cache_dir,
        len(dataset.dropped_case_ids),
    )


if __name__ == "__main__":
    main()
