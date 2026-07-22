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
_DEFAULT_DCE_ROOT = Path("/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images")
_DEFAULT_CACHE_DIR = Path(
    "/gpfs/data/karczmar-lab/workspaces/spencervenancio/gnn_cache"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--labels-path", type=Path, default=_DEFAULT_LABELS_PATH)
    parser.add_argument(
        "--dce-root",
        type=Path,
        default=_DEFAULT_DCE_ROOT,
        help="Root of the raw DCE-MRI NIfTI tree "
        "(<dce-root>/<case_id>/<case_id>_NNNN.nii.gz), used for the kinetic "
        "node features.",
    )
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
        "--kinetic-baseline-frame-count",
        type=int,
        default=None,
        help="Dataset-wide precontrast frame count (e.g. 5 for UChicago ultrafast). "
        "When set, overrides each case's run_summary.json contract; None keeps the "
        "per-case run_summary/legacy behavior (MAMA-MIA).",
    )
    parser.add_argument(
        "--kinetic-relative-enhancement",
        action="store_true",
        help="Compute relative signal change (S-S0)/S0 instead of absolute "
        "enhancement. Pair with --kinetic-baseline-frame-count for ultrafast "
        "(e.g. UChicago raw-signal DCE).",
    )
    parser.add_argument(
        "--node-mode",
        type=str,
        default="voxel",
        help="Node granularity: 'voxel' (one node per skeleton voxel) or "
        "'segment' (one node per vessel segment / line graph). See "
        "gnn/DESIGN_segment_graph.md. Note the two modes have different "
        "node-feature vocabularies, so pass a matching --node-features.",
    )
    parser.add_argument(
        "--node-features",
        type=str,
        default=None,
        help="Comma-separated node feature names (see VanguardCenterlineDataset). "
        "Defaults to the node mode's own default feature set when omitted.",
    )
    parser.add_argument(
        "--edge-features",
        type=str,
        default=None,
        help="Comma-separated edge feature names, only for --node-mode junction "
        "(the segment summary rides on the edges there). Defaults to the "
        "junction mode's own default edge feature set when omitted.",
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
        "--graph-features",
        type=str,
        default=None,
        help="Comma-separated graph-level clinical covariate names (see "
        "gnn/clinical.py for the supported vocabulary). Opt-in, mode-agnostic "
        "(unlike --node-features/--edge-features). Requires "
        "--patient-info-dir or --clinical-excel.",
    )
    parser.add_argument(
        "--patient-info-dir",
        type=Path,
        default=None,
        help="Directory of per-case patient_info JSON files, for "
        "--graph-features. Preferred over --clinical-excel when both are set.",
    )
    parser.add_argument(
        "--clinical-excel",
        type=Path,
        default=None,
        help="Excel clinical/imaging metadata file, for --graph-features "
        "(fallback if --patient-info-dir is not set).",
    )
    parser.add_argument(
        "--max-missing-clinical-frac",
        type=float,
        default=0.1,
        help="Max fraction of discovered cases allowed to be dropped for "
        "lacking a clinical row before the build raises, when "
        "--graph-features is set (see VanguardCenterlineDataset).",
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
        "--regenerate-graph-feature-inputs",
        action="store_true",
        help="Load the existing cache and (re)write only the raw "
        "graph_feature_inputs.csv sidecar, without rebuilding the graphs. Use "
        "to migrate a cache built before the per-fold graph-feature "
        "preprocessing fix (which baked whole-cohort-fit, leaky features into "
        "the graphs). Requires the same --graph-features (and clinical source) "
        "the cache was built with.",
    )
    parser.add_argument(
        "--allow-manifest-mismatch",
        action="store_true",
        help="Load an existing cache even if its cache_manifest.json records "
        "different settings (root, labels, node features, ...) than "
        "requested here, instead of failing loudly. Only use this once "
        "you've confirmed the mismatch is benign.",
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
    parser.add_argument(
        "--breast-split",
        type=str,
        default=None,
        choices=["single"],
        help="Harmonize bilateral cases down to their tumor-bearing breast "
        "(see gnn/breast_split.py, gnn/build_single_breast_skeletons.py). "
        "Native unilateral cases are always used unchanged. Requires "
        "--breast-split-skeleton-root and a clinical source "
        "(--patient-info-dir or --clinical-excel).",
    )
    parser.add_argument(
        "--breast-split-skeleton-root",
        type=Path,
        default=None,
        help="Root of precomputed single-breast skeletons, written by "
        "gnn.build_single_breast_skeletons. Required with --breast-split single.",
    )
    parser.add_argument(
        "--max-missing-breast-split-frac",
        type=float,
        default=0.1,
        help="Max fraction of discovered cases allowed to be dropped for "
        "being bilateral-but-excluded-by-the-splitter or missing a clinical "
        "row for laterality, when --breast-split is set (see "
        "VanguardCenterlineDataset).",
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
    if args.regenerate_graph_feature_inputs and (args.force_rebuild or args.no_cache):
        raise ValueError(
            "--regenerate-graph-feature-inputs only rewrites the sidecar for an "
            "existing on-disk cache; it is incompatible with --force-rebuild and "
            "--no-cache, which (re)build the graphs from source."
        )
    # None -> let the dataset resolve the node mode's own default feature set.
    node_features = tuple(args.node_features.split(",")) if args.node_features else None
    edge_features = tuple(args.edge_features.split(",")) if args.edge_features else None
    graph_features = (
        tuple(args.graph_features.split(",")) if args.graph_features else None
    )
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
        dce_root=args.dce_root,
        cache_dir=args.cache_dir,
        cases=cases,
        no_cache=args.no_cache,
        kinetic_baseline_frame_count=args.kinetic_baseline_frame_count,
        kinetic_relative_enhancement=args.kinetic_relative_enhancement,
        node_mode=args.node_mode,
        node_features=node_features,
        edge_features=edge_features,
        graph_features=graph_features,
        patient_info_dir=args.patient_info_dir,
        clinical_excel=args.clinical_excel,
        max_missing_clinical_frac=args.max_missing_clinical_frac,
        id_column=args.id_column,
        label_column=args.label_column,
        max_missing_label_frac=args.max_missing_label_frac,
        profile=not args.no_profile,
        num_workers=args.num_workers,
        allow_manifest_mismatch=args.allow_manifest_mismatch,
        breast_split_mode=args.breast_split,
        breast_split_skeleton_root=args.breast_split_skeleton_root,
        max_missing_breast_split_frac=args.max_missing_breast_split_frac,
    )
    logging.info(
        "Dataset ready: %d graph(s) cached under %s (%d dropped)",
        len(dataset),
        args.cache_dir,
        len(dataset.dropped_case_ids),
    )

    if args.regenerate_graph_feature_inputs:
        sidecar = dataset.regenerate_graph_feature_inputs()
        logging.info("Regenerated raw graph-feature-input sidecar at %s", sidecar)


if __name__ == "__main__":
    main()
