"""Batch 4D morphometry extraction for the weak-signal diagnostic pipeline.

Discovers all studies in the vessel segmentations directory, runs process_4d_study
for each, and writes a manifest of completed studies.

Usage:
    python graph_extraction/batch_process_4d.py \
        --input-dir /net/projects2/vanguard/vessel_segmentations \
        --output-dir report/4d_morphometry \
        [--study-ids STUDY1 STUDY2 ...]  # Optional: restrict to specific studies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from graph_extraction.processing import (  # noqa: E402
    process_4d_study,
)


def discover_all_study_ids(input_dir: Path) -> list[str]:
    """Discover study IDs from vessel segmentations directory layout.

    Expects: input_dir / SITE / STUDY_ID / images / *vessel_segmentation.npz
    Returns list of study_id strings (e.g. ["ISPY2_202539", "DUKE_001", ...]).
    """
    if not input_dir.exists() or not input_dir.is_dir():
        return []

    study_ids: list[str] = []
    for site_dir in sorted(input_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        for study_dir in sorted(site_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            images_dir = study_dir / "images"
            if not images_dir.exists():
                continue
            # Verify we have vessel segmentation files
            candidates = list(images_dir.glob("*vessel_segmentation*"))
            if candidates:
                study_ids.append(study_dir.name)

    return study_ids


def _merge_task_manifests(output_dir: Path) -> None:
    """Merge manifest_task_*.json files into manifest.json."""
    task_manifests = sorted(output_dir.glob("manifest_task_*.json"))
    if not task_manifests:
        print("[batch] No manifest_task_*.json files found. Nothing to merge.")
        return

    completed: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []

    for p in task_manifests:
        data = json.loads(p.read_text(encoding="utf-8"))
        completed.extend(data.get("completed", []))
        failed.extend(data.get("failed", []))
        skipped.extend(data.get("skipped", []))

    n_computed = sum(1 for m in completed if m.get("status") == "success")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
                "n_completed": n_computed,
                "n_skipped": len(skipped),
                "n_failed": len(failed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for p in task_manifests:
        p.unlink()
    print(f"[batch] Merged {len(task_manifests)} task manifests -> {manifest_path}")


def main() -> None:
    """Entry point for batch 4D morphometry extraction."""
    parser = argparse.ArgumentParser(
        description="Batch 4D skeleton extraction and morphometry for weak-signal diagnostics."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/net/projects2/vanguard/vessel_segmentations"),
        help="Base directory for vessel segmentations (layout: SITE/STUDY_ID/images/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for morphometry JSONs and manifest",
    )
    parser.add_argument(
        "--study-ids",
        nargs="*",
        help="Optional: restrict to these study IDs. If not given, discover all.",
    )
    parser.add_argument("--npy-channel", type=int, default=1)
    parser.add_argument("--threshold-low", type=float, default=0.5)
    parser.add_argument("--threshold-high", type=float, default=0.85)
    parser.add_argument("--max-temporal-radius", type=int, default=1)
    parser.add_argument("--min-voxels-per-timepoint", type=int, default=64)
    parser.add_argument("--min-anchor-fraction", type=float, default=0.005)
    parser.add_argument("--min-anchor-voxels", type=int, default=128)
    parser.add_argument("--min-temporal-support", type=int, default=2)
    parser.add_argument("--force-skeleton", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip studies that already have morphometry JSON (no process_4d_study call)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress skeletonize4d verbose output (faster, less I/O)",
    )
    parser.add_argument(
        "--save-center-manifold-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save full 4D manifold mask (large files)",
    )
    parser.add_argument(
        "--study-index",
        type=int,
        default=None,
        help="Process only study at this index (for SLURM array jobs). Writes manifest_task_<index>.json.",
    )
    parser.add_argument(
        "--merge-manifests",
        action="store_true",
        help="Merge manifest_task_*.json into manifest.json and remove task manifests. Run after array completes.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle --merge-manifests first (does not need study discovery)
    if args.merge_manifests:
        _merge_task_manifests(output_dir)
        return

    if args.study_ids:
        study_ids = args.study_ids
        print(f"[batch] Processing {len(study_ids)} specified studies")
    else:
        study_ids = discover_all_study_ids(args.input_dir)
        print(f"[batch] Discovered {len(study_ids)} studies in {args.input_dir}")

    if not study_ids:
        print("[batch] No studies to process. Exiting.")
        sys.exit(0)

    # Handle --study-index: process single study for SLURM array
    if args.study_index is not None:
        if args.study_index < 0 or args.study_index >= len(study_ids):
            print(
                f"[batch] study-index {args.study_index} out of range [0, {len(study_ids)-1}], exiting."
            )
            sys.exit(0)
        study_ids = [study_ids[args.study_index]]
        manifest_suffix = f"task_{args.study_index}"
    else:
        manifest_suffix = None

    manifest: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []

    for i, study_id in enumerate(study_ids, start=1):
        morphometry_path = output_dir / f"{study_id}_morphometry.json"
        if args.skip_existing and morphometry_path.exists():
            manifest.append(
                {
                    "study_id": study_id,
                    "status": "skipped",
                    "morphometry_path": str(morphometry_path),
                }
            )
            skipped.append({"study_id": study_id})
            print(f"[batch] {i}/{len(study_ids)}: {study_id} (skipped, exists)")
            continue

        print(f"[batch] {i}/{len(study_ids)}: {study_id}")
        try:
            result = process_4d_study(
                input_dir=args.input_dir,
                study_id=study_id,
                output_dir=output_dir,
                npy_channel=args.npy_channel,
                threshold_low=args.threshold_low,
                threshold_high=args.threshold_high,
                max_temporal_radius=args.max_temporal_radius,
                min_voxels_per_timepoint=args.min_voxels_per_timepoint,
                min_anchor_fraction=args.min_anchor_fraction,
                min_anchor_voxels=args.min_anchor_voxels,
                max_candidates=None,
                min_temporal_support=args.min_temporal_support,
                force_skeleton=args.force_skeleton,
                force_features=args.force_features,
                save_center_manifold_mask=args.save_center_manifold_mask,
                verbose=not args.quiet,
            )
            manifest.append(
                {
                    "study_id": study_id,
                    "status": "success",
                    "morphometry_path": str(result["morphometry_path"]),
                    "skeleton_voxels": result.get("skeleton_voxels"),
                }
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed.append({"study_id": study_id, "error": str(e)})

    manifest_path = output_dir / (
        f"manifest_{manifest_suffix}.json" if manifest_suffix else "manifest.json"
    )
    n_computed = sum(1 for m in manifest if m.get("status") == "success")
    manifest_path.write_text(
        json.dumps(
            {
                "completed": manifest,
                "failed": failed,
                "skipped": skipped,
                "n_completed": n_computed,
                "n_skipped": len(skipped),
                "n_failed": len(failed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[batch] Done. Manifest -> {manifest_path}")
    print(f"  Computed: {n_computed}, Skipped: {len(skipped)}, Failed: {len(failed)}")


if __name__ == "__main__":
    main()
