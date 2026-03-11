"""Batch 4D morphometry extraction for the weak-signal diagnostic pipeline.

Discovers all studies in the vessel segmentations directory, runs process_4d_study
for each, and writes a manifest of completed studies.

For SLURM array jobs with multi-study parallelization, use --study-range START END
and --studies-per-task N to process N studies in parallel within each array task.

Usage:
    python graph_extraction/batch_process_4d.py \
        --input-dir /net/projects2/vanguard/vessel_segmentations \
        --output-dir report/4d_morphometry \
        [--study-ids STUDY1 STUDY2 ...]  # Optional: restrict to specific studies
    # SLURM: process studies START..END-1 with N parallel workers
    python graph_extraction/batch_process_4d.py \
        --input-dir ... --output-dir ... \
        --study-range START END --studies-per-task N --chunk-id CHUNK_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from graph_extraction.processing import (  # noqa: E402
    DEFAULT_RADIOLOGIST_ANNOTATIONS_DIR,
    DEFAULT_TUMOR_MASK_DIR,
    process_4d_study,
)


def _run_one_study(
    study_id: str,
    input_dir: Path,
    output_dir: Path,
    skip_existing: bool,
    kwargs: dict,
) -> tuple[str, str, dict | str]:
    """Run process_4d_study for one study. Used by ProcessPoolExecutor.

    Returns (study_id, status, result) where status is 'success'|'skipped'|'failed'.
    """
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    morphometry_path = output_dir / f"{study_id}_morphometry.json"
    if skip_existing and morphometry_path.exists():
        return (
            study_id,
            "skipped",
            {"morphometry_path": str(morphometry_path)},
        )
    try:
        result = process_4d_study(
            input_dir=input_dir,
            study_id=study_id,
            output_dir=output_dir,
            **kwargs,
        )
        return (
            study_id,
            "success",
            {
                "morphometry_path": str(result["morphometry_path"]),
                "skeleton_voxels": result.get("skeleton_voxels"),
            },
        )
    except Exception as e:
        # Intentional: return failure result so batch can continue and report failed studies.
        return (study_id, "failed", str(e))


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
    parser.add_argument("--force-skeleton", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip studies that already have morphometry JSON (no process_4d_study call)",
    )
    parser.add_argument(
        "--save-exam-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write exam skeleton/support masks.",
    )
    parser.add_argument(
        "--save-center-manifold-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save full 4D manifold mask (large files)",
    )
    parser.add_argument(
        "--render-mip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render vessel coverage MIPs during batch processing.",
    )
    parser.add_argument("--mip-dpi", type=int, default=180)
    parser.add_argument(
        "--radiologist-annotations-dir",
        type=Path,
        default=DEFAULT_RADIOLOGIST_ANNOTATIONS_DIR,
        help="Base path for optional Duke supplemental annotations.",
    )
    parser.add_argument(
        "--tumor-mask-dir",
        type=Path,
        default=DEFAULT_TUMOR_MASK_DIR,
        help="Directory for optional tumor masks used in MIP overlays.",
    )
    parser.add_argument(
        "--study-index",
        type=int,
        default=None,
        help="Process only study at this index (for SLURM array jobs). Writes manifest_task_<index>.json. Deprecated: prefer --study-range with --chunk-id.",
    )
    parser.add_argument(
        "--study-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Process studies from index START to END-1 (for SLURM multi-study tasks). Requires --chunk-id for manifest filename.",
    )
    parser.add_argument(
        "--studies-per-task",
        type=int,
        default=1,
        help="Number of studies to run in parallel per array task (default 1). Use with --study-range.",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="Chunk/task ID for manifest_task_<chunk_id>.json. Used with --study-range.",
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

    # Handle SLURM array task modes: --study-range or --study-index
    manifest_suffix: str | None = None
    if args.study_range is not None:
        start_idx, end_idx = args.study_range[0], args.study_range[1]
        if start_idx < 0 or end_idx > len(study_ids) or start_idx >= end_idx:
            print(
                f"[batch] study-range {start_idx} {end_idx} out of range [0, {len(study_ids)}], exiting."
            )
            sys.exit(0)
        study_ids = study_ids[start_idx:end_idx]
        if args.chunk_id is not None:
            manifest_suffix = f"task_{args.chunk_id}"
        else:
            manifest_suffix = f"task_{start_idx}"
    elif args.study_index is not None:
        if args.study_index < 0 or args.study_index >= len(study_ids):
            print(
                f"[batch] study-index {args.study_index} out of range [0, {len(study_ids)-1}], exiting."
            )
            sys.exit(0)
        study_ids = [study_ids[args.study_index]]
        manifest_suffix = f"task_{args.study_index}"

    study_kwargs = {
        "force_skeleton": args.force_skeleton,
        "force_features": args.force_features,
        "save_exam_masks": args.save_exam_masks,
        "save_center_manifold_mask": args.save_center_manifold_mask,
        "render_mip": args.render_mip,
        "mip_dpi": args.mip_dpi,
        "radiologist_annotations_dir": args.radiologist_annotations_dir,
        "tumor_mask_dir": args.tumor_mask_dir,
    }

    manifest: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []

    max_workers = min(args.studies_per_task, len(study_ids))
    if max_workers > 1:
        # Parallel: run multiple studies across workers
        print(
            f"[batch] Processing {len(study_ids)} studies with {max_workers} parallel workers"
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_one_study,
                    study_id,
                    args.input_dir,
                    output_dir,
                    args.skip_existing,
                    study_kwargs,
                ): study_id
                for study_id in study_ids
            }
            for future in as_completed(futures):
                study_id = futures[future]
                try:
                    sid, status, data = future.result()
                    if status == "success":
                        manifest.append({"study_id": sid, "status": "success", **data})
                        print(f"[batch] {sid}: success")
                    elif status == "skipped":
                        manifest.append({"study_id": sid, "status": "skipped", **data})
                        skipped.append({"study_id": sid})
                        print(f"[batch] {sid}: skipped")
                    else:
                        failed.append({"study_id": sid, "error": str(data)})
                        print(f"[batch] {sid}: FAILED - {data}")
                except Exception as e:
                    failed.append({"study_id": study_id, "error": str(e)})
                    print(f"[batch] {study_id}: EXCEPTION - {e}")
    else:
        # Sequential: original loop
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
                    **study_kwargs,
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
