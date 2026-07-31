r"""Exploratory: run TC4D once over a single combined HR+UFAST phase sequence.

**Not the supported route.** Same caveat as ``analysis.skeletonize_ufast_vessels``
and ``analysis.merge_ufast_hr_skeletons``: the HR-mapped skeleton
(``preprocessing.pipeline.map_case``'s output) is the one trusted/supported
centerline for an exam. Those two scripts run TC4D separately on the HR and
UFAST-direct phase stacks and reconcile the two finished skeletons after the
fact -- which means TC4D never gets to use cross-route temporal consistency,
only within-route consistency. This script instead builds *one* priority
stack containing every phase from both routes, in one physical time order,
and calls the unmodified ``graph_extraction.tc4d.run_tc4d_from_priority``
exactly once, to check whether skeletonizing jointly recovers structure that
neither separate run (nor the post-hoc merge) does.

Combined phase order (physical acquisition order, one sequence):
  1. HR pre-contrast phase (HR phase index 0 -- ``preprocessing.pipeline``
     already treats HR phase 0 as the pre-contrast/motion-reference phase,
     see ``hr_motion_reference_phase`` in ``preprocessing_provenance.json``).
  2. UFAST pre-contrast frames (the first
     ``provenance["ufast_source"]["baseline_frame_count"]`` UFAST phases).
  3. All remaining UFAST post-contrast frames.
  4. Remaining HR phases (HR phase indices 1..n-1), i.e. HR post-contrast.

All phases must share one spatial grid before they can sit in one priority
stack. UFAST-direct phases already are on the UFAST grid; every HR phase is
resampled onto it individually here (linear interpolation, same
``resample_to_geometry`` call ``analysis.segment_ufast_vessels.load_hr_vessel_on_ufast_grid``
uses for its combined HR-vs-UFAST MIP, applied per-phase instead of
max-combined across phases).

Usage::

    micromamba run -n vanguard python -m analysis.skeletonize_combined_hr_ufast \\
        --preprocessing-root /gpfs/data/karczmar-lab/workspaces/saritbose/preprocessing_out \\
        --output-root /gpfs/data/karczmar-lab/workspaces/saritbose/ufast-seg \\
        --exam-id uch_nac_1_3_1004294011319227272751368633606051346978841464528
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from graph_extraction.tc4d import run_tc4d_from_priority  # noqa: E402
from preprocessing.dicom import DicomGeometry  # noqa: E402
from preprocessing.spatial import resample_to_geometry  # noqa: E402


def load_hr_vessel_phases_on_ufast_grid(
    preprocessing_root: Path, exam_id: str
) -> np.ndarray:
    """Every HR-route vessel-probability phase, individually resampled onto the UFAST grid.

    Returns ``(n_hr_phases, y, x, z)``, in HR acquisition order. Unlike
    ``analysis.segment_ufast_vessels.load_hr_vessel_on_ufast_grid``, this keeps
    phases separate (needed to place them at their own positions in the
    combined time order) rather than max-combining them into one volume.
    """
    work_dir = preprocessing_root / "work" / exam_id
    provenance = json.loads((work_dir / "preprocessing_provenance.json").read_text())
    hr_geometry = DicomGeometry.from_dict(provenance["hr_source"]["geometry"])
    ufast_geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])

    vessel_dir = work_dir / "hr_vessel_predictions"
    paths = sorted(vessel_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no HR vessel phases found under {vessel_dir}")

    phases_on_ufast_grid = []
    for path in paths:
        with np.load(path, allow_pickle=False) as loaded:
            phase_yxz = np.asarray(loaded["vessel"], dtype=np.float32)
        phase_zyx = np.transpose(phase_yxz, (2, 0, 1))
        if phase_zyx.shape != tuple(hr_geometry.shape_zyx):
            raise ValueError(
                f"case={exam_id}: HR vessel phase shape {phase_zyx.shape} != "
                f"HR geometry {hr_geometry.shape_zyx} ({path.name})"
            )
        mapped_zyx = resample_to_geometry(phase_zyx, hr_geometry, ufast_geometry)
        phases_on_ufast_grid.append(np.transpose(mapped_zyx, (1, 2, 0)))
    return np.stack(phases_on_ufast_grid, axis=0)


def load_ufast_vessel_phases(output_root: Path, exam_id: str) -> np.ndarray:
    """Every UFAST-direct vessel-probability phase, already on the UFAST grid.

    Same array order and source directory as
    ``analysis.skeletonize_ufast_vessels.load_ufast_vessel_phase_stack``.
    """
    vessel_dir = output_root / exam_id / "ufast_vessel_predictions"
    paths = sorted(vessel_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no UFAST vessel phases found under {vessel_dir}")
    phases = []
    for path in paths:
        with np.load(path, allow_pickle=False) as loaded:
            phases.append(np.asarray(loaded["vessel"], dtype=np.float32))
    return np.stack(phases, axis=0)


def build_combined_priority_stack(
    preprocessing_root: Path, output_root: Path, exam_id: str
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Assemble the single ordered (t,y,x,z) priority stack plus its phase manifest.

    Manifest entries describe, per output stack index, which route/phase it
    came from and which of the four ordering blocks it belongs to -- so a
    downstream consumer (or a human comparing this to ``merged_tc4d``) can
    trace any voxel's temporal-support contributions back to real phases.
    """
    work_dir = preprocessing_root / "work" / exam_id
    provenance = json.loads((work_dir / "preprocessing_provenance.json").read_text())
    # Read directly off provenance rather than via
    # analysis.bolus_arrival_time.resolve_baseline_frame_count: that helper
    # expects a real ExamCase and a run_summary.json-shaped dict (it prefers
    # run_summary's kinetic_feature_policy, falling back to this same
    # provenance field only for older mapped outputs). This script has
    # neither a case object nor a run_summary at this point, so read the
    # provenance field directly -- the same source pipeline.py itself uses
    # (``record.ufast_baseline_frame_count`` / this field) to record it.
    baseline_frame_count = int(provenance["ufast_source"]["baseline_frame_count"])

    hr_phases = load_hr_vessel_phases_on_ufast_grid(preprocessing_root, exam_id)
    ufast_phases = load_ufast_vessel_phases(output_root, exam_id)
    if not 1 <= baseline_frame_count < ufast_phases.shape[0]:
        raise ValueError(
            f"case={exam_id}: baseline_frame_count={baseline_frame_count} invalid "
            f"for {ufast_phases.shape[0]} UFAST phases"
        )
    #: Need at least a pre- and a post-contrast HR phase to split the block.
    min_hr_phases = 2
    if hr_phases.shape[0] < min_hr_phases:
        raise ValueError(
            f"case={exam_id}: expected at least 2 HR phases (pre + post-contrast), "
            f"found {hr_phases.shape[0]}"
        )

    hr_pre = hr_phases[0:1]
    hr_post = hr_phases[1:]
    ufast_pre = ufast_phases[:baseline_frame_count]
    ufast_post = ufast_phases[baseline_frame_count:]

    priority_tyxz = np.concatenate([hr_pre, ufast_pre, ufast_post, hr_post], axis=0)

    manifest: list[dict[str, object]] = []
    for block_name, route, phases, start_index in (
        ("hr_pre_contrast", "hr", hr_pre, 0),
        ("ufast_pre_contrast", "ufast", ufast_pre, 0),
        ("ufast_post_contrast", "ufast", ufast_post, baseline_frame_count),
        ("hr_post_contrast", "hr", hr_post, 1),
    ):
        for offset in range(phases.shape[0]):
            manifest.append(
                {
                    "block": block_name,
                    "route": route,
                    "source_phase_index": start_index + offset,
                }
            )
    for stack_index, entry in enumerate(manifest):
        entry["stack_index"] = stack_index
    return priority_tyxz, manifest


def skeletonize_exam(preprocessing_root: Path, output_root: Path, exam_id: str) -> Path:
    """Run TC4D once on one exam's combined HR+UFAST priority stack. Returns the output dir."""
    output_dir = output_root / exam_id / "combined_tc4d"
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing TC4D output: {output_dir}"
        )

    priority_tyxz, manifest = build_combined_priority_stack(
        preprocessing_root, output_root, exam_id
    )
    t_start = time.perf_counter()
    result, params, diagnostics = run_tc4d_from_priority(priority_tyxz)
    wall_seconds = float(time.perf_counter() - t_start)

    skeleton_yxz = np.asarray(result["exam_mask"], dtype=bool)
    support_yxz = np.asarray(result["support_mask"], dtype=bool)

    output_dir.mkdir(parents=True)
    try:
        np.save(
            output_dir / "exam_skeleton_zyx.npy",
            np.transpose(skeleton_yxz, (2, 0, 1)).astype(np.uint8),
        )
        np.save(
            output_dir / "exam_support_zyx.npy",
            np.transpose(support_yxz, (2, 0, 1)).astype(np.uint8),
        )
        np.save(
            output_dir / "center_manifold_4d_yxz.npy",
            np.asarray(result["mask_4d"], dtype=np.uint8),
        )
        provenance = {
            "exam_id": exam_id,
            "route": "combined_hr_ufast_tc4d (exploratory, single joint TC4D run)",
            "input_phases": int(priority_tyxz.shape[0]),
            "input_array_order": "time,y,x,z",
            "phase_order": [
                "hr_pre_contrast",
                "ufast_pre_contrast",
                "ufast_post_contrast",
                "hr_post_contrast",
            ],
            "phase_manifest": manifest,
            "skeleton_voxels": int(skeleton_yxz.sum()),
            "support_voxels": int(support_yxz.sum()),
            "effective_min_temporal_support": int(
                result["effective_min_temporal_support"]
            ),
            "params": params,
            "diagnostics": diagnostics,
            "wall_seconds": wall_seconds,
        }
        (output_dir / "combined_tc4d_provenance.json").write_text(
            json.dumps(provenance, indent=2)
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir


def is_combined_tc4d_complete(output_root: Path, exam_id: str) -> bool:
    """Whether an exam's combined-TC4D output has all expected artifacts.

    Same rationale as ``analysis.skeletonize_ufast_vessels.is_tc4d_complete``:
    a preemption kill doesn't give the except-block a chance to clean up, so a
    plain directory-exists check can silently treat a partial run as done
    forever.
    """
    output_dir = output_root / exam_id / "combined_tc4d"
    expected = (
        "exam_skeleton_zyx.npy",
        "exam_support_zyx.npy",
        "center_manifold_4d_yxz.npy",
        "combined_tc4d_provenance.json",
    )
    return all((output_dir / name).exists() for name in expected)


def discover_exam_ids(preprocessing_root: Path, output_root: Path) -> list[str]:
    """Every exam with both HR and UFAST-direct vessel predictions available."""
    exam_ids = []
    for d in sorted(output_root.iterdir()):
        if not d.is_dir() or not (d / "ufast_vessel_predictions").exists():
            continue
        if not (
            preprocessing_root / "work" / d.name / "hr_vessel_predictions"
        ).exists():
            continue
        exam_ids.append(d.name)
    return exam_ids


def main() -> None:
    """CLI entrypoint: run TC4D once on a combined HR+UFAST priority stack, per exam."""
    p = argparse.ArgumentParser(
        description=(
            "Exploratory: run one unmodified TC4D call over a single combined "
            "HR+UFAST phase sequence (see module docstring for phase order)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preprocessing-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--exam-id", action="append", dest="exam_ids")
    p.add_argument(
        "--full-cohort",
        action="store_true",
        help="Skeletonize every exam with both HR and UFAST-direct vessel predictions.",
    )
    args = p.parse_args()
    if not args.full_cohort and not args.exam_ids:
        p.error("--exam-id is required unless --full-cohort is set")

    exam_ids = (
        discover_exam_ids(args.preprocessing_root, args.output_root)
        if args.full_cohort
        else args.exam_ids
    )

    skipped: list[dict[str, str]] = []
    for exam_id in exam_ids:
        output_dir_existing = args.output_root / exam_id / "combined_tc4d"
        if output_dir_existing.exists():
            if is_combined_tc4d_complete(args.output_root, exam_id):
                print(f"[{exam_id}] already skeletonized, skipping")
                continue
            print(
                f"[{exam_id}] found incomplete output (likely killed mid-run), re-running"
            )
            shutil.rmtree(output_dir_existing)
        print(f"[{exam_id}] running TC4D on combined HR+UFAST phase sequence...")
        try:
            out_dir = skeletonize_exam(
                args.preprocessing_root, args.output_root, exam_id
            )
        except Exception as exc:  # noqa: BLE001 - log and continue a long unattended batch job
            skipped.append({"exam_id": exam_id, "reason": str(exc)})
            print(f"[{exam_id}] FAILED: {exc}")
            continue
        print(f"[{exam_id}] done -> {out_dir}")

    if skipped:
        import csv

        skipped_path = args.output_root / "combined_tc4d_skipped_exams.csv"
        with skipped_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["exam_id", "reason"])
            writer.writeheader()
            writer.writerows(skipped)
        print(f"Wrote {skipped_path} ({len(skipped)} exams skipped)")


if __name__ == "__main__":
    main()
