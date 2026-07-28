r"""Backfill work/<exam>/ufast_tc4d/ from the standalone ufast-seg tree.

The current pipeline's ``map_case`` merges HR + UFAST-direct TC4D, but no exam in this
cohort has ever had the UFAST-direct route run through ``preprocessing.pipeline`` --
``work/<exam>/ufast_tc4d/`` does not exist for any of the 180 exams. A prior standalone
workflow (``analysis/segment_ufast_vessels.py`` + ``analysis/skeletonize_ufast_vessels.py``)
already computed the equivalent output for every exam in this cohort, sitting in
``ufast-seg/<exam>/ufast_tc4d/`` -- same core algorithm
(``tc4d_graph_forward_no_learning`` / ``tc4d_auto_ved_spt4d_subtree``, per that route's own
provenance), just invoked outside the current pipeline module.

Per explicit user direction (2026-07-28): reuse that existing output rather than
re-running UFAST-direct TC4D through the pipeline, which would cost real GPU/CPU time
this backfill avoids entirely.

Also clears any stale ``"complement"`` key from the exam's provenance: every exam in
this cohort already has one, written by a complement_case run against the OLD (HR-only)
skeleton before this backfill existed. complement_case's own guard
(``if merge_only_path.exists() or "complement" in provenance: raise FileExistsError``)
would otherwise refuse to run again even after a fresh, real merge -- the stale key
alone is enough to trip it, independent of whether the file exists.

Records what it did directly in provenance under ``"tc4d_ufast_route_backfill"`` --
source path, per-file sha256, and the backfill date -- so a reader of the provenance
later knows this route was reused from another workflow, not computed by this run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_REQUIRED_FILES = ("exam_skeleton_zyx.npy", "exam_support_zyx.npy")
_OPTIONAL_FILES = ("center_manifold_4d_yxz.npy",)

#: Sentinel for a QC field this cohort's provenance genuinely cannot supply --
#: distinct from both "pass" and "review_required" so nothing downstream mistakes an
#: unevaluated check for a clean one. preprocessing.pipeline._combined_alignment_status
#: only reads these values and compares to "pass" (it never gates on "pass" vs
#: "review_required" vs anything else) and map_case only uses the combined result to
#: populate run_summary.json, so this sentinel flows through as an honest
#: "review_required" without needing to touch pipeline.py itself.
_LEGACY_QC_UNAVAILABLE = "unavailable_legacy_provenance"


def _motion_qc_status(metrics: list[dict[str, object]]) -> str:
    """Copy of preprocessing.pipeline._motion_qc_status (private, not importable).

    Kept in sync deliberately: this needs the EXACT classifier prepare_case would have
    used, applied to data (``motion_metrics``) this cohort's provenance already has in
    full -- a real derivation, not a guess. Reproduced rather than imported because the
    source function is prefixed private and the pipeline module has no public seam for
    it.
    """
    allowed_reasons = {"accepted", "correlation_gain_below_minimum"}
    for phase_metrics in metrics[1:]:
        reason = phase_metrics.get("transform_rejection_reason")
        try:
            finite = all(
                np.isfinite(float(phase_metrics[key]))
                for key in ("raw_correlation", "proposed_correlation", "corr_delta")
            )
        except (KeyError, TypeError, ValueError):
            finite = False
        if reason not in allowed_reasons or not finite:
            return "review_required"
    return "pass"


def backfill_alignment_qc_fields(provenance: dict) -> bool:
    """Complete the QC fields preprocessing.pipeline._combined_alignment_status needs.

    This cohort's provenance predates three fields current map_case requires:
    ``identity_alignment_qc.status``, ``hr_motion_qc``, and ``ufast_motion_qc``. Two of
    those cannot be honestly recovered without re-running raw-DICOM computation this
    backfill deliberately avoids (identity_alignment_qc's status needs a fresh
    correct_phase call on raw HR/UFAST phase-0 images; hr_motion_qc needs the HR
    series's own per-phase motion correction, never persisted separately). Those get an
    explicit non-"pass" sentinel rather than a fabricated value.

    ufast_motion_qc IS honestly recoverable: this cohort's provenance already has the
    complete per-phase ``motion_metrics`` list prepare_case computed, and
    ``_motion_qc_status`` is a pure function of it -- reapplying it is a real
    derivation, not an approximation.

    Returns whether anything changed (so a caller can report accurately rather than
    claim a backfill on an exam that already had these fields).
    """
    changed = False
    if "status" not in provenance.get("identity_alignment_qc", {}):
        provenance.setdefault("identity_alignment_qc", {})["status"] = (
            _LEGACY_QC_UNAVAILABLE
        )
        changed = True
    if "hr_motion_qc" not in provenance:
        provenance["hr_motion_qc"] = {
            "status": _LEGACY_QC_UNAVAILABLE,
            "note": (
                "HR per-phase motion QC was never computed under this cohort's "
                "prepare-stage schema and is not recoverable without the raw HR "
                "series; not fabricated."
            ),
        }
        changed = True
    if "ufast_motion_qc" not in provenance and "motion_metrics" in provenance:
        provenance["ufast_motion_qc"] = {
            "status": _motion_qc_status(provenance["motion_metrics"]),
            "note": (
                "Derived from this exam's own recorded motion_metrics via the same "
                "classifier prepare_case uses, not recomputed from raw data."
            ),
        }
        changed = True
    return changed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backfill_one(
    *, preprocessing_root: Path, ufast_seg_root: Path, exam_id: str
) -> dict[str, object]:
    """Copy one exam's ufast_tc4d output and annotate its provenance. Returns a summary."""
    case_root = preprocessing_root / "work" / exam_id
    source_dir = ufast_seg_root / exam_id / "ufast_tc4d"
    dest_dir = case_root / "ufast_tc4d"

    for name in _REQUIRED_FILES:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(
                f"missing required backfill source: {source_dir / name}"
            )

    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in _REQUIRED_FILES + _OPTIONAL_FILES:
        source_path = source_dir / name
        if not source_path.is_file():
            continue
        dest_path = dest_dir / name
        shutil.copy2(source_path, dest_path)
        copied[name] = _sha256(dest_path)

    provenance_path = case_root / "preprocessing_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    had_stale_complement = "complement" in provenance
    provenance.pop("complement", None)
    backfilled_alignment_qc = backfill_alignment_qc_fields(provenance)
    provenance["tc4d_ufast_route_backfill"] = {
        "source_dir": str(source_dir),
        "backfilled_files_sha256": copied,
        "date": "2026-07-28",
        "note": (
            "UFAST-direct TC4D route reused from the standalone ufast-seg workflow "
            "(analysis/segment_ufast_vessels.py + analysis/skeletonize_ufast_vessels.py), "
            "not computed by preprocessing.pipeline.tc4d_case, per explicit user "
            "direction to reuse existing cohort output rather than recompute."
        ),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2))

    return {
        "exam_id": exam_id,
        "files_copied": list(copied),
        "cleared_stale_complement_key": had_stale_complement,
        "backfilled_alignment_qc_fields": backfilled_alignment_qc,
    }


def main() -> None:
    """Backfill ufast_tc4d for every exam in the given case list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-root", type=Path, required=True)
    parser.add_argument("--ufast-seg-root", type=Path, required=True)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, copy nothing"
    )
    args = parser.parse_args()

    lines = [
        line.strip() for line in args.case_list.read_text().splitlines() if line.strip()
    ]
    exam_ids = [line.split("/")[-1] for line in lines]
    print(f"{len(exam_ids)} exams in case list", flush=True)

    if args.dry_run:
        missing = [
            exam_id
            for exam_id in exam_ids
            if not all(
                (args.ufast_seg_root / exam_id / "ufast_tc4d" / name).is_file()
                for name in _REQUIRED_FILES
            )
        ]
        print(f"exams missing required source files: {len(missing)}")
        for exam_id in missing[:10]:
            print(f"  {exam_id}")
        return

    n_stale_cleared = 0
    n_qc_backfilled = 0
    for index, exam_id in enumerate(exam_ids):
        summary = backfill_one(
            preprocessing_root=args.preprocessing_root,
            ufast_seg_root=args.ufast_seg_root,
            exam_id=exam_id,
        )
        if summary["cleared_stale_complement_key"]:
            n_stale_cleared += 1
        if summary["backfilled_alignment_qc_fields"]:
            n_qc_backfilled += 1
        if (index + 1) % 20 == 0 or index == len(exam_ids) - 1:
            print(f"  [{index + 1}/{len(exam_ids)}] {exam_id[:40]}", flush=True)

    print(
        f"\nbackfilled {len(exam_ids)} exams; cleared stale complement key on "
        f"{n_stale_cleared}; completed legacy alignment-QC fields on {n_qc_backfilled}"
    )


if __name__ == "__main__":
    main()
