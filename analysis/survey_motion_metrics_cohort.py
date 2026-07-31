r"""Cohort-wide survey of measured motion-correction residuals.

The Check-4 synthetic motion control (analysis/qc_complement_vessels.py) found that
a synthetic rigid shift of 1-2 voxels already passes the complement gate's own
calibrated thresholds on all 4 exams tested, reaching the scale of real additions by
3-5 voxels. That result is only a risk in practice if real exams actually have
residual motion in that range. This script answers that on the full cohort by reading
the motion-correction telemetry every exam already has recorded
(``preprocessing_provenance.json["motion_metrics"]``, per-UFAST-phase
``translation_voxels`` -- the rigid shift the motion-correction step measured and
applied for that phase, before which the phase was that far out of alignment).

This is NOT the same as measuring residual error AFTER correction (this codebase
does not record that directly) -- a large recorded shift means real motion was
present and was corrected for, not that correction failed. But an exam where large
shifts are common is exactly the kind of exam where under-correction is most likely,
and it is a cheap, already-computed number to survey across all 180 exams without
running SegVessel again.

Usage::

    micromamba run -n vanguard python -m analysis.survey_motion_metrics_cohort \
        --preprocessing-root /gpfs/data/.../preprocessing_out \
        --output-csv motion_metrics_cohort_survey.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

#: Shift magnitudes (voxels) that analysis.qc_complement_vessels's Check 4 showed
#: start passing real structure through the complement gate on all 4 tested exams.
_DANGER_THRESHOLDS_VOXELS = (1.0, 2.0)


def _exam_motion_summary(case_root: Path) -> dict | None:
    provenance_path = case_root / "preprocessing_provenance.json"
    if not provenance_path.exists():
        return None
    provenance = json.loads(provenance_path.read_text())
    metrics = provenance.get("motion_metrics")
    if not metrics:
        return None

    norms = []
    n_rejected = 0
    for entry in metrics:
        translation = entry.get("translation_voxels")
        if translation is None:
            continue
        norms.append(float(np.linalg.norm(translation)))
        if not entry.get("transform_accepted", True):
            n_rejected += 1
    if not norms:
        return None
    norms_arr = np.asarray(norms)

    summary = {
        "exam_id": case_root.name,
        "n_phases": len(norms_arr),
        "n_transforms_rejected": n_rejected,
        "max_translation_voxels": round(float(norms_arr.max()), 3),
        "p95_translation_voxels": round(float(np.percentile(norms_arr, 95)), 3),
        "median_translation_voxels": round(float(np.median(norms_arr)), 3),
    }
    for threshold in _DANGER_THRESHOLDS_VOXELS:
        summary[f"frac_phases_ge_{threshold:g}vox"] = round(
            float(np.mean(norms_arr >= threshold)), 4
        )
        summary[f"any_phase_ge_{threshold:g}vox"] = bool(np.any(norms_arr >= threshold))
    return summary


def main() -> None:
    """Survey every exam's motion telemetry and report the cohort-wide distribution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    work_root = args.preprocessing_root / "work"
    exam_dirs = sorted(p for p in work_root.iterdir() if p.is_dir())

    rows = []
    n_missing = 0
    for case_root in exam_dirs:
        summary = _exam_motion_summary(case_root)
        if summary is None:
            n_missing += 1
            continue
        rows.append(summary)

    if not rows:
        print("no exams had usable motion_metrics; nothing to report")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    maxima = np.array([row["max_translation_voxels"] for row in rows])
    print(
        f"cohort exams with motion_metrics: {len(rows)} (missing/unusable: {n_missing})"
    )
    print(
        f"max residual-motion-per-phase across cohort: "
        f"median={np.median(maxima):.2f}  p75={np.percentile(maxima, 75):.2f}  "
        f"p90={np.percentile(maxima, 90):.2f}  p95={np.percentile(maxima, 95):.2f}  "
        f"max={maxima.max():.2f}  (voxels, 1 mm isotropic UFAST grid)"
    )
    for threshold in _DANGER_THRESHOLDS_VOXELS:
        key = f"any_phase_ge_{threshold:g}vox"
        n_flagged = sum(1 for row in rows if row[key])
        print(
            f"exams with >=1 phase at >= {threshold:g} vox residual motion: "
            f"{n_flagged}/{len(rows)} ({100 * n_flagged / len(rows):.1f}%)"
        )
    print(f"\nwrote {args.output_csv}")
    print("\nworst 10 exams by max residual motion:")
    for row in sorted(rows, key=lambda r: -r["max_translation_voxels"])[:10]:
        print(
            f"  {row['exam_id'][:40]:<42} max={row['max_translation_voxels']:.2f}vox  "
            f"frac>=1vox={row['frac_phases_ge_1vox']:.1%}  "
            f"frac>=2vox={row['frac_phases_ge_2vox']:.1%}"
        )


if __name__ == "__main__":
    main()
