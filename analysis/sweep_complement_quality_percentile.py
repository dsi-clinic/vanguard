r"""Re-sweep DEFAULT_QUALITY_PERCENTILE now that the gate judges BRANCHES, not components.

Why this needs redoing rather than trusting the existing default: pct=10 was chosen by
a sweep run when `filter_complement` classified whole connected components, of which a
real exam has only 2-9. The gate now judges branches (runs between junctions), of which
the same exams have 55-569. A percentile is a statement about a distribution's shape, so
moving to a population 10-60x larger invalidates the calibration that picked 10.0 --
the constant's own docstring already warned it was under-validated even for the old
unit.

Sweeps the operating point on real exams and reports, per value:
  * voxels and branches kept on the REAL exam (yield);
  * voxels that pass the same bar on the NO-CONTRAST negative control, which cannot
    contain vessels, so this is a direct false-positive count at that operating point;
  * the resulting false-positive ratio, which is what should actually drive the choice
    -- more yield is only better if the control stays clean.

Runs SegVessel twice per exam (real + control) and then evaluates every percentile
against those cached outputs, so the sweep costs no more compute than a single QC run.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "/home/t-9sbose/worktrees/matlab-complement")

import SimpleITK as sitk  # noqa: E402

from analysis.qc_complement_vessels import (  # noqa: E402
    find_pre_complement_skeleton,
)
from preprocessing.complement import (  # noqa: E402
    _MIN_POINTS_FOR_ELONGATION,
    DEFAULT_CONFIRMED_OVERLAP_FRACTION,
    DEFAULT_EDGE_MARGIN_VOXELS,
    DEFAULT_MIN_COMPONENT_VOXELS,
    DEFAULT_MIN_INTERIOR_FRACTION,
    DEFAULT_MIN_NOVEL_VOXELS,
    DEFAULT_TC_TOLERANCE_VOXELS,
    MIN_CONFIRMED_COMPONENTS_FOR_CALIBRATION,
    VESSELNESS_NORMALISATION_EROSION_VOXELS,
    _branch_labels,
    _breast_mask_ufast_yxz,
    _load_dyn_subtraction,
    _load_hr_subtraction,
    _per_component_stats,
)
from preprocessing.dicom import DicomGeometry  # noqa: E402
from preprocessing.merge import find_ufast_phases  # noqa: E402
from segmentation.matlab_vessel_segmentation import SegVessel  # noqa: E402

ROOT = Path("/gpfs/data/karczmar-lab/workspaces/saritbose/preprocessing_out")
PERCENTILES = (2.0, 5.0, 10.0, 15.0, 25.0, 40.0)


def _run_segvessel(sub_dyn, sub_hr, breast, spacing):
    roi = ndi.binary_erosion(breast, iterations=VESSELNESS_NORMALISATION_EROSION_VOXELS)
    _vmask, morph = SegVessel(
        sub_dyn * breast, sub_hr * breast, spacing, verbose=False, normalisation_roi=roi
    )
    return morph["skel_label"] > 0, morph["vness_dyn"]


def _stats_for(skeleton, vesselness, enhancement, spacing, tc_near, interior):
    labels = _branch_labels(skeleton)
    if labels.max() == 0:
        return {}
    return _per_component_stats(
        labels,
        vesselness,
        enhancement,
        spacing,
        fraction_masks={"near_merged": tc_near, "interior": interior},
    )


def _evaluate(stats, thresholds, *, require_novel):
    """Voxels/branches passing the gate at a given set of thresholds."""
    kept_voxels = 0
    kept_branches = 0
    for entry in stats.values():
        if entry["size"] < DEFAULT_MIN_COMPONENT_VOXELS:
            continue
        if entry["frac_interior"] < DEFAULT_MIN_INTERIOR_FRACTION:
            continue
        if require_novel:
            novel = entry["size"] * (1.0 - entry["frac_near_merged"])
            if novel < DEFAULT_MIN_NOVEL_VOXELS:
                continue
        if (
            entry["mean_vesselness"] >= thresholds["mean_vesselness"]
            and entry["mean_enhancement"] >= thresholds["mean_enhancement"]
            and entry["elongation"] >= thresholds["elongation"]
        ):
            kept_branches += 1
            kept_voxels += int(entry["size"])
    return kept_voxels, kept_branches


def main(exam_ids) -> None:
    """Sweep the operating point on the given exams and write the summary CSV."""
    rows = []
    for exam_id in exam_ids:
        case_root = ROOT / "work" / exam_id
        print(f"\n=== {exam_id[:46]}... ===", flush=True)
        provenance = json.loads(
            (case_root / "preprocessing_provenance.json").read_text()
        )
        baseline = int(provenance["ufast_source"]["baseline_frame_count"])
        geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])
        spacing = (
            geometry.spacing_xyz_mm[1],
            geometry.spacing_xyz_mm[0],
            geometry.spacing_xyz_mm[2],
        )

        breast = _breast_mask_ufast_yxz(case_root, provenance)
        interior = ndi.binary_erosion(breast, iterations=DEFAULT_EDGE_MARGIN_VOXELS)
        sub_dyn = _load_dyn_subtraction(ROOT, exam_id, baseline)
        sub_hr = _load_hr_subtraction(case_root, provenance)

        skeleton_path = find_pre_complement_skeleton(ROOT, exam_id)
        print(f"  baseline skeleton: {skeleton_path.name}", flush=True)
        merged = np.transpose(np.load(skeleton_path).astype(bool), (1, 2, 0))
        tc_near = ndi.binary_dilation(merged, iterations=DEFAULT_TC_TOLERANCE_VOXELS)

        print("  real SegVessel...", flush=True)
        skeleton, vesselness = _run_segvessel(sub_dyn, sub_hr, breast, spacing)
        real_stats = _stats_for(
            skeleton, vesselness, sub_dyn * breast, spacing, tc_near, interior
        )

        phases = find_ufast_phases(ROOT, exam_id)
        signal = np.stack(
            [sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in phases], axis=0
        ).astype(np.float32)
        null_zyx = np.clip(
            signal[baseline - 1] - signal[: baseline - 1].mean(axis=0), 0, None
        )
        null_sub = np.transpose(null_zyx, (1, 2, 0))
        print("  null-control SegVessel...", flush=True)
        null_skeleton, null_vesselness = _run_segvessel(
            null_sub, null_sub, breast, spacing
        )
        null_stats = _stats_for(
            null_skeleton,
            null_vesselness,
            null_sub * breast,
            spacing,
            tc_near,
            interior,
        )

        calibration = [
            s
            for s in real_stats.values()
            if s["frac_near_merged"] >= DEFAULT_CONFIRMED_OVERLAP_FRACTION
            and s["size"] >= _MIN_POINTS_FOR_ELONGATION
        ]
        print(f"  calibration branches: {len(calibration)}", flush=True)
        if len(calibration) < MIN_CONFIRMED_COMPONENTS_FOR_CALIBRATION:
            print("  too few calibration branches; skipping exam")
            continue

        for percentile in PERCENTILES:
            thresholds = {
                key: float(np.percentile([s[key] for s in calibration], percentile))
                for key in ("mean_vesselness", "mean_enhancement", "elongation")
            }
            real_vox, real_br = _evaluate(real_stats, thresholds, require_novel=True)
            null_vox, null_br = _evaluate(null_stats, thresholds, require_novel=False)
            rows.append(
                {
                    "exam_id": exam_id,
                    "percentile": percentile,
                    "real_voxels": real_vox,
                    "real_branches": real_br,
                    "null_voxels": null_vox,
                    "null_branches": null_br,
                    "fp_ratio": round(null_vox / real_vox, 4) if real_vox else "",
                    "elongation_threshold": round(thresholds["elongation"], 3),
                    "vesselness_threshold": round(thresholds["mean_vesselness"], 4),
                }
            )
            print(
                f"    pct={percentile:5.1f}  real={real_vox:5d}vox/{real_br:3d}br  "
                f"null={null_vox:5d}vox/{null_br:3d}br  "
                f"fp_ratio={rows[-1]['fp_ratio']}  elong_thr={thresholds['elongation']:.2f}",
                flush=True,
            )

    if rows:
        out = Path(
            "/gpfs/data/karczmar-lab/workspaces/saritbose/qc_complement/percentile_sweep.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")

        print("\n=== pooled across exams ===")
        for percentile in PERCENTILES:
            subset = [r for r in rows if r["percentile"] == percentile]
            real = sum(r["real_voxels"] for r in subset)
            null = sum(r["null_voxels"] for r in subset)
            print(
                f"  pct={percentile:5.1f}  real={real:6d}vox  null={null:5d}vox  "
                f"fp_ratio={null / real:.4f}"
                if real
                else f"  pct={percentile:5.1f}  real=0"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
