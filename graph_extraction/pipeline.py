"""Study-level driver for graph extraction, feature writing, and debug outputs."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from graph_extraction.constants import PROCESSING_VIZ_FLIP_SPEC
from graph_extraction.core4d import (
    discover_study_timepoints,
    load_time_series_from_files,
)
from graph_extraction.feature_stats import _repair_skeleton_support_consistency
from graph_extraction.graph_outputs import build_graph_outputs_from_centerline
from graph_extraction.masks import (
    _apply_flip_spec,
    _maybe_load_breast_context_for_alignment,
    _maybe_load_radiologist_context_for_mip,
    _maybe_load_tumor_context_for_features,
)
from graph_extraction.tc4d import run_tc4d_from_priority
from graph_extraction.vessel_mip import render_vessel_coverage_mip


def run_study_pipeline(
    *,
    input_dir: Path,
    case_id: str,
    output_dir: Path,
    features_only: bool,
    force_skeleton: bool,
    force_features: bool,
    strict_qc: bool,
    save_exam_masks: bool,
    save_center_manifold_mask: bool,
    render_mip: bool,
    mip_dpi: int,
    radiologist_annotations_dir: Path,
    tumor_mask_dir: Path,
) -> dict[str, object]:
    """Run the full graph-extraction workflow for one study.

    This function is the top-level per-study driver used by
    `run_skeleton_processing.py`.

    Depending on the flags, it does one of two things:

    - full run: load the vessel time series, extract the exam-level centerline,
      then build graph outputs and tumor-focused feature JSONs
    - features-only run: skip centerline extraction and rebuild the JSON outputs
      from previously saved skeleton and support masks

    The returned dictionary is the study-level run summary written to
    `run_summary.json`. It records which stages ran, how long they took,
    whether quality checks passed, and where the main outputs were written.
    """
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton_path = output_dir / f"{case_id}_skeleton_4d_exam_mask.npy"
    support_path = output_dir / f"{case_id}_skeleton_4d_exam_support_mask.npy"
    manifold_path = output_dir / f"{case_id}_center_manifold_4d_mask.npy"
    morphometry_path = output_dir / f"{case_id}_morphometry.json"
    coverage_mip_path = output_dir / f"{case_id}_vessel_coverage_mip.png"

    skeleton_took = 0.0
    features_took = 0.0
    mip_took = 0.0

    discovered_files: list[Path] | None = None
    discovered_timepoints: list[int] | None = None
    effective_min_temporal_support: int | None = None
    tc4d_params: dict[str, object] | None = None
    tc4d_diagnostics: dict[str, object] | None = None
    priority_4d_for_kinetics: np.ndarray | None = None

    if features_only:
        if force_skeleton:
            raise ValueError(
                "`--features-only` cannot be combined with `--force-skeleton`."
            )
        if not skeleton_path.exists() or not support_path.exists():
            raise ValueError(
                "features-only mode requires existing exam masks: "
                f"skeleton={skeleton_path}, support={support_path}"
            )
        skeleton_mask = np.load(skeleton_path).astype(bool, copy=False)
        support_mask = np.load(support_path).astype(bool, copy=False)
        retained_per_t: list[int] | None = None
        skeleton_status = "loaded_existing_features_only"
    elif (
        save_exam_masks
        and skeleton_path.exists()
        and support_path.exists()
        and not force_skeleton
    ):
        skeleton_mask = np.load(skeleton_path).astype(bool, copy=False)
        support_mask = np.load(support_path).astype(bool, copy=False)
        retained_per_t = None
        skeleton_status = "loaded_existing"
    else:
        t0 = time.perf_counter()
        discovered_files, discovered_timepoints = discover_study_timepoints(
            input_dir=input_dir,
            case_id=case_id,
        )
        priority_4d = load_time_series_from_files(
            discovered_files,
        )
        priority_4d_for_kinetics = np.asarray(priority_4d, dtype=np.float32, copy=False)
        tc4d_result, tc4d_params, tc4d_diagnostics = run_tc4d_from_priority(
            priority_4d,
        )
        mask_4d = np.asarray(tc4d_result["mask_4d"], dtype=bool)
        skeleton_mask = np.asarray(tc4d_result["exam_mask"], dtype=bool)
        support_mask = np.asarray(tc4d_result["support_mask"], dtype=bool)
        effective_min_temporal_support = int(
            tc4d_result["effective_min_temporal_support"]
        )
        if not np.any(skeleton_mask):
            raise ValueError("TC4D produced an empty exam-level skeleton.")
        if not np.any(support_mask):
            raise ValueError("TC4D produced an empty support mask.")

        if save_exam_masks:
            np.save(skeleton_path, skeleton_mask.astype(np.uint8))
            np.save(support_path, support_mask.astype(np.uint8))
        if save_center_manifold_mask:
            np.save(manifold_path, mask_4d.astype(np.uint8))

        retained_per_t = [int(x) for x in np.count_nonzero(mask_4d, axis=(1, 2, 3))]
        skeleton_took = float(time.perf_counter() - t0)
        skeleton_status = "computed"

    skeleton_mask, support_mask, support_consistency = (
        _repair_skeleton_support_consistency(
            skeleton_mask_zyx=skeleton_mask,
            support_mask_zyx=support_mask,
        )
    )
    if bool(support_consistency["repaired_support"]):
        print(
            "[qc-warning] repaired skeleton/support mismatch: "
            f"outside_voxels={support_consistency['skeleton_outside_support_voxels']}"
        )
        if support_path.exists() or save_exam_masks or features_only:
            np.save(support_path, support_mask.astype(np.uint8))

    shape_zyx = tuple(int(v) for v in skeleton_mask.shape)
    tumor_graph_features_path = output_dir / f"{case_id}_tumor_graph_features.json"
    tumor_mask_model: np.ndarray | None = None
    tumor_spacing_mm_zyx: tuple[float, float, float] | None = None
    tumor_context_for_summary: dict[str, object] | None = None
    kinetic_context_for_summary: dict[str, object] | None = None

    if morphometry_path.exists() and not force_features:
        feature_stats: dict[str, object] | None = None
        features_status = "loaded_existing"
        kinetic_context_for_summary = {"status": "skipped_existing_features"}
    else:
        (
            tumor_mask_model,
            tumor_spacing_mm_zyx,
            tumor_context_for_summary,
        ) = _maybe_load_tumor_context_for_features(
            case_id=case_id,
            shape_zyx=shape_zyx,
            tumor_mask_dir=tumor_mask_dir,
            radiologist_annotations_dir=radiologist_annotations_dir,
        )
        kinetic_priority_4d: np.ndarray | None = None
        kinetic_timepoints: list[int] | None = None
        kinetic_reference_mask_zyx: np.ndarray | None = None
        if tumor_mask_model is None:
            kinetic_context_for_summary = {"status": "skipped_missing_tumor_context"}
        else:
            if priority_4d_for_kinetics is not None:
                kinetic_priority_4d = priority_4d_for_kinetics
                kinetic_timepoints = discovered_timepoints
                kinetic_context_for_summary = {
                    "status": "ok",
                    "time_series_source": "current_tc4d_input",
                    "timepoint_count": int(kinetic_priority_4d.shape[0]),
                }
            else:
                try:
                    k_files, k_timepoints = discover_study_timepoints(
                        input_dir=input_dir,
                        case_id=case_id,
                    )
                    kinetic_priority_4d = load_time_series_from_files(k_files)
                    kinetic_timepoints = list(k_timepoints)
                    if discovered_files is None:
                        discovered_files = list(k_files)
                    if discovered_timepoints is None:
                        discovered_timepoints = list(k_timepoints)
                    kinetic_context_for_summary = {
                        "status": "ok",
                        "time_series_source": "loaded_from_input_dir",
                        "timepoint_count": int(kinetic_priority_4d.shape[0]),
                    }
                except Exception as exc:
                    kinetic_context_for_summary = {
                        "status": "time_series_load_failed",
                        "error": str(exc),
                    }

            breast_reference_mask, breast_reference_context = (
                _maybe_load_breast_context_for_alignment(
                    case_id=case_id,
                    shape_zyx=shape_zyx,
                    radiologist_annotations_dir=radiologist_annotations_dir,
                )
            )
            if breast_reference_mask is not None:
                kinetic_reference_mask_zyx = breast_reference_mask
            if kinetic_context_for_summary is None:
                kinetic_context_for_summary = {}
            kinetic_context_for_summary["reference_mask_source"] = (
                "breast_mask"
                if breast_reference_mask is not None
                else "background_fallback"
            )
            kinetic_context_for_summary["breast_reference_context"] = (
                breast_reference_context
            )

        t1 = time.perf_counter()
        feature_stats = build_graph_outputs_from_centerline(
            skeleton_mask_zyx=skeleton_mask,
            vessel_reference_zyx=support_mask.astype(np.uint8, copy=False),
            output_json_path=morphometry_path,
            strict_qc=bool(strict_qc),
            tumor_mask_zyx=tumor_mask_model,
            tumor_spacing_mm_zyx=tumor_spacing_mm_zyx,
            tumor_graph_output_path=tumor_graph_features_path,
            kinetic_priority_4d=kinetic_priority_4d,
            kinetic_timepoints=kinetic_timepoints,
            kinetic_reference_mask_zyx=kinetic_reference_mask_zyx,
        )
        features_took = float(time.perf_counter() - t1)
        features_status = "computed"

    mip_status = "skipped"
    coverage_mip_diagnostics: dict[str, object] | None = None
    radiologist_context_for_summary: dict[str, object] | None = None
    if render_mip:
        t2 = time.perf_counter()
        try:
            radiologist_context = _maybe_load_radiologist_context_for_mip(
                case_id=case_id,
                shape_zyx=shape_zyx,
                radiologist_annotations_dir=radiologist_annotations_dir,
            )
            radiologist_mask_model: np.ndarray | None = None
            breast_mask_model: np.ndarray | None = None
            radiologist_mask_viz: np.ndarray | None = None
            breast_mask_viz: np.ndarray | None = None
            if radiologist_context.get("status") == "ok":
                radiologist_mask_model = np.asarray(
                    radiologist_context["radiologist_mask_zyx"], dtype=bool
                )
                breast_mask_model = np.asarray(
                    radiologist_context["breast_mask_zyx"], dtype=bool
                )
                radiologist_mask_viz = _apply_flip_spec(
                    radiologist_mask_model,
                    PROCESSING_VIZ_FLIP_SPEC,
                )
                breast_mask_viz = _apply_flip_spec(
                    breast_mask_model,
                    PROCESSING_VIZ_FLIP_SPEC,
                )

            tumor_mask_viz: np.ndarray | None = None
            if tumor_mask_model is None:
                (
                    tumor_mask_model,
                    tumor_spacing_mm_zyx,
                    tumor_context_for_summary,
                ) = _maybe_load_tumor_context_for_features(
                    case_id=case_id,
                    shape_zyx=shape_zyx,
                    tumor_mask_dir=tumor_mask_dir,
                    radiologist_annotations_dir=radiologist_annotations_dir,
                )
            if tumor_mask_model is not None:
                tumor_mask_viz = _apply_flip_spec(
                    tumor_mask_model,
                    PROCESSING_VIZ_FLIP_SPEC,
                )

            method_label = "tc4d"
            row_masks: list[tuple[str, np.ndarray]] = [
                (
                    method_label,
                    _apply_flip_spec(skeleton_mask, PROCESSING_VIZ_FLIP_SPEC),
                )
            ]
            if radiologist_mask_viz is not None:
                row_masks.append(("radiologist", radiologist_mask_viz))

            coverage_mip_diag = render_vessel_coverage_mip(
                row_masks=row_masks,
                output_path=coverage_mip_path,
                case_label=case_id,
                title_prefix=f"{method_label} vessel coverage mip",
                radiologist_mask_zyx=radiologist_mask_viz,
                breast_mask_zyx=breast_mask_viz,
                tumor_mask_zyx=tumor_mask_viz,
                vessel_color="#111827",
                dpi=int(mip_dpi),
            )
            radiologist_context_for_summary = {
                k: v
                for k, v in radiologist_context.items()
                if k not in {"breast_mask_zyx", "radiologist_mask_zyx"}
            }
            coverage_mip_diagnostics = {
                **coverage_mip_diag,
                "radiologist_context": radiologist_context_for_summary,
                "tumor_context": tumor_context_for_summary,
                "visualization_flip_spec": PROCESSING_VIZ_FLIP_SPEC,
            }
            mip_status = "computed"
        except Exception as exc:
            mip_status = "failed"
            coverage_mip_diagnostics = {"error": str(exc)}
        mip_took = float(time.perf_counter() - t2)

    feature_qc = None if feature_stats is None else feature_stats.get("qc_counters")
    feature_qc_passed = (
        None if feature_stats is None else bool(feature_stats.get("qc_passed", True))
    )
    tumor_graph_features_status = (
        None
        if feature_stats is None
        else feature_stats.get("tumor_graph_features_status")
    )
    if tumor_graph_features_status is None and tumor_graph_features_path.exists():
        tumor_graph_features_status = "existing"

    return {
        "mode": "features_only" if features_only else "tc4d",
        "case_id": case_id,
        "input_dir": str(input_dir),
        "algorithm": "tc4d",
        "features_only": bool(features_only),
        "strict_qc": bool(strict_qc),
        "effective_min_temporal_support": (
            None
            if effective_min_temporal_support is None
            else int(effective_min_temporal_support)
        ),
        "skeleton_status": skeleton_status,
        "features_status": features_status,
        "skeleton_voxels": int(np.count_nonzero(skeleton_mask)),
        "support_voxels": int(np.count_nonzero(support_mask)),
        "skeleton_path": (
            str(skeleton_path)
            if (features_only or save_exam_masks or skeleton_path.exists())
            else None
        ),
        "support_path": (
            str(support_path)
            if (features_only or save_exam_masks or support_path.exists())
            else None
        ),
        "manifold_path": str(manifold_path) if save_center_manifold_mask else None,
        "morphometry_path": str(morphometry_path),
        "tumor_graph_features_path": (
            str(tumor_graph_features_path)
            if tumor_graph_features_path.exists() or features_status == "computed"
            else None
        ),
        "tumor_graph_features_status": tumor_graph_features_status,
        "coverage_mip_path": str(coverage_mip_path)
        if mip_status == "computed"
        else None,
        "coverage_mip_status": mip_status,
        "coverage_mip_diagnostics": coverage_mip_diagnostics,
        "radiologist_context": radiologist_context_for_summary,
        "tumor_context": tumor_context_for_summary,
        "kinetic_context": kinetic_context_for_summary,
        "feature_stats": feature_stats,
        "feature_qc": feature_qc,
        "feature_qc_passed": feature_qc_passed,
        "support_consistency": support_consistency,
        "tc4d_params": tc4d_params,
        "tc4d_diagnostics": tc4d_diagnostics,
        "study_files": None
        if discovered_files is None
        else [str(p) for p in discovered_files],
        "study_timepoints": discovered_timepoints,
        "retained_per_timepoint": retained_per_t,
        "timing_seconds": {
            "skeleton": skeleton_took,
            "features": features_took,
            "coverage_mip": mip_took,
            "total": float(time.perf_counter() - start),
        },
    }
