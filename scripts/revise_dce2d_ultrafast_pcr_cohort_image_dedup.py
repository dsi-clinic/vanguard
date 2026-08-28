#!/usr/bin/env python3
"""Create Sarit's image-deduplicated pCR cohort revision.

This revision starts from the immutable v2 release, verifies both the v2 and
image-audit checksum manifests, removes one exam from every validated duplicate
pair, preserves folds for retained identities, and publishes a new compatible
release atomically.  Raw imaging and the two frozen input releases are read-only.

Retired.  This stage produced the pinned v3 release and is kept so that release has its producing
code in the repository.  Its 55 exclusions and one label reconciliation are now frozen curation
inputs to `build_dce2d_ultrafast_pcr_cohort.py`, which applies them before any per-patient fact is
measured rather than by copying a finished release forward.  Do not run it against a new build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA = "vanguard.sarit_pcr_pretreatment_cohort.v3"
FOLD_SCHEMA = "vanguard.sarit_pcr_fold_assignment.v3_image_dedup"
IDENTITY_SCHEMA = "vanguard.sarit_pcr_pretreatment_cohort.identity_audit.v2"
IMAGE_EXCLUSION_SCHEMA = "vanguard.sarit_pcr_image_duplicate_exclusion.v1"
LABEL_RECONCILIATION_SCHEMA = "vanguard.sarit_pcr_duplicate_label_reconciliation.v1"

EXPECTED_BASE_SOURCE_EXAMS = 341
EXPECTED_BASE_MAIN_EXAMS = 292
EXPECTED_DUPLICATE_PAIRS = 55
EXPECTED_EXACT_CANONICAL_PAIRS = 19
EXPECTED_CROSS_DELIVERY_PAIRS = 36
EXPECTED_RETAINED_SOURCE_EXAMS = 286
EXPECTED_RETAINED_MAIN_EXAMS = 243
EXPECTED_RETAINED_READY_EXAMS = 246
EXPECTED_RETAINED_PENDING_EXAMS = 43
EXPECTED_RETAINED_SIX_COLUMN_PAIRS = 281
EXPECTED_LABEL_RECONCILIATIONS = 1

CROSS_UFAST_MINIMUM = 0.998
CROSS_ENHANCEMENT_MINIMUM = 0.994
CROSS_HR_MINIMUM = 0.999

DEFAULT_BASE_ROOT = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v2"
)
DEFAULT_AUDIT_ROOT = Path(
    "/gpfs/data/karczmar-lab/vanguard/"
    "dce2d_internal_ultrafast_pretreatment_cohort_v2_duplicate_audit_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v3"
)

TABLE_FILES = {
    "main": "dce2d_internal_ultrafast_manifest.csv",
    "source": "source_eligible_cohort_manifest.csv",
    "accounting": "cohort_accounting.csv",
    "folds": "fold_assignments.csv",
    "paired_case": "paired_preprocessing_case_manifest.csv",
    "paired_exclusions": "paired_preprocessing_exclusions.csv",
    "paired_source": "paired_source_manifest.csv",
    "pending": "pending_unprocessed_pretreatment_sources.csv",
    "retro_dedup": "retro_patient_deduplication_exclusions.csv",
    "symlinks": "symlink_manifest.csv",
}


def _clean(value: object) -> str:
    """Return a stable stripped string for mixed CSV values."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    """Interpret common serialized booleans."""
    return _clean(value).lower() in {"1", "true", "t", "yes", "y"}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checksum_manifest(root: Path) -> dict[str, str]:
    """Parse and verify every entry in a release checksum manifest."""
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", maxsplit=1)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("checksum manifest contains an unsafe relative path")
        path = root / relative_path
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"checksum verification failed for {relative}")
        entries[relative] = digest
    if not entries:
        raise ValueError("checksum manifest is empty")
    return entries


def _read_tables(base_root: Path) -> dict[str, pd.DataFrame]:
    """Load the compatible v2 release tables as strings."""
    tables: dict[str, pd.DataFrame] = {}
    for key, name in TABLE_FILES.items():
        path = base_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        tables[key] = pd.read_csv(path, dtype=str).fillna("")
    return tables


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Fail with a concise schema error when required columns are missing."""
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _derive_duplicate_decisions(
    *,
    tables: dict[str, pd.DataFrame],
    base_root: Path,
    audit_root: Path,
    audit_checksums: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive one deterministic keep/drop decision per validated image pair."""
    score_path = audit_root / "all_selected_pair_scores.csv"
    summary_path = audit_root / "validation_summary.json"
    scores = pd.read_csv(score_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        int(summary["selected_exams"]) != EXPECTED_BASE_SOURCE_EXAMS
        or int(summary["exact_selected_exam_pixel_matches"])
        != EXPECTED_EXACT_CANONICAL_PAIRS
        or int(summary["automatic_probable_duplicate_pairs"])
        != EXPECTED_DUPLICATE_PAIRS
        or summary["preliminary_status"] != "exact_duplicate_detected"
    ):
        raise ValueError("image audit summary does not match the frozen review target")
    if audit_checksums.get("all_selected_pair_scores.csv") != _sha256_file(score_path):
        raise ValueError("pair-score hash does not match the verified audit manifest")

    _require_columns(
        scores,
        {
            "exam_id_a",
            "exam_id_b",
            "cohort_component_a",
            "cohort_component_b",
            "ufast_baseline_correlation",
            "ufast_enhancement_correlation",
            "hr_baseline_correlation",
            "combined_image_score",
            "exact_ufast_phase_pair",
            "exact_hr_baseline",
            "exact_exam_pixel_match",
            "automatic_probable_duplicate",
            "automatic_review_candidate",
        },
        "image pair scores",
    )
    probable = scores.loc[scores["automatic_probable_duplicate"].map(_truthy)].copy()
    review = scores.loc[scores["automatic_review_candidate"].map(_truthy)].copy()
    if len(probable) != EXPECTED_DUPLICATE_PAIRS or set(
        zip(probable["exam_id_a"], probable["exam_id_b"], strict=True)
    ) != set(zip(review["exam_id_a"], review["exam_id_b"], strict=True)):
        raise ValueError("automatic probable and review pair sets differ")

    source = tables["source"].set_index("exam_id", drop=False)
    if len(source) != EXPECTED_BASE_SOURCE_EXAMS or source.index.duplicated().any():
        raise ValueError("base source manifest identity contract failed")
    legacy_pairs = pd.read_csv(
        base_root / "_build/input_snapshots/canonical_legacy_pair_manifest.csv",
        dtype=str,
    ).fillna("")
    zhen_pairs = pd.read_csv(
        base_root / "_build/input_snapshots/canonical_zhen_pair_manifest.csv",
        dtype=str,
    ).fillna("")
    legacy_studies = set(legacy_pairs["study_instance_uid"].astype(str))
    zhen_studies = set(zhen_pairs["study_instance_uid"].astype(str))

    decisions: list[dict[str, Any]] = []
    pair_exam_ids: list[str] = []
    for row in probable.to_dict(orient="records"):
        exam_a = _clean(row["exam_id_a"])
        exam_b = _clean(row["exam_id_b"])
        if exam_a not in source.index or exam_b not in source.index:
            raise ValueError(
                "an audited pair is not present in the base source manifest"
            )
        pair_exam_ids.extend((exam_a, exam_b))
        component_a = _clean(row["cohort_component_a"])
        component_b = _clean(row["cohort_component_b"])
        retro_a = component_a.startswith("retro")
        retro_b = component_b.startswith("retro")
        if retro_a != retro_b:
            classification = "cross_delivery_same_exam"
            dropped = exam_a if retro_a else exam_b
            retained = exam_b if retro_a else exam_a
            if not _truthy(row["exact_hr_baseline"]):
                raise ValueError("a cross-delivery match lacks exact HR pixel identity")
            if (
                float(row["ufast_baseline_correlation"]) < CROSS_UFAST_MINIMUM
                or float(row["ufast_enhancement_correlation"])
                < CROSS_ENHANCEMENT_MINIMUM
                or float(row["hr_baseline_correlation"]) < CROSS_HR_MINIMUM
            ):
                raise ValueError("a cross-delivery match falls below review thresholds")
            policy = "drop_retro_copy_keep_established_canonical_copy_and_fold"
        else:
            classification = "canonical_reidentified_exact_exam"
            if not _truthy(row["exact_exam_pixel_match"]):
                raise ValueError(
                    "a within-canonical match is not exact across both series"
                )
            study_a = _clean(source.loc[exam_a, "study_instance_uid"])
            study_b = _clean(source.loc[exam_b, "study_instance_uid"])
            a_legacy = study_a in legacy_studies
            b_legacy = study_b in legacy_studies
            a_zhen = study_a in zhen_studies
            b_zhen = study_b in zhen_studies
            if not ((a_legacy and b_zhen) or (b_legacy and a_zhen)):
                raise ValueError("an exact canonical pair is not legacy-versus-Zhen")
            retained = exam_a if a_legacy else exam_b
            dropped = exam_b if a_legacy else exam_a
            policy = "drop_zhen_reidentified_copy_keep_established_legacy_copy_and_fold"

        dropped_pcr = int(float(source.loc[dropped, "pcr"]))
        retained_pcr = int(float(source.loc[retained, "pcr"]))
        decisions.append(
            {
                "schema": IMAGE_EXCLUSION_SCHEMA,
                "dropped_exam_id": dropped,
                "retained_exam_id": retained,
                "duplicate_classification": classification,
                "selection_status": "excluded_image_content_duplicate",
                "selection_policy": policy,
                "dropped_cohort_component": _clean(
                    source.loc[dropped, "cohort_component"]
                ),
                "retained_cohort_component": _clean(
                    source.loc[retained, "cohort_component"]
                ),
                "exact_ufast_phase_pair": _truthy(row["exact_ufast_phase_pair"]),
                "exact_hr_baseline": _truthy(row["exact_hr_baseline"]),
                "exact_exam_pixel_match": _truthy(row["exact_exam_pixel_match"]),
                "ufast_baseline_correlation": float(row["ufast_baseline_correlation"]),
                "ufast_enhancement_correlation": float(
                    row["ufast_enhancement_correlation"]
                ),
                "hr_baseline_correlation": float(row["hr_baseline_correlation"]),
                "combined_image_score": float(row["combined_image_score"]),
                "dropped_pcr": dropped_pcr,
                "retained_pcr_before_reconciliation": retained_pcr,
                "pcr_label_agreement": dropped_pcr == retained_pcr,
                "dropped_pcr_label_authority": _clean(
                    source.loc[dropped, "pcr_label_authority"]
                ),
                "retained_pcr_label_authority_before_reconciliation": _clean(
                    source.loc[retained, "pcr_label_authority"]
                ),
                "dropped_pcr_label_is_provisional": _truthy(
                    source.loc[dropped, "pcr_label_is_provisional"]
                ),
                "retained_pcr_label_is_provisional_before_reconciliation": _truthy(
                    source.loc[retained, "pcr_label_is_provisional"]
                ),
                "dropped_fold": int(float(source.loc[dropped, "fold"])),
                "retained_fold": int(float(source.loc[retained, "fold"])),
                "cross_fold_before_exclusion": int(float(source.loc[dropped, "fold"]))
                != int(float(source.loc[retained, "fold"])),
                "audit_pair_scores_path": str(score_path.resolve()),
                "audit_pair_scores_sha256": audit_checksums[
                    "all_selected_pair_scores.csv"
                ],
            }
        )

    if len(pair_exam_ids) != len(set(pair_exam_ids)):
        raise ValueError("duplicate match graph is not a set of disjoint pairs")
    decision_frame = pd.DataFrame(decisions).sort_values(
        ["duplicate_classification", "dropped_exam_id"], kind="stable"
    )
    if (
        len(decision_frame) != EXPECTED_DUPLICATE_PAIRS
        or decision_frame["dropped_exam_id"].nunique() != EXPECTED_DUPLICATE_PAIRS
        or decision_frame["retained_exam_id"].nunique() != EXPECTED_DUPLICATE_PAIRS
        or decision_frame["duplicate_classification"]
        .eq("canonical_reidentified_exact_exam")
        .sum()
        != EXPECTED_EXACT_CANONICAL_PAIRS
        or decision_frame["duplicate_classification"]
        .eq("cross_delivery_same_exam")
        .sum()
        != EXPECTED_CROSS_DELIVERY_PAIRS
    ):
        raise ValueError("derived duplicate decisions fail expected accounting")

    reconciliations: list[dict[str, Any]] = []
    conflicts = decision_frame.loc[~decision_frame["pcr_label_agreement"]]
    for row in conflicts.to_dict(orient="records"):
        retained = _clean(row["retained_exam_id"])
        dropped = _clean(row["dropped_exam_id"])
        if row["duplicate_classification"] != "canonical_reidentified_exact_exam":
            continue
        if (
            _clean(source.loc[dropped, "rcb_class"]).lower() != "pcr"
            or float(source.loc[dropped, "rcb_score"]) != 0.0
            or _truthy(source.loc[dropped, "pcr_label_is_provisional"])
            or int(float(source.loc[dropped, "pcr"])) != 1
        ):
            raise ValueError("canonical label conflict lacks explicit RCB pCR evidence")
        reconciliations.append(
            {
                "schema": LABEL_RECONCILIATION_SCHEMA,
                "retained_exam_id": retained,
                "excluded_evidence_exam_id": dropped,
                "old_pcr": int(float(source.loc[retained, "pcr"])),
                "new_pcr": 1,
                "old_pcr_label_authority": _clean(
                    source.loc[retained, "pcr_label_authority"]
                ),
                "new_pcr_label_authority": _clean(
                    source.loc[dropped, "pcr_label_authority"]
                ),
                "new_rcb_class": _clean(source.loc[dropped, "rcb_class"]),
                "new_rcb_score": float(source.loc[dropped, "rcb_score"]),
                "reason": (
                    "pixel-identical reidentified exam carried explicit non-provisional "
                    "RCB pCR / score 0.0 pathology evidence"
                ),
            }
        )
    reconciliation_frame = pd.DataFrame(reconciliations)
    if len(reconciliation_frame) != EXPECTED_LABEL_RECONCILIATIONS:
        raise ValueError("unexpected number of canonical label reconciliations")
    return decision_frame, reconciliation_frame


def _filter_release_tables(
    *,
    tables: dict[str, pd.DataFrame],
    decisions: pd.DataFrame,
    reconciliations: pd.DataFrame,
    base_root: Path,
    output_root: Path,
) -> dict[str, pd.DataFrame]:
    """Filter duplicate exams, reconcile one label, and rewrite release-local paths."""
    dropped = set(decisions["dropped_exam_id"].astype(str))
    source_index = tables["source"].set_index("exam_id", drop=False)
    dropped_patients = set(source_index.loc[list(dropped), "patient_key"].astype(str))
    revised: dict[str, pd.DataFrame] = {}
    for key, frame in tables.items():
        candidate = frame.copy()
        if "exam_id" in candidate:
            candidate = candidate.loc[~candidate["exam_id"].isin(dropped)].copy()
        if key == "folds":
            candidate = candidate.loc[
                ~candidate["patient_key"].isin(dropped_patients)
            ].copy()
        revised[key] = candidate.reset_index(drop=True)

    label_columns = [
        "pcr",
        "label_source",
        "pcr_label_authority",
        "pcr_label_confidence",
        "pcr_label_is_provisional",
        "rcb_class",
        "rcb_score",
    ]
    for reconciliation in reconciliations.to_dict(orient="records"):
        retained = _clean(reconciliation["retained_exam_id"])
        evidence = _clean(reconciliation["excluded_evidence_exam_id"])
        for table_key in ("main", "source"):
            frame = revised[table_key]
            mask = frame["exam_id"].eq(retained)
            if mask.sum() != 1:
                raise ValueError(
                    "retained reconciliation exam is missing from a manifest"
                )
            for column in label_columns:
                if column in frame:
                    frame.loc[mask, column] = source_index.loc[evidence, column]
        for table_key in ("accounting", "pending"):
            frame = revised[table_key]
            if "exam_id" not in frame:
                continue
            mask = frame["exam_id"].eq(retained)
            if not mask.any():
                continue
            frame.loc[mask, "pcr"] = "1"
            if "pcr_label_authority" in frame:
                frame.loc[mask, "pcr_label_authority"] = source_index.loc[
                    evidence, "pcr_label_authority"
                ]
            if "pcr_label_is_provisional" in frame:
                frame.loc[mask, "pcr_label_is_provisional"] = "False"
        patient_key = _clean(source_index.loc[retained, "patient_key"])
        fold_mask = revised["folds"]["patient_key"].eq(patient_key)
        if fold_mask.sum() != 1:
            raise ValueError("retained reconciliation fold row is missing")
        revised["folds"].loc[fold_mask, "pcr"] = "1"

    old_root = str(base_root.resolve())
    new_root = str(output_root.resolve())
    for table_key in ("main", "source", "symlinks"):
        frame = revised[table_key]
        for column in frame.columns:
            if frame[column].astype(str).str.contains(old_root, regex=False).any():
                frame[column] = (
                    frame[column]
                    .astype(str)
                    .str.replace(old_root, new_root, regex=False)
                )
    for table_key in ("main", "source"):
        frame = revised[table_key]
        if "cohort_build_schema" in frame:
            frame["cohort_build_schema"] = SCHEMA
        if "patient_identity_scope" in frame:
            frame["patient_identity_scope"] = (
                "source_identity_exclusive_plus_selected_exam_image_content_dedup_v1"
            )
    revised["folds"]["schema"] = FOLD_SCHEMA
    revised["pending"]["schema"] = SCHEMA
    return revised


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a stable CSV with the existing column order."""
    frame.to_csv(path, index=False, lineterminator="\n")


def _stage_release_inputs(
    *,
    stage: Path,
    base_root: Path,
    audit_root: Path,
    decisions: pd.DataFrame,
) -> None:
    """Freeze the inherited source snapshots and deduplication evidence."""
    build_root = stage / "_build"
    snapshot_root = build_root / "input_snapshots"
    shutil.copytree(base_root / "_build/input_snapshots", snapshot_root)
    shutil.copy2(Path(__file__).resolve(), build_root / Path(__file__).name)
    shutil.copy2(base_root / "SHA256SUMS", snapshot_root / "base_v2_SHA256SUMS")
    shutil.copy2(
        base_root / "provenance.json", snapshot_root / "base_v2_provenance.json"
    )
    shutil.copy2(
        base_root / "validation_summary.json",
        snapshot_root / "base_v2_validation_summary.json",
    )
    shutil.copy2(audit_root / "SHA256SUMS", snapshot_root / "image_audit_SHA256SUMS")
    shutil.copy2(
        audit_root / "validation_summary.json",
        snapshot_root / "image_audit_validation_summary.json",
    )
    shutil.copy2(
        audit_root / "provenance.json", snapshot_root / "image_audit_provenance.json"
    )
    _write_csv(decisions, snapshot_root / "selected_image_duplicate_pairs.csv")


def _create_release_symlinks(
    *, stage: Path, output_root: Path, symlink_manifest: pd.DataFrame
) -> None:
    """Create only retained ready-exam links under the staging tree."""
    for row in symlink_manifest.to_dict(orient="records"):
        final_link = Path(_clean(row["link_path"]))
        target = Path(_clean(row["target_path"]))
        try:
            relative = final_link.relative_to(output_root)
        except ValueError as error:
            raise ValueError("a symlink path is outside the output root") from error
        if not target.is_dir():
            raise FileNotFoundError("a retained phase-export target is missing")
        staged_link = stage / relative
        staged_link.parent.mkdir(parents=True, exist_ok=True)
        if staged_link.exists() or staged_link.is_symlink():
            raise FileExistsError("duplicate staged symlink path")
        staged_link.symlink_to(target, target_is_directory=True)


def _staged_path(path: Path, *, stage: Path, output_root: Path) -> Path:
    """Map a final release-local path to its pre-publication staging path."""
    try:
        return stage / path.relative_to(output_root)
    except ValueError:
        return path


def _validate_release(
    *,
    tables: dict[str, pd.DataFrame],
    decisions: pd.DataFrame,
    reconciliations: pd.DataFrame,
    stage: Path,
    output_root: Path,
    base_root: Path,
    audit_root: Path,
) -> dict[str, Any]:
    """Validate compatibility, content accounting, paths, labels, folds, and dedup."""
    main = tables["main"]
    source = tables["source"]
    folds = tables["folds"]
    symlinks = tables["symlinks"]
    pending = tables["pending"]
    paired_case = tables["paired_case"]
    paired_source = tables["paired_source"]
    accounting = tables["accounting"]
    if (
        len(source) != EXPECTED_RETAINED_SOURCE_EXAMS
        or len(main) != EXPECTED_RETAINED_MAIN_EXAMS
        or len(symlinks) != EXPECTED_RETAINED_READY_EXAMS
        or len(pending) != EXPECTED_RETAINED_PENDING_EXAMS
        or len(paired_case) != EXPECTED_RETAINED_SIX_COLUMN_PAIRS
        or len(paired_source) != EXPECTED_RETAINED_SOURCE_EXAMS
        or len(accounting) != EXPECTED_RETAINED_SOURCE_EXAMS
        or len(folds) != EXPECTED_RETAINED_SOURCE_EXAMS
    ):
        raise ValueError("revised release row accounting failed")
    if (
        source["exam_id"].duplicated().any()
        or source["patient_key"].duplicated().any()
        or main["exam_id"].duplicated().any()
        or folds["patient_key"].duplicated().any()
    ):
        raise ValueError("revised release identity uniqueness failed")
    if not set(main["exam_id"]).issubset(set(source["exam_id"])):
        raise ValueError("main manifest is not a source-manifest subset")
    if {int(float(value)) for value in main["pcr"]} != {0, 1}:
        raise ValueError("main manifest labels are not binary")
    if {int(float(value)) for value in main["fold"]} != set(range(5)):
        raise ValueError("main manifest does not contain folds 0-4")
    if not main["high_resolution_partner_layout"].eq("single_series").all():
        raise ValueError("main manifest contains a split HR partner")

    reference = pd.read_csv(
        base_root / "_build/input_snapshots/reference_v1_manifest.csv", nrows=0
    )
    if list(main.columns[:37]) != list(reference.columns[:37]):
        raise ValueError("first 37 main-manifest columns no longer match v1")
    expected_pair_columns = [
        "exam_id",
        "dataset",
        "study_instance_uid",
        "hr_series_instance_uid",
        "ufast_series_instance_uid",
        "ufast_baseline_frame_count",
    ]
    if list(paired_case.columns) != expected_pair_columns:
        raise ValueError("six-column paired manifest schema changed")

    ready_source = source.loc[source["phase_export_status"].eq("ready")]
    if len(ready_source) != EXPECTED_RETAINED_READY_EXAMS:
        raise ValueError("ready source count does not match symlink count")
    linked_phase_files = 0
    for row in ready_source.to_dict(orient="records"):
        phase_files = [Path(value) for value in json.loads(_clean(row["phase_files"]))]
        times_path = Path(_clean(row["times_path"]))
        if len(phase_files) != int(float(row["n_phases"])):
            raise ValueError("a ready exam has a phase-count mismatch")
        staged_phase_files = [
            _staged_path(path, stage=stage, output_root=output_root)
            for path in phase_files
        ]
        staged_times = _staged_path(times_path, stage=stage, output_root=output_root)
        if (
            not all(path.is_file() for path in staged_phase_files)
            or not staged_times.is_file()
        ):
            raise FileNotFoundError("a retained ready-exam artifact is missing")
        if staged_times.suffix.lower() == ".npy":
            times = np.asarray(np.load(staged_times), dtype=float)
        elif staged_times.suffix.lower() == ".json":
            times = np.asarray(
                json.loads(staged_times.read_text(encoding="utf-8")), dtype=float
            )
        else:
            raise ValueError("a retained native timing array has an unsupported format")
        if (
            times.ndim != 1
            or len(times) != len(phase_files)
            or not np.isfinite(times).all()
            or times[0] != 0.0
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("a retained native timing array is invalid")
        linked_phase_files += len(phase_files)

    retained = set(source["exam_id"])
    scores = pd.read_csv(audit_root / "all_selected_pair_scores.csv")
    residual = scores.loc[
        scores["automatic_review_candidate"].map(_truthy)
        & scores["exam_id_a"].isin(retained)
        & scores["exam_id_b"].isin(retained)
    ]
    if not residual.empty:
        raise ValueError("an image duplicate candidate remains in the revised cohort")
    fingerprints = pd.read_csv(audit_root / "exam_image_fingerprints.csv")
    retained_fingerprints = fingerprints.loc[
        fingerprints["exam_id"].isin(retained)
        & ~fingerprints["is_positive_control"].map(_truthy)
    ]
    exact_columns = [
        "ufast_baseline_sha256",
        "ufast_late_sha256",
        "hr_baseline_sha256",
    ]
    if retained_fingerprints.duplicated(exact_columns).any():
        raise ValueError("an exact image fingerprint duplicate remains")

    source_pcr = source["pcr"].astype(float).astype(int).value_counts().to_dict()
    main_pcr = main["pcr"].astype(float).astype(int).value_counts().to_dict()
    if source_pcr != {0: 175, 1: 111} or main_pcr != {0: 142, 1: 101}:
        raise ValueError("post-reconciliation label accounting differs from expected")
    if len(reconciliations) != EXPECTED_LABEL_RECONCILIATIONS:
        raise ValueError("label reconciliation accounting failed")
    return {
        "schema": SCHEMA,
        "status": "passed",
        "base_release_checksums_verified": True,
        "image_audit_checksums_verified": True,
        "source_manifest_unique_exam_ids": True,
        "source_manifest_unique_source_patient_keys": True,
        "main_manifest_is_source_subset": True,
        "main_manifest_binary_labels_only": True,
        "main_manifest_folds": sorted(set(main["fold"].astype(int))),
        "main_manifest_single_hr_series_only": True,
        "first_37_columns_match_v1": True,
        "six_column_pair_contract_preserved": True,
        "selected_image_duplicate_pairs_excluded": int(len(decisions)),
        "remaining_image_duplicate_review_candidates": int(len(residual)),
        "remaining_exact_exam_fingerprint_duplicates": 0,
        "label_reconciliations": int(len(reconciliations)),
        "ready_exam_directories": int(len(ready_source)),
        "native_time_arrays_verified": int(len(ready_source)),
        "linked_phase_files_verified_present": int(linked_phase_files),
    }


def _counts(tables: dict[str, pd.DataFrame], decisions: pd.DataFrame) -> dict[str, Any]:
    """Build release-level count summaries from validated tables."""
    source = tables["source"]
    main = tables["main"]
    return {
        "source_eligible": {
            "exams": int(len(source)),
            "source_patient_identities": int(source["patient_key"].nunique()),
            "pcr_0": int(source["pcr"].astype(float).astype(int).eq(0).sum()),
            "pcr_1": int(source["pcr"].astype(float).astype(int).eq(1).sum()),
            "provisional_labels": int(
                source["pcr_label_is_provisional"].map(_truthy).sum()
            ),
        },
        "sarit_compatible_main": {
            "exams": int(len(main)),
            "source_patient_identities": int(main["patient_key"].nunique()),
            "pcr_0": int(main["pcr"].astype(float).astype(int).eq(0).sum()),
            "pcr_1": int(main["pcr"].astype(float).astype(int).eq(1).sum()),
            "provisional_labels": int(
                main["pcr_label_is_provisional"].map(_truthy).sum()
            ),
        },
        "cohort_components_source": source["cohort_component"].value_counts().to_dict(),
        "source_readiness": source["source_readiness"].value_counts().to_dict(),
        "full_fold_counts": source["fold"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict(),
        "main_fold_counts": main["fold"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict(),
        "pending_from_main": int(len(tables["pending"])),
        "six_column_pair_rows": int(len(tables["paired_case"])),
        "ready_exam_links": int(len(tables["symlinks"])),
        "image_duplicate_exclusions": {
            "total": int(len(decisions)),
            "canonical_reidentified_exact_exam": int(
                decisions["duplicate_classification"]
                .eq("canonical_reidentified_exact_exam")
                .sum()
            ),
            "cross_delivery_same_exam": int(
                decisions["duplicate_classification"]
                .eq("cross_delivery_same_exam")
                .sum()
            ),
            "cross_fold_before_exclusion": int(
                decisions["cross_fold_before_exclusion"].map(_truthy).sum()
            ),
        },
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    """Capture the exact repository commit and dirty flag."""

    def run(*arguments: str) -> str:
        return subprocess.run(  # noqa: S603 - fixed executable and trusted arguments
            ["/usr/bin/git", *arguments],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _write_checksums(root: Path) -> None:
    """Write checksums for every regular output file except the manifest itself."""
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(root)}" for path in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_protected_permissions(root: Path) -> None:
    """Apply group-protected release permissions without touching symlink targets."""
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        path.chmod(0o2770 if path.is_dir() else 0o660)
    root.chmod(0o2770)


def _publish(arguments: argparse.Namespace) -> dict[str, Any]:
    """Build, validate, checksum, and atomically publish the revised release."""
    base_root = arguments.base_root.expanduser().resolve()
    audit_root = arguments.audit_root.expanduser().resolve()
    output_root = arguments.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {output_root}")
    stage = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(f"staging directory already exists: {stage}")
    os.umask(0o007)

    base_checksums = _read_checksum_manifest(base_root)
    audit_checksums = _read_checksum_manifest(audit_root)
    base_validation = json.loads(
        (base_root / "validation_summary.json").read_text(encoding="utf-8")
    )
    if (
        base_validation.get("status") != "passed"
        or int(pd.read_csv(base_root / TABLE_FILES["source"]).shape[0])
        != EXPECTED_BASE_SOURCE_EXAMS
        or int(pd.read_csv(base_root / TABLE_FILES["main"]).shape[0])
        != EXPECTED_BASE_MAIN_EXAMS
    ):
        raise ValueError("base v2 release is not the expected validated input")
    tables = _read_tables(base_root)
    decisions, reconciliations = _derive_duplicate_decisions(
        tables=tables,
        base_root=base_root,
        audit_root=audit_root,
        audit_checksums=audit_checksums,
    )
    revised = _filter_release_tables(
        tables=tables,
        decisions=decisions,
        reconciliations=reconciliations,
        base_root=base_root,
        output_root=output_root,
    )

    stage.mkdir(parents=True, exist_ok=False)
    try:
        _stage_release_inputs(
            stage=stage,
            base_root=base_root,
            audit_root=audit_root,
            decisions=decisions,
        )
        for key, name in TABLE_FILES.items():
            _write_csv(revised[key], stage / name)
        _write_csv(decisions, stage / "image_duplicate_exclusions.csv")
        _write_csv(reconciliations, stage / "image_duplicate_label_reconciliations.csv")
        _create_release_symlinks(
            stage=stage,
            output_root=output_root,
            symlink_manifest=revised["symlinks"],
        )
        validation = _validate_release(
            tables=revised,
            decisions=decisions,
            reconciliations=reconciliations,
            stage=stage,
            output_root=output_root,
            base_root=base_root,
            audit_root=audit_root,
        )
        counts = _counts(revised, decisions)
        identity_audit = {
            "schema": IDENTITY_SCHEMA,
            "identifier_audit_base_path": str(
                (base_root / "identity_overlap_audit.json").resolve()
            ),
            "identifier_audit_base_sha256": base_checksums[
                "identity_overlap_audit.json"
            ],
            "image_content_audit_root": str(audit_root),
            "image_content_audit_SHA256SUMS_sha256": _sha256_file(
                audit_root / "SHA256SUMS"
            ),
            "selected_exam_pair_comparisons": 57_970,
            "within_canonical_exact_exam_pairs_excluded": EXPECTED_EXACT_CANONICAL_PAIRS,
            "cross_delivery_same_exam_pairs_excluded": EXPECTED_CROSS_DELIVERY_PAIRS,
            "selected_image_duplicate_pairs_excluded_total": EXPECTED_DUPLICATE_PAIRS,
            "retained_source_exams": EXPECTED_RETAINED_SOURCE_EXAMS,
            "retained_retro_source_exams": int(
                revised["source"]["cohort_component"].str.startswith("retro").sum()
            ),
            "remaining_selected_image_duplicate_review_candidates": 0,
            "fold_exclusivity_scope": (
                "source-identity exclusive after selected-exam image-content deduplication"
            ),
            "limitation": (
                "The image audit rules out retained copies of the same selected exam at the "
                "validated thresholds. Independent de-identification can still conceal one "
                "physical patient appearing with a different acquisition; a protected patient "
                "crosswalk remains necessary for a definitive physical-patient identity claim."
            ),
        }
        (stage / "identity_overlap_audit.json").write_text(
            json.dumps(identity_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "validation_summary.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        frozen_builder = stage / "_build" / Path(__file__).name
        provenance = {
            "schema": SCHEMA,
            "created_at": datetime.now().astimezone().isoformat(),
            "command": [sys.executable, *sys.argv],
            "output_root": str(output_root),
            "base_release": {
                "path": str(base_root),
                "SHA256SUMS_sha256": _sha256_file(base_root / "SHA256SUMS"),
                "verified_entries": len(base_checksums),
            },
            "image_duplicate_audit": {
                "path": str(audit_root),
                "SHA256SUMS_sha256": _sha256_file(audit_root / "SHA256SUMS"),
                "verified_entries": len(audit_checksums),
                "pair_scores_sha256": audit_checksums["all_selected_pair_scores.csv"],
                "slurm_job_id": "14400420",
                "slurm_state_exit": "COMPLETED 0:0",
            },
            "selection_contract": {
                "canonical_exact_pairs": (
                    "keep established legacy copy and fold; exclude reidentified Zhen copy"
                ),
                "cross_delivery_pairs": (
                    "keep established canonical copy and fold; exclude Retro copy"
                ),
                "retained_fold_assignment": "inherited unchanged from v2; no reassignment",
                "label_reconciliation": (
                    "one retained legacy exam updated from explicit non-provisional RCB pCR / "
                    "score 0.0 evidence on its pixel-identical excluded copy"
                ),
            },
            "counts": counts,
            "validation": validation,
            "identity_audit": identity_audit,
            "frozen_builder_sha256": _sha256_file(frozen_builder),
            "vanguard_git": _git_state(arguments.repo_root.expanduser().resolve()),
        }
        (stage / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        readme = f"""# DCE2D internal UFAST pretreatment pCR cohort v3

This release supersedes v2 for Sarit's pCR work. It keeps the v1-compatible main manifest
format while removing selected exams shown by direct image-content review to be duplicate
deliveries under different identifiers.

## Use this manifest

`dce2d_internal_ultrafast_manifest.csv`

- {counts["sarit_compatible_main"]["exams"]} runnable exams
- pCR 0/1: {counts["sarit_compatible_main"]["pcr_0"]}/{counts["sarit_compatible_main"]["pcr_1"]}
- one retained source identity per row
- first 37 columns unchanged from Sarit's v1 manifest contract

The broader `source_eligible_cohort_manifest.csv` contains
{counts["source_eligible"]["exams"]} image-deduplicated labeled exams, including
{counts["pending_from_main"]} typed pending cases that are not yet runnable in the main
six-column/single-series contract.

## Image duplicate correction

- 19 pixel-identical legacy-versus-Zhen canonical exam pairs: kept the established legacy
  copy and fold.
- 36 canonical-versus-Retro same-exam pairs: each had a pixel-identical full HR baseline and
  UFAST whole-volume correlations of at least {CROSS_UFAST_MINIMUM:.3f}; kept the canonical
  copy and fold.
- 45/55 duplicate pairs had been assigned to different folds in v2; removing one copy from
  every pair eliminates that direct leakage path.
- One exact canonical pair disagreed on pCR. The retained row now uses the excluded copy's
  explicit non-provisional `RCB class = pCR`, `RCB score = 0.0` evidence. This changes that
  retained label from 0 to 1 and is recorded in
  `image_duplicate_label_reconciliations.csv`.

See `image_duplicate_exclusions.csv`, `identity_overlap_audit.json`, `provenance.json`, and
`validation_summary.json` for the exact protected audit trail. The frozen v2 release and the
versioned image audit remain unchanged.

## Companion files

- `paired_preprocessing_case_manifest.csv`: Sarit's exact six-column pair contract
- `paired_source_manifest.csv`: all retained source pairs, including split-HR metadata
- `pending_unprocessed_pretreatment_sources.csv`: retained typed pending sources
- `fold_assignments.csv`: v2 folds inherited unchanged for retained identities
- `symlink_manifest.csv`: {counts["ready_exam_links"]} retained ready image links
- `_build/input_snapshots/`: frozen source, v2, and image-audit evidence

Residual limitation: direct image review rules out retained copies of the same selected exam,
but independent de-identification could still conceal the same physical patient returning for
a different acquisition. That stronger identity claim requires a protected patient crosswalk.
"""
        (stage / "README.md").write_text(readme, encoding="utf-8")
        _write_checksums(stage)
        _set_protected_permissions(stage)
        stage.rename(output_root)
    except Exception:
        # Preserve a failed staging directory for diagnosis; never publish partial output.
        raise
    return {"output_root": str(output_root), "counts": counts, "validation": validation}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def main() -> None:
    """Create the checksum-verified image-deduplicated v3 release."""
    result = _publish(parse_arguments())
    counts = result["counts"]
    print(
        "[release] complete "
        f"output={result['output_root']} "
        f"source={counts['source_eligible']['exams']} "
        f"main={counts['sarit_compatible_main']['exams']} "
        f"image_duplicate_exclusions={counts['image_duplicate_exclusions']['total']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
