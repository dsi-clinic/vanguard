#!/usr/bin/env python3
"""Publish Sarit's v4 cohort with two manually adjudicated non-pCR labels.

The immutable v3 image-deduplicated release is copied into a new version.  The
only scientific changes are two pCR-positive retained canonical exams whose
pixel-identical excluded Retro-CAPS copies were non-pCR.  Their complete
OncoTrace source projections were independently reconstructed and matched the
fingerprints saved at inference.  Anna explicitly adjudicated both as non-pCR.

Raw images, v3, and the protected evidence reconstruction remain read-only.
The builder verifies every v3 checksum, changes every release table carrying
the two labels, validates the exact cell-level change surface, and publishes
v4 atomically.

Retired.  This stage produced the pinned v4 release and is kept so that release has its producing
code in the repository.  Its two adjudications are now rows in the frozen `pcr_label_overrides.csv`
that `build_dce2d_ultrafast_pcr_cohort.py` applies to the upstream label fields, which re-derives
the label status this stage had to repair separately.  Do not run it against a new build.
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

import pandas as pd

SCHEMA = "vanguard.sarit_pcr_pretreatment_cohort.v4"
FOLD_SCHEMA = "vanguard.sarit_pcr_fold_assignment.v4_manual_pcr_adjudication"
ADJUDICATION_SCHEMA = "vanguard.sarit_pcr_label_adjudication.v1"
METADATA_REPAIR_SCHEMA = "vanguard.sarit_pcr_label_metadata_repair.v1"

EXPECTED_SOURCE_EXAMS = 286
EXPECTED_MAIN_EXAMS = 243
EXPECTED_READY_EXAMS = 246
EXPECTED_PENDING_EXAMS = 43
EXPECTED_FOLD_ROWS = 286
EXPECTED_TARGETS = 2
EXPECTED_METADATA_REPAIRS = 1
EXPECTED_V3_CHECKSUM_ENTRIES = 40
EXPECTED_SOURCE_PCR = {0: 177, 1: 109}
EXPECTED_MAIN_PCR = {0: 144, 1: 99}

NEW_PCR = "0"
NEW_STATUS = "not_supported"
NEW_AUTHORITY = "anna_manual"
NEW_CONFIDENCE = "high"
NEW_PROVISIONAL = "False"
ADJUDICATED_BY = "anna"
ADJUDICATED_ON = "2026-08-25"
RECONSTRUCTION_JOB_ID = "14404126"
RECONSTRUCTION_JOB_STATE = "COMPLETED 0:0"
PCR_DEFINITION = (
    "No residual invasive carcinoma in the treated breast and no residual "
    "regional-node disease after neoadjuvant therapy; residual DCIS alone does "
    "not refute pCR."
)
USER_INSTRUCTION = "make sure in our dataset these are both called as non-pcr"

DEFAULT_BASE_ROOT = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v4"
)
DEFAULT_EVIDENCE_ROOT = Path(
    "/gpfs/data/huo-lab/Image/annawoodard/hfdp/derived_datasets/"
    "retro_caps_pcr_cohort/glm52_label_error_reconstruction_v1"
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

MANIFEST_LABEL_COLUMNS = (
    "pcr",
    "label_source",
    "pcr_label_status",
    "pcr_label_authority",
    "pcr_label_confidence",
    "pcr_label_is_provisional",
)
ACCOUNTING_LABEL_COLUMNS = (
    "pcr",
    "pcr_label_authority",
    "pcr_label_is_provisional",
)
EXPECTED_TARGET_AUTHORITIES = {
    "simbiosys_uchicago_workbook_pcr",
    "uch_brca_nac_pcr",
}
CASE_REASONS = {
    "simbiosys_uchicago_workbook_pcr": (
        "The treated breast had no residual invasive carcinoma, but definitive "
        "post-treatment pathology showed residual regional-node disease. The "
        "v4 endpoint requires both breast and node clearance."
    ),
    "uch_brca_nac_pcr": (
        "Definitive post-treatment pathology showed residual invasive carcinoma "
        "in the current treated breast. The prior contralateral cancer episode "
        "does not establish pCR for this image-linked episode."
    ),
}


def _clean(value: object) -> str:
    """Return a stable stripped string for mixed CSV values."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    """Interpret common serialized boolean values."""
    return _clean(value).lower() in {"1", "true", "t", "yes", "y"}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _read_checksum_manifest(root: Path) -> dict[str, str]:
    """Parse and verify every entry in a release checksum manifest."""
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError("base release checksum manifest is missing")
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", maxsplit=1)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("base checksum manifest contains an unsafe path")
        path = root / relative_path
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError("base release checksum verification failed")
        entries[relative] = digest
    if len(entries) != EXPECTED_V3_CHECKSUM_ENTRIES:
        raise ValueError("base release checksum entry count changed")
    return entries


def _read_tables(root: Path) -> dict[str, pd.DataFrame]:
    """Read all release contract tables as strings."""
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError("a required base release table is missing")
        tables[key] = pd.read_csv(path, dtype=str).fillna("")
    return tables


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Fail when a required table schema changed."""
    if not columns.issubset(frame.columns):
        raise ValueError(f"{label} schema changed")


def _load_evidence(evidence_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the protected reconstruction and return authority-keyed evidence."""
    summary_path = evidence_root / "reconstruction_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("protected reconstruction summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("case_count") != EXPECTED_TARGETS
        or summary.get("fingerprint_match_count") != EXPECTED_TARGETS
        or summary.get("reasoning_nonempty_model_responses") != 0
    ):
        raise ValueError("protected reconstruction did not pass its declared checks")

    by_authority: dict[str, Any] = {}
    safe_cases: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        authority = _clean(case.get("legacy_label_authority"))
        index = int(case.get("case_index", 0))
        case_root = evidence_root / f"case_{index:02d}"
        packet_path = case_root / "accessible_evidence_packet.json"
        transcript_path = case_root / "exposure_transcript.json"
        if (
            authority in by_authority
            or not case.get("fingerprint_match")
            or not case.get("row_count_match")
            or int(case.get("legacy_pcr", -1)) != 1
            or int(case.get("retro_pcr", -1)) != 0
            or case.get("retro_label_status") != NEW_STATUS
            or case.get("retro_label_authority") != "oncotrace_glm52"
            or not packet_path.is_file()
            or not transcript_path.is_file()
        ):
            raise ValueError("one reconstructed evidence case is invalid")
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        answer = transcript.get("final_answer_exact_from_unit_record") or {}
        if (
            packet.get("projection_fingerprint")
            != case.get("record_projection_fingerprint")
            or answer.get("endpoint") != NEW_STATUS
            or transcript.get("exact_persisted_unit_record_sha256")
            != case.get("record_sha256")
        ):
            raise ValueError("one evidence packet does not bind to its unit record")
        evidence = {
            "case_index": index,
            "legacy_label_authority": authority,
            "source_projection_rows": int(case["record_row_count"]),
            "source_projection_fingerprint": case["record_projection_fingerprint"],
            "unit_record_sha256": case["record_sha256"],
            "evidence_packet_path": str(packet_path.resolve()),
            "evidence_packet_sha256": _sha256_file(packet_path),
            "exposure_transcript_path": str(transcript_path.resolve()),
            "exposure_transcript_sha256": _sha256_file(transcript_path),
        }
        by_authority[authority] = evidence
        safe_cases.append(
            {
                key: value
                for key, value in evidence.items()
                if key not in {"evidence_packet_path", "exposure_transcript_path"}
            }
        )
    if set(by_authority) != EXPECTED_TARGET_AUTHORITIES:
        raise ValueError("reconstruction authorities do not match the target labels")
    verification = {
        "schema": "vanguard.sarit_pcr_glm52_reconstruction_verification.v1",
        "reconstruction_root": str(evidence_root.resolve()),
        "reconstruction_summary_sha256": _sha256_file(summary_path),
        "slurm_job_id": RECONSTRUCTION_JOB_ID,
        "slurm_state_exit": RECONSTRUCTION_JOB_STATE,
        "fingerprint_matches": EXPECTED_TARGETS,
        "cases": sorted(safe_cases, key=lambda row: int(row["case_index"])),
    }
    return by_authority, verification


def _derive_adjudications(
    *,
    base_root: Path,
    tables: dict[str, pd.DataFrame],
    evidence_by_authority: dict[str, Any],
) -> pd.DataFrame:
    """Bind the two cross-delivery label conflicts to their retained exams."""
    decisions = pd.read_csv(
        base_root / "image_duplicate_exclusions.csv", dtype=str
    ).fillna("")
    _require_columns(
        decisions,
        {
            "dropped_exam_id",
            "retained_exam_id",
            "duplicate_classification",
            "pcr_label_agreement",
            "dropped_pcr",
            "retained_pcr_before_reconciliation",
            "dropped_pcr_label_authority",
            "retained_pcr_label_authority_before_reconciliation",
        },
        "image duplicate exclusions",
    )
    targets = decisions.loc[
        decisions["duplicate_classification"].eq("cross_delivery_same_exam")
        & ~decisions["pcr_label_agreement"].map(_truthy)
    ].copy()
    if len(targets) != EXPECTED_TARGETS:
        raise ValueError("expected two cross-delivery pCR conflicts")
    source = tables["source"].set_index("exam_id", drop=False)
    rows: list[dict[str, Any]] = []
    for target in targets.sort_values(
        "retained_pcr_label_authority_before_reconciliation", kind="stable"
    ).to_dict(orient="records"):
        retained = _clean(target["retained_exam_id"])
        excluded = _clean(target["dropped_exam_id"])
        authority = _clean(target["retained_pcr_label_authority_before_reconciliation"])
        if (
            retained not in source.index
            or int(float(source.loc[retained, "pcr"])) != 1
            or _clean(source.loc[retained, "pcr_label_authority"]) != authority
            or int(float(target["retained_pcr_before_reconciliation"])) != 1
            or int(float(target["dropped_pcr"])) != 0
            or _clean(target["dropped_pcr_label_authority"]) != "oncotrace_glm52"
            or authority not in evidence_by_authority
        ):
            raise ValueError("a target label no longer matches the adjudication basis")
        evidence = evidence_by_authority[authority]
        rows.append(
            {
                "schema": ADJUDICATION_SCHEMA,
                "retained_exam_id": retained,
                "excluded_retro_evidence_exam_id": excluded,
                "same_physical_exam_evidence": (
                    "pixel-identical full HR baseline plus validated UFAST image match"
                ),
                "old_pcr": 1,
                "old_pcr_label_status": "supported",
                "old_pcr_label_authority": authority,
                "new_pcr": 0,
                "new_pcr_label_status": NEW_STATUS,
                "new_pcr_label_authority": NEW_AUTHORITY,
                "new_pcr_label_confidence": NEW_CONFIDENCE,
                "new_pcr_label_is_provisional": False,
                "pcr_endpoint_definition": PCR_DEFINITION,
                "adjudication_reason": CASE_REASONS[authority],
                "adjudicated_by": ADJUDICATED_BY,
                "adjudicated_on": ADJUDICATED_ON,
                "adjudication_basis": (
                    "explicit PI decision after review of the case-safe summary from "
                    "a fingerprint-matched protected evidence reconstruction"
                ),
                "direct_packet_review_by_adjudicator": False,
                "explicit_user_instruction": USER_INSTRUCTION,
                "reconstruction_slurm_job_id": RECONSTRUCTION_JOB_ID,
                "reconstruction_slurm_state_exit": RECONSTRUCTION_JOB_STATE,
                **evidence,
            }
        )
    result = pd.DataFrame(rows)
    if (
        len(result) != EXPECTED_TARGETS
        or result["retained_exam_id"].nunique() != EXPECTED_TARGETS
        or result["excluded_retro_evidence_exam_id"].nunique() != EXPECTED_TARGETS
    ):
        raise ValueError("label adjudication identities are not one-to-one")
    return result


def _derive_v3_status_repairs(
    *, base_root: Path, tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Identify the one pCR-status field missed by the v3 reconciliation."""
    reconciliations = pd.read_csv(
        base_root / "image_duplicate_label_reconciliations.csv", dtype=str
    ).fillna("")
    if len(reconciliations) != EXPECTED_METADATA_REPAIRS:
        raise ValueError("the inherited duplicate-label reconciliation count changed")
    source = tables["source"].set_index("exam_id", drop=False)
    rows: list[dict[str, Any]] = []
    for reconciliation in reconciliations.to_dict(orient="records"):
        retained = _clean(reconciliation.get("retained_exam_id"))
        if (
            retained not in source.index
            or int(float(reconciliation.get("new_pcr", -1))) != 1
            or _clean(reconciliation.get("new_rcb_class")).lower() != "pcr"
            or float(reconciliation.get("new_rcb_score", -1)) != 0.0
            or int(float(source.loc[retained, "pcr"])) != 1
            or _clean(source.loc[retained, "pcr_label_status"]) != NEW_STATUS
            or _clean(source.loc[retained, "pcr_label_authority"])
            != _clean(reconciliation.get("new_pcr_label_authority"))
        ):
            raise ValueError("the inherited v3 status mismatch changed unexpectedly")
        rows.append(
            {
                "schema": METADATA_REPAIR_SCHEMA,
                "exam_id": retained,
                "pcr": 1,
                "old_pcr_label_status": NEW_STATUS,
                "new_pcr_label_status": "supported",
                "pcr_label_authority": _clean(
                    source.loc[retained, "pcr_label_authority"]
                ),
                "rcb_class": _clean(source.loc[retained, "rcb_class"]),
                "rcb_score": _clean(source.loc[retained, "rcb_score"]),
                "reason": (
                    "v3 copied the explicit non-provisional RCB pCR label but omitted "
                    "the corresponding pcr_label_status field"
                ),
                "source_audit_table": "image_duplicate_label_reconciliations.csv",
            }
        )
    repairs = pd.DataFrame(rows)
    if len(repairs) != EXPECTED_METADATA_REPAIRS:
        raise ValueError("unexpected inherited label-status repair count")
    return repairs


def _replace_release_root(
    frame: pd.DataFrame, *, base_root: Path, output_root: Path
) -> pd.DataFrame:
    """Rewrite embedded release-local paths from v3 to v4."""
    result = frame.copy()
    old = str(base_root.resolve())
    new = str(output_root.resolve())
    for column in result.columns:
        values = result[column].astype(str)
        if values.str.contains(old, regex=False).any():
            result[column] = values.str.replace(old, new, regex=False)
    return result


def _revise_tables(
    *,
    tables: dict[str, pd.DataFrame],
    adjudications: pd.DataFrame,
    metadata_repairs: pd.DataFrame,
    base_root: Path,
    output_root: Path,
) -> dict[str, pd.DataFrame]:
    """Apply the two labels consistently while preserving membership and folds."""
    revised = {key: frame.copy() for key, frame in tables.items()}
    targets = set(adjudications["retained_exam_id"].astype(str))
    repair_exams = set(metadata_repairs["exam_id"].astype(str))
    if targets.intersection(repair_exams):
        raise ValueError("a manual adjudication overlaps an inherited metadata repair")
    source_before = tables["source"].set_index("exam_id", drop=False)

    for key in ("main", "source"):
        frame = revised[key]
        mask = frame["exam_id"].isin(targets)
        if int(mask.sum()) != EXPECTED_TARGETS:
            raise ValueError("a target is absent from a delivered manifest")
        frame.loc[mask, "pcr"] = NEW_PCR
        frame.loc[mask, "label_source"] = NEW_AUTHORITY
        frame.loc[mask, "pcr_label_status"] = NEW_STATUS
        frame.loc[mask, "pcr_label_authority"] = NEW_AUTHORITY
        frame.loc[mask, "pcr_label_confidence"] = NEW_CONFIDENCE
        frame.loc[mask, "pcr_label_is_provisional"] = NEW_PROVISIONAL
        repair_mask = frame["exam_id"].isin(repair_exams)
        if int(repair_mask.sum()) != EXPECTED_METADATA_REPAIRS:
            raise ValueError("the inherited status repair is absent from a manifest")
        frame.loc[repair_mask, "pcr_label_status"] = "supported"
        if "cohort_build_schema" not in frame:
            raise ValueError("manifest build schema column is missing")
        frame["cohort_build_schema"] = SCHEMA
        revised[key] = _replace_release_root(
            frame, base_root=base_root, output_root=output_root
        )

    accounting = revised["accounting"]
    accounting_mask = accounting["exam_id"].isin(targets)
    if int(accounting_mask.sum()) != EXPECTED_TARGETS:
        raise ValueError("a target is absent from cohort accounting")
    accounting.loc[accounting_mask, "pcr"] = NEW_PCR
    accounting.loc[accounting_mask, "pcr_label_authority"] = NEW_AUTHORITY
    accounting.loc[accounting_mask, "pcr_label_is_provisional"] = NEW_PROVISIONAL

    pending = revised["pending"]
    if pending["exam_id"].isin(targets).any():
        raise ValueError("an adjudicated target unexpectedly appears in pending rows")
    pending["schema"] = SCHEMA

    target_patients = {
        _clean(source_before.loc[exam_id, "patient_key"]) for exam_id in targets
    }
    folds = revised["folds"]
    fold_mask = folds["patient_key"].isin(target_patients)
    if int(fold_mask.sum()) != EXPECTED_TARGETS:
        raise ValueError("a target patient is absent from fold assignments")
    folds.loc[fold_mask, "pcr"] = NEW_PCR
    folds["schema"] = FOLD_SCHEMA

    revised["symlinks"] = _replace_release_root(
        revised["symlinks"], base_root=base_root, output_root=output_root
    )
    return revised


def _normalize_root(
    frame: pd.DataFrame, *, base_root: Path, output_root: Path
) -> pd.DataFrame:
    """Map v4 paths back to v3 paths for exact comparison."""
    result = frame.copy()
    old = str(output_root.resolve())
    new = str(base_root.resolve())
    for column in result.columns:
        values = result[column].astype(str)
        if values.str.contains(old, regex=False).any():
            result[column] = values.str.replace(old, new, regex=False)
    return result


def _pcr_counts(frame: pd.DataFrame) -> dict[int, int]:
    """Return binary pCR counts with integer keys."""
    return {
        int(key): int(value)
        for key, value in frame["pcr"]
        .astype(float)
        .astype(int)
        .value_counts()
        .to_dict()
        .items()
    }


def _validate_revision(
    *,
    base: dict[str, pd.DataFrame],
    revised: dict[str, pd.DataFrame],
    adjudications: pd.DataFrame,
    metadata_repairs: pd.DataFrame,
    stage: Path,
    base_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Prove that membership, folds, images, and all other labels are unchanged."""
    expected_lengths = {
        "source": EXPECTED_SOURCE_EXAMS,
        "main": EXPECTED_MAIN_EXAMS,
        "accounting": EXPECTED_SOURCE_EXAMS,
        "folds": EXPECTED_FOLD_ROWS,
        "pending": EXPECTED_PENDING_EXAMS,
        "symlinks": EXPECTED_READY_EXAMS,
    }
    for key, expected in expected_lengths.items():
        if len(base[key]) != expected or len(revised[key]) != expected:
            raise ValueError("release row accounting changed")
    for key in TABLE_FILES:
        if list(base[key].columns) != list(revised[key].columns):
            raise ValueError("an inherited CSV schema changed")

    targets = set(adjudications["retained_exam_id"].astype(str))
    repair_exams = set(metadata_repairs["exam_id"].astype(str))
    base_source = base["source"].set_index("exam_id", drop=False)
    target_patients = {
        _clean(base_source.loc[exam_id, "patient_key"]) for exam_id in targets
    }

    for key in ("main", "source"):
        before = base[key].copy()
        after = _normalize_root(
            revised[key], base_root=base_root, output_root=output_root
        )
        if not before["exam_id"].equals(after["exam_id"]):
            raise ValueError("manifest membership or ordering changed")
        target_mask = before["exam_id"].isin(targets)
        repair_mask = before["exam_id"].isin(repair_exams)
        for column in before.columns:
            if column in MANIFEST_LABEL_COLUMNS:
                allowed_mask = target_mask.copy()
                if column == "pcr_label_status":
                    allowed_mask |= repair_mask
                if not before.loc[~allowed_mask, column].equals(
                    after.loc[~allowed_mask, column]
                ):
                    raise ValueError("a non-target manifest label changed")
            elif column == "cohort_build_schema":
                if not after[column].eq(SCHEMA).all():
                    raise ValueError("manifest build schema was not advanced to v4")
            elif not before[column].equals(after[column]):
                raise ValueError("a non-label manifest value changed")
        if not (
            before.loc[target_mask, "pcr"].eq("1").all()
            and after.loc[target_mask, "pcr"].eq(NEW_PCR).all()
            and after.loc[target_mask, "label_source"].eq(NEW_AUTHORITY).all()
            and after.loc[target_mask, "pcr_label_status"].eq(NEW_STATUS).all()
            and after.loc[target_mask, "pcr_label_authority"].eq(NEW_AUTHORITY).all()
            and after.loc[target_mask, "pcr_label_confidence"].eq(NEW_CONFIDENCE).all()
            and after.loc[target_mask, "pcr_label_is_provisional"]
            .eq(NEW_PROVISIONAL)
            .all()
        ):
            raise ValueError("target manifest labels do not match adjudication")
        if not (
            int(repair_mask.sum()) == EXPECTED_METADATA_REPAIRS
            and before.loc[repair_mask, "pcr"].eq("1").all()
            and after.loc[repair_mask, "pcr"].eq("1").all()
            and before.loc[repair_mask, "pcr_label_status"].eq(NEW_STATUS).all()
            and after.loc[repair_mask, "pcr_label_status"].eq("supported").all()
        ):
            raise ValueError("the inherited pCR status metadata repair failed")

    before_accounting = base["accounting"]
    after_accounting = revised["accounting"]
    accounting_targets = before_accounting["exam_id"].isin(targets)
    if not before_accounting["exam_id"].equals(after_accounting["exam_id"]):
        raise ValueError("cohort accounting membership changed")
    for column in before_accounting.columns:
        if column in ACCOUNTING_LABEL_COLUMNS:
            if not before_accounting.loc[~accounting_targets, column].equals(
                after_accounting.loc[~accounting_targets, column]
            ):
                raise ValueError("a non-target accounting label changed")
        elif not before_accounting[column].equals(after_accounting[column]):
            raise ValueError("a non-label accounting value changed")
    if not (
        after_accounting.loc[accounting_targets, "pcr"].eq(NEW_PCR).all()
        and after_accounting.loc[accounting_targets, "pcr_label_authority"]
        .eq(NEW_AUTHORITY)
        .all()
        and after_accounting.loc[accounting_targets, "pcr_label_is_provisional"]
        .eq(NEW_PROVISIONAL)
        .all()
    ):
        raise ValueError("target accounting labels do not match adjudication")

    before_folds = base["folds"]
    after_folds = revised["folds"]
    fold_targets = before_folds["patient_key"].isin(target_patients)
    if not (
        before_folds["patient_key"].equals(after_folds["patient_key"])
        and before_folds["fold"].equals(after_folds["fold"])
        and before_folds["assignment_source"].equals(after_folds["assignment_source"])
        and before_folds.loc[~fold_targets, "pcr"].equals(
            after_folds.loc[~fold_targets, "pcr"]
        )
        and before_folds.loc[fold_targets, "pcr"].eq("1").all()
        and after_folds.loc[fold_targets, "pcr"].eq(NEW_PCR).all()
        and after_folds["schema"].eq(FOLD_SCHEMA).all()
    ):
        raise ValueError("fold membership or label propagation changed unexpectedly")

    before_pending = base["pending"].copy()
    after_pending = revised["pending"].copy()
    before_pending["schema"] = SCHEMA
    if not before_pending.equals(after_pending):
        raise ValueError("pending rows changed beyond the release schema")

    normalized_symlinks = _normalize_root(
        revised["symlinks"], base_root=base_root, output_root=output_root
    )
    if not base["symlinks"].equals(normalized_symlinks):
        raise ValueError("symlink manifest changed beyond its release-local paths")
    for key in ("paired_case", "paired_exclusions", "paired_source", "retro_dedup"):
        if not base[key].equals(revised[key]):
            raise ValueError("an inherited companion table changed")

    source = revised["source"]
    main = revised["main"]
    if (
        _pcr_counts(source) != EXPECTED_SOURCE_PCR
        or _pcr_counts(main) != EXPECTED_MAIN_PCR
        or source["exam_id"].duplicated().any()
        or source["patient_key"].duplicated().any()
        or main["exam_id"].duplicated().any()
        or not set(main["exam_id"]).issubset(set(source["exam_id"]))
        or not (main["pcr"].eq("1") == main["pcr_label_status"].eq("supported")).all()
    ):
        raise ValueError("final manifest label or identity accounting failed")

    link_count = 0
    for row in revised["symlinks"].to_dict(orient="records"):
        final_link = Path(_clean(row["link_path"]))
        target = Path(_clean(row["target_path"]))
        try:
            staged_link = stage / final_link.relative_to(output_root)
        except ValueError as error:
            raise ValueError("a release symlink path is outside v4") from error
        if (
            not staged_link.is_symlink()
            or not staged_link.is_dir()
            or not target.is_dir()
        ):
            raise ValueError("a retained release image link is invalid")
        link_count += 1
    if link_count != EXPECTED_READY_EXAMS:
        raise ValueError("ready image-link accounting changed")

    return {
        "schema": SCHEMA,
        "status": "passed",
        "base_v3_checksums_verified": True,
        "source_projection_fingerprints_matched": EXPECTED_TARGETS,
        "label_adjudications": EXPECTED_TARGETS,
        "target_labels_non_pcr_in_main": EXPECTED_TARGETS,
        "target_labels_non_pcr_in_source": EXPECTED_TARGETS,
        "target_labels_non_pcr_in_accounting": EXPECTED_TARGETS,
        "target_labels_non_pcr_in_folds": EXPECTED_TARGETS,
        "inherited_pcr_status_metadata_repairs": EXPECTED_METADATA_REPAIRS,
        "cohort_membership_changed": False,
        "fold_assignment_changed": False,
        "image_links_changed": False,
        "non_target_numeric_pcr_labels_changed": False,
        "source_manifest_unique_exam_ids": True,
        "source_manifest_unique_source_patient_keys": True,
        "main_manifest_is_source_subset": True,
        "ready_exam_links_verified": link_count,
        "source_pcr_0": EXPECTED_SOURCE_PCR[0],
        "source_pcr_1": EXPECTED_SOURCE_PCR[1],
        "main_pcr_0": EXPECTED_MAIN_PCR[0],
        "main_pcr_1": EXPECTED_MAIN_PCR[1],
    }


def _counts(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Build release-level counts from the validated tables."""
    source = tables["source"]
    main = tables["main"]
    return {
        "source_eligible": {
            "exams": int(len(source)),
            "pcr_0": int(_pcr_counts(source).get(0, 0)),
            "pcr_1": int(_pcr_counts(source).get(1, 0)),
            "provisional_labels": int(
                source["pcr_label_is_provisional"].map(_truthy).sum()
            ),
            "anna_manual_labels": int(
                source["pcr_label_authority"].eq(NEW_AUTHORITY).sum()
            ),
        },
        "sarit_compatible_main": {
            "exams": int(len(main)),
            "pcr_0": int(_pcr_counts(main).get(0, 0)),
            "pcr_1": int(_pcr_counts(main).get(1, 0)),
            "provisional_labels": int(
                main["pcr_label_is_provisional"].map(_truthy).sum()
            ),
            "anna_manual_labels": int(
                main["pcr_label_authority"].eq(NEW_AUTHORITY).sum()
            ),
        },
        "fold_counts": {
            str(int(key)): int(value)
            for key, value in main["fold"]
            .astype(int)
            .value_counts()
            .sort_index()
            .items()
        },
        "pending_exams": int(len(tables["pending"])),
        "ready_image_links": int(len(tables["symlinks"])),
        "label_adjudications": EXPECTED_TARGETS,
        "inherited_pcr_status_metadata_repairs": EXPECTED_METADATA_REPAIRS,
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    """Capture the repository commit and dirty flag."""

    def run(*arguments: str) -> str:
        return subprocess.run(  # noqa: S603 - fixed executable and arguments
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


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write one stable CSV while preserving column order."""
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_checksums(root: Path) -> None:
    """Write checksums for every regular output file except SHA256SUMS."""
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(root)}" for path in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_release_permissions(root: Path) -> None:
    """Apply the existing group-protected release permission contract."""
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        path.chmod(0o2770 if path.is_dir() else 0o660)
    root.chmod(0o2770)


def _write_snapshots(
    *,
    stage: Path,
    base_root: Path,
    evidence_root: Path,
    evidence_verification: dict[str, Any],
) -> Path:
    """Freeze the builder and the exact upstream provenance surfaces."""
    build_root = stage / "_build"
    snapshot_root = build_root / "input_snapshots"
    frozen_builder = build_root / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_builder)
    shutil.copy2(base_root / "SHA256SUMS", snapshot_root / "base_v3_SHA256SUMS")
    shutil.copy2(
        base_root / "provenance.json", snapshot_root / "base_v3_provenance.json"
    )
    shutil.copy2(
        base_root / "validation_summary.json",
        snapshot_root / "base_v3_validation_summary.json",
    )
    shutil.copy2(
        evidence_root / "README.md",
        snapshot_root / "glm52_reconstruction_README.md",
    )
    (snapshot_root / "glm52_reconstruction_verification.json").write_text(
        json.dumps(evidence_verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frozen_builder


def _readme(counts: dict[str, Any]) -> str:
    """Return the user-facing v4 release description."""
    main = counts["sarit_compatible_main"]
    source = counts["source_eligible"]
    return f"""# DCE2D internal UFAST pretreatment pCR cohort v4

This release supersedes v3 for Sarit's pCR work. It preserves every v3 exam, image link,
and fold while changing two image-linked pCR labels from positive to non-pCR after explicit
manual adjudication of fingerprint-matched protected evidence reconstructions.

## Use this manifest

`dce2d_internal_ultrafast_manifest.csv`

- {main["exams"]} runnable exams
- pCR 0/1: {main["pcr_0"]}/{main["pcr_1"]}
- the same membership, images, row order, and folds as v3
- first 37 columns unchanged from Sarit's v1 manifest contract

The broader `source_eligible_cohort_manifest.csv` contains {source["exams"]}
image-deduplicated labeled exams with pCR 0/1 = {source["pcr_0"]}/{source["pcr_1"]}.

## Two manual non-pCR adjudications

Both retained canonical exams are pixel-matched to excluded Retro-CAPS copies whose full
clinical evidence projections were reconstructed exactly: both row counts and inference-time
content fingerprints matched. The exact saved tool-result traces support non-pCR under this
release's endpoint definition: no residual invasive carcinoma in the treated breast **and**
no residual regional-node disease after neoadjuvant therapy.

- One current treatment episode has residual invasive breast carcinoma; a patient-level
  legacy positive label appears to refer to a different cancer episode/laterality.
- One has breast response but residual regional-node disease; the legacy binary source did
  not retain a nodal endpoint definition.

Anna explicitly directed that both be called non-pCR. They are recorded as `pcr = 0`,
`pcr_label_status = not_supported`, `pcr_label_authority = anna_manual`, high confidence,
and non-provisional in every delivered label table. `pcr_label_adjudications.csv` binds each
change to the exact evidence-packet, transcript, and source-record hashes. It also records
that the decision used the case-safe reconstruction summary rather than direct packet review.

## Provenance and compatibility

- v3 remains immutable and all {EXPECTED_V3_CHECKSUM_ENTRIES} of its checksum entries were
  verified before this release was built.
- No cohort membership, image link, fold assignment, or other numeric pCR label changed.
- v4 also repairs one inherited metadata inconsistency: v3's explicit RCB pCR
  reconciliation had `pcr = 1` but retained the old textual `not_supported` status.
  `pcr_label_metadata_repairs.csv` records the status-only correction to `supported`.
- `validation_summary.json` records the exact-change checks.
- `_build/input_snapshots/` freezes v3 provenance, checksums, the reconstruction verification,
  and the publishing script.

The v3 duplicate audit and its original pre-reconciliation label fields remain unchanged as
historical evidence. The v4 manual decisions live in the separate adjudication table.
"""


def _preflight(arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify all immutable inputs and derive the two adjudications."""
    base_root = arguments.base_root.expanduser().resolve()
    evidence_root = arguments.evidence_root.expanduser().resolve()
    base_validation = json.loads(
        (base_root / "validation_summary.json").read_text(encoding="utf-8")
    )
    if base_validation.get("status") != "passed":
        raise ValueError("base v3 validation is not passed")
    checksums = _read_checksum_manifest(base_root)
    tables = _read_tables(base_root)
    evidence, evidence_verification = _load_evidence(evidence_root)
    adjudications = _derive_adjudications(
        base_root=base_root,
        tables=tables,
        evidence_by_authority=evidence,
    )
    metadata_repairs = _derive_v3_status_repairs(base_root=base_root, tables=tables)
    return {
        "base_root": base_root,
        "evidence_root": evidence_root,
        "checksums": checksums,
        "tables": tables,
        "evidence_verification": evidence_verification,
        "adjudications": adjudications,
        "metadata_repairs": metadata_repairs,
    }


def _publish(arguments: argparse.Namespace) -> dict[str, Any]:
    """Copy v3, apply the two labels, validate, checksum, and publish v4."""
    inputs = _preflight(arguments)
    base_root: Path = inputs["base_root"]
    evidence_root: Path = inputs["evidence_root"]
    base_tables: dict[str, pd.DataFrame] = inputs["tables"]
    adjudications: pd.DataFrame = inputs["adjudications"]
    metadata_repairs: pd.DataFrame = inputs["metadata_repairs"]
    output_root = arguments.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("refusing to overwrite an existing release")
    stage = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if stage.exists():
        raise FileExistsError("release staging directory already exists")
    os.umask(0o007)

    revised = _revise_tables(
        tables=base_tables,
        adjudications=adjudications,
        metadata_repairs=metadata_repairs,
        base_root=base_root,
        output_root=output_root,
    )
    shutil.copytree(base_root, stage, symlinks=True, copy_function=shutil.copy2)
    try:
        for key, filename in TABLE_FILES.items():
            _write_csv(revised[key], stage / filename)
        _write_csv(adjudications, stage / "pcr_label_adjudications.csv")
        _write_csv(metadata_repairs, stage / "pcr_label_metadata_repairs.csv")
        frozen_builder = _write_snapshots(
            stage=stage,
            base_root=base_root,
            evidence_root=evidence_root,
            evidence_verification=inputs["evidence_verification"],
        )
        validation = _validate_revision(
            base=base_tables,
            revised=revised,
            adjudications=adjudications,
            metadata_repairs=metadata_repairs,
            stage=stage,
            base_root=base_root,
            output_root=output_root,
        )
        counts = _counts(revised)
        (stage / "validation_summary.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provenance = {
            "schema": SCHEMA,
            "created_at": datetime.now().astimezone().isoformat(),
            "command": [sys.executable, *sys.argv],
            "output_root": str(output_root),
            "base_release": {
                "path": str(base_root),
                "SHA256SUMS_sha256": _sha256_file(base_root / "SHA256SUMS"),
                "verified_entries": len(inputs["checksums"]),
            },
            "label_adjudication": {
                "count": EXPECTED_TARGETS,
                "adjudicated_by": ADJUDICATED_BY,
                "adjudicated_on": ADJUDICATED_ON,
                "explicit_user_instruction": USER_INSTRUCTION,
                "new_pcr": int(NEW_PCR),
                "new_status": NEW_STATUS,
                "new_authority": NEW_AUTHORITY,
                "new_confidence": NEW_CONFIDENCE,
                "new_is_provisional": False,
                "pcr_endpoint_definition": PCR_DEFINITION,
                "audit_table": "pcr_label_adjudications.csv",
                "direct_packet_review_by_adjudicator": False,
            },
            "evidence_reconstruction": inputs["evidence_verification"],
            "change_scope": {
                "cohort_membership": "unchanged from v3",
                "image_links": "unchanged from v3",
                "fold_assignments": "unchanged from v3",
                "labels": "exactly two retained exams changed from pCR 1 to pCR 0",
                "inherited_metadata_repair": (
                    "one previously reconciled pCR-positive row changed only from "
                    "pcr_label_status not_supported to supported"
                ),
                "other_numeric_pcr_labels": "unchanged from v3",
            },
            "counts": counts,
            "validation": validation,
            "frozen_builder_sha256": _sha256_file(frozen_builder),
            "vanguard_git": _git_state(arguments.repo_root.expanduser().resolve()),
        }
        (stage / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "README.md").write_text(_readme(counts), encoding="utf-8")
        _write_checksums(stage)
        _set_release_permissions(stage)
        stage.rename(output_root)
    except Exception:
        # Preserve a protected failed stage for diagnosis; never publish partial data.
        raise
    return {"output_root": str(output_root), "counts": counts, "validation": validation}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "publish"))
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def main() -> None:
    """Run the read-only preflight or publish the versioned release."""
    arguments = parse_arguments()
    if arguments.action == "check":
        inputs = _preflight(arguments)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "base_checksum_entries": len(inputs["checksums"]),
                    "label_adjudications": len(inputs["adjudications"]),
                    "inherited_pcr_status_metadata_repairs": len(
                        inputs["metadata_repairs"]
                    ),
                    "source_projection_fingerprint_matches": inputs[
                        "evidence_verification"
                    ]["fingerprint_matches"],
                },
                sort_keys=True,
            )
        )
        return
    result = _publish(arguments)
    counts = result["counts"]["sarit_compatible_main"]
    print(
        "[release] complete "
        f"output={result['output_root']} "
        f"main={counts['exams']} "
        f"pcr_0={counts['pcr_0']} "
        f"pcr_1={counts['pcr_1']} "
        f"adjudications={EXPECTED_TARGETS}",
        flush=True,
    )


if __name__ == "__main__":
    main()
