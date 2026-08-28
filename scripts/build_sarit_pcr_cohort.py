#!/usr/bin/env python
"""Build Sarit's versioned UChicago + retro-CAPS pretreatment pCR cohort.

The consumer manifest is intentionally stricter than the source-eligible cohort:
it contains only one-row-per-source-patient exams with a binary pCR label, a
validated ultrafast/high-resolution pair representable by Sarit's legacy
six-column pair contract, and an existing raw-signal ultrafast phase export with
its native clock. Source-eligible cases awaiting phase export or split-series HR
intensity scaling remain visible in typed accounting tables.

No image payload is copied or opened. The release links existing phase directories
and validates file presence plus the small native-time arrays only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SCHEMA = "vanguard.sarit_pcr_pretreatment_cohort.v2"
ACCOUNTING_SCHEMA = "vanguard.sarit_pcr_pretreatment_cohort.accounting.v2"
FOLD_SCHEMA = "vanguard.sarit_pcr_pretreatment_cohort.folds.v1"
# Used only when a previous release's fold assignments are supplied. The retained-fold semantics are
# genuinely different from v1 — a patient's fold is now a property of the first release that assigned
# it, not of the current selection — so the table says so rather than reusing the v1 name.
FOLD_SCHEMA_RETAINED = "vanguard.sarit_pcr_pretreatment_cohort.folds.v2"
IDENTITY_SCHEMA = "vanguard.sarit_pcr_pretreatment_cohort.identity_audit.v1"
N_FOLDS = 5
MIN_TIMEPOINTS = 2
POLICY_BASELINE_FRAMES = 5
SINGLE_SERIES = "single_series"
SPLIT_SERIES = "split_precontrast_postcontrast_pair"
LABEL_MAP = {"not_supported": 0, "supported": 1}

PAIR_COLUMNS = [
    "exam_id",
    "dataset",
    "study_instance_uid",
    "hr_series_instance_uid",
    "ufast_series_instance_uid",
    "ufast_baseline_frame_count",
]

EXTRA_MANIFEST_COLUMNS = [
    "cohort_component",
    "source_readiness",
    "included_in_sarit_manifest",
    "phase_export_status",
    "pcr_label_status",
    "pcr_label_authority",
    "pcr_label_confidence",
    "pcr_label_is_provisional",
    "pair_gate_schema",
    "pair_gate_source",
    "ultrafast_series_instance_uid",
    "high_resolution_series_instance_uid",
    "high_resolution_precontrast_series_instance_uid",
    "high_resolution_partner_layout",
    "high_resolution_baseline_frame_count",
    "high_resolution_timed_phases",
    "requires_cross_series_scaling",
    "tumor_bearing_status",
    "tumor_bearing_basis",
    "patient_deduplication_policy",
    "patient_identity_scope",
]

SNAPSHOT_ARGUMENTS = {
    "reference_manifest": "reference_v1_manifest.csv",
    "canonical_manifest": "canonical_137_manifest.csv",
    "legacy_pair_manifest": "canonical_legacy_pair_manifest.csv",
    "zhen_pair_manifest": "canonical_zhen_pair_manifest.csv",
    "zhen_staging_pair_manifest": "canonical_zhen_staging_pair_manifest.csv",
    "retro_gate": "retro_gate_readiness.csv",
    "retro_gate_summary": "retro_gate_readiness_summary.json",
    "retro_gate_readme": "retro_gate_README.md",
    "retro_inventory_exams": "retro_exam_inventory.parquet",
    "retro_inventory_series": "retro_dicom_series_inventory.parquet",
    "retro_metadata": "retro_landed_exam_metadata.csv",
    "retro_metadata_readme": "retro_landed_exam_metadata_README.md",
    "retro_roles": "retro_exam_bucket_assignment.csv",
    "retro_raw_cache_manifest": "retro_raw_signal_cache_manifest.csv",
    "retro_raw_ingest_manifest": "retro_raw_signal_ingest_manifest.csv",
    "cross_delivery_roster": "cross_delivery_acquisition_roster.csv",
}

# Snapshotted like the rest when supplied, but a build is valid without them, so they cannot be
# required inputs. `_input_paths` includes an entry only when its argument was actually given.
OPTIONAL_SNAPSHOT_ARGUMENTS = {
    "previous_fold_assignments": "previous_release_fold_assignments.csv",
}

ALL_SNAPSHOT_ARGUMENTS = {**SNAPSHOT_ARGUMENTS, **OPTIONAL_SNAPSHOT_ARGUMENTS}


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>", "nat"} else text


def _truthy(value: object) -> bool:
    return _clean(value).lower() in {"true", "1", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _token(namespace: str, value: object, length: int = 20) -> str:
    return hashlib.sha256(f"{namespace}|{_clean(value)}".encode()).hexdigest()[:length]


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _normalize_description(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _calendar_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(  # noqa: S603 -- fixed executable and internal arguments
            ("git", *arguments),
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _ensure_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks required columns: {missing}")


def _input_paths(arguments: argparse.Namespace) -> dict[str, Path]:
    paths = {
        name: Path(getattr(arguments, name)).expanduser().resolve()
        for name in SNAPSHOT_ARGUMENTS
    }
    for name in OPTIONAL_SNAPSHOT_ARGUMENTS:
        value = getattr(arguments, name, None)
        if _clean(value):
            paths[name] = Path(value).expanduser().resolve()
    return paths


def _validate_input_paths(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input files: {missing}")


def _canonical_pairs(paths: dict[str, Path], canonical: pd.DataFrame) -> pd.DataFrame:
    frames = [
        _read_csv(paths["legacy_pair_manifest"]),
        _read_csv(paths["zhen_pair_manifest"]),
        _read_csv(paths["zhen_staging_pair_manifest"]),
    ]
    for frame in frames:
        _ensure_columns(frame, set(PAIR_COLUMNS), "canonical pair manifest")
    pairs = pd.concat(frames, ignore_index=True)
    identity = PAIR_COLUMNS[2:]
    conflicts = (
        pairs.groupby("study_instance_uid", dropna=False)[identity]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        raise ValueError("canonical pair inputs disagree for a StudyInstanceUID")
    pairs = pairs.drop_duplicates("study_instance_uid", keep="first")
    pairs = pairs[
        pairs["study_instance_uid"].isin(canonical["study_instance_uid"])
    ].copy()
    if len(pairs) != len(canonical):
        raise ValueError(
            f"canonical pair coverage is {len(pairs)}/{len(canonical)}, expected complete"
        )
    if pairs["study_instance_uid"].duplicated().any():
        raise ValueError("canonical pairs contain duplicate studies")
    return pairs


def _raw_phase_rows(
    paths: dict[str, Path],
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    source_specs = [
        ("released_cache", paths["retro_raw_cache_manifest"]),
        ("ingest_174", paths["retro_raw_ingest_manifest"]),
    ]
    selected: dict[str, dict[str, object]] = {}
    overlap_checks = Counter()
    comparison_columns = [
        "n_phases",
        "times_seconds_json",
        "policy_name",
        "canonical_orientation_policy",
        "source_phase_files_already_canonical",
        "source_phase_spatial_transform",
    ]
    for priority, (source_name, path) in enumerate(source_specs):
        frame = _read_csv(path)
        _ensure_columns(
            frame,
            {
                "exam_id",
                "dataset",
                "study_instance_uid",
                "laterality",
                "n_phases",
                "times_path",
                "times_source",
                "times_seconds_json",
                "phase_files_json",
                "policy_name",
            },
            source_name,
        )
        if frame["study_instance_uid"].duplicated().any():
            raise ValueError(f"{source_name} has duplicate StudyInstanceUID rows")
        for row in frame.to_dict(orient="records"):
            uid = _clean(row["study_instance_uid"])
            record: dict[str, object] = {
                **row,
                "_raw_source_name": source_name,
                "_raw_source_manifest": str(path),
                "_raw_priority": priority,
            }
            if uid in selected:
                overlap_checks["overlapping_studies"] += 1
                prior = selected[uid]
                for column in comparison_columns:
                    if _clean(prior.get(column)) != _clean(record.get(column)):
                        raise ValueError(
                            f"raw phase manifests disagree on {column} for an overlapping study"
                        )
                overlap_checks["metadata_equivalent_overlaps"] += 1
                continue
            selected[uid] = record
    return selected, dict(overlap_checks)


def _join_retro_candidates(
    *,
    paths: dict[str, Path],
    raw_by_study: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    gate = _read_csv(paths["retro_gate"])
    metadata = _read_csv(paths["retro_metadata"])
    roles = _read_csv(paths["retro_roles"])
    inventory = pd.read_parquet(paths["retro_inventory_exams"])
    inventory = inventory.astype(dict.fromkeys(inventory.columns, "string"))
    inventory = inventory.fillna("")
    _ensure_columns(
        gate,
        {
            "study_instance_uid",
            "patient_id_fs",
            "reason",
            "ultrafast_series_instance_uid",
            "high_resolution_series_instance_uid",
            "high_resolution_precontrast_series_instance_uid",
            "high_resolution_partner_layout",
            "requires_cross_series_scaling",
            "high_resolution_baseline_frame_count",
            "high_resolution_timed_phases",
            "already_released_selected_acquisition",
        },
        "retro gate",
    )
    _ensure_columns(
        inventory,
        {"study_instance_uid", "patient_id_fs", "visit_folder"},
        "retro exam inventory",
    )
    _ensure_columns(
        metadata,
        {
            "study_id",
            "exam_relative_path",
            "exam_datetime",
            "exam_description",
            "pcr_label",
            "pcr_label_authority",
            "pcr_confidence",
            "neoadjuvant_actual_start",
            "subtype",
            "er_status",
            "pr_status",
            "her2_status",
        },
        "retro metadata",
    )
    _ensure_columns(
        roles,
        {"study_id", "exam_datetime", "proc_name", "role", "bucket"},
        "retro role assignment",
    )
    if gate["study_instance_uid"].duplicated().any():
        raise ValueError("retro gate contains duplicate StudyInstanceUID rows")
    if inventory["study_instance_uid"].duplicated().any():
        raise ValueError(
            "retro exam inventory contains duplicate StudyInstanceUID rows"
        )
    gate = gate.merge(
        inventory[["study_instance_uid", "patient_id_fs", "visit_folder"]].rename(
            columns={"patient_id_fs": "inventory_patient_id_fs"}
        ),
        on="study_instance_uid",
        how="left",
        validate="one_to_one",
    )
    if gate["visit_folder"].eq("").any():
        raise ValueError("a gate row does not resolve to the frozen exam inventory")
    gate["_exam_relative_path"] = (
        gate["inventory_patient_id_fs"]
        .where(gate["inventory_patient_id_fs"].ne(""), gate["patient_id_fs"])
        .str.strip("/")
        + "/"
        + gate["visit_folder"].str.strip("/")
    )
    metadata["_exam_relative_path"] = metadata["exam_relative_path"].str.strip("/")
    if metadata["_exam_relative_path"].duplicated().any():
        raise ValueError("retro metadata contains duplicate exam-relative paths")
    joined = gate.merge(
        metadata,
        on="_exam_relative_path",
        how="left",
        validate="many_to_one",
        suffixes=("", "_metadata"),
    )
    passed = joined[
        joined["reason"].map(_clean).eq("") & joined["pcr_label"].isin(LABEL_MAP)
    ].copy()
    passed["_calendar_date"] = _calendar_date(passed["exam_datetime"])
    passed["_description_normalized"] = _normalize_description(
        passed["exam_description"]
    )
    roles["_calendar_date"] = _calendar_date(roles["exam_datetime"])
    roles["_description_normalized"] = _normalize_description(roles["proc_name"])
    role_keys = ["study_id", "_calendar_date", "_description_normalized"]
    grouped = roles.groupby(role_keys, dropna=False)
    role_counts = grouped["role"].nunique(dropna=False)
    role_summary = grouped.agg(
        role=(
            "role",
            lambda values: "|".join(sorted({_clean(value) for value in values})),
        ),
        bucket=(
            "bucket",
            lambda values: "|".join(sorted({_clean(value) for value in values})),
        ),
    ).reset_index()
    role_summary["_role_count"] = role_summary.set_index(role_keys).index.map(
        role_counts
    )
    passed = passed.merge(
        role_summary,
        on=role_keys,
        how="left",
        validate="many_to_one",
    )
    if passed["role"].map(_clean).eq("").any():
        raise ValueError(
            "a binary-label gate-pass exam does not resolve to a visit role"
        )
    if passed["_role_count"].fillna(0).astype(int).gt(1).any():
        raise ValueError(
            "a binary-label gate-pass exam resolves to conflicting visit roles"
        )
    pretreatment = passed[passed["role"].eq("pretreatment")].copy()
    if pretreatment["study_id"].map(_clean).eq("").any():
        raise ValueError(
            "a pretreatment candidate lacks its protected project patient alias"
        )
    pretreatment["_phase_ready"] = pretreatment["study_instance_uid"].isin(raw_by_study)
    pretreatment["_single_series"] = pretreatment["high_resolution_partner_layout"].eq(
        SINGLE_SERIES
    )
    pretreatment["_released"] = pretreatment[
        "already_released_selected_acquisition"
    ].map(_truthy)
    pretreatment["_hr_timed_phases"] = pd.to_numeric(
        pretreatment["high_resolution_timed_phases"], errors="coerce"
    ).fillna(-1)
    pretreatment["_tie_break"] = pretreatment["study_instance_uid"].map(
        lambda value: _token("retro-patient-dedup-v1", value, 64)
    )

    duplicate_sizes = pretreatment.groupby("study_id").size()
    duplicate_ids = set(duplicate_sizes[duplicate_sizes.gt(1)].index)
    for _patient, block in pretreatment[
        pretreatment["study_id"].isin(duplicate_ids)
    ].groupby("study_id"):
        if (
            len(set(block["_calendar_date"])) != 1
            or len(set(block["_description_normalized"])) != 1
            or len(set(block["pcr_label"])) != 1
        ):
            raise ValueError(
                "one source patient has multiple non-equivalent pretreatment candidates; "
                "technical deduplication is not valid"
            )
    ordered = pretreatment.sort_values(
        [
            "study_id",
            "_phase_ready",
            "_single_series",
            "_released",
            "_hr_timed_phases",
            "_tie_break",
        ],
        ascending=[True, False, False, False, False, True],
        kind="stable",
    )
    selected = ordered.drop_duplicates("study_id", keep="first").copy()
    chosen_studies = set(selected["study_instance_uid"])
    dropped = ordered[~ordered["study_instance_uid"].isin(chosen_studies)].copy()
    selected["patient_key"] = selected["study_id"].map(
        lambda value: f"retro_{_token('retro-caps-project-alias-v1', value)}"
    )
    selected["exam_id"] = selected["study_instance_uid"].map(
        lambda value: f"retro_caps_{_token('retro-caps-study-v1', value, 24)}"
    )
    patient_key_map = dict(
        zip(selected["study_id"], selected["patient_key"], strict=True)
    )
    dropped["patient_key"] = dropped["study_id"].map(patient_key_map)
    dropped["exam_id"] = dropped["study_instance_uid"].map(
        lambda value: f"retro_caps_{_token('retro-caps-study-v1', value, 24)}"
    )
    dropped = dropped.assign(
        selection_status="duplicate_same_patient_date_description_not_selected",
        reason=(
            "one-per-source-patient technical deduplication preferred an existing phase export, "
            "then a single-series HR partner, a released acquisition, more HR timed phases, "
            "and finally a deterministic StudyInstanceUID hash"
        ),
    )[
        [
            "exam_id",
            "patient_key",
            "study_instance_uid",
            "pcr_label",
            "selection_status",
            "reason",
        ]
    ]
    audit = {
        "gate_rows": int(len(gate)),
        "gate_pass_rows": int(gate["reason"].map(_clean).eq("").sum()),
        "gate_rows_with_binary_label": int(len(passed)),
        "pretreatment_candidate_exams": int(len(pretreatment)),
        "pretreatment_candidate_source_patients": int(
            pretreatment["study_id"].nunique()
        ),
        "duplicate_source_patient_groups": int(len(duplicate_ids)),
        "duplicate_exam_rows_removed": int(len(dropped)),
        "selected_retro_exams": int(len(selected)),
    }
    return selected, dropped, audit


def _prior_folds(prior: pd.DataFrame | None) -> dict[str, int]:
    """Read a released `fold_assignments.csv` into patient_key -> fold.

    The greedy assignment below is order dependent: each patient is placed against the balance as it
    stands at that moment, and the iteration order is a hash of the patient key. Adding patients
    therefore interleaves newcomers among the patients an earlier release already placed, changing the
    balance those patients saw and moving them to different folds. Measured on the v5 assignments,
    extending the cohort re-runs 126 of the 168 additively-assigned patients into a different fold.

    Nothing in the manifest would flag that. A patient silently changing fold between releases
    invalidates every cross-validated result computed against the earlier one, so a fold, once
    released, is treated here as a property of the release that assigned it.
    """

    if prior is None:
        return {}
    _ensure_columns(prior, {"patient_key", "fold"}, "previous fold assignments")
    frame = prior[["patient_key", "fold"]].copy()
    frame["patient_key"] = frame["patient_key"].map(_clean)
    if frame["patient_key"].eq("").any():
        raise ValueError("previous fold assignments contain a blank patient_key")
    if frame["patient_key"].duplicated().any():
        raise ValueError(
            "previous fold assignments name a patient twice, so its released fold is ambiguous"
        )
    folds = frame["fold"].astype(float).astype(int)
    outside = sorted(set(folds[~folds.between(0, N_FOLDS - 1)]))
    if outside:
        raise ValueError(f"previous fold assignments hold folds outside 0..{N_FOLDS - 1}: {outside}")
    return dict(zip(frame["patient_key"], folds))


def _assign_folds(
    canonical: pd.DataFrame,
    retro: pd.DataFrame,
    prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    canonical_assignments = canonical[["patient_key", "pcr", "fold"]].copy()
    canonical_assignments["pcr"] = (
        canonical_assignments["pcr"].astype(float).astype(int)
    )
    canonical_assignments["fold"] = (
        canonical_assignments["fold"].astype(float).astype(int)
    )
    if canonical_assignments["patient_key"].duplicated().any():
        raise ValueError("canonical patient keys are not unique")
    canonical_assignments["assignment_source"] = "canonical_fold_preserved"
    schema = FOLD_SCHEMA if prior is None else FOLD_SCHEMA_RETAINED
    canonical_assignments["schema"] = schema
    prior_folds = _prior_folds(prior)

    # A canonical patient's fold comes from the canonical manifest, and the previous release also
    # recorded one. They describe the same assignment, so disagreement means the two inputs are not
    # the same lineage and neither can be trusted to be the released fold.
    conflicts = sorted(
        str(row.patient_key)
        for row in canonical_assignments.itertuples(index=False)
        if str(row.patient_key) in prior_folds
        and prior_folds[str(row.patient_key)] != int(row.fold)
    )
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} canonical patients have a different fold in the previous release, so "
            f"the canonical manifest and the previous release disagree: {conflicts[:5]}"
        )

    balance = canonical_assignments[["patient_key", "pcr", "fold"]].copy()
    records = canonical_assignments.to_dict(orient="records")
    requested = retro[["patient_key", "pcr"]].drop_duplicates().copy()
    requested["_order"] = requested["patient_key"].map(
        lambda value: _token("sarit-pcr-fold-v2", value, 64)
    )

    # The two arms must name disjoint patients. The original code relied on the duplicate check at the
    # end of this function to catch a collision; with retained folds now short-circuiting part of the
    # loop that check can no longer see every case, so the invariant is asserted directly.
    canonical_keys = set(canonical_assignments["patient_key"].map(_clean))
    collisions = sorted(
        canonical_keys.intersection(requested["patient_key"].map(_clean))
    )
    if collisions:
        raise ValueError(
            f"{len(collisions)} patients appear in both the canonical and retro arms, so one patient "
            f"would receive two fold assignments: {collisions[:5]}"
        )

    # Retained folds are seated before any newcomer is placed, so the greedy pass below sees the whole
    # released cohort as fixed context and only ever chooses folds for patients that have none.
    retained = requested[
        requested["patient_key"].map(lambda value: _clean(value) in prior_folds)
    ]
    for row in retained.sort_values("_order").itertuples(index=False):
        record = {
            "patient_key": row.patient_key,
            "pcr": int(row.pcr),
            "fold": int(prior_folds[_clean(row.patient_key)]),
            "assignment_source": "prior_release_fold_preserved",
            "schema": schema,
        }
        records.append(record)
        balance = pd.concat(
            [
                balance,
                pd.DataFrame(
                    [{key: record[key] for key in ("patient_key", "pcr", "fold")}]
                ),
            ],
            ignore_index=True,
        )

    requested = requested[
        requested["patient_key"].map(lambda value: _clean(value) not in prior_folds)
    ]
    for row in requested.sort_values("_order").itertuples(index=False):
        label_counts = balance.loc[
            balance["pcr"].eq(int(row.pcr)), "fold"
        ].value_counts()
        total_counts = balance["fold"].value_counts()
        fold = min(
            range(N_FOLDS),
            key=lambda candidate: (
                int(label_counts.get(candidate, 0)),
                int(total_counts.get(candidate, 0)),
                candidate,
            ),
        )
        record = {
            "patient_key": row.patient_key,
            "pcr": int(row.pcr),
            "fold": int(fold),
            "assignment_source": "additive_label_balanced_hash_order_v1",
            "schema": schema,
        }
        records.append(record)
        balance = pd.concat(
            [
                balance,
                pd.DataFrame(
                    [{key: record[key] for key in ("patient_key", "pcr", "fold")}]
                ),
            ],
            ignore_index=True,
        )
    assignments = pd.DataFrame(records)
    if assignments["patient_key"].duplicated().any():
        raise ValueError("fold assignments contain duplicate source patient keys")
    return assignments.sort_values("patient_key").reset_index(drop=True)


def _identity_audit(
    *,
    paths: dict[str, Path],
    canonical: pd.DataFrame,
    canonical_pairs: pd.DataFrame,
    retro: pd.DataFrame,
) -> dict[str, object]:
    roster = _read_csv(paths["cross_delivery_roster"])
    _ensure_columns(
        roster,
        {"series_instance_uid", "delivery_patient_key", "cohort", "patient_link_group"},
        "cross-delivery roster",
    )
    series_inventory = pd.read_parquet(
        paths["retro_inventory_series"],
        columns=["series_instance_uid", "patient_id_fs"],
    ).astype(str)
    signal = roster[roster["cohort"].eq("signal_enhancement_released_cache")].copy()
    series_to_group = dict(
        zip(signal["series_instance_uid"], signal["patient_link_group"], strict=False)
    )
    canonical_link = canonical[["study_instance_uid", "patient_key"]].merge(
        canonical_pairs[["study_instance_uid", "ufast_series_instance_uid"]],
        on="study_instance_uid",
        how="left",
        validate="one_to_one",
    )
    canonical_link["_patient_link_group"] = canonical_link[
        "ufast_series_instance_uid"
    ].map(series_to_group)
    if canonical_link["_patient_link_group"].isna().any():
        raise ValueError(
            "not every canonical UFAST acquisition resolves to the identity roster"
        )
    shared = series_inventory.merge(
        signal[["series_instance_uid", "patient_link_group"]],
        on="series_instance_uid",
        how="inner",
        validate="many_to_many",
    )
    options = (
        shared.groupby("patient_id_fs")["patient_link_group"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    ambiguous = sum(len(values) > 1 for values in options.values())
    if ambiguous:
        raise ValueError(
            "a current retro patient alias links to multiple released-cache patients"
        )
    retro_groups = retro["patient_id_fs"].map(options)
    retro_groups = retro_groups.map(
        lambda values: values[0]
        if isinstance(values, list) and len(values) == 1
        else ""
    )
    canonical_groups = set(canonical_link["_patient_link_group"])
    mapped_retro_groups = set(retro_groups[retro_groups.ne("")])
    overlap = canonical_groups & mapped_retro_groups
    exact_studies = set(canonical["study_instance_uid"]) & set(
        retro["study_instance_uid"]
    )
    exact_ufast = set(canonical_pairs["ufast_series_instance_uid"]) & set(
        retro["ultrafast_series_instance_uid"]
    )
    if exact_studies or exact_ufast or overlap:
        raise ValueError(
            "the canonical and retro components overlap by a known study, UFAST acquisition, "
            "or cross-delivery patient link group"
        )
    return {
        "schema": IDENTITY_SCHEMA,
        "exact_study_instance_uid_overlap": 0,
        "exact_selected_ufast_series_instance_uid_overlap": 0,
        "canonical_source_patients_resolved_to_cross_delivery_roster": int(
            canonical_link["_patient_link_group"].notna().sum()
        ),
        "retro_source_patients_with_any_shared_acquisition_to_released_cache": int(
            retro_groups.ne("").sum()
        ),
        "retro_source_patients_without_cross_delivery_physical_identity_evidence": int(
            retro_groups.eq("").sum()
        ),
        "known_cross_delivery_patient_link_groups_overlapping_components": 0,
        "combined_patient_count_unit": "source patient identities",
        "fold_exclusivity_scope": "patient-exclusive within source identity namespaces",
        "limitation": (
            "CAPS and the released UChicago delivery were de-identified independently. "
            "Shared acquisition UIDs prove identity when present, but their absence cannot prove "
            "different physical patients; unresolved retro source identities therefore require "
            "a protected crosswalk before final pooled-CV claims."
        ),
        "cross_delivery_roster_path": str(paths["cross_delivery_roster"]),
        "cross_delivery_roster_sha256": _sha256(paths["cross_delivery_roster"]),
    }


def _timing_fields(times: list[float]) -> dict[str, object]:
    values = np.asarray(times, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < MIN_TIMEPOINTS
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "native ultrafast times must be a finite one-dimensional vector"
        )
    if not math.isclose(float(values[0]), 0.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("native ultrafast time must start at zero")
    diffs = np.diff(values)
    if not (diffs > 0).all():
        raise ValueError("native ultrafast times must be strictly increasing")
    return {
        "times_seconds_json": _json([float(value) for value in values]),
        "acquisition_times": _json([float(value) for value in values]),
        "ultrafast_median_dt_sec": float(np.median(diffs)),
        "ultrafast_min_positive_dt_sec": float(np.min(diffs)),
        "ultrafast_max_positive_dt_sec": float(np.max(diffs)),
    }


def _rewrite_ready_paths(
    *,
    row: dict[str, object],
    output_root: Path,
    dataset: str,
    exam_id: str,
    source_case_dir: Path,
    phase_paths: list[Path],
    times_path: Path,
) -> dict[str, object]:
    destination = output_root / "images" / dataset / exam_id
    rewritten_phases = []
    for path in phase_paths:
        try:
            relative = path.relative_to(source_case_dir)
        except ValueError as error:
            raise ValueError(
                "a phase file is not inside its declared preprocessing directory"
            ) from error
        rewritten_phases.append(str(destination / relative))
    try:
        rewritten_times = destination / times_path.relative_to(source_case_dir)
    except ValueError as error:
        raise ValueError("the time file is not inside its phase directory") from error
    row.update(
        {
            "phase_files": _json(rewritten_phases),
            "times_path": str(rewritten_times),
            "preproc_root": str(output_root / "images"),
            "preproc_exam_dir": str(destination),
        }
    )
    return row


def _build_tables(
    *,
    arguments: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, object]:
    reference = _read_csv(paths["reference_manifest"])
    canonical = _read_csv(paths["canonical_manifest"])
    if list(reference.columns[:37]) != list(canonical.columns[:37]):
        raise ValueError(
            "the canonical successor no longer preserves Sarit's first 37 columns"
        )
    if len(canonical) != arguments.expected_canonical:
        raise ValueError(
            f"canonical count changed: {len(canonical)} != {arguments.expected_canonical}"
        )
    if (
        canonical["study_instance_uid"].duplicated().any()
        or canonical["patient_key"].duplicated().any()
    ):
        raise ValueError("canonical manifest is not one unique exam per source patient")
    canonical_pairs = _canonical_pairs(paths, canonical)
    raw_by_study, raw_overlap_audit = _raw_phase_rows(paths)
    retro, dedup_exclusions, retro_join_audit = _join_retro_candidates(
        paths=paths, raw_by_study=raw_by_study
    )
    if len(retro) != arguments.expected_retro:
        raise ValueError(
            f"retro count changed: {len(retro)} != {arguments.expected_retro}"
        )
    retro["pcr"] = retro["pcr_label"].map(LABEL_MAP).astype(int)
    prior_folds_frame = (
        _read_csv(paths["previous_fold_assignments"])
        if "previous_fold_assignments" in paths
        else None
    )
    folds = _assign_folds(canonical, retro, prior=prior_folds_frame)
    print(
        "fold assignment: "
        + ", ".join(
            f"{source} {count}"
            for source, count in folds["assignment_source"]
            .value_counts()
            .sort_index()
            .items()
        ),
        file=sys.stderr,
    )
    fold_index = folds.set_index("patient_key")
    retro["fold"] = retro["patient_key"].map(fold_index["fold"]).astype(int)
    identity_audit = _identity_audit(
        paths=paths,
        canonical=canonical,
        canonical_pairs=canonical_pairs,
        retro=retro,
    )

    manifest_columns = list(canonical.columns)
    manifest_columns.extend(
        column for column in EXTRA_MANIFEST_COLUMNS if column not in manifest_columns
    )
    pair_index = canonical_pairs.set_index("study_instance_uid")
    source_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, str]] = []
    paired_source_rows: list[dict[str, object]] = []
    roles_sha256 = _sha256(paths["retro_roles"])

    for source in canonical.to_dict(orient="records"):
        row = {column: source.get(column, "") for column in manifest_columns}
        exam_id = _clean(source["exam_id"])
        dataset = _clean(source["dataset"])
        source_case = Path(_clean(source["preproc_exam_dir"]))
        phases = [Path(value) for value in json.loads(_clean(source["phase_files"]))]
        times_path = Path(_clean(source["times_path"]))
        if (
            not source_case.is_dir()
            or not times_path.is_file()
            or not all(path.is_file() for path in phases)
        ):
            raise FileNotFoundError(
                "a canonical ready case is missing phases or its time file"
            )
        row = _rewrite_ready_paths(
            row=row,
            output_root=Path(arguments.output_root).resolve(),
            dataset=dataset,
            exam_id=exam_id,
            source_case_dir=source_case,
            phase_paths=phases,
            times_path=times_path,
        )
        pair = pair_index.loc[_clean(source["study_instance_uid"])]
        pcr = int(float(source["pcr"]))
        row.update(
            {
                "pcr": pcr,
                "fold": int(float(source["fold"])),
                "cohort_component": "uchicago_ultrafast_pretreatment_cohort_v1",
                "source_readiness": "sarit_manifest_runnable",
                "included_in_sarit_manifest": True,
                "phase_export_status": "ready",
                "pcr_label_status": "supported" if pcr == 1 else "not_supported",
                "pcr_label_authority": _clean(source.get("label_source"))
                or "canonical_reviewed_label",
                "pcr_label_confidence": "",
                "pcr_label_is_provisional": False,
                "pair_gate_schema": "canonical_paired_preprocessing_contract",
                "pair_gate_source": str(paths["canonical_manifest"]),
                "ultrafast_series_instance_uid": _clean(
                    pair["ufast_series_instance_uid"]
                ),
                "high_resolution_series_instance_uid": _clean(
                    pair["hr_series_instance_uid"]
                ),
                "high_resolution_precontrast_series_instance_uid": "",
                "high_resolution_partner_layout": SINGLE_SERIES,
                "high_resolution_baseline_frame_count": "",
                "high_resolution_timed_phases": "",
                "requires_cross_series_scaling": False,
                "tumor_bearing_status": True,
                "tumor_bearing_basis": "published_nonempty_pretreatment_tumor_mask",
                "patient_deduplication_policy": "canonical_one_exam_per_patient",
                "patient_identity_scope": "canonical_mrn_hash",
            }
        )
        source_rows.append(row)
        link_rows.append(
            {
                "exam_id": exam_id,
                "dataset": dataset,
                "link_path": str(
                    Path(arguments.output_root).resolve() / "images" / dataset / exam_id
                ),
                "target_path": str(source_case),
                "cohort_component": "canonical_137",
            }
        )
        paired_source_rows.append(
            {
                "exam_id": exam_id,
                "dataset": dataset,
                "study_instance_uid": _clean(source["study_instance_uid"]),
                "hr_series_instance_uid": _clean(pair["hr_series_instance_uid"]),
                "ufast_series_instance_uid": _clean(pair["ufast_series_instance_uid"]),
                "ufast_baseline_frame_count": int(
                    float(pair["ufast_baseline_frame_count"])
                ),
                "hr_precontrast_series_instance_uid": "",
                "hr_baseline_frame_count": "",
                "hr_partner_layout": SINGLE_SERIES,
                "requires_cross_series_scaling": False,
                "paired_preprocessing_status": "six_column_contract_representable",
                "cohort_component": "canonical_137",
            }
        )

    gate_summary = json.loads(paths["retro_gate_summary"].read_text(encoding="utf-8"))
    if gate_summary.get("dirty") is not False:
        raise ValueError(
            "retro gate snapshot was not produced from a clean hfdp worktree"
        )
    gate_schema = _clean(gate_summary.get("schema"))
    series_inventory = pd.read_parquet(paths["retro_inventory_series"])
    series_inventory["series_instance_uid"] = series_inventory[
        "series_instance_uid"
    ].astype(str)
    series_lookup = (
        series_inventory.sort_values("series_instance_uid")
        .drop_duplicates("series_instance_uid")
        .set_index("series_instance_uid")
    )
    for source in retro.to_dict(orient="records"):
        row = dict.fromkeys(manifest_columns, "")
        exam_id = _clean(source["exam_id"])
        dataset = "retro_caps_pcr"
        patient_key = _clean(source["patient_key"])
        study_uid = _clean(source["study_instance_uid"])
        raw = raw_by_study.get(study_uid)
        phase_ready = raw is not None
        layout = _clean(source["high_resolution_partner_layout"])
        if layout not in {SINGLE_SERIES, SPLIT_SERIES}:
            raise ValueError(
                "a selected retro case has an unsupported HR partner layout"
            )
        split = layout == SPLIT_SERIES
        included = phase_ready and not split
        if phase_ready:
            phase_paths = [
                Path(value) for value in json.loads(_clean(raw["phase_files_json"]))
            ]
            times_path = Path(_clean(raw["times_path"]))
            source_case = phase_paths[0].parent
            if (
                len(phase_paths) != int(float(_clean(raw["n_phases"])))
                or len({path.parent for path in phase_paths}) != 1
                or times_path.parent != source_case
                or not times_path.is_file()
                or not all(path.is_file() for path in phase_paths)
            ):
                raise ValueError("a retro raw phase row fails its file/count contract")
            times = [
                float(value) for value in json.loads(_clean(raw["times_seconds_json"]))
            ]
            if len(times) != len(phase_paths):
                raise ValueError(
                    "a retro raw phase row has a phase/time length mismatch"
                )
            timing = _timing_fields(times)
            row.update(timing)
            row.update(
                {
                    "n_phases": len(phase_paths),
                    "n_phases_exported": len(phase_paths),
                    "source_manifest": _clean(raw["_raw_source_manifest"]),
                    "laterality": _clean(raw.get("laterality")),
                    "laterality_exam": _clean(raw.get("laterality")),
                    "policy_name": _clean(raw.get("policy_name")),
                    "times_source": _clean(raw.get("times_source")),
                    "preprocessed_exam_id": exam_id,
                    "preproc_exam_core": exam_id,
                    "derived_output_root": str(source_case.parent),
                }
            )
            row = _rewrite_ready_paths(
                row=row,
                output_root=Path(arguments.output_root).resolve(),
                dataset=dataset,
                exam_id=exam_id,
                source_case_dir=source_case,
                phase_paths=phase_paths,
                times_path=times_path,
            )
            link_rows.append(
                {
                    "exam_id": exam_id,
                    "dataset": dataset,
                    "link_path": str(
                        Path(arguments.output_root).resolve()
                        / "images"
                        / dataset
                        / exam_id
                    ),
                    "target_path": str(source_case),
                    "cohort_component": "retro_caps",
                }
            )
        else:
            row.update(
                {
                    "n_phases": "",
                    "n_phases_exported": "",
                    "phase_files": "[]",
                    "times_path": "",
                    "acquisition_times": "[]",
                    "times_seconds_json": "[]",
                    "source_manifest": str(paths["retro_gate"]),
                    "policy_name": "",
                    "times_source": "",
                }
            )
        ufast_uid = _clean(source["ultrafast_series_instance_uid"])
        manufacturer = ""
        if ufast_uid in series_lookup.index:
            manufacturer = _clean(series_lookup.loc[ufast_uid].get("manufacturer"))
        exam_dt = pd.to_datetime(_clean(source.get("exam_datetime")), errors="coerce")
        treatment_dt = pd.to_datetime(
            _clean(source.get("neoadjuvant_actual_start")), errors="coerce"
        )
        days_before = ""
        if pd.notna(exam_dt) and pd.notna(treatment_dt):
            days_before = float((treatment_dt - exam_dt).total_seconds() / 86400.0)
            if days_before <= 0:
                raise ValueError("a selected pretreatment exam is not before treatment")
        authority = _clean(source["pcr_label_authority"])
        provisional = authority != "anna_manual"
        if split and not _truthy(source["requires_cross_series_scaling"]):
            raise ValueError(
                "a split-series HR partner is not marked for cross-series scaling"
            )
        if not split and _truthy(source["requires_cross_series_scaling"]):
            raise ValueError(
                "a single-series HR partner unexpectedly requires cross-series scaling"
            )
        if included:
            readiness = "sarit_manifest_runnable"
        elif phase_ready:
            readiness = "split_hr_scaling_required"
        elif split:
            readiness = "phase_export_and_split_hr_scaling_required"
        else:
            readiness = "ultrafast_phase_export_pending"
        row.update(
            {
                "exam_id": exam_id,
                "dataset": dataset,
                "patient_id": patient_key,
                "patient_key": patient_key,
                "fold": int(source["fold"]),
                "pcr": int(source["pcr"]),
                "study_instance_uid": study_uid,
                "study_description": _clean(source.get("exam_description")),
                "manufacturer": manufacturer,
                "source_kind": "filesystem_dicom",
                "label_source": authority,
                "image_source": "retro_caps_delivered_dicom",
                "exam_id_core": exam_id,
                "exam_uid_core": "",
                "study_uid_core": "".join(
                    character for character in study_uid if character.isdigit()
                ),
                "source_zip_stem": "",
                "mrn_patient_key": "",
                "mrn_identity_match_sources": "",
                "subtype_role": "retro_caps_regex_provisional",
                "subtype": _clean(source.get("subtype")),
                "subtype_unavailable_reason": (
                    ""
                    if _clean(source.get("subtype"))
                    else "not_available_in_landed_metadata"
                ),
                "er_consensus": _clean(source.get("er_status")),
                "pr_consensus": _clean(source.get("pr_status")),
                "her2_consensus": _clean(source.get("her2_status")),
                "clinical_provenance_role": "retro_caps_landed_exam_metadata_snapshot",
                "post_hoc_only": True,
                "accession_specific_subtype_claim": False,
                "cohort_projection_schema": SCHEMA,
                "pretreatment_status": "role_confirmed_pretreatment",
                "pretreatment_selection_reason": (
                    "patient_alias_calendar_date_normalized_description_transfer_role"
                ),
                "days_before_treatment": days_before,
                "days_treatment_to_surgery": "",
                "timepoint_evidence_sha256": roles_sha256,
                "timepoint_provenance_role": "retro_caps_transfer_bucket_assignment",
                "baseline_selection_used_aif_or_model_quality": False,
                "one_exam_per_patient": True,
                "cohort_source": "retro_caps_pcr_imaging_cohort",
                "visit_role": "pretreatment",
                "study_date": _clean(source.get("_calendar_date")),
                "baseline_dates_json": _json([_clean(source.get("_calendar_date"))]),
                "is_declared_baseline": True,
                "baseline_match_method": (
                    "patient_alias_calendar_date_normalized_description"
                ),
                "source_name": "retro_caps",
                "tumor_laterality": "",
                "histologic_type": "",
                "grade": "",
                "pathology_subtype": "",
                "rcb_class": "",
                "rcb_score": "",
                "centerline_available": False,
                "cohort_build_schema": SCHEMA,
                "cohort_component": "retro_caps_current_transfer",
                "source_readiness": readiness,
                "included_in_sarit_manifest": included,
                "phase_export_status": "ready" if phase_ready else "pending",
                "pcr_label_status": _clean(source["pcr_label"]),
                "pcr_label_authority": authority,
                "pcr_label_confidence": _clean(source["pcr_confidence"]),
                "pcr_label_is_provisional": provisional,
                "pair_gate_schema": gate_schema,
                "pair_gate_source": str(paths["retro_gate"]),
                "ultrafast_series_instance_uid": ufast_uid,
                "high_resolution_series_instance_uid": _clean(
                    source["high_resolution_series_instance_uid"]
                ),
                "high_resolution_precontrast_series_instance_uid": _clean(
                    source["high_resolution_precontrast_series_instance_uid"]
                ),
                "high_resolution_partner_layout": layout,
                "high_resolution_baseline_frame_count": _clean(
                    source["high_resolution_baseline_frame_count"]
                ),
                "high_resolution_timed_phases": _clean(
                    source["high_resolution_timed_phases"]
                ),
                "requires_cross_series_scaling": _truthy(
                    source["requires_cross_series_scaling"]
                ),
                "tumor_bearing_status": True,
                "tumor_bearing_basis": "retro_caps_tumor_bearing_pretreatment_transfer_role",
                "patient_deduplication_policy": (
                    "same_date_description_reexport_dedup_phase_single_released_hrphases_hash_v1"
                ),
                "patient_identity_scope": "retro_caps_project_alias_hash",
            }
        )
        source_rows.append(row)
        paired_source_rows.append(
            {
                "exam_id": exam_id,
                "dataset": dataset,
                "study_instance_uid": study_uid,
                "hr_series_instance_uid": _clean(
                    source["high_resolution_series_instance_uid"]
                ),
                "ufast_series_instance_uid": ufast_uid,
                "ufast_baseline_frame_count": POLICY_BASELINE_FRAMES,
                "hr_precontrast_series_instance_uid": _clean(
                    source["high_resolution_precontrast_series_instance_uid"]
                ),
                "hr_baseline_frame_count": _clean(
                    source["high_resolution_baseline_frame_count"]
                ),
                "hr_partner_layout": layout,
                "requires_cross_series_scaling": split,
                "paired_preprocessing_status": (
                    "six_column_contract_representable"
                    if not split
                    else "split_hr_scaling_required"
                ),
                "cohort_component": "retro_caps",
            }
        )

    source_manifest = pd.DataFrame(source_rows, columns=manifest_columns)
    source_manifest["pcr"] = source_manifest["pcr"].astype(int)
    source_manifest["fold"] = source_manifest["fold"].astype(int)
    source_manifest = source_manifest.sort_values(
        ["cohort_component", "patient_key", "exam_id"], kind="stable"
    ).reset_index(drop=True)
    main_manifest = source_manifest[
        source_manifest["included_in_sarit_manifest"].map(_truthy)
    ].copy()
    if len(main_manifest) != arguments.expected_main:
        raise ValueError(
            f"main count changed: {len(main_manifest)} != {arguments.expected_main}"
        )
    if list(main_manifest.columns[:37]) != list(reference.columns[:37]):
        raise ValueError(
            "the v2 main manifest does not preserve Sarit's first 37 columns"
        )
    if (
        source_manifest["exam_id"].duplicated().any()
        or source_manifest["patient_key"].duplicated().any()
    ):
        raise ValueError(
            "source cohort is not one unique exam per source patient identity"
        )
    if not set(main_manifest["exam_id"]).issubset(set(source_manifest["exam_id"])):
        raise ValueError("main manifest is not a subset of the source cohort")
    if not main_manifest["high_resolution_partner_layout"].eq(SINGLE_SERIES).all():
        raise ValueError("the Sarit-compatible main manifest includes split-series HR")
    if not main_manifest["phase_export_status"].eq("ready").all():
        raise ValueError(
            "the Sarit-compatible main manifest includes a pending phase export"
        )

    paired_source = pd.DataFrame(paired_source_rows)
    if paired_source["study_instance_uid"].duplicated().any():
        raise ValueError("paired source manifest contains duplicate studies")
    paired_legacy = paired_source[
        paired_source["paired_preprocessing_status"].eq(
            "six_column_contract_representable"
        )
    ][PAIR_COLUMNS].copy()
    paired_exclusions = paired_source[
        paired_source["paired_preprocessing_status"].ne(
            "six_column_contract_representable"
        )
    ].assign(
        selection_status="split_series_not_runnable",
        reason=(
            "split precontrast/postcontrast HR pair requires cross-series intensity scaling "
            "and is not representable by Sarit's six-column single-HR-series contract"
        ),
    )[["exam_id", "dataset", "selection_status", "reason"]]
    if set(main_manifest["study_instance_uid"]) - set(
        paired_legacy["study_instance_uid"]
    ):
        raise ValueError(
            "a main-manifest case is absent from the six-column pair contract"
        )

    pending = source_manifest[
        ~source_manifest["included_in_sarit_manifest"].map(_truthy)
    ].copy()
    pending_table = pd.DataFrame(
        {
            "schema": ACCOUNTING_SCHEMA,
            "exam_id": pending["exam_id"],
            "patient_key": pending["patient_key"],
            "dataset": pending["dataset"],
            "pcr": pending["pcr"],
            "fold": pending["fold"],
            "status": pending["source_readiness"],
            "reason": pending["source_readiness"].map(
                {
                    "ultrafast_phase_export_pending": (
                        "verified single-series HR/UFAST source pair; raw-signal UFAST phase "
                        "export has not been published"
                    ),
                    "split_hr_scaling_required": (
                        "raw-signal UFAST phases exist, but split HR halves require explicit "
                        "cross-series intensity scaling"
                    ),
                    "phase_export_and_split_hr_scaling_required": (
                        "raw-signal UFAST phase export and split-HR cross-series scaling are pending"
                    ),
                }
            ),
            "phase_export_status": pending["phase_export_status"],
            "hr_partner_layout": pending["high_resolution_partner_layout"],
            "requires_cross_series_scaling": pending["requires_cross_series_scaling"],
            "pcr_label_authority": pending["pcr_label_authority"],
            "pcr_label_is_provisional": pending["pcr_label_is_provisional"],
            "tumor_bearing_status": pending["tumor_bearing_status"],
        }
    )
    accounting = pd.DataFrame(
        {
            "exam_id": source_manifest["exam_id"],
            "dataset": source_manifest["dataset"],
            "study_instance_uid": source_manifest["study_instance_uid"],
            "patient_key": source_manifest["patient_key"],
            "pcr": source_manifest["pcr"],
            "subtype": source_manifest["subtype"].where(
                source_manifest["subtype"].map(_clean).ne(""), "unavailable"
            ),
            "pretreatment_status": source_manifest["pretreatment_status"],
            "pretreatment_selection_reason": source_manifest[
                "pretreatment_selection_reason"
            ],
            "paired_preprocessing_status": source_manifest["source_readiness"],
            "paired_preprocessing_reason": source_manifest["source_readiness"].where(
                ~source_manifest["source_readiness"].eq("sarit_manifest_runnable"), ""
            ),
            "fold": source_manifest["fold"],
            "cohort_component": source_manifest["cohort_component"],
            "included_in_sarit_manifest": source_manifest["included_in_sarit_manifest"],
            "pcr_label_authority": source_manifest["pcr_label_authority"],
            "pcr_label_is_provisional": source_manifest["pcr_label_is_provisional"],
            "tumor_bearing_status": source_manifest["tumor_bearing_status"],
            "tumor_bearing_basis": source_manifest["tumor_bearing_basis"],
        }
    )

    return {
        "reference_columns": list(reference.columns),
        "canonical": canonical,
        "retro": retro,
        "source_manifest": source_manifest,
        "main_manifest": main_manifest,
        "paired_source": paired_source,
        "paired_legacy": paired_legacy,
        "paired_exclusions": paired_exclusions,
        "pending": pending_table,
        "accounting": accounting,
        "folds": folds,
        "dedup_exclusions": dedup_exclusions,
        "links": pd.DataFrame(link_rows),
        "identity_audit": identity_audit,
        "raw_overlap_audit": raw_overlap_audit,
        "retro_join_audit": retro_join_audit,
        "gate_summary": gate_summary,
    }


def _translate_to_stage(path: Path, output_root: Path, stage: Path) -> Path:
    try:
        return stage / path.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            f"output path is not rooted under the release: {path}"
        ) from error


def _materialize_links(links: pd.DataFrame, *, output_root: Path, stage: Path) -> None:
    for row in links.to_dict(orient="records"):
        target = Path(_clean(row["target_path"]))
        final_link = Path(_clean(row["link_path"]))
        stage_link = _translate_to_stage(final_link, output_root, stage)
        if not target.is_dir():
            raise FileNotFoundError(target)
        stage_link.parent.mkdir(parents=True, exist_ok=True)
        if stage_link.exists() or stage_link.is_symlink():
            raise FileExistsError(stage_link)
        stage_link.symlink_to(target, target_is_directory=True)


def _validate_ready_paths(
    source_manifest: pd.DataFrame, *, output_root: Path, stage: Path
) -> dict[str, int]:
    ready = source_manifest[source_manifest["phase_export_status"].eq("ready")]
    phase_files = 0
    clocks = 0
    for row in ready.to_dict(orient="records"):
        phases = [Path(value) for value in json.loads(_clean(row["phase_files"]))]
        times_json = np.asarray(
            json.loads(_clean(row["times_seconds_json"])), dtype=np.float64
        )
        if len(phases) != int(row["n_phases"]) or len(times_json) != len(phases):
            raise ValueError("a ready row has a phase/time/count mismatch")
        stage_phases = [
            _translate_to_stage(path, output_root, stage) for path in phases
        ]
        if not all(path.is_file() for path in stage_phases):
            raise FileNotFoundError("a linked phase file is missing")
        time_path = _translate_to_stage(
            Path(_clean(row["times_path"])), output_root, stage
        )
        if not time_path.is_file():
            raise FileNotFoundError("a linked native-time file is missing")
        if time_path.suffix == ".npy":
            native_payload = np.load(time_path, allow_pickle=False)
        elif time_path.suffix == ".json":
            native_payload = json.loads(time_path.read_text(encoding="utf-8"))
        else:
            raise ValueError(
                f"unsupported native-time sidecar format: {time_path.suffix}"
            )
        native = np.asarray(native_payload, dtype=np.float64).reshape(-1)
        if native.shape != times_json.shape or not np.allclose(
            native, times_json, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(
                "a linked native-time file differs from its manifest clock"
            )
        _timing_fields([float(value) for value in native])
        phase_files += len(phases)
        clocks += 1
    return {
        "ready_exam_directories": int(len(ready)),
        "linked_phase_files_verified_present": int(phase_files),
        "native_time_arrays_verified": int(clocks),
    }


def _counts(tables: dict[str, object]) -> dict[str, object]:
    source = tables["source_manifest"]
    main = tables["main_manifest"]
    pending = tables["pending"]
    paired_legacy = tables["paired_legacy"]
    paired_exclusions = tables["paired_exclusions"]
    assert isinstance(source, pd.DataFrame)
    assert isinstance(main, pd.DataFrame)
    assert isinstance(pending, pd.DataFrame)
    assert isinstance(paired_legacy, pd.DataFrame)
    assert isinstance(paired_exclusions, pd.DataFrame)

    def frame_counts(frame: pd.DataFrame) -> dict[str, object]:
        return {
            "exams": int(len(frame)),
            "source_patient_identities": int(frame["patient_key"].nunique()),
            "pcr_0": int(frame["pcr"].astype(int).eq(0).sum()),
            "pcr_1": int(frame["pcr"].astype(int).eq(1).sum()),
            "tumor_bearing": int(frame["tumor_bearing_status"].map(_truthy).sum()),
            "provisional_labels": int(
                frame["pcr_label_is_provisional"].map(_truthy).sum()
            ),
        }

    return {
        "source_eligible": frame_counts(source),
        "sarit_compatible_main": frame_counts(main),
        "pending_from_main": int(len(pending)),
        "six_column_pair_rows": int(len(paired_legacy)),
        "split_hr_pair_exclusions": int(len(paired_exclusions)),
        "cohort_components_source": {
            str(key): int(value)
            for key, value in source["cohort_component"]
            .value_counts()
            .sort_index()
            .items()
        },
        "source_readiness": {
            str(key): int(value)
            for key, value in source["source_readiness"]
            .value_counts()
            .sort_index()
            .items()
        },
        "retro_label_authorities": {
            str(key): int(value)
            for key, value in source.loc[
                source["cohort_component"].eq("retro_caps_current_transfer"),
                "pcr_label_authority",
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "full_fold_counts": {
            str(int(key)): int(value)
            for key, value in source["fold"].value_counts().sort_index().items()
        },
        "main_fold_counts": {
            str(int(key)): int(value)
            for key, value in main["fold"].value_counts().sort_index().items()
        },
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.chmod(0o660)


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(0o660)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o660)


def _snapshot_inputs(
    *, paths: dict[str, Path], stage: Path
) -> dict[str, dict[str, str]]:
    snapshot_root = stage / "_build" / "input_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, str]] = {}
    for name, source in paths.items():
        destination = snapshot_root / ALL_SNAPSHOT_ARGUMENTS[name]
        shutil.copy2(source, destination)
        destination.chmod(0o660)
        source_hash = _sha256(source)
        snapshot_hash = _sha256(destination)
        if source_hash != snapshot_hash:
            raise ValueError(f"snapshot hash differs for {name}")
        records[name] = {
            "source_path": str(source),
            "source_sha256": source_hash,
            "snapshot_relative_path": str(destination.relative_to(stage)),
            "snapshot_sha256": snapshot_hash,
        }
    builder_destination = stage / "_build" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), builder_destination)
    builder_destination.chmod(0o660)
    records["builder_script"] = {
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "snapshot_relative_path": str(builder_destination.relative_to(stage)),
        "snapshot_sha256": _sha256(builder_destination),
    }
    return records


def _readme(
    *, output_root: Path, counts: dict[str, object], provenance: dict[str, object]
) -> str:
    source = counts["source_eligible"]
    main = counts["sarit_compatible_main"]
    identity = provenance["identity_audit"]
    return f"""# Sarit pretreatment pCR cohort v2

Use `dce2d_internal_ultrafast_manifest.csv` for Sarit's current pipeline. It preserves the
first 37 columns and their order from the v1 manifest, then carries the canonical successor
columns and explicit readiness/provenance fields. The companion
`paired_preprocessing_case_manifest.csv` keeps the exact legacy six-column schema.

## What changed

Sarit previously used the v1 release with 83 exams (50 non-pCR, 33 pCR); 82 had a runnable
single-HR-series pair. This v2 starts from the canonical 137-exam successor and adds the latest
frozen retro-CAPS delivery/label snapshot.

- source-eligible cohort: {source["exams"]} exams / {source["source_patient_identities"]} source
  patient identities; {source["pcr_0"]} non-pCR and {source["pcr_1"]} pCR
- immediately Sarit-compatible main manifest: {main["exams"]} exams / {main["source_patient_identities"]}
  source patient identities; {main["pcr_0"]} non-pCR and {main["pcr_1"]} pCR
- typed pending rows excluded from the main manifest: {counts["pending_from_main"]}
- exact six-column single-HR-series source pairs: {counts["six_column_pair_rows"]}
- split-HR pairs requiring cross-series scaling: {counts["split_hr_pair_exclusions"]}

Every source-eligible row is clinically tumor-bearing by cohort definition ({source["tumor_bearing"]}/{source["exams"]}).
The canonical 137 have non-empty published tumor masks; retro-CAPS rows use the tumor-bearing
pretreatment transfer role and do not claim an image-derived tumor mask or centerline.

## Files

- `dce2d_internal_ultrafast_manifest.csv`: strict consumer manifest; every row has a binary label,
  one selected pretreatment exam per source identity, an existing native-clock UFAST phase export,
  and a single-series HR/UFAST pair representable by Sarit's current pair contract
- `source_eligible_cohort_manifest.csv`: all source-eligible rows, including typed pending cases;
  do not pass this broader table to the current adapter
- `paired_preprocessing_case_manifest.csv`: exact legacy six-column pair manifest for every
  single-series source pair, including cases queued for phase export
- `paired_source_manifest.csv`: extended all-source pair table; preserves split HR precontrast and
  postcontrast series identities and the scaling requirement
- `pending_unprocessed_pretreatment_sources.csv`: why each source-eligible row is not yet in the
  strict consumer manifest
- `cohort_accounting.csv`, `fold_assignments.csv`, `retro_patient_deduplication_exclusions.csv`:
  cohort, split, and one-per-source-patient accounting
- `identity_overlap_audit.json`, `validation_summary.json`, `provenance.json`: leakage boundary,
  validation, and exact input/code hashes
- `_build/input_snapshots/`: protected frozen source tables used for this release

## Label policy

Canonical labels retain their reviewed source authority. Retro labels use
`anna_manual > oncotrace_glm52 > regex_proxy`; only `anna_manual` is marked non-provisional.
The retro snapshot contains {counts["retro_label_authorities"].get("anna_manual", 0)} manual,
{counts["retro_label_authorities"].get("oncotrace_glm52", 0)} first-draft model, and
{counts["retro_label_authorities"].get("regex_proxy", 0)} regex labels. Audit provisional labels
before treating a final model result as clinical ground truth.

## Dynamic and pair fidelity

No DCE phase was dropped, temporally resampled, clipped, or normalized by this build. Image
directories are symlinked read-only from their existing raw-signal exports. Validation requires
one native timestamp per phase, finite strictly increasing clocks starting at zero, and equality
between each manifest clock and its small `times_path` array. Split HR pairs remain out of the
consumer manifest until their separately reconstructed halves are placed on one intensity scale.

## Patient identity limitation

There is no exact StudyInstanceUID or selected UFAST SeriesInstanceUID overlap between the two
components, and the available shared-acquisition identity roster finds zero known overlapping
patient link groups. However, {identity["retro_source_patients_without_cross_delivery_physical_identity_evidence"]}
of {source["exams"] - 137} retro source identities have no shared acquisition that can bridge the
independently de-identified namespaces. Therefore `{source["source_patient_identities"]}` is a count
of **source patient identities**, not a proven count of distinct physical people. The five folds
are patient-exclusive within those source namespaces; resolve a protected crosswalk before a final
pooled-CV claim.

## Rebuild

The exact producing script and all small input tables are frozen under `_build/`; `provenance.json`
pins their SHA-256 hashes. Rebuild to a new versioned path rather than modifying this release in
place. This release is rooted at `{output_root}`.
"""


def _write_release(
    *,
    arguments: argparse.Namespace,
    paths: dict[str, Path],
    tables: dict[str, object],
) -> dict[str, object]:
    output_root = Path(arguments.output_root).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite versioned output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.building-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o770)
    stage.chmod(0o2770)  # noqa: S103 -- protected lab group, no world access
    try:
        snapshots = _snapshot_inputs(paths=paths, stage=stage)
        links = tables["links"]
        assert isinstance(links, pd.DataFrame)
        _materialize_links(links, output_root=output_root, stage=stage)
        validation = _validate_ready_paths(
            tables["source_manifest"], output_root=output_root, stage=stage
        )
        counts = _counts(tables)
        validation.update(
            {
                "schema": SCHEMA,
                "status": "passed",
                "first_37_columns_match_v1": True,
                "source_manifest_unique_exam_ids": True,
                "source_manifest_unique_source_patient_keys": True,
                "main_manifest_is_source_subset": True,
                "main_manifest_single_hr_series_only": True,
                "main_manifest_binary_labels_only": True,
                "main_manifest_folds": list(range(N_FOLDS)),
                "raw_manifest_overlap_audit": tables["raw_overlap_audit"],
                "retro_join_audit": tables["retro_join_audit"],
            }
        )

        _write_csv(
            tables["main_manifest"], stage / "dce2d_internal_ultrafast_manifest.csv"
        )
        _write_csv(
            tables["source_manifest"], stage / "source_eligible_cohort_manifest.csv"
        )
        _write_csv(
            tables["paired_legacy"], stage / "paired_preprocessing_case_manifest.csv"
        )
        _write_csv(tables["paired_source"], stage / "paired_source_manifest.csv")
        _write_csv(
            tables["paired_exclusions"], stage / "paired_preprocessing_exclusions.csv"
        )
        _write_csv(
            tables["pending"], stage / "pending_unprocessed_pretreatment_sources.csv"
        )
        _write_csv(tables["accounting"], stage / "cohort_accounting.csv")
        _write_csv(tables["folds"], stage / "fold_assignments.csv")
        _write_csv(
            tables["dedup_exclusions"],
            stage / "retro_patient_deduplication_exclusions.csv",
        )
        _write_csv(links, stage / "symlink_manifest.csv")
        _write_json(tables["identity_audit"], stage / "identity_overlap_audit.json")
        _write_json(validation, stage / "validation_summary.json")

        provenance = {
            "schema": SCHEMA,
            "created_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
            "output_root": str(output_root),
            "command": " ".join(sys.argv),
            "vanguard_git_commit": _git("rev-parse", "HEAD"),
            "vanguard_git_dirty": bool(_git("status", "--porcelain")),
            "frozen_builder_sha256": snapshots["builder_script"]["snapshot_sha256"],
            "inputs": snapshots,
            "counts": counts,
            "identity_audit": tables["identity_audit"],
            "selection_contract": {
                "timepoint": "role-confirmed pretreatment",
                "outcome": "binary pCR supported/not_supported",
                "retro_label_priority": [
                    "anna_manual",
                    "oncotrace_glm52",
                    "regex_proxy",
                ],
                "one_exam_per_source_patient": True,
                "main_requires_phase_export": True,
                "main_requires_single_series_hr_pair": True,
                "main_requires_cross_series_scaling": False,
                "ufast_baseline_frame_count": POLICY_BASELINE_FRAMES,
                "temporal_resampling": False,
                "phase_dropping": False,
                "signal_clipping_or_normalization": False,
            },
            "fold_contract": {
                "n_folds": N_FOLDS,
                "canonical_folds_preserved": True,
                "retro_assignment": "additive label-balanced deterministic hash order",
                "patient_identity_scope": (
                    "source-identity exclusive; cross-delivery physical identity unresolved "
                    "where no acquisition is shared"
                ),
            },
            "validation": validation,
        }
        _write_json(provenance, stage / "provenance.json")
        _write_text(
            _readme(output_root=output_root, counts=counts, provenance=provenance),
            stage / "README.md",
        )

        checksum_rows = []
        for path in sorted(stage.rglob("*")):
            if path.is_symlink() or not path.is_file() or path.name == "SHA256SUMS":
                continue
            checksum_rows.append(f"{_sha256(path)}  {path.relative_to(stage)}")
        _write_text("\n".join(checksum_rows) + "\n", stage / "SHA256SUMS")
        for directory in [stage, *[path for path in stage.rglob("*") if path.is_dir()]]:
            if not directory.is_symlink():
                directory.chmod(0o2770)  # noqa: S103 -- protected lab group, no world access
        stage.rename(output_root)
        return {
            "counts": counts,
            "validation": validation,
            "output_root": str(output_root),
        }
    except Exception:
        # This staging directory is owned by this invocation and has never been published.
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/gpfs/data/karczmar-lab/vanguard/"
            "dce2d_internal_ultrafast_pretreatment_cohort_v2"
        ),
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path(
            "/gpfs/data/karczmar-lab/vanguard/"
            "dce2d_internal_ultrafast_pretreatment_cohort_v1/"
            "dce2d_internal_ultrafast_manifest.csv"
        ),
    )
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        default=Path(
            "/gpfs/data/karczmar-lab/vanguard/"
            "uchicago_ultrafast_pretreatment_cohort_v1/"
            "dce2d_internal_ultrafast_manifest.csv"
        ),
    )
    parser.add_argument(
        "--legacy-pair-manifest",
        type=Path,
        default=Path(
            "/gpfs/data/karczmar-lab/vanguard/"
            "dce2d_internal_ultrafast_pretreatment_cohort_v1/"
            "paired_preprocessing_case_manifest.csv"
        ),
    )
    parser.add_argument(
        "--zhen-pair-manifest",
        type=Path,
        default=Path(
            "/gpfs/data/karczmar-lab/vanguard/uchicago_ultrafast_longitudinal_cohort_v1/"
            "_build/zhen_extension/preprocessing_case_manifest.csv"
        ),
    )
    parser.add_argument(
        "--zhen-staging-pair-manifest",
        type=Path,
        default=Path(
            "/gpfs/data/karczmar-lab/vanguard/uchicago_ultrafast_longitudinal_cohort_v1/"
            "_build/zhen_staging_extension/preprocessing_case_manifest.csv"
        ),
    )
    parser.add_argument(
        "--retro-gate",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/"
            "retro_caps_two_series_gate_readiness_resolved_identity/gate_readiness.csv"
        ),
    )
    parser.add_argument(
        "--retro-gate-summary",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/"
            "retro_caps_two_series_gate_readiness_resolved_identity/"
            "gate_readiness_summary.json"
        ),
    )
    parser.add_argument(
        "--retro-gate-readme",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/"
            "retro_caps_two_series_gate_readiness_resolved_identity/README.md"
        ),
    )
    parser.add_argument(
        "--retro-inventory-exams",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/"
            "retro_caps_pcr_imaging_inventory_stability_gated/exam_manifest.parquet"
        ),
    )
    parser.add_argument(
        "--retro-inventory-series",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/"
            "retro_caps_pcr_imaging_inventory_stability_gated/dicom_series_manifest.parquet"
        ),
    )
    parser.add_argument(
        "--retro-metadata",
        type=Path,
        default=Path(
            "/gpfs/data/karczmar-lab/DR_662135_Karczmar_Breast_MRI_ML/"
            "retro_caps_pcr_imaging_cohort/landed_exam_metadata.csv"
        ),
    )
    parser.add_argument(
        "--retro-metadata-readme",
        type=Path,
        default=Path(
            "/gpfs/data/karczmar-lab/DR_662135_Karczmar_Breast_MRI_ML/"
            "retro_caps_pcr_imaging_cohort/landed_exam_metadata_README.md"
        ),
    )
    parser.add_argument(
        "--retro-roles",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/derived_datasets/"
            "retro_caps_pcr_cohort/transfer_buckets/exam_bucket_assignment.csv"
        ),
    )
    parser.add_argument(
        "--retro-raw-cache-manifest",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/derived_datasets/hfdp/"
            "retro_caps_pcr_raw_signal_cache/raw_signal_manifest.csv"
        ),
    )
    parser.add_argument(
        "--retro-raw-ingest-manifest",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/runs/hfdp/"
            "retro_caps_ultrafast_raw_signal_ingest_174/phase_files/"
            "preprocessed_dynamic_t1_manifest.csv"
        ),
    )
    parser.add_argument(
        "--cross-delivery-roster",
        type=Path,
        default=Path(
            "/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/runs/hfdp/"
            "two_series_cohort_roster_full_delivery/acquisitions.csv"
        ),
    )
    parser.add_argument(
        "--previous-fold-assignments",
        type=Path,
        default=None,
        help=(
            "a released fold_assignments.csv whose folds are retained verbatim. Any patient it names "
            "keeps that fold and is recorded as prior_release_fold_preserved; only patients it does "
            "not name are placed by the label-balanced hash-order pass. Omit to reproduce the "
            "pre-v6 behaviour, which re-places every retro patient on every build."
        ),
    )
    parser.add_argument("--expected-canonical", type=int, default=137)
    parser.add_argument("--expected-retro", type=int, default=204)
    parser.add_argument("--expected-main", type=int, default=292)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="load, join, and validate metadata without creating the release",
    )
    return parser


def main() -> int:
    """Build or metadata-validate the requested versioned release."""
    arguments = _parser().parse_args()
    paths = _input_paths(arguments)
    _validate_input_paths(paths)
    tables = _build_tables(arguments=arguments, paths=paths)
    if arguments.plan_only:
        print(
            json.dumps(
                {
                    "status": "plan_validated",
                    "output_root": str(Path(arguments.output_root).resolve()),
                    "counts": _counts(tables),
                    "identity_audit": tables["identity_audit"],
                    "retro_join_audit": tables["retro_join_audit"],
                    "raw_overlap_audit": tables["raw_overlap_audit"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = _write_release(arguments=arguments, paths=paths, tables=tables)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
