"""Stage reviewed high-resolution DICOM series beside the UFAST cohort.

The source archives are opened read-only. DICOM payloads are copied byte for
byte into one restricted, restartable ZIP per exam; filenames and shared
inventory columns intentionally omit patient identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pandas as pd

INVENTORY_COLUMNS = [
    "study_instance_uid",
    "series_instance_uid",
    "archive_path",
    "archive_member",
    "read_ok",
    "temporal_position_identifier",
    "instance_number",
    "sop_instance_uid",
    "file_size_bytes",
]
SHARED_INVENTORY_COLUMNS = [
    "exam_id",
    "dataset",
    "series_role",
    "study_instance_uid",
    "series_instance_uid",
    "archive_path",
    "archive_member",
    "read_ok",
    "temporal_position_identifier",
    "instance_number",
    "sop_instance_uid",
    "file_size_bytes",
]


def _safe_component(value: str, *, field: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field} must be one safe path component, got {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "exam_id",
        "dataset",
        "study_instance_uid",
        "series_instance_uid",
        "series_role",
        "selection_status",
        "source_inventory",
        "expected_n_instances",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"selection manifest missing columns: {sorted(missing)}")
    if frame[list(required - {"expected_n_instances"})].eq("").any().any():
        raise ValueError("selection manifest identifiers cannot be empty")
    for column in ("exam_id", "dataset", "series_instance_uid"):
        frame[column] = [
            _safe_component(str(value), field=column) for value in frame[column]
        ]
    frame["expected_n_instances"] = frame["expected_n_instances"].astype(int)
    duplicated = frame.duplicated(
        ["exam_id", "series_instance_uid"], keep=False
    )
    if duplicated.any():
        raise ValueError("selection manifest repeats an exam/series pair")
    return frame


def _inventory_rows(selection: pd.DataFrame) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for source, source_selection in selection.groupby("source_inventory", sort=False):
        series_uids = source_selection["series_instance_uid"].tolist()
        inventory = pd.read_parquet(
            source,
            columns=INVENTORY_COLUMNS,
            filters=[("series_instance_uid", "in", series_uids)],
        )
        inventory = inventory.loc[inventory["read_ok"].fillna(False)].copy()
        chunks.append(inventory)
    rows = pd.concat(chunks, ignore_index=True)
    roles = selection[["series_instance_uid", "series_role"]]
    rows = rows.merge(roles, on="series_instance_uid", validate="many_to_one")
    expected_study = set(selection["study_instance_uid"])
    if set(rows["study_instance_uid"].astype(str)) != expected_study:
        raise ValueError("source inventory study UID does not match selection")
    counts = rows.groupby("series_instance_uid").size().to_dict()
    for record in selection.itertuples(index=False):
        actual = int(counts.get(record.series_instance_uid, 0))
        if actual != record.expected_n_instances:
            raise ValueError(
                f"{record.series_instance_uid}: expected "
                f"{record.expected_n_instances} files, found {actual}"
            )
    if rows["archive_member"].isna().any() or rows["archive_path"].isna().any():
        raise ValueError("selected inventory rows require ZIP archive members")
    return rows


def _sort_rows(rows: pd.DataFrame) -> pd.DataFrame:
    sortable = rows.assign(
        _temporal=pd.to_numeric(
            rows["temporal_position_identifier"], errors="coerce"
        ).fillna(-1),
        _instance=pd.to_numeric(rows["instance_number"], errors="coerce").fillna(-1),
    )
    return sortable.sort_values(
        ["series_role", "series_instance_uid", "_temporal", "_instance", "archive_member"]
    ).drop(columns=["_temporal", "_instance"])


def stage_exam(selection_path: Path, destination: Path, index: int) -> None:
    """Stage one exam selected by its stable sorted array index."""
    selection = _selection(selection_path)
    exam_ids = sorted(selection["exam_id"].unique())
    if index < 0 or index >= len(exam_ids):
        raise IndexError(f"array index {index} outside [0, {len(exam_ids) - 1}]")
    exam_id = exam_ids[index]
    selected = selection.loc[selection["exam_id"] == exam_id].copy()
    if selected["dataset"].drop_duplicates().shape[0] != 1:
        raise ValueError(f"{exam_id}: expected exactly one dataset")

    dataset = selected["dataset"].iloc[0]
    archive = destination / "archives" / dataset / f"{exam_id}.zip"
    shard = destination / "inventory_shards" / dataset / f"{exam_id}.parquet"
    provenance = destination / "provenance_shards" / dataset / f"{exam_id}.json"
    for parent in (archive.parent, shard.parent, provenance.parent):
        parent.mkdir(parents=True, exist_ok=True)

    if archive.exists() and shard.exists() and provenance.exists():
        print(f"[skip] complete outputs already exist for {exam_id}", flush=True)
        return

    rows = _sort_rows(_inventory_rows(selected))
    temp_archive = archive.with_suffix(f".zip.tmp.{os.getpid()}")
    temp_shard = shard.with_suffix(f".parquet.tmp.{os.getpid()}")
    temp_provenance = provenance.with_suffix(f".json.tmp.{os.getpid()}")
    for temporary in (temp_archive, temp_shard, temp_provenance):
        temporary.unlink(missing_ok=True)

    source_archives: dict[str, zipfile.ZipFile] = {}
    output_rows: list[dict[str, object]] = []
    payload_digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(
            temp_archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as output_zip:
            counters: dict[str, int] = {}
            for row in rows.itertuples(index=False):
                source_path = str(row.archive_path)
                if source_path not in source_archives:
                    source_archives[source_path] = zipfile.ZipFile(source_path, mode="r")
                payload = source_archives[source_path].read(str(row.archive_member))
                if len(payload) != int(row.file_size_bytes):
                    raise ValueError(
                        f"source size mismatch for {row.series_instance_uid}/"
                        f"{row.archive_member}"
                    )
                counter = counters.get(row.series_instance_uid, 0)
                counters[row.series_instance_uid] = counter + 1
                member = f"series/{row.series_instance_uid}/{counter:06d}.dcm"
                output_zip.writestr(member, payload)
                payload_digest.update(row.series_instance_uid.encode())
                payload_digest.update(member.encode())
                payload_digest.update(payload)
                output_rows.append(
                    {
                        "exam_id": exam_id,
                        "dataset": dataset,
                        "series_role": row.series_role,
                        "study_instance_uid": str(row.study_instance_uid),
                        "series_instance_uid": str(row.series_instance_uid),
                        "archive_path": str(archive.resolve()),
                        "archive_member": member,
                        "read_ok": True,
                        "temporal_position_identifier": row.temporal_position_identifier,
                        "instance_number": row.instance_number,
                        "sop_instance_uid": str(row.sop_instance_uid),
                        "file_size_bytes": len(payload),
                    }
                )
    finally:
        for source_zip in source_archives.values():
            source_zip.close()

    output = pd.DataFrame(output_rows, columns=SHARED_INVENTORY_COLUMNS)
    output.to_parquet(temp_shard, index=False)
    metadata = {
        "exam_id": exam_id,
        "dataset": dataset,
        "selection_status": sorted(set(selected["selection_status"])),
        "series_instance_uids": selected["series_instance_uid"].tolist(),
        "series_roles": selected["series_role"].tolist(),
        "n_files": len(output),
        "payload_sha256": payload_digest.hexdigest(),
        "archive_sha256": _sha256(temp_archive),
        "archive_bytes": temp_archive.stat().st_size,
    }
    temp_provenance.write_text(json.dumps(metadata, indent=2) + "\n")
    temp_archive.chmod(0o640)
    temp_shard.chmod(0o640)
    temp_provenance.chmod(0o640)
    temp_archive.replace(archive)
    temp_shard.replace(shard)
    temp_provenance.replace(provenance)
    print(
        f"[complete] {exam_id}: {len(output)} files, "
        f"{archive.stat().st_size / 1024**3:.2f} GiB",
        flush=True,
    )


def finalize(selection_path: Path, destination: Path, ufast_manifest: Path) -> None:
    """Merge completed shards and create HR-linked cohort manifests."""
    selection = _selection(selection_path)
    exam_ids = sorted(selection["exam_id"].unique())
    shard_paths = [
        destination / "inventory_shards" / dataset / f"{exam_id}.parquet"
        for exam_id, dataset in (
            selection[["exam_id", "dataset"]].drop_duplicates().itertuples(index=False)
        )
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} inventory shards: {missing[:5]}")
    inventory = pd.concat(
        [pd.read_parquet(path) for path in shard_paths], ignore_index=True
    )
    expected_files = int(selection["expected_n_instances"].sum())
    if len(inventory) != expected_files:
        raise ValueError(f"expected {expected_files} inventory rows, got {len(inventory)}")
    if inventory["exam_id"].nunique() != len(exam_ids):
        raise ValueError("merged inventory does not cover every selected exam")

    combined_path = destination / "dicom_file_manifest.parquet"
    temp_combined = combined_path.with_suffix(".parquet.tmp")
    inventory.to_parquet(temp_combined, index=False)
    temp_combined.chmod(0o640)
    temp_combined.replace(combined_path)

    grouped_rows: list[dict[str, object]] = []
    for exam_id, group in selection.groupby("exam_id", sort=True):
        dataset = group["dataset"].iloc[0]
        statuses = sorted(set(group["selection_status"]))
        archive = destination / "archives" / dataset / f"{exam_id}.zip"
        paired_ufast_uid = ""
        if "paired_ufast_series_instance_uid" in group:
            ufast_uids = sorted(
                set(group["paired_ufast_series_instance_uid"].dropna()) - {""}
            )
            if len(ufast_uids) > 1:
                raise ValueError(f"{exam_id}: multiple paired UFAST series UIDs")
            if ufast_uids:
                paired_ufast_uid = ufast_uids[0]
        grouped_rows.append(
            {
                "exam_id": exam_id,
                "dataset": dataset,
                "study_instance_uid": group["study_instance_uid"].iloc[0],
                "hr_series_instance_uids": "|".join(group["series_instance_uid"]),
                "hr_series_roles": "|".join(group["series_role"]),
                "hr_selection_status": "|".join(statuses),
                "ufast_series_instance_uid": paired_ufast_uid,
                "hr_source_archive_path": str(archive.resolve()),
                "hr_inventory_path": str(combined_path.resolve()),
            }
        )
    cases = pd.DataFrame(grouped_rows)
    cases_path = destination / "high_resolution_case_manifest.csv"
    cases.to_csv(cases_path, index=False)
    cases_path.chmod(0o640)

    cohort = pd.read_csv(ufast_manifest, dtype=str, keep_default_na=False)
    hr_columns = cases.drop(columns="study_instance_uid")
    enriched = cohort.merge(
        hr_columns, on=["exam_id", "dataset"], how="left", validate="one_to_one"
    )
    if enriched["hr_source_archive_path"].eq("").any() or enriched[
        "hr_source_archive_path"
    ].isna().any():
        raise ValueError("HR selection does not cover the complete UFAST cohort")
    enriched_path = ufast_manifest.with_name(
        "dce2d_internal_ultrafast_with_high_resolution_manifest.csv"
    )
    enriched.to_csv(enriched_path, index=False)
    enriched_path.chmod(0o640)

    checksum_path = destination / "SHA256SUMS"
    with checksum_path.open("w") as stream:
        for path in sorted(destination.glob("provenance_shards/*/*.json")):
            metadata = json.loads(path.read_text())
            archive = destination / "archives" / metadata["dataset"] / f"{metadata['exam_id']}.zip"
            stream.write(f"{metadata['archive_sha256']}  {archive.relative_to(destination)}\n")
    checksum_path.chmod(0o640)

    readme_path = destination / "README.md"
    readme_path.write_text(
        "# High-resolution DCE source data\n\n"
        "This directory contains the reviewed native high-resolution DCE series "
        "associated with the 181-exam UFAST manifest. DICOM payloads were copied "
        "byte-for-byte from read-only source ZIPs and recompressed into one ZIP "
        "per exam. The archive layout and shared manifests omit source filenames "
        "and patient columns, but the DICOM payloads are **not deidentified**; keep "
        "this data within the restricted Karczmar-lab share.\n\n"
        "- `dicom_file_manifest.parquet`: minimal ZIP-backed inventory accepted by "
        "Vanguard's DICOM loader.\n"
        "- `high_resolution_case_manifest.csv`: one HR association row per UFAST "
        "exam.\n"
        "- `selection_manifest.csv`: reviewed series-level selection and source "
        "provenance.\n"
        "- `SHA256SUMS`: archive checksums relative to this directory.\n"
        "- `archives/<dataset>/<exam_id>.zip`: selected source DICOM payloads.\n\n"
        "The original `dce2d_internal_ultrafast_manifest.csv` was left unchanged. "
        "Use the adjacent "
        "`dce2d_internal_ultrafast_with_high_resolution_manifest.csv` for the "
        "one-to-one HR/UFAST linkage. Of 181 exams, 179 have one complete native "
        "HR series. One HITS exam has no distinct HR acquisition and is marked "
        "`shared_hr_ufast_no_distinct_hr`. One Siemens exam stores pre/post phases "
        "as separate series; all seven phases plus its static high-resolution "
        "series are retained and marked `split_series_not_runnable`. Do not treat "
        "its legacy 84 exported images as 84 physical UFAST phases.\n"
    )
    readme_path.chmod(0o640)
    print(
        f"[complete] merged {len(inventory)} files across {len(exam_ids)} exams",
        flush=True,
    )


def main() -> None:
    """Run one array task or finalize all completed tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "finalize"))
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--ufast-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "stage":
        index = args.index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        stage_exam(args.selection, args.destination, index)
    else:
        if args.ufast_manifest is None:
            parser.error("finalize requires --ufast-manifest")
        finalize(args.selection, args.destination, args.ufast_manifest)


if __name__ == "__main__":
    main()
