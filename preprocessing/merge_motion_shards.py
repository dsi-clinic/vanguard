"""Validate and merge all exam-level motion-correction shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageDraw

from preprocessing.motion import motion_shard_dir
from preprocessing.spgr import ExamRecord, read_manifest, read_member_payload


class _HashingWriter:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return self.stream.write(data)

    def flush(self) -> None:
        self.stream.flush()

    def tell(self) -> int:
        return self.stream.tell()


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = int(size)
    info.mode = 0o640
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _read_source_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"source manifest is empty: {path}")
    return fields, rows


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def _load_metadata(record: ExamRecord, *, output_root: Path) -> dict[str, object]:
    directory = motion_shard_dir(output_root, record)
    metadata_path = directory / "metadata.json"
    shard_path = directory / "phase_images.tar"
    if not metadata_path.is_file() or not shard_path.is_file():
        raise FileNotFoundError(f"missing required shard outputs: {directory}")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("status") != "complete"
        or metadata.get("exam_id") != record.exam_id
        or int(metadata.get("row_index", -1)) != record.row_index
        or int(metadata.get("n_phases", -1)) != record.n_phases
    ):
        raise ValueError(f"shard metadata contract mismatch: {metadata_path}")
    metadata["shard_path"] = str(shard_path)
    return dict(metadata)


def _motion_manifest_row(
    source_row: dict[str, str],
    metadata: dict[str, object],
    *,
    source_manifest: Path,
    output_archive: Path,
) -> dict[str, object]:
    row: dict[str, object] = dict(source_row)
    row["source_phase_archive_path"] = row.pop("phase_archive_path")
    row["source_phase_archive_members_json"] = row.pop("phase_archive_members_json")
    row["source_phase_member_bytes_json"] = row.pop("phase_member_bytes_json")
    row["source_phase_member_sha256_json"] = row.pop("phase_member_sha256_json")
    row["source_phase_archive_sha256"] = row.pop("phase_archive_sha256")
    row["source_motion_correction_applied"] = row.get(
        "motion_correction_applied",
        "false",
    )
    row["phase_archive_path"] = str(output_archive.resolve())
    row["phase_archive_members_json"] = json.dumps(
        metadata["output_phase_archive_members"],
        separators=(",", ":"),
    )
    row["phase_member_bytes_json"] = json.dumps(
        metadata["output_phase_member_bytes"],
        separators=(",", ":"),
    )
    row["phase_member_sha256_json"] = json.dumps(
        metadata["output_phase_member_sha256"],
        separators=(",", ":"),
    )
    row["phase_archive_sha256"] = "PENDING"
    row["motion_correction_applied"] = "true"
    row["motion_correction_method"] = metadata["motion_correction_method"]
    row["motion_correction_reference_phase_index"] = metadata["reference_phase_index"]
    row["motion_correction_source_manifest"] = str(source_manifest.resolve())
    row["motion_transforms_accepted"] = metadata["transforms_accepted"]
    row["motion_transforms_rejected"] = metadata["transforms_rejected"]
    metrics = list(metadata["registration_metrics"])
    row["motion_max_proposed_translation_mm"] = max(
        (float(item["proposed_translation_norm_mm"]) for item in metrics),
        default=0.0,
    )
    row["motion_max_saved_translation_mm"] = max(
        (float(item["translation_norm_mm"]) for item in metrics),
        default=0.0,
    )
    row["motion_qc_panel"] = metadata.get("qc_panel", "")
    return row


def _build_contact_sheet(
    metadata_rows: list[dict[str, object]],
    *,
    output_path: Path,
    selection_csv: Path,
    maximum_panels: int,
) -> None:
    selected = sorted(
        metadata_rows,
        key=lambda item: float(item.get("qc_score", 0.0)),
        reverse=True,
    )[: int(maximum_panels)]
    selection_rows = [
        {
            "rank": rank,
            "exam_id": item["exam_id"],
            "dataset": item["dataset"],
            "qc_score": item.get("qc_score", 0.0),
            "qc_phase_index": item.get("qc_phase_index", 0),
            "qc_panel": item.get("qc_panel", ""),
        }
        for rank, item in enumerate(selected, start=1)
    ]
    _write_csv(
        selection_csv,
        selection_rows,
        ["rank", "exam_id", "dataset", "qc_score", "qc_phase_index", "qc_panel"],
    )
    images: list[tuple[str, Image.Image]] = []
    for row in selection_rows:
        path = Path(str(row["qc_panel"]))
        if not path.is_file():
            raise FileNotFoundError(f"missing QC panel: {path}")
        with Image.open(path) as image:
            copy = image.convert("RGB")
            copy.thumbnail((720, 360), Image.Resampling.LANCZOS)
            images.append((f"#{row['rank']} {row['exam_id']}", copy.copy()))
    if not images:
        raise ValueError("no QC panels available for contact sheet")
    columns = 2
    cell_width = 740
    cell_height = 400
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        x_position = (index % columns) * cell_width
        y_position = (index // columns) * cell_height
        draw.text((x_position + 8, y_position + 6), label, fill=(0, 0, 0))
        sheet.paste(image, (x_position + 8, y_position + 30))
    partial = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    sheet.save(partial, format="PNG")
    partial.replace(output_path)


def merge_motion_shards(
    *,
    manifest_path: Path,
    output_root: Path,
    maximum_qc_panels: int,
) -> dict[str, object]:
    """Merge all expected shards and fail if any semantic artifact is missing."""
    output_root = output_root.resolve()
    output_archive = output_root / "phase_images.tar"
    output_manifest = output_root / "manifest.csv"
    metrics_csv = output_root / "registration_metrics.csv"
    summary_path = output_root / "summary.json"
    contact_sheet = output_root / "motion_qc_contact_sheet.png"
    selection_csv = output_root / "motion_qc_selection.csv"
    final_paths = [
        output_archive,
        output_manifest,
        metrics_csv,
        summary_path,
        contact_sheet,
        selection_csv,
    ]
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite completed outputs: {existing}")

    records = read_manifest(manifest_path)
    source_fields, source_rows = _read_source_rows(manifest_path)
    if len(records) != len(source_rows):
        raise ValueError("validated records and source CSV rows differ")
    metadata_rows = [
        _load_metadata(record, output_root=output_root) for record in records
    ]
    archive_partial = output_archive.with_name(
        f".{output_archive.name}.partial.{os.getpid()}"
    )
    archive_partial.unlink(missing_ok=True)
    with archive_partial.open("wb") as archive_stream:
        hashing_stream = _HashingWriter(archive_stream)
        with tarfile.open(fileobj=hashing_stream, mode="w|") as output_tar:
            for metadata in metadata_rows:
                members = list(metadata["output_phase_archive_members"])
                hashes = list(metadata["output_phase_member_sha256"])
                sizes = list(metadata["output_phase_member_bytes"])
                if not (len(members) == len(hashes) == len(sizes)):
                    raise ValueError(
                        f"shard list lengths differ: {metadata['exam_id']}"
                    )
                with tarfile.open(str(metadata["shard_path"]), mode="r") as shard_tar:
                    for member, checksum, size in zip(
                        members,
                        hashes,
                        sizes,
                        strict=True,
                    ):
                        payload = read_member_payload(
                            shard_tar,
                            str(member),
                            expected_sha256=str(checksum),
                        )
                        if len(payload) != int(size):
                            raise ValueError(f"shard member size mismatch: {member}")
                        output_tar.addfile(
                            _tar_info(str(member), len(payload)),
                            io.BytesIO(payload),
                        )
        hashing_stream.flush()
        os.fsync(archive_stream.fileno())
        archive_sha256 = hashing_stream.digest.hexdigest()

    output_rows = [
        _motion_manifest_row(
            source_row,
            metadata,
            source_manifest=manifest_path,
            output_archive=output_archive,
        )
        for source_row, metadata in zip(source_rows, metadata_rows, strict=True)
    ]
    for row in output_rows:
        row["phase_archive_sha256"] = archive_sha256
    manifest_fields = list(output_rows[0])
    metrics_rows = [
        dict(metric)
        for metadata in metadata_rows
        for metric in list(metadata["registration_metrics"])
    ]
    metrics_fields = list(metrics_rows[0]) if metrics_rows else []
    _write_csv(output_manifest, output_rows, manifest_fields)
    _write_csv(metrics_csv, metrics_rows, metrics_fields)
    _build_contact_sheet(
        metadata_rows,
        output_path=contact_sheet,
        selection_csv=selection_csv,
        maximum_panels=maximum_qc_panels,
    )
    archive_partial.replace(output_archive)

    accepted = sum(int(item["transforms_accepted"]) for item in metadata_rows)
    rejected = sum(int(item["transforms_rejected"]) for item in metadata_rows)
    all_metrics = [
        metric
        for metadata in metadata_rows
        for metric in list(metadata["registration_metrics"])
    ]
    summary: dict[str, object] = {
        "status": "complete",
        "source_manifest": str(manifest_path.resolve()),
        "output_root": str(output_root),
        "motion_corrected_manifest": str(output_manifest),
        "motion_corrected_archive": str(output_archive),
        "motion_corrected_archive_bytes": output_archive.stat().st_size,
        "motion_corrected_archive_sha256": archive_sha256,
        "exams": len(records),
        "phases": sum(record.n_phases for record in records),
        "transforms_accepted": accepted,
        "transforms_rejected": rejected,
        "minimum_saved_correlation_delta": min(
            float(metric["corr_delta"]) for metric in all_metrics
        ),
        "maximum_proposed_translation_mm": max(
            float(metric["proposed_translation_norm_mm"]) for metric in all_metrics
        ),
        "maximum_saved_translation_mm": max(
            float(metric["translation_norm_mm"]) for metric in all_metrics
        ),
        "registration_metrics": str(metrics_csv),
        "motion_qc_contact_sheet": str(contact_sheet),
        "motion_qc_selection": str(selection_csv),
    }
    summary["manifest_sha256"] = hashlib.sha256(
        output_manifest.read_bytes()
    ).hexdigest()
    partial_summary = summary_path.with_name(
        f".{summary_path.name}.partial.{os.getpid()}"
    )
    partial_summary.write_text(json.dumps(summary, indent=2) + "\n")
    partial_summary.replace(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-qc-panels", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    """Merge all shards selected by command-line arguments."""
    args = _parse_args()
    summary = merge_motion_shards(
        manifest_path=args.manifest,
        output_root=args.output_root,
        maximum_qc_panels=args.maximum_qc_panels,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
