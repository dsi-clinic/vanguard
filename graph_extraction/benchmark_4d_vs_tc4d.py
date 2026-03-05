#!/usr/bin/env python3
"""Run per-study 4d-vs-tc4d benchmark jobs and collect records.

This script is designed for both:
1. Head-node orchestration (`--manifest-only` / multi-study loop)
2. One-study Slurm array tasks (`--manifest-file` + `--task-index`)

Each processed study writes a standalone record file under:
  <output_dir>/studies/<study_id>/benchmark_record.json

A rotating 3D visualization MP4 is expected to be produced by the compare script.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTATION_DIR = Path("/net/projects2/vanguard/vessel_segmentations")
DEFAULT_OUTPUT_DIR = Path("/net/projects2/vanguard/benchmarks/4d_vs_tc4d")
DEFAULT_MANIFEST_NAME = "study_ids.txt"
DEFAULT_RECORDS_JSONL_NAME = "benchmark_records.jsonl"
DEFAULT_RECORDS_CSV_NAME = "benchmark_records.csv"
DEFAULT_RUNNER_SUMMARY_NAME = "runner_summary.json"

STUDY_FILE_PATTERN = re.compile(
    r"(?P<study_id>.+)_(?P<timepoint>\d{4})_vessel_segmentation\.npy$",
    flags=re.IGNORECASE,
)

COMPARE_SCRIPT_CANDIDATES = (
    REPO_ROOT / "graph_extraction" / "debug_compare_4d_vs_tc4d.py",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run 4d-vs-tc4d benchmark comparisons per study and persist records."
        )
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="Directory containing *_<timepoint>_vessel_segmentation.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for manifests, records, and study artifacts.",
    )
    parser.add_argument(
        "--compare-script",
        type=Path,
        default=None,
        help=(
            "Compare script path. If omitted, defaults to the 4d-vs-tc4d compare script."
        ),
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to invoke the compare script.",
    )
    parser.add_argument(
        "--study-id",
        action="append",
        default=None,
        help=(
            "Explicit study ID to process. Repeat flag for multiple IDs. "
            "If omitted, discovery/manifest is used."
        ),
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="Optional newline-delimited list of study IDs.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Optional destination path for writing the resolved study list.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Resolve and write manifest, then exit without running benchmark jobs.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help=(
            "Optional index into the resolved study list. "
            "Used for Slurm array jobs (one task => one study)."
        ),
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Optional limit on number of resolved studies.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing per-study summary/record when available.",
    )
    parser.add_argument(
        "--compare-arg",
        action="append",
        default=None,
        help="Extra argument to forward to compare script. Repeat flag as needed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions and write records without executing compare script.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failed study.",
    )
    return parser.parse_args()


def discover_study_ids(segmentation_dir: Path) -> list[str]:
    """Discover unique study IDs from segmentation files."""
    if not segmentation_dir.exists():
        raise ValueError(f"Segmentation directory does not exist: {segmentation_dir}")
    if not segmentation_dir.is_dir():
        raise ValueError(f"Segmentation path is not a directory: {segmentation_dir}")

    study_ids: set[str] = set()
    for path in sorted(segmentation_dir.glob("*_vessel_segmentation.npy")):
        match = STUDY_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        study_ids.add(match.group("study_id"))

    if not study_ids:
        raise ValueError(
            "No study IDs discovered. Expected files matching "
            "'<study_id>_<timepoint>_vessel_segmentation.npy'."
        )

    return sorted(study_ids)


def load_manifest(manifest_file: Path) -> list[str]:
    """Load newline-delimited study IDs from a manifest file."""
    if not manifest_file.exists():
        raise ValueError(f"Manifest file does not exist: {manifest_file}")

    study_ids = [
        line.strip()
        for line in manifest_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not study_ids:
        raise ValueError(f"Manifest file is empty: {manifest_file}")

    return study_ids


def write_manifest(study_ids: list[str], manifest_file: Path) -> None:
    """Write resolved study IDs to manifest file."""
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(study_ids)
    manifest_file.write_text(f"{lines}\n", encoding="utf-8")


def resolve_compare_script(explicit_script: Path | None) -> Path:
    """Resolve compare script path with explicit override and fallback candidates."""
    if explicit_script is not None:
        if not explicit_script.exists():
            raise ValueError(f"Compare script does not exist: {explicit_script}")
        return explicit_script

    for candidate in COMPARE_SCRIPT_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates = ", ".join(str(path) for path in COMPARE_SCRIPT_CANDIDATES)
    raise ValueError(
        "No compare script found. Provide --compare-script or add one of: "
        f"{candidates}"
    )


def get_nested_value(payload: dict[str, Any], dotted_path: str) -> Any:  # noqa: ANN401
    """Get nested dict value by dotted path; return None if missing."""
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def first_value(payload: dict[str, Any], candidates: list[str]) -> Any:  # noqa: ANN401
    """Return first non-None value among candidate dotted paths."""
    for candidate in candidates:
        value = get_nested_value(payload, candidate)
        if value is not None:
            return value
    return None


def coerce_float(value: Any) -> float | None:  # noqa: ANN401
    """Convert value to float when possible; otherwise return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:  # noqa: ANN401
    """Convert value to int when possible; otherwise return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(summary: dict[str, Any]) -> dict[str, float | int | str | None]:
    """Extract normalized 4d-vs-tc4d metrics from summary.json."""
    return {
        "4d_exam_voxels": coerce_int(
            first_value(
                summary,
                [
                    "4d.exam_voxels",
                ],
            )
        ),
        "tc4d_exam_voxels": coerce_int(
            first_value(
                summary,
                [
                    "tc4d.exam_voxels",
                ],
            )
        ),
        "4d_exam_components_26": coerce_int(
            first_value(
                summary,
                [
                    "4d.exam_components_26",
                ],
            )
        ),
        "tc4d_exam_components_26": coerce_int(
            first_value(
                summary,
                [
                    "tc4d.exam_components_26",
                ],
            )
        ),
        "overlap_voxels": coerce_int(
            first_value(
                summary,
                [
                    "overlap.intersection_voxels",
                ],
            )
        ),
        "overlap_dice": coerce_float(first_value(summary, ["overlap.dice"])),
        "overlap_jaccard": coerce_float(first_value(summary, ["overlap.jaccard"])),
        "compare_elapsed_seconds": coerce_float(
            first_value(
                summary,
                [
                    "timing_seconds.total",
                ],
            )
        ),
        "radiologist_alignment_flip": first_value(
            summary,
            [
                "radiologist_coverage.alignment_flip_to_model",
            ],
        ),
        "4d_exam_radiologist_hits": coerce_int(
            first_value(
                summary,
                [
                    "radiologist_coverage.4d_exam.hits",
                ],
            )
        ),
        "tc4d_exam_radiologist_hits": coerce_int(
            first_value(
                summary,
                [
                    "radiologist_coverage.tc4d_exam.hits",
                ],
            )
        ),
        "4d_exam_radiologist_hit_rate": coerce_float(
            first_value(
                summary,
                [
                    "radiologist_coverage.4d_exam.hit_rate",
                ],
            )
        ),
        "tc4d_exam_radiologist_hit_rate": coerce_float(
            first_value(
                summary,
                [
                    "radiologist_coverage.tc4d_exam.hit_rate",
                ],
            )
        ),
    }


def resolve_visualization_path(
    summary: dict[str, Any],
    study_dir: Path,
) -> str | None:
    """Resolve rotating 3D visualization path from summary outputs when available."""
    mp4_path = first_value(
        summary,
        [
            "outputs.rotation_compare_4d_vs_tc4d_core_mp4",
        ],
    )

    if not isinstance(mp4_path, str) or not mp4_path.strip():
        return None

    path_obj = Path(mp4_path)
    if not path_obj.is_absolute():
        path_obj = study_dir / path_obj

    return str(path_obj)


def tail_text(path: Path, max_chars: int = 4000) -> str:
    """Read tail text from file; return empty string when unavailable."""
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_compare_command(
    *,
    python_executable: Path,
    compare_script: Path,
    segmentation_dir: Path,
    study_id: str,
    artifacts_dir: Path,
    compare_args: list[str],
) -> list[str]:
    """Build compare-script subprocess command."""
    command = [
        str(python_executable),
        str(compare_script),
        "--input-dir",
        str(segmentation_dir),
        "--study-id",
        study_id,
        "--output-dir",
        str(artifacts_dir),
    ]

    if compare_args:
        command.extend(compare_args)

    return command


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object at {path}, got {type(payload).__name__}"
        )
    return payload


def run_study(
    *,
    study_id: str,
    compare_script: Path,
    python_executable: Path,
    segmentation_dir: Path,
    output_dir: Path,
    compare_args: list[str],
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Run benchmark compare step for one study and persist a record."""
    study_dir = output_dir / "studies" / study_id
    artifacts_dir = study_dir / "artifacts"
    record_path = study_dir / "benchmark_record.json"
    summary_path = artifacts_dir / "summary.json"
    stdout_log = study_dir / "compare_stdout.log"
    stderr_log = study_dir / "compare_stderr.log"

    study_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    command = build_compare_command(
        python_executable=python_executable,
        compare_script=compare_script,
        segmentation_dir=segmentation_dir,
        study_id=study_id,
        artifacts_dir=artifacts_dir,
        compare_args=compare_args,
    )

    if dry_run:
        record = {
            "study_id": study_id,
            "status": "dry_run",
            "command": command,
            "compare_script": str(compare_script),
            "summary_path": str(summary_path),
            "visualization_mp4": None,
            "metrics": {},
            "runner_elapsed_seconds": 0.0,
            "error": None,
        }
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    if skip_existing and record_path.exists() and summary_path.exists():
        record = load_json(record_path)
        record["status"] = "reused"
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    started = time.perf_counter()

    with stdout_log.open("w", encoding="utf-8") as stdout_handle:
        with stderr_log.open("w", encoding="utf-8") as stderr_handle:
            result = subprocess.run(  # noqa: S603
                command,
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )

    elapsed = float(time.perf_counter() - started)

    if result.returncode != 0:
        error_tail = tail_text(stderr_log)
        record = {
            "study_id": study_id,
            "status": "failed",
            "command": command,
            "compare_script": str(compare_script),
            "summary_path": str(summary_path),
            "visualization_mp4": None,
            "metrics": {},
            "runner_elapsed_seconds": elapsed,
            "error": (
                f"compare script exited with code {result.returncode}. "
                f"stderr tail:\n{error_tail}"
            ),
        }
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    if not summary_path.exists():
        record = {
            "study_id": study_id,
            "status": "failed",
            "command": command,
            "compare_script": str(compare_script),
            "summary_path": str(summary_path),
            "visualization_mp4": None,
            "metrics": {},
            "runner_elapsed_seconds": elapsed,
            "error": f"Expected summary JSON not found: {summary_path}",
        }
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    summary = load_json(summary_path)
    metrics = extract_metrics(summary)
    visualization_mp4 = resolve_visualization_path(summary, artifacts_dir)

    record = {
        "study_id": study_id,
        "status": "ok",
        "command": command,
        "compare_script": str(compare_script),
        "summary_path": str(summary_path),
        "visualization_mp4": visualization_mp4,
        "metrics": metrics,
        "runner_elapsed_seconds": elapsed,
        "error": None,
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def write_records_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write run records to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def write_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write run records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "study_id",
        "status",
        "compare_script",
        "summary_path",
        "visualization_mp4",
        "runner_elapsed_seconds",
        "4d_exam_voxels",
        "tc4d_exam_voxels",
        "4d_exam_components_26",
        "tc4d_exam_components_26",
        "overlap_voxels",
        "overlap_dice",
        "overlap_jaccard",
        "compare_elapsed_seconds",
        "radiologist_alignment_flip",
        "4d_exam_radiologist_hits",
        "tc4d_exam_radiologist_hits",
        "4d_exam_radiologist_hit_rate",
        "tc4d_exam_radiologist_hit_rate",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics", {})
            row = {
                "study_id": record.get("study_id"),
                "status": record.get("status"),
                "compare_script": record.get("compare_script"),
                "summary_path": record.get("summary_path"),
                "visualization_mp4": record.get("visualization_mp4"),
                "runner_elapsed_seconds": record.get("runner_elapsed_seconds"),
                "4d_exam_voxels": metrics.get("4d_exam_voxels"),
                "tc4d_exam_voxels": metrics.get("tc4d_exam_voxels"),
                "4d_exam_components_26": metrics.get("4d_exam_components_26"),
                "tc4d_exam_components_26": metrics.get("tc4d_exam_components_26"),
                "overlap_voxels": metrics.get("overlap_voxels"),
                "overlap_dice": metrics.get("overlap_dice"),
                "overlap_jaccard": metrics.get("overlap_jaccard"),
                "compare_elapsed_seconds": metrics.get("compare_elapsed_seconds"),
                "radiologist_alignment_flip": metrics.get("radiologist_alignment_flip"),
                "4d_exam_radiologist_hits": metrics.get("4d_exam_radiologist_hits"),
                "tc4d_exam_radiologist_hits": metrics.get("tc4d_exam_radiologist_hits"),
                "4d_exam_radiologist_hit_rate": metrics.get(
                    "4d_exam_radiologist_hit_rate"
                ),
                "tc4d_exam_radiologist_hit_rate": metrics.get(
                    "tc4d_exam_radiologist_hit_rate"
                ),
                "error": record.get("error"),
            }
            writer.writerow(row)


def resolve_study_ids(args: argparse.Namespace) -> list[str]:
    """Resolve ordered study ID list from explicit list, manifest, or discovery."""
    if args.study_id:
        ordered = sorted(set(args.study_id))
    elif args.manifest_file is not None:
        ordered = load_manifest(args.manifest_file)
    else:
        ordered = discover_study_ids(args.segmentation_dir)

    if args.max_studies is not None:
        if args.max_studies <= 0:
            raise ValueError("--max-studies must be positive when provided")
        ordered = ordered[: args.max_studies]

    if not ordered:
        raise ValueError("Resolved study list is empty")

    return ordered


def select_task_studies(study_ids: list[str], task_index: int | None) -> list[str]:
    """Select a single study for array-task execution when task index is provided."""
    if task_index is None:
        return study_ids

    if task_index < 0:
        raise ValueError("--task-index cannot be negative")

    if task_index >= len(study_ids):
        raise ValueError(
            f"--task-index={task_index} out of range for {len(study_ids)} studies"
        )

    return [study_ids[task_index]]


def summarize_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build top-level run summary counts."""
    total = len(records)
    ok = sum(1 for record in records if record.get("status") == "ok")
    reused = sum(1 for record in records if record.get("status") == "reused")
    failed = sum(1 for record in records if record.get("status") == "failed")
    dry_run = sum(1 for record in records if record.get("status") == "dry_run")

    return {
        "total": total,
        "ok": ok,
        "reused": reused,
        "failed": failed,
        "dry_run": dry_run,
    }


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    resolved_study_ids = resolve_study_ids(args)

    manifest_path = args.write_manifest
    if manifest_path is None and args.manifest_only:
        manifest_path = args.output_dir / DEFAULT_MANIFEST_NAME

    if manifest_path is not None:
        write_manifest(resolved_study_ids, manifest_path)
        print(
            f"[manifest] wrote {len(resolved_study_ids)} study IDs to {manifest_path}",
        )

    if args.manifest_only:
        return

    target_study_ids = select_task_studies(resolved_study_ids, args.task_index)
    compare_script = resolve_compare_script(args.compare_script)

    compare_args = args.compare_arg if args.compare_arg is not None else []

    print(f"[runner] compare script: {compare_script}")
    print(f"[runner] studies to process: {len(target_study_ids)}")

    records: list[dict[str, Any]] = []
    for index, study_id in enumerate(target_study_ids, start=1):
        print(f"[runner] ({index}/{len(target_study_ids)}) study={study_id}")
        record = run_study(
            study_id=study_id,
            compare_script=compare_script,
            python_executable=args.python_executable,
            segmentation_dir=args.segmentation_dir,
            output_dir=args.output_dir,
            compare_args=compare_args,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        records.append(record)
        print(
            "[runner] "
            f"study={study_id} status={record.get('status')} "
            f"record={args.output_dir / 'studies' / study_id / 'benchmark_record.json'}"
        )

        if args.fail_fast and record.get("status") == "failed":
            break

    is_single_task_mode = args.task_index is not None
    if is_single_task_mode:
        return

    jsonl_path = args.output_dir / DEFAULT_RECORDS_JSONL_NAME
    csv_path = args.output_dir / DEFAULT_RECORDS_CSV_NAME
    summary_path = args.output_dir / DEFAULT_RUNNER_SUMMARY_NAME

    write_records_jsonl(records, jsonl_path)
    write_records_csv(records, csv_path)

    summary = summarize_run(records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[runner] wrote JSONL: {jsonl_path}")
    print(f"[runner] wrote CSV: {csv_path}")
    print(f"[runner] wrote summary: {summary_path}")
    print(
        "[runner] status counts "
        f"ok={summary['ok']} reused={summary['reused']} "
        f"failed={summary['failed']} dry_run={summary['dry_run']}"
    )


if __name__ == "__main__":
    main()
