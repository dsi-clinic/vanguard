#!/usr/bin/env python3
"""Reduce per-study 4d-vs-tc4d benchmark records into summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARK_DIR = Path("/net/projects2/vanguard/benchmarks/4d_vs_tc4d")
DEFAULT_PER_STUDY_CSV = "benchmark_reduced_per_study.csv"
DEFAULT_SUMMARY_JSON = "benchmark_reduced_summary.json"

NUMERIC_METRIC_FIELDS = (
    "4d_exam_voxels",
    "tc4d_exam_voxels",
    "4d_exam_components_26",
    "tc4d_exam_components_26",
    "overlap_voxels",
    "overlap_dice",
    "overlap_jaccard",
    "compare_elapsed_seconds",
    "runner_elapsed_seconds",
    "4d_exam_radiologist_hits",
    "tc4d_exam_radiologist_hits",
    "4d_exam_radiologist_hit_rate",
    "tc4d_exam_radiologist_hit_rate",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Reduce benchmark record files into per-study and aggregate summaries."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Benchmark root containing studies/*/benchmark_record.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reduced CSV/JSON. Defaults to --benchmark-dir.",
    )
    parser.add_argument(
        "--per-study-csv",
        type=str,
        default=DEFAULT_PER_STUDY_CSV,
        help="Filename for reduced per-study CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=DEFAULT_SUMMARY_JSON,
        help="Filename for aggregate summary JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON object from file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object at {path}, got {type(payload).__name__}"
        )
    return payload


def collect_records(benchmark_dir: Path) -> list[dict[str, Any]]:
    """Collect per-study record JSON files from benchmark directory."""
    studies_dir = benchmark_dir / "studies"
    if not studies_dir.exists() or not studies_dir.is_dir():
        raise ValueError(f"Missing studies directory: {studies_dir}")

    record_files = sorted(studies_dir.glob("*/benchmark_record.json"))
    if not record_files:
        raise ValueError(f"No benchmark records found in: {studies_dir}")

    records = [load_json(path) for path in record_files]
    return records


def numeric_values(records: list[dict[str, Any]], metric_key: str) -> list[float]:
    """Collect numeric values for one metric from successful/reused records."""
    values: list[float] = []
    for record in records:
        status = record.get("status")
        if status not in {"ok", "reused"}:
            continue

        if metric_key == "runner_elapsed_seconds":
            candidate = record.get(metric_key)
        else:
            metrics = record.get("metrics", {})
            candidate = metrics.get(metric_key)

        if isinstance(candidate, int | float):
            values.append(float(candidate))

    return values


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    """Compute descriptive stats for numeric metric values."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_pairwise_comparison(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count tc4d-vs-4d wins based on exam voxel counts."""
    tc4d_better = 0
    baseline_4d_better = 0
    tied = 0
    comparable = 0

    for record in records:
        if record.get("status") not in {"ok", "reused"}:
            continue

        metrics = record.get("metrics", {})
        baseline_4d_voxels = metrics.get("4d_exam_voxels")
        tc4d_voxels = metrics.get("tc4d_exam_voxels")

        if not isinstance(baseline_4d_voxels, int | float):
            continue
        if not isinstance(tc4d_voxels, int | float):
            continue

        comparable += 1
        if tc4d_voxels > baseline_4d_voxels:
            tc4d_better += 1
        elif tc4d_voxels < baseline_4d_voxels:
            baseline_4d_better += 1
        else:
            tied += 1

    return {
        "comparable_studies": comparable,
        "tc4d_exam_voxels_greater": tc4d_better,
        "4d_exam_voxels_greater": baseline_4d_better,
        "voxels_tied": tied,
    }


def write_per_study_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write flattened per-study records to CSV."""
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

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


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate JSON summary from benchmark records."""
    total = len(records)
    ok = sum(1 for record in records if record.get("status") == "ok")
    reused = sum(1 for record in records if record.get("status") == "reused")
    failed = sum(1 for record in records if record.get("status") == "failed")
    dry_run = sum(1 for record in records if record.get("status") == "dry_run")

    failed_studies = [
        {
            "study_id": record.get("study_id"),
            "error": record.get("error"),
        }
        for record in records
        if record.get("status") == "failed"
    ]

    numeric_summary = {
        metric_key: summarize_numeric(numeric_values(records, metric_key))
        for metric_key in NUMERIC_METRIC_FIELDS
    }

    pairwise = build_pairwise_comparison(records)

    return {
        "counts": {
            "total": total,
            "ok": ok,
            "reused": reused,
            "failed": failed,
            "dry_run": dry_run,
        },
        "pairwise_voxel_comparison": pairwise,
        "numeric_metrics": numeric_summary,
        "failed_studies": failed_studies,
    }


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.benchmark_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.benchmark_dir)

    per_study_csv_path = output_dir / args.per_study_csv
    summary_json_path = output_dir / args.summary_json

    write_per_study_csv(records, per_study_csv_path)
    summary = build_summary(records)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[reduce] loaded records: {len(records)}")
    print(f"[reduce] per-study CSV: {per_study_csv_path}")
    print(f"[reduce] summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
