#!/usr/bin/env python3
"""Per-study pipeline inventory for the VANGUARD vessel-network pipeline.

For every study found under --input-root, checks which of the three pipeline
stages have been completed and prints a summary table with per-study status.

Stages checked
--------------
1. segmentation   -- vessel_segmentation.npz files exist and are non-empty
2. centerlines    -- skeleton_4d_exam_mask.npy exists under --output-root
3. graph_features -- tumor_graph_features.json exists under --output-root

Study discovery mirrors graph_extraction/slurm/submit_tc4d_array.sh:
iterate cohort/ → study_id/ subdirectories under --input-root.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# ── path defaults (match slurm script fallbacks) ────────────────────────────

_DEFAULT_INPUT_ROOT = Path("/scratch/t-9sbose/vessel_segmentations")
_DEFAULT_OUTPUT_ROOT = Path("/scratch/t-9sbose/centerlines_tc4d/studies")

STAGES = ("segmentation", "centerlines", "graph_features")


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class StudyRecord:
    study_id: str
    cohort: str
    seg_timepoints: int
    centerline_voxels: int  # -1 = file missing/unreadable; >=0 = nonzero voxel count
    graph_features: bool

    @property
    def segmentation(self) -> bool:
        return self.seg_timepoints > 0

    @property
    def centerlines(self) -> bool:
        return self.centerline_voxels >= 0

    @property
    def status(self) -> str:
        flags = (self.segmentation, self.centerlines, self.graph_features)
        n = sum(flags)
        if n == 3:
            return "complete"
        if n == 0:
            return "missing"
        return "partial"

    def stage_flag(self, stage: str) -> bool:
        return getattr(self, stage)


# ── study discovery ──────────────────────────────────────────────────────────

def discover_studies(input_root: Path) -> list[tuple[str, str]]:
    """Return sorted list of (cohort, case_id) pairs from input_root."""
    pairs: list[tuple[str, str]] = []
    if not input_root.exists():
        print(
            f"WARNING: input-root does not exist: {input_root}",
            file=sys.stderr,
        )
        return pairs
    for site_dir in sorted(input_root.iterdir()):
        if not site_dir.is_dir():
            continue
        for study_dir in sorted(site_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            pairs.append((site_dir.name, study_dir.name))
    return pairs


# ── stage checks ─────────────────────────────────────────────────────────────

def _nonzero(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


def check_segmentation(input_root: Path, cohort: str, case_id: str) -> int:
    """Return the number of non-empty *_vessel_segmentation.npz timepoint files."""
    images_dir = input_root / cohort / case_id / "images"
    if not images_dir.is_dir():
        return 0
    return sum(
        1 for f in images_dir.iterdir()
        if f.name.endswith("_vessel_segmentation.npz") and _nonzero(f)
    )


def count_centerline_voxels(output_root: Path, cohort: str, case_id: str) -> int:
    """Return number of nonzero voxels in skeleton_4d_exam_mask.npy, or -1 if missing."""
    import numpy as np  # lazy — keeps startup fast when numpy isn't needed
    p = output_root / cohort / case_id / f"{case_id}_skeleton_4d_exam_mask.npy"
    if not p.exists():
        return -1
    try:
        return int(np.count_nonzero(np.load(p, allow_pickle=False)))
    except Exception:
        return -1


def check_graph_features(output_root: Path, cohort: str, case_id: str) -> bool:
    p = output_root / cohort / case_id / f"{case_id}_tumor_graph_features.json"
    return p.exists()


def build_record(
    cohort: str,
    case_id: str,
    input_root: Path,
    output_root: Path,
) -> StudyRecord:
    return StudyRecord(
        study_id=case_id,
        cohort=cohort,
        seg_timepoints=check_segmentation(input_root, cohort, case_id),
        centerline_voxels=count_centerline_voxels(output_root, cohort, case_id),
        graph_features=check_graph_features(output_root, cohort, case_id),
    )


# ── bottleneck detection ─────────────────────────────────────────────────────

def find_bottleneck(records: list[StudyRecord]) -> tuple[str, int] | None:
    """Return (stage_name, count) for the stage blocking the most studies.

    A study is blocked at stage N if all stages before N are done but stage N
    is not.  The bottleneck is whichever stage has the highest such count.
    """
    # Ordered pipeline stages with their prerequisite stages
    pipeline = [
        ("segmentation", []),
        ("centerlines", ["segmentation"]),
        ("graph_features", ["segmentation", "centerlines"]),
    ]
    best_stage, best_count = "", 0
    for stage, prereqs in pipeline:
        count = sum(
            1
            for r in records
            if not r.stage_flag(stage)
            and all(r.stage_flag(p) for p in prereqs)
        )
        if count > best_count:
            best_count, best_stage = count, stage
    return (best_stage, best_count) if best_count > 0 else None


# ── printing ──────────────────────────────────────────────────────────────────

def _yn(flag: bool) -> str:
    return "yes" if flag else "no "


def _cl_vox(n: int) -> str:
    return "-" if n < 0 else str(n)


def print_table(records: list[StudyRecord], stage_filter: str | None) -> None:
    if not records:
        print("No studies found.")
        return

    if stage_filter:
        if stage_filter == "segmentation":
            header = f"{'study_id':<18}  {'cohort':<8}  {'segmentation':<14}  {'seg_tp':<7}  status"
        elif stage_filter == "centerlines":
            header = f"{'study_id':<18}  {'cohort':<8}  {'centerlines':<12}  {'cl_voxels':<12}  status"
        else:
            header = f"{'study_id':<18}  {'cohort':<8}  {stage_filter:<16}  status"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for r in records:
            if stage_filter == "segmentation":
                print(f"{r.study_id:<18}  {r.cohort:<8}  {_yn(r.segmentation):<14}  {r.seg_timepoints:<7}  {r.status}")
            elif stage_filter == "centerlines":
                print(f"{r.study_id:<18}  {r.cohort:<8}  {_yn(r.centerlines):<12}  {_cl_vox(r.centerline_voxels):<12}  {r.status}")
            else:
                flag = _yn(r.stage_flag(stage_filter))
                print(f"{r.study_id:<18}  {r.cohort:<8}  {flag:<16}  {r.status}")
    else:
        header = (
            f"{'study_id':<18}  {'cohort':<8}  "
            f"{'segmentation':<14}  {'seg_tp':<7}  {'centerlines':<12}  "
            f"{'cl_voxels':<12}  {'graph_features':<15}  status"
        )
        sep = "-" * len(header)
        print(header)
        print(sep)
        for r in records:
            print(
                f"{r.study_id:<18}  {r.cohort:<8}  "
                f"{_yn(r.segmentation):<14}  {r.seg_timepoints:<7}  {_yn(r.centerlines):<12}  "
                f"{_cl_vox(r.centerline_voxels):<12}  {_yn(r.graph_features):<15}  {r.status}"
            )


def print_summary(records: list[StudyRecord], stage_filter: str | None) -> None:
    total = len(records)
    if total == 0:
        return

    by_status: dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    for r in records:
        by_status[r.status] += 1

    print()
    print(f"Total studies : {total}")
    for status, count in by_status.items():
        pct = count / total * 100
        print(f"  {status:<10}: {count:>5}  ({pct:5.1f}%)")

    if stage_filter is None:
        bottleneck = find_bottleneck(records)
        if bottleneck:
            stage, count = bottleneck
            print()
            print(f"Bottleneck: {stage}  ({count} studies are ready for this stage but it is not done)")


def write_csv(records: list[StudyRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "study_id", "cohort", "segmentation", "seg_timepoints",
                "centerlines", "centerline_voxels", "graph_features", "status",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "study_id": r.study_id,
                "cohort": r.cohort,
                "segmentation": "yes" if r.segmentation else "no",
                "seg_timepoints": r.seg_timepoints,
                "centerlines": "yes" if r.centerlines else "no",
                "centerline_voxels": r.centerline_voxels if r.centerline_voxels >= 0 else "",
                "graph_features": "yes" if r.graph_features else "no",
                "status": r.status,
            })
    print(f"\nCSV written to: {path}")


def print_cohort_distribution(records: list[StudyRecord]) -> None:
    """Print per-cohort distribution of centerline voxel counts."""
    import statistics

    with_cl = [r for r in records if r.centerline_voxels >= 0]
    if not with_cl:
        return

    cohorts: dict[str, list[int]] = {}
    for r in with_cl:
        cohorts.setdefault(r.cohort, []).append(r.centerline_voxels)

    print()
    print("Centerline voxel count distribution by cohort")
    print("(studies with centerlines only)")
    header = f"{'cohort':<10}  {'n':>5}  {'min':>8}  {'P25':>8}  {'median':>8}  {'P75':>8}  {'max':>8}"
    print(header)
    print("-" * len(header))
    for cohort in sorted(cohorts):
        vals = sorted(cohorts[cohort])
        n = len(vals)
        lo, hi = vals[0], vals[-1]
        med = statistics.median(vals)
        p25 = vals[max(0, int(n * 0.25))]
        p75 = vals[min(n - 1, int(n * 0.75))]
        print(f"{cohort:<10}  {n:>5}  {lo:>8}  {p25:>8}  {int(med):>8}  {p75:>8}  {hi:>8}")


# ── config loading ────────────────────────────────────────────────────────────

def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (input_root, output_root) after applying config + CLI priority."""
    input_root: Path | None = args.input_root
    output_root: Path | None = args.output_root

    if args.config is not None:
        try:
            # Import lazily so the script works without the full vanguard env on PATH
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from config import load_config  # noqa: PLC0415
            cfg = load_config(args.config)
            dp = cfg.data_paths
            if input_root is None and dp.get("vessel_segmentation_root"):
                input_root = Path(dp["vessel_segmentation_root"])
            if output_root is None and dp.get("centerline_root"):
                output_root = Path(dp["centerline_root"])
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not load config {args.config}: {exc}", file=sys.stderr)

    return (
        input_root if input_root is not None else _DEFAULT_INPUT_ROOT,
        output_root if output_root is not None else _DEFAULT_OUTPUT_ROOT,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory which VANGUARD pipeline stages are complete for each study."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Vessel segmentation output root.  "
            "Studies are discovered as {cohort}/{case_id} subdirectories here.  "
            f"Defaults to {_DEFAULT_INPUT_ROOT} (or config vessel_segmentation_root)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Graph/centerline output root.  "
            "Skeleton and feature JSON files are expected at "
            "{output_root}/{cohort}/{case_id}/{case_id}_*.  "
            f"Defaults to {_DEFAULT_OUTPUT_ROOT} (or config centerline_root)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML config file (configs/*.yaml).  "
            "data_paths.vessel_segmentation_root and data_paths.centerline_root "
            "are used as fallbacks when the explicit --input-root / --output-root "
            "flags are not given."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default=None,
        help="Filter the table to show only this stage's column.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the inventory table to a CSV file at this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root, output_root = _resolve_paths(args)

    print(f"input-root  : {input_root}")
    print(f"output-root : {output_root}")
    print()

    if not output_root.exists():
        print(
            f"WARNING: output-root does not exist: {output_root}",
            file=sys.stderr,
        )

    pairs = discover_studies(input_root)
    if not pairs:
        print("No studies discovered. Check --input-root.", file=sys.stderr)
        sys.exit(1)

    records: list[StudyRecord] = []
    for cohort, case_id in pairs:
        records.append(build_record(cohort, case_id, input_root, output_root))

    if args.stage:
        filtered = [r for r in records if not r.stage_flag(args.stage)]
        print(f"Studies missing stage '{args.stage}' ({len(filtered)} of {len(records)}):\n")
        print_table(filtered, args.stage)
        print_summary(filtered, args.stage)
    else:
        print_table(records, stage_filter=None)
        print_summary(records, stage_filter=None)
        print_cohort_distribution(records)

    if args.output_csv:
        write_csv(records, args.output_csv)


if __name__ == "__main__":
    main()
