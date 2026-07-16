"""Run one restartable Vanguard preprocessing stage for a Slurm array task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from preprocessing.cases import CaseRecord, read_case_manifest
from preprocessing.pipeline import (
    infer_case,
    map_case,
    prepare_case,
    qc_case,
    tc4d_case,
)


def select_array_case(case_manifest: Path, index: int) -> CaseRecord:
    """Select one case using a stable exam-ID sort order."""
    records = sorted(read_case_manifest(case_manifest), key=lambda record: record.exam_id)
    if index < 0 or index >= len(records):
        raise IndexError(f"array index {index} outside [0, {len(records) - 1}]")
    return records[index]


def _provenance(case_root: Path) -> dict[str, object]:
    path = case_root / "preprocessing_provenance.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _stage_complete(stage: str, case_root: Path) -> bool:
    provenance = _provenance(case_root)
    if stage == "prepare":
        return provenance.get("status") == "prepared"
    if stage == "infer":
        return "inference" in provenance
    if stage == "tc4d":
        return "tc4d" in provenance
    if stage == "map":
        return "mapping" in provenance
    if stage == "qc":
        return "mapping" in provenance and (
            case_root.parents[1]
            / "centerlines"
            / str(provenance["case"]["dataset"])
            / case_root.name
            / "mapping_qc.png"
        ).exists()
    if stage == "postprocess":
        return all(_stage_complete(item, case_root) for item in ("tc4d", "map", "qc"))
    raise ValueError(f"unknown pipeline stage: {stage}")


def run_stage(
    *,
    stage: str,
    record: CaseRecord,
    inventory: Path,
    case_manifest: Path,
    output_root: Path,
    breast_model: Path,
    vessel_model: Path,
    batch_size: int,
) -> None:
    """Run one stage, skipping an already complete case without overwriting it."""
    case_root = output_root.expanduser().resolve() / "work" / record.exam_id
    if _stage_complete(stage, case_root):
        print(f"[skip] {record.exam_id} {stage} already complete", flush=True)
        return
    if stage == "prepare":
        prepare_case(
            inventory_path=inventory,
            case_manifest=case_manifest,
            exam_id=record.exam_id,
            output_root=output_root,
        )
    elif stage == "infer":
        infer_case(
            case_root=case_root,
            breast_model=breast_model,
            vessel_model=vessel_model,
            batch_size=batch_size,
        )
    elif stage == "tc4d":
        tc4d_case(case_root=case_root)
    elif stage == "map":
        map_case(case_root=case_root)
    elif stage == "qc":
        qc_case(case_root=case_root)
    elif stage == "postprocess":
        for item in ("tc4d", "map", "qc"):
            if not _stage_complete(item, case_root):
                run_stage(
                    stage=item,
                    record=record,
                    inventory=inventory,
                    case_manifest=case_manifest,
                    output_root=output_root,
                    breast_model=breast_model,
                    vessel_model=vessel_model,
                    batch_size=batch_size,
                )
    else:
        raise ValueError(f"unknown pipeline stage: {stage}")
    print(f"[complete] {record.exam_id} {stage}", flush=True)


def main() -> None:
    """Resolve a Slurm array index and run one exact reviewed case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "infer", "tc4d", "map", "qc", "postprocess")
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--array-index", type=int)
    model_root = (
        Path(__file__).resolve().parents[1]
        / "vanguard-blood-vessel-segmentation"
        / "trained_models"
    )
    parser.add_argument(
        "--breast-model", type=Path, default=model_root / "breast_model.pth"
    )
    parser.add_argument(
        "--vessel-model", type=Path, default=model_root / "dv_model.pth"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    index = args.array_index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    record = select_array_case(args.case_manifest, index)
    run_stage(
        stage=args.stage,
        record=record,
        inventory=args.inventory,
        case_manifest=args.case_manifest,
        output_root=args.output_root,
        breast_model=args.breast_model,
        vessel_model=args.vessel_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
