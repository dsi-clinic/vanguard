"""Run one restartable Vanguard preprocessing stage for a Slurm array task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from preprocessing.cases import CaseRecord, read_case_manifest
from preprocessing.model import model_subject_id
from preprocessing.pipeline import (
    POLICY_NAME,
    complement_case,
    infer_case,
    map_case,
    prepare_case,
    qc_case,
    tc4d_case,
)


def select_array_case(case_manifest: Path, index: int) -> CaseRecord:
    """Select one case using a stable exam-ID sort order."""
    records = sorted(
        read_case_manifest(case_manifest), key=lambda record: record.exam_id
    )
    if index < 0 or index >= len(records):
        raise IndexError(f"array index {index} outside [0, {len(records) - 1}]")
    return records[index]


def _provenance(case_root: Path) -> dict[str, object]:
    path = case_root / "preprocessing_provenance.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_equal(*, label: str, recorded: object, current: object) -> None:
    if recorded != current:
        raise RuntimeError(
            f"refusing to reuse preprocessing with changed {label}: "
            f"recorded={recorded!r}, current={current!r}; use a fresh output root"
        )


def _assert_reuse_contract(
    *,
    case_root: Path,
    record: CaseRecord,
    inventory: Path,
    case_manifest: Path,
    breast_model: Path,
    vessel_model: Path,
) -> None:
    """Reject partial, stale, or differently configured existing case outputs."""
    if not case_root.exists():
        return
    provenance = _provenance(case_root)
    if not provenance:
        raise RuntimeError(
            f"refusing to reuse {case_root} without preprocessing provenance; "
            "use a fresh output root"
        )
    policy = provenance.get("policy")
    policy_name = policy.get("name") if isinstance(policy, dict) else None
    _assert_equal(
        label="preprocessing policy", recorded=policy_name, current=POLICY_NAME
    )
    _assert_equal(
        label="case-manifest record",
        recorded=provenance.get("case"),
        current=record.__dict__,
    )
    resolved_inventory = inventory.expanduser().resolve()
    resolved_manifest = case_manifest.expanduser().resolve()
    _assert_equal(
        label="inventory path",
        recorded=provenance.get("inventory_path"),
        current=str(resolved_inventory),
    )
    _assert_equal(
        label="inventory checksum",
        recorded=provenance.get("inventory_sha256"),
        current=_sha256(resolved_inventory),
    )
    _assert_equal(
        label="case-manifest path",
        recorded=provenance.get("case_manifest_path"),
        current=str(resolved_manifest),
    )
    _assert_equal(
        label="case-manifest checksum",
        recorded=provenance.get("case_manifest_sha256"),
        current=_sha256(resolved_manifest),
    )
    inference = provenance.get("inference")
    if isinstance(inference, dict):
        for label, path in (
            ("breast model", breast_model),
            ("vessel model", vessel_model),
        ):
            resolved = path.expanduser().resolve()
            key = label.replace(" ", "_")
            _assert_equal(
                label=f"{label} path",
                recorded=inference.get(key),
                current=str(resolved),
            )
            _assert_equal(
                label=f"{label} checksum",
                recorded=inference.get(f"{key}_sha256"),
                current=_sha256(resolved),
            )


def _require_files(paths: list[Path], *, stage: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"provenance marks {stage} complete but {len(missing)} artifacts are "
            f"missing: {missing[:3]}; use a fresh output root"
        )


def _stage_complete(stage: str, case_root: Path) -> bool:
    provenance = _provenance(case_root)
    if stage == "prepare":
        complete = provenance.get("status") == "prepared"
        if complete:
            hr_phases = len(provenance["hr_source"]["times_seconds"])
            ufast_phases = len(provenance["ufast_source"]["times_seconds"])
            dce_dir = case_root.parents[1] / "dce" / case_root.name
            _require_files(
                [case_root / "hr_times_seconds.npy"]
                + [
                    case_root / "hr_model_inputs" / f"{model_subject_id(index)}.npy"
                    for index in range(hr_phases)
                ]
                + [dce_dir / "ufast_times_seconds.npy"]
                + [
                    dce_dir / f"{case_root.name}_{index:04d}.nii.gz"
                    for index in range(ufast_phases)
                ],
                stage=stage,
            )
        return complete
    if stage == "infer":
        inference = provenance.get("inference")
        complete = isinstance(inference, dict) and {
            "hr_phases",
            "ufast_phases",
        }.issubset(inference)
        if complete:
            hr_phases = int(inference["hr_phases"])
            ufast_phases = int(inference["ufast_phases"])
            exam_id = case_root.name
            _require_files(
                [
                    case_root
                    / "hr_breast_predictions"
                    / f"{model_subject_id(index)}.npy"
                    for index in range(hr_phases)
                ]
                + [
                    case_root
                    / "hr_vessel_predictions"
                    / f"{model_subject_id(index)}.npz"
                    for index in range(hr_phases)
                ]
                + [
                    case_root
                    / "ufast_breast_predictions"
                    / f"{exam_id}_{index:04d}.npy"
                    for index in range(ufast_phases)
                ]
                + [
                    case_root
                    / "ufast_vessel_predictions"
                    / f"{exam_id}_{index:04d}.npz"
                    for index in range(ufast_phases)
                ],
                stage=stage,
            )
        return complete
    if stage == "tc4d":
        complete = "tc4d" in provenance and "ufast_tc4d" in provenance
        if complete:
            _require_files(
                [
                    case_root / "hr_tc4d" / "exam_skeleton_zyx.npy",
                    case_root / "hr_tc4d" / "exam_support_zyx.npy",
                    case_root / "hr_tc4d" / "center_manifold_4d_yxz.npy",
                    case_root / "ufast_tc4d" / "exam_skeleton_zyx.npy",
                    case_root / "ufast_tc4d" / "exam_support_zyx.npy",
                    case_root / "ufast_tc4d" / "center_manifold_4d_yxz.npy",
                ],
                stage=stage,
            )
        return complete
    if stage == "map":
        complete = "mapping" in provenance and "merge" in provenance
        if complete:
            output_dir = (
                case_root.parents[1]
                / "centerlines"
                / str(provenance["case"]["dataset"])
                / case_root.name
            )
            _require_files(
                [
                    output_dir / f"{case_root.name}_skeleton_4d_exam_mask.npy",
                    output_dir / f"{case_root.name}_skeleton_4d_exam_support_mask.npy",
                    output_dir / f"{case_root.name}_skeleton_4d_exam_mask_hr_only.npy",
                    output_dir / f"{case_root.name}_merge_provenance_label_zyx.npy",
                    output_dir / "merge_provenance.json",
                    output_dir / "run_summary.json",
                ],
                stage=stage,
            )
        return complete
    if stage == "complement":
        complete = "complement" in provenance
        if complete:
            output_dir = (
                case_root.parents[1]
                / "centerlines"
                / str(provenance["case"]["dataset"])
                / case_root.name
            )
            _require_files(
                [
                    output_dir
                    / f"{case_root.name}_skeleton_4d_exam_mask_hr_ufast_merge_only.npy",
                    output_dir / "complement_provenance.json",
                ],
                stage=stage,
            )
        return complete
    if stage == "qc":
        if "mapping" not in provenance or "merge" not in provenance:
            return False
        centerline_dir = (
            case_root.parents[1]
            / "centerlines"
            / str(provenance["case"]["dataset"])
            / case_root.name
        )
        return (centerline_dir / "mapping_qc.png").exists() and (
            centerline_dir / "temporal_mip.png"
        ).exists()
    if stage == "postprocess":
        return all(
            _stage_complete(item, case_root)
            for item in ("tc4d", "map", "complement", "qc")
        )
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
    _assert_reuse_contract(
        case_root=case_root,
        record=record,
        inventory=inventory,
        case_manifest=case_manifest,
        breast_model=breast_model,
        vessel_model=vessel_model,
    )
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
    elif stage == "complement":
        complement_case(case_root=case_root)
    elif stage == "qc":
        qc_case(case_root=case_root)
    elif stage == "postprocess":
        for item in ("tc4d", "map", "complement", "qc"):
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
        "stage",
        choices=("prepare", "infer", "tc4d", "map", "complement", "qc", "postprocess"),
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
