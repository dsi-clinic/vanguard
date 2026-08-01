"""Manifest-driven temporal graph cache for UChicago ultrafast DCE exams.

This is separate from the historical Duke forecasting smoke. UChicago exam
identity, longitudinal patient grouping, pCR labels, and provided outer folds
come from its manifests; graph/DCE pixels remain read-only inputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from gnn.data_loader import (
    _CENTERLINE_SUFFIX,
    _MODE_DEFAULT_FEATURES,
    _VOXEL_MODE,
    _build_case,
    _git_commit,
)

REQUIRED_ID_COLUMNS = ("exam_id", "patient_key", "dataset")


@dataclass(frozen=True)
class TemporalCacheRecord:
    """One cached UChicago exam and its longitudinal/evaluation identity."""

    exam_id: str
    patient_key: str
    dataset: str
    graph_path: str
    n_frames: int
    duration_seconds: float
    median_dt_seconds: float
    fold: int | None
    pcr: int | None


def load_exam_manifest(path: str | Path) -> pd.DataFrame:
    """Load one row per UChicago exam and validate longitudinal identity."""
    frame = pd.read_csv(path)
    missing = set(REQUIRED_ID_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"UChicago manifest is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in REQUIRED_ID_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(
                f"UChicago manifest column {column!r} contains missing values"
            )
        frame[column] = frame[column].astype(str)
    duplicated = frame.loc[frame["exam_id"].duplicated(), "exam_id"].tolist()
    if duplicated:
        raise ValueError(
            f"UChicago manifest has duplicate exam_id values: {duplicated}"
        )
    return frame


def exclude_labeled_patients(
    pretrain_manifest: pd.DataFrame, labeled_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Remove every labeled-cohort patient once, before SSL splitting."""
    labeled_patients = set(labeled_manifest["patient_key"].astype(str))
    keep = ~pretrain_manifest["patient_key"].astype(str).isin(labeled_patients)
    filtered = pretrain_manifest.loc[keep].copy()
    overlap = set(filtered["patient_key"]) & labeled_patients
    if overlap:
        raise RuntimeError(
            f"labeled patients remain in pretraining data: {sorted(overlap)}"
        )
    if filtered.empty:
        raise ValueError(
            "excluding labeled patients removed the entire pretraining cohort"
        )
    return filtered


def _resolve_centerline(centerline_root: Path, dataset: str, exam_id: str) -> Path:
    filename = f"{exam_id}{_CENTERLINE_SUFFIX}"
    candidates = (
        centerline_root / dataset / exam_id / filename,
        centerline_root / exam_id / filename,
    )
    found = [path for path in candidates if path.exists()]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(f"multiple centerline paths resolve for {exam_id}: {found}")
    matches = list(centerline_root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one centerline for {exam_id} under {centerline_root}; "
            f"found {matches}"
        )
    return matches[0]


def build_temporal_exam(
    row: pd.Series,
    *,
    centerline_root: Path,
    dce_root: Path,
) -> object:
    """Build one voxel graph under the final temporal preprocessing contract."""
    exam_id = str(row["exam_id"])
    label = None
    if "pcr" in row.index and pd.notna(row["pcr"]):
        label = int(row["pcr"])
    mask_path = _resolve_centerline(centerline_root, str(row["dataset"]), exam_id)
    data, _ = _build_case(
        exam_id,
        mask_path,
        label,
        dce_root=dce_root,
        node_features=_MODE_DEFAULT_FEATURES[_VOXEL_MODE],
        node_mode=_VOXEL_MODE,
        attach_temporal_contract=True,
        baseline_floor_frac=0.0,
    )
    data.patient_key = str(row["patient_key"])
    data.exam_id = exam_id
    data.dataset = str(row["dataset"])
    if "fold" in row.index and pd.notna(row["fold"]):
        data.fold = int(row["fold"])
    return data


def build_temporal_cache(
    *,
    manifest_path: str | Path,
    centerline_root: str | Path,
    dce_root: str | Path,
    cache_dir: str | Path,
    exclude_labeled_manifest: str | Path | None = None,
    cases: list[str] | None = None,
    resume: bool = False,
    write_manifest: bool = True,
) -> list[TemporalCacheRecord]:
    """Build per-exam graphs and an auditable JSON manifest.

    This function is intentionally sequential; the checked-in Slurm entrypoint
    owns resource allocation, and parallel case building can be added only after
    a small UChicago smoke validates the final contract.
    """
    source = load_exam_manifest(manifest_path)
    if exclude_labeled_manifest is not None:
        labeled = load_exam_manifest(exclude_labeled_manifest)
        source = exclude_labeled_patients(source, labeled)
    if cases is not None:
        requested = set(cases)
        available = set(source["exam_id"])
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                f"requested exam_id values are absent after filtering: {missing}"
            )
        source = source.loc[source["exam_id"].isin(requested)].copy()
    cache_dir = Path(cache_dir)
    graph_dir = cache_dir / "exams"
    graph_dir.mkdir(parents=True, exist_ok=True)
    records: list[TemporalCacheRecord] = []
    for row in source.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        exam_id = str(row_series["exam_id"])
        graph_path = graph_dir / f"{exam_id}.pt"
        if resume and graph_path.exists():
            logging.info("Reusing cached UChicago temporal graph: %s", exam_id)
            data = torch.load(graph_path)
            cached_exam_id = str(getattr(data, "exam_id", ""))
            cached_patient_key = str(getattr(data, "patient_key", ""))
            if cached_exam_id != exam_id or cached_patient_key != str(
                row_series["patient_key"]
            ):
                raise ValueError(
                    f"cached graph identity mismatch for {exam_id}: "
                    f"exam_id={cached_exam_id!r}, patient_key={cached_patient_key!r}"
                )
        else:
            logging.info("Building UChicago temporal graph: %s", exam_id)
            data = build_temporal_exam(
                row_series,
                centerline_root=Path(centerline_root),
                dce_root=Path(dce_root),
            )
            torch.save(data, graph_path)
        times = data.times_seconds.detach().cpu()
        records.append(
            TemporalCacheRecord(
                exam_id=exam_id,
                patient_key=str(row_series["patient_key"]),
                dataset=str(row_series["dataset"]),
                graph_path=str(graph_path),
                n_frames=int(times.numel()),
                duration_seconds=float(times[-1] - times[0]),
                median_dt_seconds=float(torch.median(torch.diff(times))),
                fold=(
                    int(row_series["fold"])
                    if "fold" in row_series and pd.notna(row_series["fold"])
                    else None
                ),
                pcr=(
                    int(row_series["pcr"])
                    if "pcr" in row_series and pd.notna(row_series["pcr"])
                    else None
                ),
            )
        )
    manifest = {
        "source_manifest": str(Path(manifest_path)),
        "excluded_labeled_manifest": (
            str(Path(exclude_labeled_manifest))
            if exclude_labeled_manifest is not None
            else None
        ),
        "requested_cases": sorted(cases) if cases is not None else None,
        "resumed_existing_graphs": resume,
        "centerline_root": str(Path(centerline_root)),
        "dce_root": str(Path(dce_root)),
        "git_commit": _git_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing_contract": {
            "enhancement": "(S-S0_imputed)/S0_imputed",
            "baseline_imputation": "simultaneous_iterative_neighbor_median",
            "validity_threshold": "S0 > 0.05 * median(finite S0)",
            "future_frame_eligibility": False,
            "per_curve_normalization": False,
            "clipping": False,
        },
        "records": [asdict(record) for record in records],
    }
    if write_manifest:
        (cache_dir / "temporal_cache_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    return records


def load_temporal_cache_manifest(cache_dir: str | Path) -> dict[str, object]:
    """Load and minimally validate a temporal cache manifest."""
    path = Path(cache_dir) / "temporal_cache_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"temporal cache manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if not manifest.get("records"):
        raise ValueError(f"temporal cache manifest has no records: {path}")
    return manifest
