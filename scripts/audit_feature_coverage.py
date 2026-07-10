#!/usr/bin/env python3
"""Audit per-case tabular feature artifact coverage across datasets (Issue #150).

For every centerline study directory under ``centerline_root``, this script
records whether each upstream artifact required by the tabular pipeline is
present:

- pCR label (from ``pcr_labels.csv``)
- centerline ``.npy`` skeleton file
- tumor mask ``.nii.gz``
- patient-info JSON ``{case_id}.json`` used for clinical metadata
- ``*_morphometry.json``
- ``*_tumor_graph_features.json``
- the ``status == "ok"`` field inside the tumor-graph JSON

A per-case CSV (one row per ``case_id``) is written to ``--out-csv``, and a
compact per-dataset summary is printed to stdout and written to
``--out-summary`` if provided.

The script is read-only and does not depend on any config. Paths default to
the same locations used by ``configs/ispy2.yaml``, but can be overridden.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CENTERLINE_ROOT = Path("/net/projects2/vanguard/centerlines_tc4d/studies")
DEFAULT_TUMOR_MASK_ROOT = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/segmentations/expert"
)
DEFAULT_LABELS_CSV = Path("/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/pcr_labels.csv")
DEFAULT_PATIENT_INFO_DIR = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/patient_info_files"
)
DEFAULT_RADIOMICS_LABELS_CSV = Path("radiomics/labels.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--centerline-root",
        type=Path,
        default=DEFAULT_CENTERLINE_ROOT,
        help="Root containing one subdir per dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--tumor-mask-root",
        type=Path,
        default=DEFAULT_TUMOR_MASK_ROOT,
        help="Root containing tumor masks named '{case_id}.nii.gz' (default: %(default)s).",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="CSV mapping case ids to pCR labels (default: %(default)s).",
    )
    parser.add_argument(
        "--patient-info-dir",
        type=Path,
        default=DEFAULT_PATIENT_INFO_DIR,
        help=(
            "Directory containing clinical metadata JSONs named '{case_id}.json' "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--radiomics-labels-csv",
        type=Path,
        default=DEFAULT_RADIOMICS_LABELS_CSV,
        help="Repo-tracked radiomics labels file for cross-check (default: %(default)s).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/feature_coverage_audit.csv"),
        help="Per-case audit CSV output path (default: %(default)s).",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("results/feature_coverage_summary.csv"),
        help="Per-dataset summary CSV output path (default: %(default)s).",
    )
    parser.add_argument(
        "--centerline-pattern",
        default="{case_id}_skeleton_4d_exam_mask.npy",
        help="Filename pattern for the centerline .npy (default: %(default)s).",
    )
    return parser.parse_args()


def _load_label_index(path: Path) -> dict[str, int]:
    """Load a {case_id: label} dict from a pcr labels CSV.

    Handles common id-column variants (``case_id``, ``patient_id``).
    """
    if not path.exists():
        logging.warning("Labels CSV not found: %s", path)
        return {}
    labels_df = pd.read_csv(path)
    id_col = next(
        (c for c in ("case_id", "patient_id", "study_id") if c in labels_df.columns),
        None,
    )
    if id_col is None or "pcr" not in labels_df.columns:
        logging.warning(
            "Labels CSV %s missing expected columns (need id col + 'pcr'). Got: %s",
            path,
            list(labels_df.columns),
        )
        return {}
    labels_df = labels_df.dropna(subset=[id_col, "pcr"])
    return {str(r[id_col]): int(r["pcr"]) for _, r in labels_df.iterrows()}


def _read_tumor_graph_status(path: Path) -> tuple[bool, str | None]:
    """Return (exists, status_string_or_None) for a tumor-graph features JSON."""
    if not path.exists():
        return False, None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to parse %s: %s", path, exc)
        return True, "parse_error"
    status = payload.get("status")
    if status is None and isinstance(payload.get("metadata"), dict):
        status = payload["metadata"].get("status")
    return True, str(status) if status is not None else "unknown"


def audit_one_study(
    *,
    dataset_name: str,
    study_dir: Path,
    case_id: str,
    centerline_pattern: str,
    tumor_mask_root: Path,
    patient_info_dir: Path,
    label_index: dict[str, int],
    radiomics_label_index: dict[str, int],
) -> dict[str, Any]:
    """Compute the audit row for a single study directory."""
    centerline_path = study_dir / centerline_pattern.format(case_id=case_id)
    morphometry_path = study_dir / f"{case_id}_morphometry.json"
    tumor_graph_path = study_dir / f"{case_id}_tumor_graph_features.json"
    tumor_mask_path = tumor_mask_root / f"{case_id}.nii.gz"
    patient_info_path = patient_info_dir / f"{case_id}.json"

    tumor_graph_exists, tumor_graph_status = _read_tumor_graph_status(tumor_graph_path)

    row: dict[str, Any] = {
        "case_id": case_id,
        "dataset": dataset_name,
        "has_centerline_npy": centerline_path.exists(),
        "has_morphometry_json": morphometry_path.exists(),
        "has_tumor_graph_json": tumor_graph_exists,
        "tumor_graph_status": tumor_graph_status,
        "has_tumor_mask": tumor_mask_path.exists(),
        "has_patient_info_json": patient_info_path.exists(),
        "has_pcr_label": case_id in label_index,
        "pcr": label_index.get(case_id),
        "has_radiomics_label": case_id in radiomics_label_index,
    }
    row["has_all_artifacts"] = bool(
        row["has_centerline_npy"]
        and row["has_morphometry_json"]
        and row["has_tumor_graph_json"]
        and (row["tumor_graph_status"] == "ok")
        and row["has_tumor_mask"]
        and row["has_patient_info_json"]
        and row["has_pcr_label"]
    )
    return row


def main() -> None:
    """Walk every study dir, write per-case + per-dataset audit tables."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    label_index = _load_label_index(args.labels_csv)
    radiomics_label_index = _load_label_index(args.radiomics_labels_csv)

    logging.info(
        "Loaded %d pCR labels from %s; %d from %s",
        len(label_index),
        args.labels_csv,
        len(radiomics_label_index),
        args.radiomics_labels_csv,
    )

    centerline_root = args.centerline_root
    if not centerline_root.exists():
        raise SystemExit(f"Centerline root does not exist: {centerline_root}")

    rows: list[dict[str, Any]] = []
    for dataset_dir in sorted(centerline_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        study_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
        logging.info(
            "Auditing %s: %d study directories under %s",
            dataset_name,
            len(study_dirs),
            dataset_dir,
        )
        for study_dir in study_dirs:
            rows.append(
                audit_one_study(
                    dataset_name=dataset_name,
                    study_dir=study_dir,
                    case_id=study_dir.name,
                    centerline_pattern=args.centerline_pattern,
                    tumor_mask_root=args.tumor_mask_root,
                    patient_info_dir=args.patient_info_dir,
                    label_index=label_index,
                    radiomics_label_index=radiomics_label_index,
                )
            )

    audit_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(args.out_csv, index=False)
    logging.info("Wrote per-case audit: %s (rows=%d)", args.out_csv, len(audit_df))

    summary_records: list[dict[str, Any]] = []
    for dataset_name, sub in audit_df.groupby("dataset", dropna=False):
        n = len(sub)
        summary_records.append(
            {
                "dataset": dataset_name,
                "n_study_dirs": n,
                "n_centerline_npy": int(sub["has_centerline_npy"].sum()),
                "n_morphometry_json": int(sub["has_morphometry_json"].sum()),
                "n_tumor_graph_json": int(sub["has_tumor_graph_json"].sum()),
                "n_tumor_graph_status_ok": int(
                    (sub["tumor_graph_status"] == "ok").sum()
                ),
                "n_tumor_mask": int(sub["has_tumor_mask"].sum()),
                "n_patient_info_json": int(sub["has_patient_info_json"].sum()),
                "n_pcr_label": int(sub["has_pcr_label"].sum()),
                "n_radiomics_label": int(sub["has_radiomics_label"].sum()),
                "n_all_artifacts": int(sub["has_all_artifacts"].sum()),
            }
        )
    summary_df = pd.DataFrame(summary_records).sort_values("dataset")
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)
    logging.info("Wrote per-dataset summary: %s", args.out_summary)

    print("\n=== Per-dataset feature coverage summary ===")
    print(summary_df.to_string(index=False))

    totals = summary_df.drop(columns=["dataset"]).sum(numeric_only=True)
    totals_line = ", ".join(f"{k}={int(v)}" for k, v in totals.items())
    print(f"\n=== Totals across datasets ===\n{totals_line}")

    missing = audit_df[~audit_df["has_all_artifacts"]]
    if not missing.empty:
        print(f"\n=== Cases missing one or more artifacts ({len(missing)} total) ===")
        print(missing.groupby("dataset").size().to_string())


if __name__ == "__main__":
    main()
