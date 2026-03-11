"""Feature extraction from morphometry JSONs with patient_info metadata join.

Produces features_raw.csv and features_with_metadata.csv for weak-signal diagnostics.
Requires: morphometry JSONs (from batch_process_4d) and patient_info JSONs.

Usage:
    python graph_extraction/build_features_with_metadata.py \
        --morphometry-dir report/4d_morphometry \
        --patient-info-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files \
        --output-dir report
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL_METADATA = Path(
    "/net/projects2/vanguard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.clinic_metadata import load_clinic_metadata_excel  # noqa: E402

# ml_pipeline has hyphen; load via importlib
_spec = importlib.util.spec_from_file_location(
    "pcr_prediction",
    REPO_ROOT / "ml_pipeline" / "pcr_prediction.py",
)
_pcr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pcr)
build_features_from_feature_jsons = _pcr.build_features_from_feature_jsons


def get_patient_id(path: Path, js: dict[str, Any]) -> str:
    """Return patient_id from JSON; fallback to filename stem."""
    pid = js.get("patient_id", path.stem)
    return str(pid) if pid else path.stem


def get_site(js: dict[str, Any]) -> str:
    """Extract imaging site; 'UNKNOWN' if unavailable."""
    site = js.get("imaging_data", {}).get("site", "")
    return str(site).strip().upper() if site else "UNKNOWN"


def get_dataset(js: dict[str, Any]) -> str:
    """Extract dataset name; 'UNKNOWN' if unavailable."""
    ds = js.get("imaging_data", {}).get("dataset", "")
    return str(ds).strip().upper() if ds else "UNKNOWN"


def get_manufacturer(js: dict[str, Any]) -> str:
    """Extract manufacturer; 'UNKNOWN' if unavailable."""
    m = js.get("imaging_data", {}).get("scanner_manufacturer", "")
    return str(m).strip() if m else "UNKNOWN"


def get_model(js: dict[str, Any]) -> str:
    """Extract scanner model; 'UNKNOWN' if unavailable."""
    m = js.get("imaging_data", {}).get("scanner_model", "")
    return str(m).strip() if m else "UNKNOWN"


def get_field_strength(js: dict[str, Any]) -> float | None:
    """Extract field strength (T); None if missing."""
    fs = js.get("imaging_data", {}).get("field_strength", None)
    try:
        return float(fs) if fs is not None else None
    except (TypeError, ValueError):
        return None


def get_pcr(js: dict[str, Any]) -> int | None:
    """Extract pCR label 0/1; None if missing."""
    pcr = js.get("primary_lesion", {}).get("pcr", None)
    if pcr is None:
        return None
    if isinstance(pcr, bool):
        return 1 if pcr else 0
    s = str(pcr).strip().lower()
    if s in ("true", "yes", "1"):
        return 1
    if s in ("false", "no", "0"):
        return 0
    try:
        v = int(float(pcr))
        return 1 if v else 0
    except (TypeError, ValueError):
        return None


def load_excel_laterality(
    excel_path: Path, *, id_col: str = "patient_id"
) -> pd.DataFrame | None:
    """Load bilateral_mri from Excel and map to laterality.

    bilateral_mri: 0 -> unilateral, 1 -> bilateral, NaN/missing -> unknown.
    Returns DataFrame with id_col and 'laterality'; None if file missing/invalid.
    """
    if not excel_path.exists():
        logging.warning("Excel metadata not found: %s", excel_path)
        return None
    try:
        excel_df = load_clinic_metadata_excel(excel_path)
    except Exception as exc:
        # Intentional: log and return None so caller can proceed without this file.
        logging.warning("Failed to load Excel %s: %s", excel_path, exc)
        return None
    if "bilateral_mri" not in excel_df.columns:
        logging.warning("Excel %s has no 'bilateral_mri' column", excel_path)
        return None
    if id_col not in excel_df.columns:
        logging.warning("Excel %s has no '%s' column", excel_path, id_col)
        return None

    def _map(v: object) -> str:
        if pd.isna(v):
            return "unknown"
        try:
            n = int(float(v))
            return "bilateral" if n == 1 else "unilateral"
        except (TypeError, ValueError):
            return "unknown"

    laterality = excel_df["bilateral_mri"].map(_map)
    return pd.DataFrame(
        {id_col: excel_df[id_col].astype(str), "laterality": laterality}
    )


def load_patient_info_metadata(patient_info_dir: Path) -> pd.DataFrame:
    """Load patient_info JSONs into a DataFrame with metadata columns."""
    rows: list[dict[str, Any]] = []
    for p in sorted(patient_info_dir.glob("*.json")):
        try:
            js = json.loads(p.read_text())
        except Exception as exc:
            # Intentional: skip invalid JSON and continue loading other files.
            logging.warning("Skipping invalid JSON %s: %s", p, exc)
            continue
        rows.append(
            {
                "patient_id": get_patient_id(p, js),
                "site": get_site(js),
                "dataset": get_dataset(js),
                "manufacturer": get_manufacturer(js),
                "model": get_model(js),
                "field_strength": get_field_strength(js),
                "pcr": get_pcr(js),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point for feature extraction with metadata join."""
    parser = argparse.ArgumentParser(
        description="Extract morphometry features and join with patient_info metadata."
    )
    parser.add_argument(
        "--morphometry-dir",
        type=Path,
        required=True,
        help="Directory with per-case morphometry JSONs (e.g. from batch_process_4d)",
    )
    parser.add_argument(
        "--patient-info-dir",
        type=Path,
        default=Path("/net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files"),
        help="Directory with patient_info JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report"),
        help="Output directory for features_raw.csv and features_with_metadata.csv",
    )
    parser.add_argument(
        "--excel-metadata",
        type=Path,
        default=DEFAULT_EXCEL_METADATA,
        help="Excel file with bilateral_mri (0=unilateral, 1=bilateral). Default: MAMA-MIA clinical_and_imaging_info.xlsx",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="patient_id",
        help="Excel column for patient/sample ID (default: patient_id)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1.2a: Extract features from morphometry JSONs
    print(f"[features] Loading morphometry from {args.morphometry_dir}")
    feats_df = build_features_from_feature_jsons(args.morphometry_dir)
    raw_path = output_dir / "features_raw.csv"
    feats_df.to_csv(raw_path, index=False)
    print(
        f"[features] features_raw.csv -> {raw_path} ({feats_df.shape[0]} cases, {feats_df.shape[1]} cols)"
    )

    # Phase 1.2b: Load patient_info metadata
    if not args.patient_info_dir.exists():
        print(
            f"[features] WARNING: patient_info dir not found: {args.patient_info_dir}"
        )
        print("[features] Writing features_raw.csv only (no metadata join)")
        return

    print(f"[features] Loading patient_info from {args.patient_info_dir}")
    meta_df = load_patient_info_metadata(args.patient_info_dir)
    print(f"[features] Loaded metadata for {len(meta_df)} patients")

    # Phase 1.2c: Join on case_id = patient_id
    merged = feats_df.merge(
        meta_df,
        left_on="case_id",
        right_on="patient_id",
        how="left",
    )
    # Drop duplicate patient_id if case_id was kept
    if "patient_id" in merged.columns and "case_id" in merged.columns:
        merged = merged.drop(columns=["patient_id"])

    # Phase 1.2d: Merge laterality from Excel (bilateral_mri: 0=unilateral, 1=bilateral)
    lat_df = load_excel_laterality(args.excel_metadata, id_col=args.id_col)
    if lat_df is not None:
        merged = merged.merge(
            lat_df,
            left_on="case_id",
            right_on=args.id_col,
            how="left",
            suffixes=("", "_excel"),
        )
        if args.id_col in merged.columns and args.id_col != "case_id":
            merged = merged.drop(columns=[args.id_col])
        n_with_lat = merged["laterality"].notna().sum()
        n_lat_known = (merged["laterality"].isin(["unilateral", "bilateral"])).sum()
        print(
            f"[features] Laterality from Excel: {n_lat_known} with known value ({n_with_lat} non-null)"
        )

    out_path = output_dir / "features_with_metadata.csv"
    merged.to_csv(out_path, index=False)
    print(f"[features] features_with_metadata.csv -> {out_path}")
    n_with_meta = merged["site"].notna().sum() if "site" in merged.columns else 0
    print(f"[features] Cases with metadata: {n_with_meta}/{len(merged)}")


if __name__ == "__main__":
    main()
