"""Clinical feature loading helpers for the pCR modeling pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _load_clinical_from_patient_info(patient_info_dir: Path) -> pd.DataFrame:
    """Load case-level clinical/imaging metadata from patient_info JSON files."""
    rows = []
    for fp in sorted(patient_info_dir.glob("*.json")):
        data = json.loads(fp.read_text())
        clinical_data = data.get("clinical_data", {})
        lesion_data = data.get("primary_lesion", {})
        imaging_data = data.get("imaging_data", {})

        rows.append(
            {
                "case_id": data.get("case_id") or data.get("patient_id"),
                "age": clinical_data.get("age"),
                "menopausal_status": clinical_data.get("menopausal_status"),
                "breast_density": clinical_data.get("breast_density"),
                "tumor_subtype": lesion_data.get("tumor_subtype"),
                "dataset": imaging_data.get("dataset"),
                "site": imaging_data.get("site"),
                "bilateral": imaging_data.get("bilateral"),
                "field_strength": imaging_data.get("field_strength"),
                "echo_time": imaging_data.get("echo_time"),
                "repetition_time": imaging_data.get("repetition_time"),
                "scanner_manufacturer": imaging_data.get("scanner_manufacturer"),
                "scanner_model": imaging_data.get("scanner_model"),
            }
        )

    df = pd.DataFrame(rows)
    if "case_id" not in df.columns:
        raise ValueError("patient_info JSONs did not produce a usable case_id column.")
    return df


def _load_clinical_from_excel(excel_path: Path) -> pd.DataFrame:
    """Load clinical metadata from Excel with basic column normalization."""
    df = pd.read_excel(excel_path)
    rename_map = {
        "menopause": "menopausal_status",
        "menopausal status": "menopausal_status",
        "patient_id": "case_id",
        "patientid": "case_id",
        "patient id": "case_id",
    }
    normalized_cols = {c: c.strip().lower() for c in df.columns}
    reverse_lookup = {v: k for k, v in normalized_cols.items()}

    for src, dst in rename_map.items():
        if src in reverse_lookup and dst not in df.columns:
            df = df.rename(columns={reverse_lookup[src]: dst})

    if "case_id" not in df.columns:
        raise ValueError(f"{excel_path} must contain a case_id column.")

    # Keep only columns we actively use in the pipeline if present.
    wanted = [
        "case_id",
        "age",
        "menopausal_status",
        "breast_density",
        "tumor_subtype",
        "dataset",
        "site",
        "bilateral",
        "field_strength",
        "echo_time",
        "repetition_time",
        "scanner_manufacturer",
        "scanner_model",
    ]
    keep = [c for c in wanted if c in df.columns]
    return df[keep].copy()


def get_clinical_features(config: dict) -> pd.DataFrame:
    """Load clinical features from configured sources.

    Preference order:
    1) patient_info JSON directory (data_paths.patient_info_dir)
    2) Excel metadata file (data_paths.clinical_excel)
    """
    data_paths = config.get("data_paths", {})

    patient_info_dir = data_paths.get("patient_info_dir")
    if patient_info_dir:
        patient_info_path = Path(patient_info_dir)
        if patient_info_path.exists():
            return _load_clinical_from_patient_info(patient_info_path)

    clinical_excel = data_paths.get("clinical_excel")
    if clinical_excel:
        excel_path = Path(clinical_excel)
        if excel_path.exists():
            return _load_clinical_from_excel(excel_path)

    msg = (
        "No clinical source found. Set data_paths.patient_info_dir or "
        "data_paths.clinical_excel in your training config, such as configs/ispy2.yaml."
    )
    raise FileNotFoundError(msg)
