#!/usr/bin/env python3
"""make_labels_from_json.py

Helper script (optional) to build a labels.csv from patient-level JSON
metadata, e.g.:

{
  "case_id": "DUKE_001",
  "primary_lesion": {
    "pcr": true,
    "tumor_subtype": "HR+ HER2-"
  }
}

This script walks a directory of JSONs, picks out case_id, pcr,
and tumor_subtype, and writes a labels.csv that the rest of the pipeline can read.
"""

import argparse
import csv
import json
from pathlib import Path


def truthy_to_int(value: object) -> int:
    """Convert a truthy-like value to 0/1.

    Handles booleans, numeric types, and common string representations
    like "yes"/"no", "true"/"false", "1"/"0", and "y"/"n".
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value != 0)
    string = str(value).strip().lower()
    return 1 if string in {"1", "true", "yes", "y"} else 0


def main() -> None:
    """CLI entry point: read JSON metadata and write a labels CSV."""
    ap = argparse.ArgumentParser(
        description="Build labels.csv from patient_info JSON files."
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Folder with patient_info .json files",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV path (labels.csv)",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        msg = f"Input directory does not exist or is not a directory: {input_dir}"
        raise NotADirectoryError(msg)

    rows: list[dict[str, object]] = []
    for json_path in input_dir.iterdir():
        if json_path.suffix != ".json":
            continue

        with json_path.open() as f:
            data = json.load(f)

        pid = data.get("case_id")
        primary_lesion = data.get("primary_lesion", {}) or {}
        pcr_raw = primary_lesion.get("pcr")
        subtype = primary_lesion.get("tumor_subtype")

        if pid is None or pcr_raw is None:
            print(f"[WARN] skipping {json_path.name}: missing case_id or pcr")
            continue

        rows.append(
            {
                "case_id": pid,
                "pcr": truthy_to_int(pcr_raw),
                "subtype": subtype,
            }
        )

    rows.sort(key=lambda row: str(row["case_id"]))

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "pcr", "subtype"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
