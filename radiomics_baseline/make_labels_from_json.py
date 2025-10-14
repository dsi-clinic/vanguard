#!/usr/bin/env python3
import argparse, json, os, csv

def truthy_to_int(v):
    # handles True/False, "yes"/"no", "1"/"0"
    if isinstance(v, bool): return int(v)
    if isinstance(v, (int, float)): return int(v != 0)
    s = str(v).strip().lower()
    return 1 if s in {"1","true","yes","y"} else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder with patient_info .json files")
    ap.add_argument("--out", required=True, help="Output CSV path (labels.csv)")
    args = ap.parse_args()

    rows = []
    for fname in os.listdir(args.input_dir):
        if not fname.endswith(".json"): continue
        with open(os.path.join(args.input_dir, fname), "r") as f:
            data = json.load(f)
        pid = data.get("patient_id")
        pcr_raw = data.get("primary_lesion", {}).get("pcr", None)
        subtype = data.get("primary_lesion", {}).get("tumor_subtype", None)
        if pid is None or pcr_raw is None:
            print(f"[WARN] skipping {fname}: missing patient_id or pcr")
            continue
        rows.append({"patient_id": pid, "pcr": truthy_to_int(pcr_raw), "subtype": subtype})

    rows.sort(key=lambda r: r["patient_id"])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id","pcr","subtype"])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
