"""Extract case_id + pcr columns from the MAMA-MIA Excel file and write pcr_labels.csv.

The Excel file uses 'patient_id' as the primary key; configs expect 'case_id'.
Rows with null pcr are dropped (15 cases in the 1,506-row dataset).
"""
import pandas as pd
import sys
from pathlib import Path

EXCEL = "/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx"
OUT = "/ess/scratch/scratch1/t-9sbose/pcr_labels.csv"


def main():
    df = pd.read_excel(EXCEL, usecols=["patient_id", "pcr"])
    n_before = len(df)
    df = df.dropna(subset=["pcr"]).rename(columns={"patient_id": "case_id"})
    df["pcr"] = df["pcr"].astype(int)
    df = df[["case_id", "pcr"]].reset_index(drop=True)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    n_dropped = n_before - len(df)
    print(f"Read {n_before} rows, dropped {n_dropped} null-pcr rows")
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"pcr distribution:\n{df['pcr'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
