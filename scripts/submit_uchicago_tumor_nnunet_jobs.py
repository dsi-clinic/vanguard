#!/usr/bin/env python3
"""Submit MAMA-MIA nnU-Net tumor inference jobs for a prepared UChicago run."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

DEFAULT_SLURM = Path("slurm/submit_nnunet_tumor_inference.slurm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--slurm-script", type=Path, default=DEFAULT_SLURM)
    parser.add_argument("--jobs-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    jobs_csv = args.jobs_csv or run_root / "tumor_inference_slurm_jobs.csv"
    cases = pd.read_csv(run_root / "tumor_case_manifest.csv")

    rows = []
    for _, row in cases.iterrows():
        input_dir = Path(row["input_dir"]).resolve()
        output_dir = Path(row["output_dir"]).resolve()
        cmd = [
            "sbatch",
            "--parsable",
            "--export=ALL,"
            f"INPUT_DIR={input_dir},"
            f"OUTPUT_DIR={output_dir}",
            str(args.slurm_script),
        ]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        out_row = dict(row)
        out_row["job_id"] = job_id
        out_row["slurm_script"] = str(args.slurm_script)
        rows.append(out_row)
        print(f"{job_id}\t{row['dataset']}\t{row['exam_id']}")

    pd.DataFrame(rows).to_csv(jobs_csv, index=False)
    print(f"wrote {jobs_csv}")


if __name__ == "__main__":
    main()
