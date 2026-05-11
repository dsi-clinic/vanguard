"""Merge sharded Deep Sets manifest parts into one manifest CSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from deepsets_runtime import stage_timer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for merging Deep Sets manifest shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Merge manifest shard CSVs into one sorted manifest file."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    output_dir = args.output_dir
    parts_dir = output_dir / "manifest_parts"
    part_paths = sorted(parts_dir.glob("deepsets_manifest_part_*.csv"))
    if not part_paths:
        raise FileNotFoundError(f"No manifest parts found in {parts_dir}")
    with stage_timer(
        "merge",
        output_dir=output_dir,
        num_parts=len(part_paths),
    ):
        frames = [pd.read_csv(path) for path in part_paths]
        manifest_df = pd.concat(frames, ignore_index=True)
        if "case_id" in manifest_df.columns:
            manifest_df["case_id"] = manifest_df["case_id"].astype(str)
            manifest_df = manifest_df.sort_values("case_id").reset_index(drop=True)
        manifest_path = output_dir / "deepsets_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        logging.info(
            "Merged Deep Sets manifest rows=%d parts=%d path=%s",
            len(manifest_df),
            len(part_paths),
            manifest_path,
        )


if __name__ == "__main__":
    main()
