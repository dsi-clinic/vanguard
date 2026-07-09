r"""Step 2 validation gate: adapter-on vs. adapter-off feature parity.

Runs the tabular feature build twice on the same config and data — once with no
dataset adapter (today's behavior) and once with the ``MamaMiaDataset`` adapter
selected through the real run-config factory (the same seam ``tabular/train.py``
uses) — then asserts the two ``features_engineered_labeled.csv`` files are
byte-identical. This is the merge gate for Step 2 of the multi-dataset migration
(see ``cohorts/README.md``).

Data paths are injected at runtime rather than committed, because the centerline
outputs currently live only in per-user workspaces (there is no canonical shared
``centerline_root`` yet). Point the script at reachable data via flags or env:

    CENTERLINE_ROOT=/gpfs/.../centerlines_tc4d/studies \\
    MAMAMIA_ROOT=/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042 \\
    python scripts/validate_adapter_feature_parity.py

Exit code 0 = identical (gate passes), 1 = differs (gate fails).

Note on ``--cohort``: ``all_datasets.yaml`` builds one table over all four
cohorts, and the identity method this stage uses (``case_dataset_name``) is
cohort-independent, so the cohort passed to ``MamaMiaDataset`` does not affect
the result here. It is required only to satisfy the factory's constructor.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from cohorts.factory import build_adapter_from_config
from load_cohort import load_config
from tabular.cohort import prepare_data

if TYPE_CHECKING:
    from cohorts.base import DatasetAdapter
    from config import ConfigNode

_OUTPUT_NAME = "features_engineered_labeled.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments, defaulting path inputs to environment variables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/all_datasets.yaml"),
        help="Run config to build features from (default: configs/all_datasets.yaml).",
    )
    parser.add_argument(
        "--centerline-root",
        default=os.environ.get("CENTERLINE_ROOT"),
        help="Override data_paths.centerline_root (or set CENTERLINE_ROOT).",
    )
    parser.add_argument(
        "--mamamia-root",
        default=os.environ.get(
            "MAMAMIA_ROOT", "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042"
        ),
        help="MAMA-MIA root for masks/clinical/labels (or set MAMAMIA_ROOT).",
    )
    parser.add_argument(
        "--cohort",
        default="ispy2",
        help="Cohort for the MamaMiaDataset factory (immaterial here; see module docstring).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/adapter_parity_check"),
        help="Where to write the two output tables for inspection.",
    )
    return parser.parse_args()


def _apply_path_overrides(
    config: ConfigNode, centerline_root: str | None, mamamia_root: str
) -> None:
    """Point the config's data_paths at reachable data (in place)."""
    dp = config.data_paths
    if centerline_root:
        dp["centerline_root"] = str(centerline_root)
    root = Path(mamamia_root)
    dp["tumor_mask_root"] = str(root / "segmentations" / "expert")
    dp["patient_info_dir"] = str(root / "patient_info_files")
    dp["clinical_excel"] = str(root / "clinical_and_imaging_info.xlsx")
    dp["labels_csv"] = str(root / "pcr_labels.csv")


def _build(config: ConfigNode, outdir: Path, adapter: DatasetAdapter | None) -> Path:
    """Run prepare_data into ``outdir`` and return the labeled-table path."""
    outdir.mkdir(parents=True, exist_ok=True)
    prepare_data(config, outdir, adapter=adapter)
    return outdir / _OUTPUT_NAME


def main() -> int:
    """Run both builds and compare the labeled tables byte-for-byte."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()

    off_dir = args.outdir / "adapter_off"
    on_dir = args.outdir / "adapter_on"

    # --- adapter OFF: today's behavior (identity from directory name) ---
    config_off = load_config(args.config)
    _apply_path_overrides(config_off, args.centerline_root, args.mamamia_root)
    logging.info("Building WITHOUT adapter -> %s", off_dir)
    csv_off = _build(config_off, off_dir, adapter=None)

    # --- adapter ON: identity through the real run-config factory ---
    config_on = load_config(args.config)
    _apply_path_overrides(config_on, args.centerline_root, args.mamamia_root)
    config_on.dataset["name"] = "mamamia"
    config_on.dataset["cohort"] = args.cohort
    config_on.dataset["root"] = args.mamamia_root
    adapter = build_adapter_from_config(config_on)
    logging.info("Building WITH adapter (%s) -> %s", type(adapter).__name__, on_dir)
    csv_on = _build(config_on, on_dir, adapter=adapter)

    # --- compare ---
    bytes_off = csv_off.read_bytes()
    bytes_on = csv_on.read_bytes()
    df_off = pd.read_csv(csv_off, low_memory=False)
    df_on = pd.read_csv(csv_on, low_memory=False)

    logging.info("adapter-off table: shape=%s", df_off.shape)
    logging.info("adapter-on  table: shape=%s", df_on.shape)
    logging.info(
        "dataset value_counts (off): %s",
        df_off["dataset"].value_counts().to_dict() if "dataset" in df_off else "n/a",
    )

    if bytes_off == bytes_on:
        print(f"PASS: {_OUTPUT_NAME} is byte-identical with and without the adapter.")
        print(f"      rows={len(df_off)}, cols={df_off.shape[1]}")
        return 0

    print("FAIL: outputs differ.")
    print(f"  off: {csv_off} ({len(bytes_off)} bytes, shape {df_off.shape})")
    print(f"  on : {csv_on} ({len(bytes_on)} bytes, shape {df_on.shape})")
    if df_off.shape == df_on.shape and list(df_off.columns) == list(df_on.columns):
        try:
            pd.testing.assert_frame_equal(df_off, df_on)
        except AssertionError as exc:  # noqa: BLE001
            print("  first cell-level difference:")
            print(f"  {exc}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
