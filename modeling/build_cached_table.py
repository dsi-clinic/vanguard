"""Build a cached labeled feature table for ablation-array runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from run_ablation_matrix import _normalize_ablation_arms, _prepare_full_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    args.outdir.mkdir(parents=True, exist_ok=True)
    arms = _normalize_ablation_arms(config)
    _prepare_full_dataset(config, arms, args.outdir)


if __name__ == "__main__":
    main()
