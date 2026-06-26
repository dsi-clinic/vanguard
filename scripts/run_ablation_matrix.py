"""Run a reproducible pCR feature-block ablation matrix."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from modeling.ablation import run_ablation_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run feature-block ablation matrix.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to base YAML config with optional ablation_arms block.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Root output directory for shared features and per-arm runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    run_ablation_matrix(config, args.outdir)


if __name__ == "__main__":
    main()
