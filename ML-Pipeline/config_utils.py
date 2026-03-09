"""Module for loading and standardizing Vanguard project configurations."""

from pathlib import Path
from typing import Any

import yaml  # type: ignore


def load_pipeline_config(
    config_path: str = "ML-Pipeline/config_pcr.yaml",
) -> tuple[dict[str, Any], Path]:
    """Load configuration and initialize a project output directory.

    This function reads the YAML configuration, creates a timestamped output
    directory, and copies the config file for reproducibility.
    """
    with Path(config_path).open(encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    return config, Path()
