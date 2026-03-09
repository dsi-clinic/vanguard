"""Module for loading and standardizing Vanguard project configurations."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore


def load_pipeline_config(
    config_path: str = "ML-Pipeline/config_pcr.yaml",
) -> tuple[dict[str, Any], Path]:
    """Load configuration and initialize a project output directory.

    This function reads the YAML configuration, creates a timestamped output
    directory, and copies the config file for reproducibility.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A tuple containing the configuration dictionary and the output directory path.
    """
    with Path(config_path).open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name: str = config["experiment_setup"]["name"]
    outdir: Path = (
        Path(config["experiment_setup"]["base_outdir"]) / f"{exp_name}_{timestamp}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, outdir / "config_used.yaml")

    return config, outdir
