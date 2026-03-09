"""Module for loading and standardizing Vanguard project configurations."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config(
    config_path: str = "ML-Pipeline/config_pcr.yaml",
) -> tuple[dict[str, Any], Path]:
    """Load configuration and initialize a project output directory."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = config["experiment_setup"]["name"]
    outdir = Path(config["experiment_setup"]["base_outdir"]) / f"{exp_name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, outdir / "config_used.yaml")

    return config, outdir
