"""Shared cohort/config helpers for modeling entrypoints."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config from disk."""
    with Path(config_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_run_output_dir(
    *,
    config: dict[str, Any],
    outdir_override: Path | None = None,
) -> Path:
    """Resolve the output directory for one run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = (
        Path(outdir_override)
        if outdir_override is not None
        else Path(config["experiment_setup"]["base_outdir"])
    )
    run_name = f"{config['experiment_setup']['name']}_{timestamp}"
    return base_out / run_name


def write_config_snapshot(
    *,
    config: dict[str, Any],
    outdir: Path,
    config_source: Path | None = None,
) -> None:
    """Write the config used for a run to the output directory."""
    outdir.mkdir(parents=True, exist_ok=True)
    if config_source is not None:
        outdir.joinpath("config_used.yaml").write_text(
            Path(config_source).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return

    with outdir.joinpath("config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
