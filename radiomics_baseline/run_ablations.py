#!/usr/bin/env python3
"""Generate and run a grid of radiomics ablation experiments.

Reads a "sweep" YAML that names a base experiment config and declares a
parameter grid.  Every combination in the grid becomes its own config file
written into a ``configs/generated/`` directory.  The runner is smart about
extraction: configs that share identical extraction parameters are pointed at
the same extraction output directory, so feature extraction only runs once per
unique extraction setup.  After all experiments finish, a summary CSV is
written next to the sweep YAML.

Usage
-----
    # Full sweep
    python run_ablations.py configs/sweep_example.yaml

    # Print generated configs + commands without executing
    python run_ablations.py configs/sweep_example.yaml --dry-run

    # Force re-extraction even when outputs exist
    python run_ablations.py configs/sweep_example.yaml --force

    # Write generated configs somewhere else
    python run_ablations.py configs/sweep_example.yaml --generated-dir my_sweep
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Make sure run_experiment is importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import run_single_experiment  # noqa: E402

# Keys that fully determine the extraction output.  Two configs whose values
# for these keys are identical can safely share one extraction directory.
_EXTRACT_FINGERPRINT_KEYS = (
    "extract.image_patterns",
    "extract.mask_pattern",
    "extract.peri_radius_mm",
    "extract.n_jobs",
    "extract.label_override",
    "paths.params_yaml",
)


# Nested-dict helpers
def deep_get(d: dict, dotted_key: str, default: Any = None) -> Any:  # noqa: ANN401
    """Retrieve a value from a nested dict via a dot-separated key path.

    Example: ``deep_get(cfg, "extract.peri_radius_mm")``
    returns ``cfg["extract"]["peri_radius_mm"]``.
    """
    node = d
    for part in dotted_key.split("."):
        if isinstance(node, dict):
            node = node.get(part, default)
        else:
            return default
    return node


def deep_set(d: dict, dotted_key: str, value: Any) -> None:  # noqa: ANN401
    """Set a value inside a nested dict via a dot-separated key path,
    creating intermediate dicts as needed.

    Example: ``deep_set(cfg, "train.classifier", "rf")``
    sets ``cfg["train"]["classifier"] = "rf"``.
    """  # noqa: D205
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


# Extraction-sharing logic
def extraction_fingerprint(cfg: dict[str, Any]) -> str:
    """Return a stable string that uniquely identifies the extraction portion
    of a config.  Configs with the same fingerprint can share one extraction
    output directory.
    """  # noqa: D205
    parts = []
    for key in _EXTRACT_FINGERPRINT_KEYS:
        val = deep_get(cfg, key, "")
        parts.append(f"{key}={val!r}")
    return "|".join(parts)


def _value_label(value: Any) -> str:  # noqa: ANN401
    """Short, filesystem-safe label for a single sweep value.

    * Lists become their length (e.g. a 2-element image_patterns → ``2``).
    * Everything else is stringified and stripped of non-alphanumeric chars
      (except underscores, hyphens, and dots).
    """
    if isinstance(value, list):
        return str(len(value))
    return re.sub(r"[^\w.\-]", "", str(value))


# Config generation
def generate_configs(
    sweep: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, str]]]:
    """Expand a sweep definition into concrete (config, param_values) pairs.

    Steps
    -----
    1. Load the base config YAML referenced by ``sweep["base_config"]``.
    2. Compute the Cartesian product of every list in ``sweep["sweep"]``.
    3. For each combination, deep-copy the base config, apply the overrides,
       and generate a unique ``experiment_name`` and ``outdir``.

    Returns a list of ``(config_dict, param_values_dict)`` tuples where
    ``param_values_dict`` maps each sweep key to its string value (used later
    for the summary CSV columns).
    """
    with open(sweep["base_config"]) as fh:  # noqa: PTH123
        base_cfg = yaml.safe_load(fh)

    param_keys = list(sweep["sweep"].keys())
    value_lists = [sweep["sweep"][k] for k in param_keys]
    base_name = base_cfg.get("experiment_name", "exp")
    base_outdir = Path(base_cfg["paths"]["outdir"])

    configs: list[tuple[dict[str, Any], dict[str, str]]] = []

    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(base_cfg)
        label_parts: list[str] = []

        for key, value in zip(param_keys, combo):
            deep_set(cfg, key, value)
            short = key.split(".")[-1]  # e.g. "peri_radius_mm"
            label_parts.append(f"{short}-{_value_label(value)}")

        # Unique name and output directory derived from the base + overrides
        cfg["experiment_name"] = f"{base_name}_{'_'.join(label_parts)}"
        cfg["paths"]["outdir"] = str(base_outdir.parent / cfg["experiment_name"])

        param_values = {k: str(v) for k, v in zip(param_keys, combo)}
        configs.append((cfg, param_values))

    return configs


def assign_shared_extract_outdirs(
    configs: list[tuple[dict[str, Any], dict[str, str]]],
    base_outdir: Path,
) -> None:
    """Group configs by extraction fingerprint and point each group at the
    same ``extract_outdir``.

    The shared directory is named after the *first* experiment in each group
    so the path is human-readable.  This means extraction only runs once per
    unique set of extraction parameters across the entire sweep.
    """  # noqa: D205
    fingerprint_to_dir: dict[str, str] = {}

    for cfg, _ in configs:
        fp = extraction_fingerprint(cfg)
        if fp not in fingerprint_to_dir:
            fingerprint_to_dir[fp] = str(
                base_outdir / "shared_extraction" / cfg["experiment_name"]
            )
        cfg["paths"]["extract_outdir"] = fingerprint_to_dir[fp]


# CLI entry point
def main() -> None:
    """Parse sweep YAML, generate configs, run experiments, write summary."""
    ap = argparse.ArgumentParser(
        description="Run an ablation sweep defined by a sweep YAML.",
    )
    ap.add_argument(
        "sweep_config",
        help="Path to the sweep YAML file.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run extraction even if feature CSVs already exist.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print commands without executing.",
    )
    ap.add_argument(
        "--generated-dir",
        default="configs/generated",
        help=(
            "Directory to write generated per-experiment configs "
            "(default: configs/generated)."
        ),
    )

    args = ap.parse_args()
    scripts_dir = Path(__file__).resolve().parent

    # load sweep
    with open(args.sweep_config) as fh:  # noqa: PTH123
        sweep = yaml.safe_load(fh)

    if "base_config" not in sweep:
        msg = f"[{args.sweep_config}] Missing 'base_config' key."
        raise ValueError(msg)
    if not sweep.get("sweep"):
        msg = f"[{args.sweep_config}] Missing or empty 'sweep' key."
        raise ValueError(msg)

    # generate configs
    configs = generate_configs(sweep)

    # Point configs with identical extraction params at a shared extract dir
    with open(sweep["base_config"]) as fh:  # noqa: PTH123
        base_cfg = yaml.safe_load(fh)
    base_outdir = Path(base_cfg["paths"]["outdir"]).parent
    assign_shared_extract_outdirs(configs, base_outdir)

    # Write every generated config to disk (committed to git for reproducibility)
    gen_dir = Path(args.generated_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ABLATION] Generated {len(configs)} configs → {gen_dir}/\n")

    config_paths: list[str] = []
    for cfg, _ in configs:
        path = gen_dir / f"exp_{cfg['experiment_name']}.yaml"
        with open(path, "w") as fh:  # noqa: PTH123
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
        config_paths.append(str(path))
        print(f"  {path}")

    # run all experiments
    results: list[dict[str, Any]] = []
    for cfg_path, (_, param_values) in zip(config_paths, configs):
        try:
            result = run_single_experiment(
                cfg_path,
                scripts_dir,
                force=args.force,
                dry_run=args.dry_run,
            )
            result["sweep_params"] = param_values
            results.append(result)
        except (ValueError, FileNotFoundError) as exc:
            print(f"\n[ERROR] {exc}")
            results.append(
                {
                    "experiment_name": cfg_path,
                    "status": "validation_error",
                    "error": str(exc),
                    "sweep_params": param_values,
                }
            )

    # write summary CSV
    if results and not args.dry_run:
        all_param_keys = sorted({k for r in results for k in r.get("sweep_params", {})})
        metric_keys = [
            "auc_test",
            "auc_train",
            "auc_train_cv",
            "n_features_used",
        ]
        fieldnames = ["experiment_name", "status"] + all_param_keys + metric_keys

        summary_path = Path(args.sweep_config).parent / "ablation_summary.csv"
        with open(summary_path, "w", newline="") as fh:  # noqa: PTH123
            writer = csv.DictWriter(
                fh,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            for r in results:
                row: dict[str, Any] = {
                    "experiment_name": r.get("experiment_name", ""),
                    "status": r.get("status", ""),
                }
                for k in all_param_keys:
                    row[k] = r.get("sweep_params", {}).get(k, "")
                for k in metric_keys:
                    row[k] = r.get(k, "")
                writer.writerow(row)

        print(f"\n[ABLATION] Summary CSV → {summary_path}")

    # print summary table
    print(f"\n{'=' * 60}")
    print("ABLATION SUMMARY")
    print("=" * 60)
    for r in results:
        auc = f"  auc_test={r['auc_test']}" if r.get("auc_test") else ""
        print(f"  {r.get('experiment_name', '?'):60s} [{r['status']}]{auc}")


if __name__ == "__main__":
    main()
