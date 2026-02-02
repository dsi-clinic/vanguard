#!/usr/bin/env python3
"""Run one or more radiomics experiments from YAML config files.

Each experiment config defines all paths, extraction parameters, and training
parameters in one place.  The runner validates the config, runs extraction
(unless outputs already exist), then runs training.  A snapshot of the config
used is written into the output directory so every result is fully
reproducible from its own directory.

Usage
-----
    # Single experiment
    python run_experiment.py configs/exp_peri5_multiphase_logreg.yaml

    # Multiple experiments (shell glob)
    python run_experiment.py configs/exp_*.yaml

    # Re-run extraction even if feature CSVs already exist
    python run_experiment.py configs/exp_foo.yaml --force

    # Print the commands that *would* run without executing them
    python run_experiment.py configs/exp_foo.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Validation constants
_REQUIRED_TOP     = {"experiment_name", "paths", "extract", "train"}
_REQUIRED_PATHS   = {"images", "masks", "labels", "splits", "outdir"}
_REQUIRED_EXTRACT = {"image_patterns", "mask_pattern"}
_VALID_CLASSIFIERS = {"logistic", "rf", "xgb"}


# Validation
def validate_config(cfg: dict[str, Any], label: str = "") -> None:
    """Raise ValueError / FileNotFoundError on any config problem.

    Checks
    ------
    * All required top-level, paths, extract, and train keys are present.
    * Input file/directory paths actually exist on disk.
    * image_patterns is a non-empty list.
    * classifier is one of the supported values.
    """
    prefix = f"[{label}] " if label else ""

    # top-level keys
    for key in _REQUIRED_TOP:
        if key not in cfg:
            msg = f"{prefix}Missing required top-level key: '{key}'"
            raise ValueError(msg)

    # paths
    paths = cfg["paths"]
    for key in _REQUIRED_PATHS:
        if key not in paths:
            msg = f"{prefix}Missing required paths.{key}"
            raise ValueError(msg)

    # Input paths must already exist; outdir / extract_outdir will be created.
    for key in ("images", "masks", "labels", "splits"):
        p = Path(paths[key])
        if not p.exists():
            msg = f"{prefix}paths.{key} does not exist: {p}"
            raise FileNotFoundError(msg)

    if paths.get("params_yaml"):
        p = Path(paths["params_yaml"])
        if not p.exists():
            msg = f"{prefix}paths.params_yaml does not exist: {p}"
            raise FileNotFoundError(msg)

    # extract
    extract = cfg["extract"]
    for key in _REQUIRED_EXTRACT:
        if key not in extract:
            msg = f"{prefix}Missing required extract.{key}"
            raise ValueError(msg)
    if not isinstance(extract["image_patterns"], list) or not extract["image_patterns"]:
        msg = f"{prefix}extract.image_patterns must be a non-empty list"
        raise ValueError(msg)

    # train
    train = cfg["train"]
    if "classifier" not in train:
        msg = f"{prefix}Missing required train.classifier"
        raise ValueError(msg)
    if train["classifier"] not in _VALID_CLASSIFIERS:
        msg = (
            f"{prefix}train.classifier must be one of "
            f"{sorted(_VALID_CLASSIFIERS)}, got '{train['classifier']}'"
        )
        raise ValueError(msg)


# Directory helpers

def get_extract_outdir(cfg: dict[str, Any]) -> Path:
    """Return the extraction output directory.

    If ``paths.extract_outdir`` is explicitly set (the ablation runner does
    this so that experiments sharing extraction parameters reuse the same
    output), that path is used.  Otherwise it defaults to ``{outdir}/extraction``.
    """
    explicit = cfg["paths"].get("extract_outdir")
    if explicit:
        return Path(explicit)
    return Path(cfg["paths"]["outdir"]) / "extraction"


def extraction_outputs_exist(cfg: dict[str, Any]) -> bool:
    """True when both final feature CSVs are already on disk."""
    d = get_extract_outdir(cfg)
    return (
        (d / "features_train_final.csv").exists()
        and (d / "features_test_final.csv").exists()
    )



# CLI command builders
def build_extract_cmd(cfg: dict[str, Any], scripts_dir: Path) -> list[str]:
    """Translate the extraction section of a config into a
    radiomics_extract.py command-line invocation.
    """  # noqa: D205
    paths   = cfg["paths"]
    extract = cfg["extract"]

    cmd = [
        sys.executable,
        str(scripts_dir / "radiomics_extract.py"),
        "--images",         paths["images"],
        "--masks",          paths["masks"],
        "--labels",         paths["labels"],
        "--splits",         paths["splits"],
        "--output",         str(get_extract_outdir(cfg)),
        "--image-pattern",  ",".join(extract["image_patterns"]),
        "--mask-pattern",   extract["mask_pattern"],
        "--peri-radius-mm", str(extract.get("peri_radius_mm", 0)),
        "--n-jobs",         str(extract.get("n_jobs", 1)),
    ]

    if paths.get("params_yaml"):
        cmd.extend(["--params", paths["params_yaml"]])
    if extract.get("label_override") is not None:
        cmd.extend(["--label-override", str(extract["label_override"])])

    # Non-scalar vector handling
    if extract.get("non_scalar_handling"):
        cmd.extend(["--non-scalar-handling", extract["non_scalar_handling"]])
    if extract.get("aggregate_stats"):
        cmd.extend(["--aggregate-stats", ",".join(extract["aggregate_stats"])])
    if extract.get("hybrid_concat_threshold") is not None:
        cmd.extend(["--hybrid-concat-threshold", str(extract["hybrid_concat_threshold"])])

    return cmd


def build_train_cmd(cfg: dict[str, Any], scripts_dir: Path) -> list[str]:
    """Translate the training section of a config into a
    ``radiomics_train.py`` command-line invocation.

    Train/test feature paths are derived from the extraction output directory
    so they stay consistent with wherever extraction wrote its CSVs.
    """  # noqa: D205
    paths       = cfg["paths"]
    train       = cfg["train"]
    extract_out = get_extract_outdir(cfg)
    train_out   = Path(paths["outdir"]) / "training"

    cmd = [
        sys.executable,
        str(scripts_dir / "radiomics_train.py"),
        "--train-features", str(extract_out / "features_train_final.csv"),
        "--test-features",  str(extract_out / "features_test_final.csv"),
        "--labels",         paths["labels"],
        "--output",         str(train_out),
        "--classifier",     train["classifier"],
    ]

    # classifier-specific flags
    if train["classifier"] == "logistic":
        for key, flag in (
            ("logreg_penalty",  "--logreg-penalty"),
            ("logreg_l1_ratio", "--logreg-l1-ratio"),
            ("logreg_C",        "--logreg-C"),
        ):
            if key in train:
                cmd.extend([flag, str(train[key])])

    elif train["classifier"] == "rf":
        for key, flag in (
            ("rf_n_estimators",      "--rf-n-estimators"),
            ("rf_max_depth",         "--rf-max-depth"),
            ("rf_min_samples_leaf",  "--rf-min-samples-leaf"),
            ("rf_min_samples_split", "--rf-min-samples-split"),
            ("rf_max_features",      "--rf-max-features"),
            ("rf_ccp_alpha",         "--rf-ccp-alpha"),
        ):
            if key in train:
                cmd.extend([flag, str(train[key])])

    elif train["classifier"] == "xgb":
        for key, flag in (
            ("xgb_n_estimators",     "--xgb-n-estimators"),
            ("xgb_max_depth",        "--xgb-max-depth"),
            ("xgb_learning_rate",    "--xgb-learning-rate"),
            ("xgb_subsample",        "--xgb-subsample"),
            ("xgb_colsample_bytree", "--xgb-colsample-bytree"),
            ("xgb_reg_lambda",       "--xgb-reg-lambda"),
            ("xgb_reg_alpha",        "--xgb-reg-alpha"),
            ("xgb_scale_pos_weight", "--xgb-scale-pos-weight"),
        ):
            if key in train:
                cmd.extend([flag, str(train[key])])

    # shared flags
    if train.get("corr_threshold"):
        cmd.extend(["--corr-threshold", str(train["corr_threshold"])])
    if train.get("k_best"):
        cmd.extend(["--k-best", str(train["k_best"])])
    if train.get("grid_search"):
        cmd.append("--grid-search")
    if train.get("cv_folds"):
        cmd.extend(["--cv-folds", str(train["cv_folds"])])
    if train.get("include_subtype"):
        cmd.append("--include-subtype")

    return cmd


# Execution helpers
def run_cmd(cmd: list[str], label: str, *, dry_run: bool) -> int:
    """Pretty-print a command, then run it (unless *dry_run* is set).

    Returns the subprocess exit code (0 on success), or 0 for dry runs.
    """
    print(f"\n{'=' * 60}")
    print(f"[{label}]  command:")
    print("  " + " \\\n    ".join(cmd))
    print("=" * 60)

    if dry_run:
        print("[DRY-RUN] Skipping execution.\n")
        return 0

    start  = time.time()
    result = subprocess.run(cmd, check=False)  # noqa: S603
    elapsed = time.time() - start
    print(f"[{label}] Finished in {elapsed:.1f}s — exit code {result.returncode}\n")
    return result.returncode



# Single-experiment orchestrator (also used by run_ablations.py)
def run_single_experiment(
    config_path: str,
    scripts_dir: Path,
    *,
    force:   bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load, validate, and execute one experiment end-to-end.

    Returns a summary dict that includes status and (when available) the key
    metrics from ``metrics.json``.  This function is the shared core used by
    both the CLI entry point and the ablation runner.
    """
    print(f"\n{'#' * 60}")
    print(f"#  {config_path}")
    print(f"{'#' * 60}")

    with open(config_path) as fh:  # noqa: PTH123
        cfg = yaml.safe_load(fh)

    validate_config(cfg, label=config_path)

    exp_name = cfg["experiment_name"]
    outdir   = Path(cfg["paths"]["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # Snapshot the exact config that produced this run
    if not dry_run:
        shutil.copy(config_path, outdir / "run_config.yaml")

    result: dict[str, Any] = {
        "experiment_name": exp_name,
        "config_path":     config_path,
        "status":          "success",
        "extract_skipped": False,
    }

    # extraction
    if not force and extraction_outputs_exist(cfg):
        print(
            f"[EXTRACT] Outputs already present in {get_extract_outdir(cfg)}. "
            "Skipping.  (pass --force to re-run)"
        )
        result["extract_skipped"] = True
    else:
        rc = run_cmd(build_extract_cmd(cfg, scripts_dir), "EXTRACT", dry_run=dry_run)
        if rc != 0:
            result["status"] = "extract_failed"
            return result

    # training
    rc = run_cmd(build_train_cmd(cfg, scripts_dir), "TRAIN", dry_run=dry_run)
    if rc != 0:
        result["status"] = "train_failed"
        return result

    # pull key metrics
    if not dry_run:
        metrics_path = outdir / "training" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as fh:  # noqa: PTH123
                metrics = json.load(fh)
            for key in ("auc_test", "auc_train", "auc_train_cv", "n_features_used"):
                result[key] = metrics.get(key)

    return result


# CLI entry point
def main() -> None:
    """Parse CLI arguments and run one or more experiments."""
    ap = argparse.ArgumentParser(
        description="Run radiomics experiments defined by YAML config files.",
    )
    ap.add_argument(
        "configs",
        nargs="+",
        help=(
            "One or more YAML config file paths.  "
            "Shell globs (e.g. configs/exp_*.yaml) are expanded by your shell."
        ),
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run extraction even if feature CSVs already exist.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )

    args       = ap.parse_args()
    scripts_dir = Path(__file__).resolve().parent

    results: list[dict[str, Any]] = []
    for cfg_path in args.configs:
        try:
            results.append(
                run_single_experiment(
                    cfg_path, scripts_dir, force=args.force, dry_run=args.dry_run,
                ),
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"\n[ERROR] {exc}")
            results.append({
                "experiment_name": cfg_path,
                "status": "validation_error",
                "error": str(exc),
            })

    # summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        auc = f"  auc_test={r['auc_test']}" if r.get("auc_test") else ""
        print(f"  {r.get('experiment_name', '?'):50s} [{r['status']}]{auc}")


if __name__ == "__main__":
    main()