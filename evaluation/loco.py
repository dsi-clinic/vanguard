"""LOCO (Leave-One-Covariate-Out) feature-importance orchestrator.

LOCO is a *retrain-based* feature-importance method: for each covariate, a
model is retrained with that one covariate excluded and its cross-validated
performance is compared against a "full" baseline trained with every
covariate. The performance drop when a covariate is removed is its LOCO
importance score (``delta_mean`` in ``loco_summary.csv``: more negative means
more important). This is distinct from permutation importance, which
perturbs a fixed already-trained model instead of retraining.

This module holds the model-family-agnostic core: the ``LOCOAdapter``
callback contract, the ``run_loco_sweep`` driver that runs one baseline +
N leave-one-out training runs and turns their ``KFoldResults`` into a
comparison table, and the results/README writers. Per-family glue (how to
turn "drop this covariate" into an actual config/dataset for a given model
family) lives in the sibling modules ``evaluation/loco_tabular.py``,
``evaluation/loco_gnn.py``, and ``evaluation/loco_deepsets.py``.

See ``docs/loco_feature_importance.md`` for the full design, including why
this avoids a cache rebuild per covariate for GNN/DeepSets (build the cache
once with the full feature superset, then select a subset at load time).

Usage::

    python -m evaluation.loco --config configs/loco_gnn_junction.yaml --outdir experiments/loco_gnn_junction
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import yaml

from config import ConfigNode, load_config, to_plain_data
from evaluation.types import KFoldResults


@dataclass
class LOCORunResult:
    """One trained run inside a LOCO sweep (the full baseline or a leave-one-out arm)."""

    covariate: str | None  # None for the full-baseline run
    node_mode: str | None  # GNN only; None for tabular/deepsets
    feature_group: str  # "node" or "edge" (edge only meaningful for GNN junction mode)
    kfold_results: KFoldResults
    run_dir: Path


class LOCOAdapter(Protocol):
    """Per-family callback: train + evaluate with one covariate dropped (or none).

    Implementations MUST derive folds via
    ``evaluation.build_splits.create_splits_for_dataframe`` using identical
    split-relevant ``model_params`` (``split_mode``/``split_col``/``n_splits``/
    ``random_state``/``group_col``) across every call within one sweep, so
    fold membership is identical for the baseline and every leave-one-out
    run -- ``run_loco_sweep`` never touches those config keys itself and
    relies on the adapter not to either. See ``evaluation/loco_gnn.py`` and
    ``evaluation/loco_deepsets.py`` for concrete implementations; tabular
    does not implement this Protocol (see its module docstring for why).
    """

    def __call__(
        self,
        *,
        drop_covariate: str | None,
        feature_group: str,
        base_config: ConfigNode,
        run_outdir: Path,
    ) -> KFoldResults: ...


def run_loco_sweep(
    *,
    covariates: list[str],
    adapter: LOCOAdapter,
    base_config: ConfigNode,
    outdir: Path,
    family: str,
    node_mode: str | None = None,
    feature_group: str = "node",
) -> pd.DataFrame:
    """Run one full-baseline + ``len(covariates)`` leave-one-out training runs.

    Writes ``loco_summary.csv``, ``loco_fold_deltas.csv``, ``loco_config_used.yaml``,
    and ``README.md`` under ``outdir`` (per-run evaluator outputs live under
    ``outdir/runs/<covariate or _full_baseline>/``), and returns the summary
    table. See ``docs/loco_feature_importance.md`` for the results schema.
    """
    if not covariates:
        raise ValueError("covariates must be non-empty")

    outdir.mkdir(parents=True, exist_ok=True)
    runs_root = outdir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    logging.info(
        "LOCO sweep: family=%s node_mode=%s feature_group=%s covariates=%s",
        family,
        node_mode,
        feature_group,
        covariates,
    )

    baseline_dir = runs_root / "_full_baseline"
    baseline_kfold = adapter(
        drop_covariate=None,
        feature_group=feature_group,
        base_config=base_config,
        run_outdir=baseline_dir,
    )
    baseline_result = LOCORunResult(
        covariate=None,
        node_mode=node_mode,
        feature_group=feature_group,
        kfold_results=baseline_kfold,
        run_dir=baseline_dir,
    )

    run_results: list[LOCORunResult] = []
    for covariate in covariates:
        logging.info("LOCO run: dropping covariate=%r", covariate)
        run_dir = runs_root / covariate
        kfold_results = adapter(
            drop_covariate=covariate,
            feature_group=feature_group,
            base_config=base_config,
            run_outdir=run_dir,
        )
        run_results.append(
            LOCORunResult(
                covariate=covariate,
                node_mode=node_mode,
                feature_group=feature_group,
                kfold_results=kfold_results,
                run_dir=run_dir,
            )
        )

    summary_df, fold_df = build_loco_tables(
        family=family, baseline=baseline_result, runs=run_results
    )
    summary_df.to_csv(outdir / "loco_summary.csv", index=False)
    fold_df.to_csv(outdir / "loco_fold_deltas.csv", index=False)
    with (outdir / "loco_config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_data(base_config), handle, sort_keys=False)
    write_loco_readme(
        outdir=outdir,
        family=family,
        node_mode=node_mode,
        feature_group=feature_group,
        covariates=covariates,
        summary_df=summary_df,
    )
    logging.info("Wrote LOCO summary to %s", outdir / "loco_summary.csv")
    return summary_df


def build_loco_tables(
    *,
    family: str,
    baseline: LOCORunResult,
    runs: list[LOCORunResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the ``loco_summary``/``loco_fold_deltas`` tables from in-memory results.

    One summary row per (covariate, metric); every metric present in the
    baseline's ``aggregated_metrics`` is reported -- driven by whatever
    ``evaluation.metrics.METRIC_REGISTRY`` produced for this run (not
    hardcoded to AUC), so this stays in sync as metrics are added.
    ``delta_mean = loco_mean - baseline_mean``: negative means performance
    dropped when the covariate was removed, i.e. the covariate was important.
    """
    baseline_agg = baseline.kfold_results.aggregated_metrics
    metric_names = sorted(baseline_agg.keys())

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for run in runs:
        run_agg = run.kfold_results.aggregated_metrics
        for metric in metric_names:
            base_stats = baseline_agg.get(metric, {})
            run_stats = run_agg.get(metric, {})
            base_mean = base_stats.get("mean")
            run_mean = run_stats.get("mean")
            delta_mean = (
                run_mean - base_mean
                if base_mean is not None and run_mean is not None
                else None
            )
            summary_rows.append(
                {
                    "family": family,
                    "node_mode": run.node_mode,
                    "feature_group": run.feature_group,
                    "covariate": run.covariate,
                    "metric": metric,
                    "baseline_mean": base_mean,
                    "baseline_std": base_stats.get("std"),
                    "loco_mean": run_mean,
                    "loco_std": run_stats.get("std"),
                    "delta_mean": delta_mean,
                    "n_splits": run.kfold_results.n_splits,
                }
            )

        n_folds = min(
            len(baseline.kfold_results.fold_metrics),
            len(run.kfold_results.fold_metrics),
        )
        for fold_idx in range(n_folds):
            base_fold = baseline.kfold_results.fold_metrics[fold_idx]
            run_fold = run.kfold_results.fold_metrics[fold_idx]
            for metric in metric_names:
                base_val = base_fold.get(metric)
                run_val = run_fold.get(metric)
                fold_rows.append(
                    {
                        "family": family,
                        "node_mode": run.node_mode,
                        "feature_group": run.feature_group,
                        "covariate": run.covariate,
                        "fold": fold_idx,
                        "metric": metric,
                        "baseline_value": base_val,
                        "loco_value": run_val,
                        "delta": (
                            run_val - base_val
                            if base_val is not None and run_val is not None
                            else None
                        ),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def _git_commit() -> str:
    """Return the current HEAD commit hash, for README provenance."""
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table without a tabulate dependency."""
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


_README_TEMPLATE = """# LOCO results: {family}{node_mode_suffix}

Leave-One-Covariate-Out feature importance (retrain-based). Each
`loco_summary.csv` row's `delta_mean` is (LOCO run's metric mean) - (full
baseline run's metric mean); a more negative delta means removing that
covariate hurt performance more, i.e. the covariate was more important.
See `../../docs/loco_feature_importance.md` for the full method design.

## Rerun

```
python -m evaluation.loco --config {config_path} --outdir {outdir}
```

- family: {family}
- node_mode: {node_mode}
- feature_group: {feature_group}
- covariates swept: {covariates}
- git commit: {git_commit}

## Files

- `loco_summary.csv` -- one row per (covariate, metric): baseline_mean/std, loco_mean/std, delta_mean
- `loco_fold_deltas.csv` -- one row per (covariate, fold, metric), for variance inspection
- `loco_config_used.yaml` -- config snapshot used for this sweep
- `runs/_full_baseline/`, `runs/<covariate>/` -- normal evaluator outputs (predictions.csv, metrics.json, plots/) per run

## Top covariates by AUC delta (most important first)

{top_table}
"""


def write_loco_readme(
    *,
    outdir: Path,
    family: str,
    node_mode: str | None,
    feature_group: str,
    covariates: list[str],
    summary_df: pd.DataFrame,
    config_path: str | Path | None = None,
) -> None:
    """Write ``outdir/README.md`` documenting this LOCO sweep, per repo convention."""
    auc_rows = summary_df[summary_df["metric"] == "auc"].sort_values("delta_mean")
    top_table = _markdown_table(
        auc_rows[["covariate", "baseline_mean", "loco_mean", "delta_mean"]]
    )
    node_mode_suffix = f" ({node_mode}, {feature_group})" if node_mode else ""
    content = _README_TEMPLATE.format(
        family=family,
        node_mode_suffix=node_mode_suffix,
        node_mode=node_mode or "n/a",
        feature_group=feature_group,
        covariates=", ".join(covariates),
        config_path=config_path or "<config used for this run>",
        outdir=str(outdir),
        git_commit=_git_commit(),
        top_table=top_table,
    )
    (outdir / "README.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a LOCO (Leave-One-Covariate-Out) feature-importance sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to a YAML config with a top-level `loco:` block "
            "(family, covariates, and family-specific settings -- see "
            "docs/loco_feature_importance.md)."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to write loco_summary.csv/loco_fold_deltas.csv/README.md/runs/ into.",
    )
    return parser.parse_args()


def main() -> None:
    """Dispatch to the family-specific LOCO driver named in ``config.loco.family``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    if not config.get("loco"):
        raise ValueError(
            f"{args.config} has no top-level `loco:` block; see "
            "docs/loco_feature_importance.md for the expected schema."
        )
    family = str(config.loco.family).strip().lower()
    if family == "tabular":
        from evaluation.loco_tabular import run_tabular_loco

        run_tabular_loco(config, args.outdir)
    elif family == "gnn":
        from evaluation.loco_gnn import run_gnn_loco

        run_gnn_loco(config, args.outdir)
    elif family == "deepsets":
        from evaluation.loco_deepsets import run_deepsets_loco

        run_deepsets_loco(config, args.outdir)
    else:
        raise ValueError(
            f"Unknown loco.family={family!r}; expected one of tabular/gnn/deepsets"
        )


if __name__ == "__main__":
    main()
