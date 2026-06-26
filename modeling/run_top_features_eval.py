"""Issue #151 entrypoint: baseline vs globally-ranked top-K vessel features.

This script:
1. Materializes the labeled feature superset once (same pathway as
   ``run_ablation_matrix.py``).
2. Runs a **global** univariate / tree / linear screening ranker on the pool.
3. Writes ``feature_ranking_global.csv`` plus a YAML snapshot of generated arms.
4. Delegates training to ``run_ablation_matrix.run_ablation_matrix``.

The ranking step uses all labeled rows together (fast screening default). It is
not safe to treat p-values from the downstream CV as if the feature list were
chosen inside each outer fold; use the ranking table only for exploratory
comparison or follow up with nested / per-fold ranking if you need strict ML
inferential hygiene.
"""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.feature_ranking import RankingMethod  # noqa: E402
from analysis.top_k_arms import assemble_top_features_experiment_arms  # noqa: E402
from config import load_config, to_plain_data  # noqa: E402
from modeling.ablation import _prepare_full_dataset, run_ablation_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Top-K feature screening vs baseline (Issue #151)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML with top_features_eval block (see configs/top_features_eval.yaml).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Experiment directory for features, ranks, and ablation runs.",
    )
    return parser.parse_args()


def _coerce_ranking_method(raw: object) -> RankingMethod:
    method = str(raw or "univariate_auc").strip().lower()
    if method in ("univariate_auc", "xgb_gain", "lr_abs_coef"):
        return cast(RankingMethod, method)
    raise ValueError(
        f"Unsupported ranking_method {raw!r}. Use 'univariate_auc', "
        "'xgb_gain', or 'lr_abs_coef'."
    )


def main() -> None:
    """Generate ranked top-K arms and run the ablation matrix."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    base_config = load_config(args.config)
    tfe = base_config.get("top_features_eval")
    if not tfe or not tfe.get("enabled"):
        raise SystemExit(
            "Set top_features_eval.enabled: true in the config "
            "(see configs/top_features_eval.yaml)."
        )

    config = deepcopy(base_config)
    label_col = str(config.data_paths.label_column)

    baseline_blocks = list(tfe["baseline_blocks"])
    pool_blocks = list(tfe["ranking_pool_blocks"])
    k_values = [int(k) for k in tfe.get("k_values", [5, 10, 20])]
    method = _coerce_ranking_method(tfe.get("ranking_method", "univariate_auc"))

    ranker_kwargs = tfe.get("ranker_kwargs")
    if ranker_kwargs is not None and not isinstance(ranker_kwargs, dict):
        raise ValueError("top_features_eval.ranker_kwargs must be a mapping if set.")

    superset_raw = tfe.get("superset_blocks")
    if superset_raw is None:
        superset_blocks: list[str] | None = None
    else:
        superset_blocks = [str(b).strip() for b in superset_raw if str(b).strip()]

    placeholder_arms: list[dict[str, Any]] = [
        {
            "name": "_top_features_superset",
            "selected_features": sorted(set(baseline_blocks) | set(pool_blocks)),
        }
    ]
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    full_df = _prepare_full_dataset(config, placeholder_arms, outdir)

    arms, ranked = assemble_top_features_experiment_arms(
        full_df,
        label_col=label_col,
        baseline_blocks=baseline_blocks,
        ranking_pool_blocks=pool_blocks,
        k_values=k_values,
        ranking_method=method,
        baseline_arm_name=str(tfe.get("baseline_arm_name", "baseline")),
        arm_name_prefix=str(tfe.get("arm_name_prefix", "topk")),
        superset_blocks=superset_blocks,
        ranker_kwargs=ranker_kwargs,
        baseline_model_params_override=tfe.get("baseline_model_params_override"),
        topk_model_params_override=tfe.get("topk_model_params_override"),
    )

    ranked.to_csv(outdir / "feature_ranking_global.csv", index=False)
    with (outdir / "top_features_eval_arms.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_data(arms), handle, sort_keys=False)

    passthrough_keys = (
        "model_families",
        "split_mode_matrix",
        "model_family_overrides",
        "export_subtype_summary",
    )
    for key in passthrough_keys:
        if key in tfe and tfe.get(key) is not None:
            config[key] = deepcopy(tfe[key])

    b_arm = tfe.get("baseline_arm_name")
    if b_arm:
        config.baseline_arm_name = str(b_arm)
    b_run = tfe.get("baseline_run_name")
    if b_run:
        config.baseline_run_name = str(b_run)

    config.ablation_arms = arms
    run_ablation_matrix(config, outdir)


if __name__ == "__main__":
    main()
