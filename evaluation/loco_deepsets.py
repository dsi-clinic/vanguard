"""DeepSets adapter for LOCO (Leave-One-Covariate-Out) feature importance.

Relies on the ``deepsets_point_feature_set: loco_superset`` regime (see
``deepsets/build_dataset.py::DEEPSETS_FEATURE_LOCO_SUPERSET``, the union of
every other regime's columns) being built once, and on
``deepsets.data.SavedSetLookup``'s ``keep_features`` param to select a
column subset from that superset at load time -- so a LOCO sweep over every
covariate needs only one manifest build, not one per covariate. See
``docs/loco_feature_importance.md``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from config import ConfigNode
from evaluation.loco import LOCOAdapter, run_loco_sweep
from evaluation.types import KFoldResults


def ensure_deepsets_superset_manifest(base_config: ConfigNode) -> None:
    """Fail loudly with a LOCO-specific hint if the manifest isn't the superset regime.

    Cheap relative to the GNN equivalent (``evaluation.loco_gnn.
    ensure_gnn_superset_cache``): DeepSets manifests are per-case point sets,
    not full graph caches, so peeking at one case's ``feature_names`` is a
    lightweight check worth doing eagerly rather than only discovering a
    mismatch mid-sweep.
    """
    from deepsets.build_dataset import (
        DEEPSETS_FEATURE_LOCO_SUPERSET,
        deepsets_point_feature_names,
    )
    from deepsets.train import build_deepsets_dataset, load_deepsets_manifest

    manifest_df = load_deepsets_manifest(base_config)
    dataset = build_deepsets_dataset(manifest_df, base_config)
    first_case_id = str(manifest_df["case_id"].iloc[0])
    payload = dataset.get(first_case_id)
    cached_names = [str(n) for n in payload.get("feature_names", [])]
    superset = deepsets_point_feature_names(DEEPSETS_FEATURE_LOCO_SUPERSET)
    if set(cached_names) != set(superset):
        raise ValueError(
            f"DeepSets LOCO requires data_paths.deepsets_manifest_csv="
            f"{base_config.data_paths.deepsets_manifest_csv!r} to be built with "
            f"model_params.deepsets_point_feature_set: {DEEPSETS_FEATURE_LOCO_SUPERSET!r} "
            f"(expected columns {superset}, found {cached_names} on case "
            f"{first_case_id!r}). Rebuild the manifest with that regime once, "
            "then rerun LOCO."
        )


def deepsets_loco_adapter(
    *,
    drop_covariate: str | None,
    feature_group: str,
    base_config: ConfigNode,
    run_outdir: Path,
) -> KFoldResults:
    """``LOCOAdapter`` for the DeepSets track: vary which superset columns are kept.

    Points at the same (already-built) ``loco_superset`` manifest on every
    call -- only ``keep_features`` changes, so no rebuild happens between
    runs (see module docstring).
    """
    from deepsets.build_dataset import (
        DEEPSETS_FEATURE_LOCO_SUPERSET,
        deepsets_point_feature_names,
    )
    from deepsets.train import run_deepsets_pipeline

    if feature_group != "node":
        raise ValueError(
            f"DeepSets point features have no edge/node distinction; "
            f"feature_group must be 'node', got {feature_group!r}"
        )

    superset = deepsets_point_feature_names(DEEPSETS_FEATURE_LOCO_SUPERSET)
    keep_features = (
        superset
        if drop_covariate is None
        else [f for f in superset if f != drop_covariate]
    )

    run_config = deepcopy(base_config)
    run_config.model_params.deepsets_point_feature_set = DEEPSETS_FEATURE_LOCO_SUPERSET
    run_config.experiment_setup.name = f"deepsets_loco_{drop_covariate or 'full'}"
    return run_deepsets_pipeline(run_config, run_outdir, keep_features=keep_features)


# Static type-check that deepsets_loco_adapter satisfies the LOCOAdapter Protocol.
_: LOCOAdapter = deepsets_loco_adapter


def run_deepsets_loco(config: ConfigNode, outdir: Path) -> None:
    """Config-driven LOCO sweep driver for the DeepSets track.

    Reads the optional ``loco.covariates`` key to restrict the sweep (default:
    every column in the ``loco_superset`` regime, 28 total -- see
    ``docs/loco_feature_importance.md``).
    """
    from deepsets.build_dataset import (
        DEEPSETS_FEATURE_LOCO_SUPERSET,
        deepsets_point_feature_names,
    )

    superset = deepsets_point_feature_names(DEEPSETS_FEATURE_LOCO_SUPERSET)

    loco_cfg = config.loco
    requested_covariates = loco_cfg.get("covariates")
    covariates = list(requested_covariates) if requested_covariates else list(superset)
    unknown = [c for c in covariates if c not in superset]
    if unknown:
        raise ValueError(
            f"loco.covariates {unknown} not in loco_superset vocabulary: {superset}"
        )

    run_loco_sweep(
        covariates=covariates,
        adapter=deepsets_loco_adapter,
        base_config=config,
        outdir=outdir,
        family="deepsets",
        node_mode=None,
        feature_group="node",
    )
