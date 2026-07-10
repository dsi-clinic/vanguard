"""GNN adapter for LOCO (Leave-One-Covariate-Out) feature importance.

Generic across all three ``gnn_node_mode`` values (voxel/segment/junction):
the node/edge feature vocabulary for a mode comes straight from
``gnn.data_loader``'s own per-mode dicts, so this module never hardcodes a
feature list. Relies on the ``gnn/data_loader.py`` cache-manifest fix (see
that module's ``_check_cache_manifest``/``_reslice_for_requested_features``)
that lets a cache built once with the full feature superset serve any
narrower ``node_features``/``edge_features`` request without a rebuild --
see ``docs/loco_feature_importance.md`` for why this works.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from config import ConfigNode
from evaluation.loco import LOCOAdapter, run_loco_sweep
from evaluation.types import KFoldResults


def superset_features_for_mode(node_mode: str) -> tuple[str, ...]:
    """All node-feature names for a mode, from ``gnn.data_loader``'s own vocabulary.

    Excludes ``pcr_dummy`` (voxel mode's leakage-canary feature, see
    ``gnn/data_loader.py``) -- it is never a real modeling feature and must
    not be eligible for LOCO drop/keep.
    """
    from gnn.data_loader import _MODE_FEATURE_ATTR

    names = sorted(_MODE_FEATURE_ATTR[node_mode])
    return tuple(n for n in names if n != "pcr_dummy")


def superset_edge_features_for_mode(node_mode: str) -> tuple[str, ...]:
    """All edge-feature names for a mode (empty unless ``node_mode == 'junction'``)."""
    from gnn.data_loader import _MODE_EDGE_FEATURE_ATTR

    return tuple(sorted(_MODE_EDGE_FEATURE_ATTR.get(node_mode, {})))


def ensure_gnn_superset_cache(
    *,
    base_config: ConfigNode,
    node_mode: str,
    superset_node_features: tuple[str, ...],
    superset_edge_features: tuple[str, ...],
) -> None:
    """Fail loudly with a LOCO-specific hint if the superset cache isn't built yet.

    This is a check, not a builder: it never kicks off a rebuild itself (that
    could be a multi-hour job -- see ``gnn/build_dataset.py``). Constructing
    ``VanguardCenterlineDataset`` with the full feature superset is itself
    the check: if the cache at ``data_paths.gnn_cache_dir`` was never built,
    or was built with a feature list that isn't a superset of what's
    requested, ``_check_cache_manifest`` raises ``RuntimeError``. This
    re-raises with a concrete ``gnn/build_dataset.py`` command instead of
    leaving the caller to interpret a generic cache-mismatch error mid-sweep.

    Not called automatically by ``run_gnn_loco`` (the baseline run there
    would hit the same failure first anyway, without the cost of loading the
    cache twice) -- intended for a cheap pre-flight check between a build job
    and a train array in a Slurm pipeline, or in tests.
    """
    from gnn.data_loader import VanguardCenterlineDataset

    data_paths = base_config.data_paths
    try:
        VanguardCenterlineDataset(
            root=data_paths.gnn_centerline_root,
            labels_path=data_paths.gnn_labels_path,
            dce_root=data_paths.gnn_dce_root,
            cache_dir=data_paths.gnn_cache_dir,
            cases=list(data_paths.gnn_cases) if data_paths.gnn_cases else None,
            node_mode=node_mode,
            node_features=superset_node_features,
            edge_features=superset_edge_features or None,
            id_column=data_paths.gnn_id_column,
            label_column=data_paths.gnn_label_column,
        )
    except RuntimeError as exc:
        edge_flag = (
            f" --edge-features {','.join(superset_edge_features)}"
            if superset_edge_features
            else ""
        )
        raise RuntimeError(
            f"GNN LOCO requires a cache at {data_paths.gnn_cache_dir} built with "
            f"the full node_mode={node_mode!r} feature superset "
            f"{list(superset_node_features)}"
            + (
                f" + edge features {list(superset_edge_features)}"
                if superset_edge_features
                else ""
            )
            + ". Build it once with:\n"
            f"  python -m gnn.build_dataset --node-mode {node_mode} "
            f"--node-features {','.join(superset_node_features)}{edge_flag} "
            f"--cache-dir {data_paths.gnn_cache_dir}\n"
            "then rerun LOCO."
        ) from exc


def gnn_loco_adapter(
    *,
    drop_covariate: str | None,
    feature_group: str,
    base_config: ConfigNode,
    run_outdir: Path,
) -> KFoldResults:
    """``LOCOAdapter`` for the GNN track: vary ``gnn_node_features``/``gnn_edge_features``.

    Points at the same (already-built) superset cache dir on every call --
    only the requested feature subset changes, so no rebuild happens between
    runs (see module docstring).
    """
    from gnn.train import run_gnn_pipeline

    node_mode = str(base_config.model_params.gnn_node_mode)
    superset_nf = list(superset_features_for_mode(node_mode))
    superset_ef = list(superset_edge_features_for_mode(node_mode))

    run_config = deepcopy(base_config)
    if feature_group == "node":
        run_config.model_params.gnn_node_features = (
            superset_nf
            if drop_covariate is None
            else [f for f in superset_nf if f != drop_covariate]
        )
        run_config.model_params.gnn_edge_features = superset_ef
    elif feature_group == "edge":
        if not superset_ef:
            raise ValueError(f"node_mode={node_mode!r} has no edge features to sweep")
        run_config.model_params.gnn_node_features = superset_nf
        run_config.model_params.gnn_edge_features = (
            superset_ef
            if drop_covariate is None
            else [f for f in superset_ef if f != drop_covariate]
        )
    else:
        raise ValueError(
            f"feature_group must be 'node' or 'edge', got {feature_group!r}"
        )

    run_config.experiment_setup.name = (
        f"gnn_loco_{drop_covariate or 'full'}_{feature_group}"
    )
    return run_gnn_pipeline(run_config, run_outdir)


# Static type-check that gnn_loco_adapter satisfies the LOCOAdapter Protocol.
_: LOCOAdapter = gnn_loco_adapter


def run_gnn_loco(config: ConfigNode, outdir: Path) -> None:
    """Config-driven LOCO sweep driver for the GNN track.

    Reads ``model_params.gnn_node_mode`` to determine the feature vocabulary
    generically (voxel/segment/junction), and the optional
    ``loco.covariates``/``loco.feature_group`` keys to scope the sweep --
    see ``docs/loco_feature_importance.md``. Defaults to sweeping every
    feature in the mode's vocabulary (GNN's per-mode vocabularies are small
    enough that this is tractable, unlike tabular's ~800-column kinematic
    block), and to sweeping both node and edge features separately when the
    mode has edge features (junction mode).
    """
    node_mode = str(config.model_params.gnn_node_mode)
    superset_nf = superset_features_for_mode(node_mode)
    superset_ef = superset_edge_features_for_mode(node_mode)

    loco_cfg = config.loco
    requested_group = loco_cfg.get("feature_group")
    groups_to_run = (
        [str(requested_group)]
        if requested_group
        else (["node", "edge"] if superset_ef else ["node"])
    )

    for feature_group in groups_to_run:
        vocab = superset_nf if feature_group == "node" else superset_ef
        if not vocab:
            raise ValueError(
                f"feature_group={feature_group!r} requested but node_mode={node_mode!r} "
                "has no such features"
            )
        requested_covariates = loco_cfg.get("covariates")
        covariates = list(requested_covariates) if requested_covariates else list(vocab)
        unknown = [c for c in covariates if c not in vocab]
        if unknown:
            raise ValueError(
                f"loco.covariates {unknown} not in node_mode={node_mode!r} "
                f"{feature_group} vocabulary: {list(vocab)}"
            )
        sweep_outdir = outdir / feature_group if len(groups_to_run) > 1 else outdir
        run_loco_sweep(
            covariates=covariates,
            adapter=gnn_loco_adapter,
            base_config=config,
            outdir=sweep_outdir,
            family="gnn",
            node_mode=node_mode,
            feature_group=feature_group,
        )
