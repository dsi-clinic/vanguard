"""Build Duke voxel forecasting graphs and run the §7.i/§7.ii gates (smoke only).

Duke is MAMA-MIA-derived **placeholder** data. Per Spencer's instruction it is used
here for a *plumbing* smoke test of the contrast-forecasting pilot -- "don't expect
it to work" -- not as a scientific result. The UChicago pipeline (the real target)
is not ready and is never touched.

Pipeline: discover a whitelist of Duke cases (skeleton mask + pCR label), build each
one's voxel graph WITH its per-node contrast series
(``gnn.data_loader._build_case(attach_node_series=True)``), convert to a
``ForecastGraph``, and run ``gnn.pretrain.train.run_pretrain_gates`` (trained GNN vs.
trivial baselines vs. the graph-free ablation). Each case loads a 4D DCE volume, so
this is **Slurm-only**, never the login node (see
``gnn/slurm/submit_duke_forecast_smoke.slurm``).

Resolution reuses only the module constant ``_CENTERLINE_SUFFIX`` glob (same as
``VanguardCenterlineDataset._discover_cases``) plus a direct labels-CSV read -- it
does not instantiate the dataset, whose in-memory ``collate`` cannot stack a
variable-length ``node_series`` across cases. ``_build_case`` is imported as a
module-private helper deliberately: it is the single source of truth for
"case files -> voxel graph", and duplicating it would risk silent divergence.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from gnn.data_loader import (
    _CENTERLINE_SUFFIX,
    _JUNCTION_MODE,
    _MODE_DEFAULT_FEATURES,
    _SEGMENT_MODE,
    _VOXEL_MODE,
    _build_case,
    _git_commit,
)
from gnn.pretrain.forecast import ForecastHorizon, tile_forecast_windows
from gnn.pretrain.train import ForecastGraph, run_pretrain_gates


def discover_forecast_tasks(
    centerline_root: Path,
    labels_path: Path | None,
    cases: list[str],
    *,
    id_column: str = "case_id",
    label_column: str = "pcr",
) -> list[tuple[str, Path, int | None]]:
    """Resolve each whitelisted case to ``(case_id, skeleton_mask_path, label)``.

    Globs ``centerline_root`` for ``*{_CENTERLINE_SUFFIX}`` masks (identical to
    ``VanguardCenterlineDataset._discover_cases``), keeping only ``cases``, then
    joins the pCR label from ``labels_path`` **when one is given**. Fails loudly --
    rather than silently dropping -- if a whitelisted case has no skeleton mask, or
    (when labels are supplied) no label row, since the whitelist is explicit and a
    miss means a wrong id or a stale labels file.

    ``labels_path=None`` is the **label-free pretraining** mode (design review
    issue 2): the large unlabeled ``uc-uf-pretrain`` cohort is the whole reason for
    self-supervision, so no clinical label is required to build a forecasting case.
    Each task's label is then ``None``. Returns tasks in the order of ``cases``.
    """
    wanted = list(dict.fromkeys(cases))  # de-dupe, preserve order
    mask_by_case: dict[str, Path] = {}
    for mask_path in sorted(Path(centerline_root).rglob(f"*{_CENTERLINE_SUFFIX}")):
        case_id = mask_path.name[: -len(_CENTERLINE_SUFFIX)]
        if case_id in wanted and case_id not in mask_by_case:
            mask_by_case[case_id] = mask_path

    label_by_case: dict[str, object] = {}
    if labels_path is not None:
        labels_df = pd.read_csv(labels_path)
        label_by_case = dict(
            zip(labels_df[id_column].astype(str), labels_df[label_column], strict=True)
        )

    tasks: list[tuple[str, Path, int | None]] = []
    for case_id in wanted:
        if case_id not in mask_by_case:
            raise ValueError(
                f"case {case_id!r} has no skeleton mask "
                f"(*{_CENTERLINE_SUFFIX}) under {centerline_root}"
            )
        if labels_path is None:
            label: int | None = None
        elif case_id not in label_by_case:
            raise ValueError(
                f"case {case_id!r} has no label in {labels_path} "
                f"(column {label_column!r})"
            )
        else:
            label = int(label_by_case[case_id])
        tasks.append((case_id, mask_by_case[case_id], label))
    return tasks


def data_to_forecast_graphs(
    data: object, horizon: ForecastHorizon, *, keep_baseline_context: int = 1
) -> list[ForecastGraph]:
    """Tile a built ``Data``'s ``node_series`` into non-overlapping ``ForecastGraph``s.

    ``data.node_series`` ``(N, T)`` and ``data.node_times`` ``(T,)`` (attached by
    ``_build_case(attach_node_series=True)``) are tiled by ``tile_forecast_windows``
    over the post-baseline region: the leading baseline frames (``data.baseline_frame_count``,
    all but ``keep_baseline_context`` of them) are dropped as windows so the pretext
    task is dominated by contrast dynamics rather than flat baseline (issue 3b).
    Each tile becomes one ``ForecastGraph`` (same ``edge_index`` -- anatomy is
    static within a scan) carrying its own baseline-relative signal and per-tile
    elapsed times. Fails loudly if any required attribute is absent.
    """
    node_series = getattr(data, "node_series", None)
    if node_series is None:
        raise ValueError("data has no node_series; build with attach_node_series=True")
    node_times = getattr(data, "node_times", None)
    if node_times is None:
        raise ValueError("data has no node_times; build with attach_node_series=True")
    baseline_frame_count = getattr(data, "baseline_frame_count", None)
    if baseline_frame_count is None:
        raise ValueError(
            "data has no baseline_frame_count; build with attach_node_series=True"
        )
    tiles = tile_forecast_windows(
        node_series,
        node_times,
        horizon,
        baseline_frame_count=int(baseline_frame_count),
        keep_baseline_context=keep_baseline_context,
    )
    return [
        ForecastGraph(
            x_seq=inputs.unsqueeze(-1),
            target=target,
            edge_index=data.edge_index,
            input_times=input_times,
            target_times=target_times,
        )
        for inputs, target, input_times, target_times in tiles
    ]


def build_case_data(
    case_id: str,
    mask_path: Path,
    label: int | None = None,
    *,
    dce_root: Path,
    node_features: tuple[str, ...],
    node_mode: str = _VOXEL_MODE,
) -> object:
    """Build one case's ``Data`` (carrying ``node_series``) for ``node_mode``.

    Split into build-vs-window (rather than returning a ``ForecastGraph``
    directly) so the caller can inspect every case's frame count ``T`` before
    committing a horizon -- ``T`` varies across cases (see
    ``resolve_smoke_horizon``), so the horizon can only be fixed once all series
    lengths are known. ``node_mode`` is ``"voxel"``, ``"segment"``, or
    ``"junction"``. ``label=None`` builds a label-free forecasting case (issue 2);
    forecasting never reads ``data.y``, so an unlabeled case is fully usable.
    """
    data, _ = _build_case(
        case_id,
        mask_path,
        label,
        dce_root=Path(dce_root),
        node_features=node_features,
        node_mode=node_mode,
        attach_node_series=True,
    )
    return data


def resolve_smoke_horizon(
    series_lengths: list[int], horizon: ForecastHorizon, policy: str
) -> ForecastHorizon:
    """Reconcile a requested horizon with variable per-case ``T`` (smoke only).

    Duke cases have different DCE frame counts (observed T=5 and T=4). This is
    the §8b variable-length question, which is Spencer's to decide for the real
    pipeline (pad-mask / truncate / resample). For the *smoke* only:

    - ``policy="strict"`` (default): if the requested window exceeds the minimum
      ``T``, raise -- keep the fail-fast contract, don't silently shrink.
    - ``policy="truncate"``: shrink ``input_len`` so ``window == min(T)`` (keeping
      ``target_len``), i.e. truncate every case to the shortest one's length.
      No imputation, no resampling -- the least-opinionated expedient. Raises if
      ``target_len`` alone already exceeds ``min(T)``.
    """
    t_min = min(series_lengths)
    if horizon.window <= t_min:
        return horizon
    if policy != "truncate":
        raise ValueError(
            f"requested window={horizon.window} exceeds min case T={t_min}. "
            "Duke has variable-length series (§8b). Use --min-frames-policy "
            "truncate for the smoke, or choose a shorter horizon. The real "
            "variable-length policy is a design decision (docs/design/"
            "contrast_pretraining.md §8b), not defaulted here."
        )
    new_input_len = t_min - horizon.target_len
    if new_input_len < 1:
        raise ValueError(
            f"target_len={horizon.target_len} already >= min case T={t_min}; "
            "cannot fit an input horizon. Pick a smaller --target-len."
        )
    logging.warning(
        "min-frames-policy=truncate: min case T=%d < requested window=%d; "
        "shrinking input_len %d -> %d (target_len=%d unchanged). SMOKE EXPEDIENT, "
        "not the real §8b policy.",
        t_min,
        horizon.window,
        horizon.input_len,
        new_input_len,
        horizon.target_len,
    )
    return ForecastHorizon(input_len=new_input_len, target_len=horizon.target_len)


def _write_readme(
    outdir: Path,
    *,
    node_mode: str,
    argv: list[str],
    tasks: list[tuple[str, Path, int]],
    horizon: ForecastHorizon,
    report: dict[str, float],
    train_ids: list[str],
    val_ids: list[str],
) -> None:
    """Write the results README (repo rule: every results dir is self-documenting)."""
    lines = [
        f"# Duke {node_mode} contrast-forecasting smoke (PLACEHOLDER DATA)",
        "",
        "Plumbing smoke test of the voxel forecasting pilot "
        "(`docs/design/contrast_pretraining.md`) on Duke MAMA-MIA placeholder data. "
        "**Not a scientific result** -- Duke is not the target cohort and the pretext "
        "task is not expected to work here; this only checks the build+gate pipeline "
        "runs end-to-end on real graphs.",
        "",
        f"- **Commit:** {_git_commit()}",
        f"- **Command:** `python -m gnn.pretrain.duke_forecast {' '.join(argv)}`",
        f"- **Horizon:** input_len={horizon.input_len}, target_len={horizon.target_len}",
        f"- **Cases ({len(tasks)}):** " + ", ".join(c for c, _, _ in tasks),
        f"- **Train graphs:** {', '.join(train_ids)}",
        f"- **Val graphs:** {', '.join(val_ids)}",
        "",
        "## Gate report (held-out MAE)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key in ("gnn", "per_node", "last_frame", "temporal_mean"):
        lines.append(f"| {key} | {report[key]:.4f} |")
    lines += [
        f"| §7.i beats_trivial | {bool(report['beats_trivial'])} |",
        f"| §7.ii graph_helps | {bool(report['graph_helps'])} |",
        "",
        "Full report: `gate_report.json`. Regenerate: rerun the command above "
        "(build is deterministic given the same cases/commit).",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the Duke forecasting smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help=(
            "pCR labels CSV. Optional: omit for label-free pretraining (issue 2) "
            "-- forecasting never reads the label, so unlabeled cohorts "
            "(e.g. uc-uf-pretrain) build fine without it."
        ),
    )
    parser.add_argument(
        "--cases", type=str, required=True, help="Comma-separated Duke case ids"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--node-mode",
        choices=(_VOXEL_MODE, _SEGMENT_MODE, _JUNCTION_MODE),
        default=_VOXEL_MODE,
        help="Graph node definition: 'voxel', 'segment', or 'junction'.",
    )
    # Placeholder default horizon (2/2), NOT tuned -- a horizon-validation sweep
    # is a committed follow-up (see docs/design/contrast_pretraining_params.md).
    parser.add_argument("--input-len", type=int, default=2)
    parser.add_argument("--target-len", type=int, default=2)
    parser.add_argument(
        "--min-frames-policy",
        choices=("strict", "truncate"),
        default="strict",
        help=(
            "How to handle cases with fewer frames than the horizon window. "
            "'strict' (default) fails loudly; 'truncate' shrinks input_len to the "
            "minimum case T (smoke expedient, not the real §8b policy)."
        ),
    )
    parser.add_argument(
        "--val-cases", type=int, default=1, help="How many cases to hold out for val"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id-column", type=str, default="case_id")
    parser.add_argument("--label-column", type=str, default="pcr")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build Duke forecasting graphs, run the gates, and write the results dir."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args(argv)
    requested_horizon = ForecastHorizon(
        input_len=args.input_len, target_len=args.target_len
    )
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    if args.val_cases < 1 or args.val_cases >= len(cases):
        raise ValueError(
            f"--val-cases must be in [1, {len(cases) - 1}] for {len(cases)} cases, "
            f"got {args.val_cases}"
        )

    node_features = _MODE_DEFAULT_FEATURES[args.node_mode]
    tasks = discover_forecast_tasks(
        args.centerline_root,
        args.labels_path,
        cases,
        id_column=args.id_column,
        label_column=args.label_column,
    )

    # Build every case's Data (with node_series) first, so we can see each one's
    # frame count T before committing a horizon -- Duke's T varies across cases.
    built: list[object] = []
    for case_id, mask_path, label in tasks:
        logging.info("Building %s graph for %s", args.node_mode, case_id)
        data = build_case_data(
            case_id,
            mask_path,
            label,
            dce_root=args.dce_root,
            node_features=node_features,
            node_mode=args.node_mode,
        )
        series_len = int(data.node_series.shape[1])
        logging.info("  %s: %d nodes, T=%d frames", case_id, data.num_nodes, series_len)
        built.append(data)

    series_lengths = [int(d.node_series.shape[1]) for d in built]
    horizon = resolve_smoke_horizon(
        series_lengths, requested_horizon, args.min_frames_policy
    )
    logging.info(
        "case frame counts T=%s -> using horizon input_len=%d target_len=%d",
        series_lengths,
        horizon.input_len,
        horizon.target_len,
    )
    # One case (patient) -> several non-overlapping tiles.
    per_case_graphs = [data_to_forecast_graphs(d, horizon) for d in built]
    tile_counts = [len(g) for g in per_case_graphs]
    logging.info("tiles per case: %s (total %d)", tile_counts, sum(tile_counts))

    # Patient-level split: ALL tiles of the last --val-cases cases are held out,
    # so no patient's tiles straddle train and val (avoids subject leakage).
    split = len(per_case_graphs) - args.val_cases
    train_graphs = [g for case_tiles in per_case_graphs[:split] for g in case_tiles]
    val_graphs = [g for case_tiles in per_case_graphs[split:] for g in case_tiles]
    train_ids = [t[0] for t in tasks[:split]]
    val_ids = [t[0] for t in tasks[split:]]

    report = run_pretrain_gates(
        train_graphs,
        val_graphs,
        horizon,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        seed=args.seed,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_readme(
        args.outdir,
        node_mode=args.node_mode,
        argv=argv if argv is not None else _argv_for_readme(args),
        tasks=tasks,
        horizon=horizon,
        report=report,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    logging.info("Gate report: %s", json.dumps(report))
    logging.info("Wrote results to %s", args.outdir)


def _argv_for_readme(args: argparse.Namespace) -> list[str]:
    """Reconstruct a readable command line from parsed args (for the README)."""
    labels_flag = (
        [f"--labels-path {args.labels_path}"] if args.labels_path is not None else []
    )
    return [
        f"--centerline-root {args.centerline_root}",
        f"--dce-root {args.dce_root}",
        *labels_flag,
        f"--cases {args.cases}",
        f"--outdir {args.outdir}",
        f"--node-mode {args.node_mode}",
        f"--input-len {args.input_len}",
        f"--target-len {args.target_len}",
        f"--min-frames-policy {args.min_frames_policy}",
        f"--val-cases {args.val_cases}",
        f"--epochs {args.epochs}",
        f"--hidden-dim {args.hidden_dim}",
        f"--seed {args.seed}",
    ]


if __name__ == "__main__":
    main()
