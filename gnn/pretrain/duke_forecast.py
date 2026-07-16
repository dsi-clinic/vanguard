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
    _MODE_DEFAULT_FEATURES,
    _SEGMENT_MODE,
    _VOXEL_MODE,
    _build_case,
    _git_commit,
)
from gnn.pretrain.forecast import ForecastHorizon, split_forecast_window
from gnn.pretrain.train import ForecastGraph, run_pretrain_gates


def discover_forecast_tasks(
    centerline_root: Path,
    labels_path: Path,
    cases: list[str],
    *,
    id_column: str = "case_id",
    label_column: str = "pcr",
) -> list[tuple[str, Path, int]]:
    """Resolve each whitelisted case to ``(case_id, skeleton_mask_path, label)``.

    Globs ``centerline_root`` for ``*{_CENTERLINE_SUFFIX}`` masks (identical to
    ``VanguardCenterlineDataset._discover_cases``), keeping only ``cases``, then
    joins the pCR label from ``labels_path``. Fails loudly -- rather than silently
    dropping -- if a whitelisted case has no skeleton mask or no label row, since
    the whitelist is explicit and a miss means a wrong id or a stale labels file.
    Returns tasks in the order of ``cases``.
    """
    wanted = list(dict.fromkeys(cases))  # de-dupe, preserve order
    mask_by_case: dict[str, Path] = {}
    for mask_path in sorted(Path(centerline_root).rglob(f"*{_CENTERLINE_SUFFIX}")):
        case_id = mask_path.name[: -len(_CENTERLINE_SUFFIX)]
        if case_id in wanted and case_id not in mask_by_case:
            mask_by_case[case_id] = mask_path

    labels_df = pd.read_csv(labels_path)
    label_by_case = dict(
        zip(labels_df[id_column].astype(str), labels_df[label_column], strict=True)
    )

    tasks: list[tuple[str, Path, int]] = []
    for case_id in wanted:
        if case_id not in mask_by_case:
            raise ValueError(
                f"case {case_id!r} has no skeleton mask "
                f"(*{_CENTERLINE_SUFFIX}) under {centerline_root}"
            )
        if case_id not in label_by_case:
            raise ValueError(
                f"case {case_id!r} has no label in {labels_path} "
                f"(column {label_column!r})"
            )
        tasks.append((case_id, mask_by_case[case_id], int(label_by_case[case_id])))
    return tasks


def data_to_forecast_graph(data: object, horizon: ForecastHorizon) -> ForecastGraph:
    """Split a built ``Data``'s ``node_series`` into a ``ForecastGraph``.

    ``data.node_series`` (shape ``(N, T)``, attached by
    ``_build_case(attach_node_series=True)``) is cut at the horizon; the input
    horizon becomes ``x_seq`` with a trailing channel dim (the single enhancement
    channel), the following frames become ``target``. Fails loudly if the series
    is absent -- the build must have attached it.
    """
    node_series = getattr(data, "node_series", None)
    if node_series is None:
        raise ValueError("data has no node_series; build with attach_node_series=True")
    inputs, target = split_forecast_window(node_series, horizon)
    return ForecastGraph(
        x_seq=inputs.unsqueeze(-1), target=target, edge_index=data.edge_index
    )


def build_case_data(
    case_id: str,
    mask_path: Path,
    label: int,
    *,
    dce_root: Path,
    node_features: tuple[str, ...],
    node_mode: str = _VOXEL_MODE,
) -> object:
    """Build one Duke case's ``Data`` (carrying ``node_series``) for ``node_mode``.

    Split into build-vs-window (rather than returning a ``ForecastGraph``
    directly) so the caller can inspect every case's frame count ``T`` before
    committing a horizon -- Duke's ``T`` varies across cases (see
    ``resolve_smoke_horizon``), so the horizon can only be fixed once all series
    lengths are known. ``node_mode`` is ``"voxel"`` or ``"segment"`` (junction
    forecasting is not wired yet).
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
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument(
        "--cases", type=str, required=True, help="Comma-separated Duke case ids"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--node-mode",
        choices=(_VOXEL_MODE, _SEGMENT_MODE),
        default=_VOXEL_MODE,
        help="Graph node definition: 'voxel' or 'segment' (junction not wired yet).",
    )
    parser.add_argument("--input-len", type=int, default=3)
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
    graphs = [data_to_forecast_graph(d, horizon) for d in built]

    # Graph-level split: last --val-cases held out (order = --cases order).
    split = len(graphs) - args.val_cases
    train_graphs, val_graphs = graphs[:split], graphs[split:]
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
    return [
        f"--centerline-root {args.centerline_root}",
        f"--dce-root {args.dce_root}",
        f"--labels-path {args.labels_path}",
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
