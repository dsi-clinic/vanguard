"""Production UChicago temporal pretraining on manifest-grouped exams.

Unlike :mod:`gnn.pretrain.duke_forecast`, this entrypoint is longitudinal and
manifest-driven. It loads one exam at a time, samples one eligible 3+2 window
per exam per epoch, and averages validation windows within exam before averaging
exams so graph size and series length cannot reweight the metric.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch

from gnn.pretrain.baselines import last_frame_forecast
from gnn.pretrain.checkpoint import (
    DEFAULT_PREPROCESSING_CONTRACT,
    load_checkpoint,
    save_encoder_checkpoint,
)
from gnn.pretrain.forecast import ForecastHorizon
from gnn.pretrain.loss import masked_mae
from gnn.pretrain.model import ContrastForecastGNN, PerNodeForecaster
from gnn.pretrain.resume import (
    atomic_json_save,
    load_arm_completion,
    load_epoch_progress,
    make_run_fingerprint,
    save_epoch_progress,
    validate_run_completion,
    write_arm_completion,
    write_run_completion,
)
from gnn.pretrain.train import ForecastGraph
from gnn.uchicago_temporal_data import load_temporal_cache_manifest

HORIZON = ForecastHorizon(input_len=3, target_len=2)
_MIN_SPLIT_PATIENTS = 2
_NDIM_TEMPORAL = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> tuple[str, str]:
    commit = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(  # noqa: S603
        ["git", "status", "--short"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip()
    return commit, status


def _resolved_run_config(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    git_commit: str,
    git_status: str,
) -> dict[str, object]:
    """Return every run-level field that must match for an exact resume."""
    return {
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "outdir": str(Path(args.outdir).resolve()),
        "cache_manifest_sha256": _sha256(manifest_path),
        "git_commit": git_commit,
        "git_status": git_status,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "num_gcn_layers": args.num_gcn_layers,
        "num_gru_layers": args.num_gru_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "val_fraction": args.val_fraction,
        "pretraining_task": "future_frame_forecast",
        "seed": args.seed,
        "requested_arm": args.arm,
        "device": args.device,
        "forecast_horizon": {
            "input_len": HORIZON.input_len,
            "target_len": HORIZON.target_len,
        },
        "preprocessing_contract": DEFAULT_PREPROCESSING_CONTRACT,
    }


def _reproduction_command(args: argparse.Namespace) -> str:
    values = (
        ("cache-dir", args.cache_dir),
        ("outdir", args.outdir),
        ("epochs", args.epochs),
        ("hidden-dim", args.hidden_dim),
        ("num-gcn-layers", args.num_gcn_layers),
        ("num-gru-layers", args.num_gru_layers),
        ("dropout", args.dropout),
        ("learning-rate", args.learning_rate),
        ("val-fraction", args.val_fraction),
        ("seed", args.seed),
        ("arm", args.arm),
        ("device", args.device),
    )
    flags = " ".join(f"--{name} {value}" for name, value in values)
    return f"python -u -m gnn.pretrain.uchicago {flags}"


def write_pretraining_readme(
    *,
    args: argparse.Namespace,
    results: dict[str, object],
    manifest: dict[str, object],
) -> Path:
    """Write the mandatory self-contained provenance note for one result directory."""
    commit, status = _git_provenance()
    manifest_path = Path(args.cache_dir) / "temporal_cache_manifest.json"
    arm_names = [name for name in ("gnn", "graph_free") if name in results]
    rows = []
    artifacts = []
    for arm in arm_names:
        arm_result = results[arm]
        rows.append(
            f"| {arm} | {int(arm_result['best_epoch'])} | "
            f"{float(arm_result['validation_mae']):.10f} | "
            f"{float(arm_result['persistence_mae']):.10f} | "
            f"{bool(arm_result['beats_persistence'])} |"
        )
        checkpoint_path = args.outdir / f"{arm}_encoder.pt"
        artifacts.append(
            f"- `{checkpoint_path.name}` SHA-256 `{_sha256(checkpoint_path)}`"
        )
    results_path = args.outdir / "pretraining_results.json"
    artifacts.append(f"- `{results_path.name}` SHA-256 `{_sha256(results_path)}`")
    status_text = status if status else "clean"
    readme = f"""# UChicago temporal pretraining result

## Status and question

Complete. Requested arm: `{args.arm}`. This run tests future-frame DCE forecasting;
it is not a downstream pCR result.

## Results

Lower MAE is better. Validation averages eligible windows within exam and then exams.

| Arm | Best epoch | Validation MAE | Persistence MAE | Beats persistence |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

## Data and split

- Cache: `{args.cache_dir}`
- Cache manifest SHA-256: `{_sha256(manifest_path)}`
- Source manifest: `{manifest.get('source_manifest')}`
- Excluded labeled manifest: `{manifest.get('excluded_labeled_manifest')}`
- Records: {len(manifest['records'])}
- Train patients: {results['train_patient_count']}
- Validation patients: {results['val_patient_count']}
- Split: patient-level, validation fraction {args.val_fraction}, seed {args.seed}

## Configuration

- Epochs: {args.epochs}
- Hidden dimension: {args.hidden_dim}
- Per-frame layers: {args.num_gcn_layers}
- GRU layers: {args.num_gru_layers}
- Dropout: {args.dropout}
- Learning rate: {args.learning_rate}
- Device: {args.device}
- Forecast window: 3 input frames -> 2 target frames
- Time anchor: minutes since last baseline acquisition

## Scheduler and artifacts

- Slurm job ID: `{os.environ.get('SLURM_JOB_ID', 'not available')}`
{chr(10).join(artifacts)}

## Reproduce

```bash
{_reproduction_command(args)}
```

## Code provenance

- Git commit: `{commit}`
- Dirty-worktree status at README creation:

```text
{status_text}
```

## Interpretation limit

This single split/seed forecasting result does not establish downstream pCR benefit or
that graph connectivity is universally useful or useless. Compare matched downstream
random/pretrained arms with patient-level OOF metrics and uncertainty intervals.
"""
    path = args.outdir / "README.md"
    path.write_text(readme)
    return path


def _record_groups(
    records: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["patient_key"])].append(record)
    return dict(grouped)


def split_records_by_patient(
    records: list[dict[str, object]], *, val_fraction: float, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Randomly split unique patients, keeping all their exams together."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    grouped = _record_groups(records)
    patients = sorted(grouped)
    if len(patients) < _MIN_SPLIT_PATIENTS:
        raise ValueError("patient-level pretraining split requires at least 2 patients")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(patients), generator=generator).tolist()
    n_val = max(1, round(len(patients) * val_fraction))
    val_patients = {patients[i] for i in order[:n_val]}
    train = [r for r in records if str(r["patient_key"]) not in val_patients]
    val = [r for r in records if str(r["patient_key"]) in val_patients]
    if {str(r["patient_key"]) for r in train} & val_patients:
        raise RuntimeError("patient leakage in pretraining split")
    return train, val


def eligible_starts(data: object, horizon: ForecastHorizon = HORIZON) -> list[int]:
    """Enumerate starts from B-1 through the last complete window."""
    baseline_count = int(data.baseline_frame_count)
    total_frames = int(data.node_series.shape[1])
    starts = list(range(baseline_count - 1, total_frames - horizon.window + 1))
    if not starts:
        raise ValueError(
            f"exam {getattr(data, 'exam_id', '')} has no eligible window: "
            f"B={baseline_count}, T={total_frames}, window={horizon.window}"
        )
    return starts


def make_window(
    data: object, start: int, horizon: ForecastHorizon = HORIZON
) -> ForecastGraph:
    """Materialize one masked window using minutes from the last baseline."""
    if data.node_series.ndim != _NDIM_TEMPORAL or data.node_series.shape[-1] != 1:
        raise ValueError("node_series must be (num_nodes, T, 1)")
    if start not in eligible_starts(data, horizon):
        raise ValueError(f"start={start} is not eligible for this exam")
    baseline_count = int(data.baseline_frame_count)
    times = (data.times_seconds - data.times_seconds[baseline_count - 1]) / 60.0
    cut = start + horizon.input_len
    end = start + horizon.window
    inputs = data.node_series[:, start:cut, :]
    target = data.node_series[:, cut:end, 0]
    target_mask = data.node_valid[:, None].bool() & torch.isfinite(target)
    return ForecastGraph(
        x_seq=torch.nan_to_num(inputs),
        target=torch.nan_to_num(target),
        edge_index=data.edge_index,
        input_times=times[start:cut],
        target_times=times[cut:end],
        target_mask=target_mask,
        baseline_observed=data.node_valid.bool(),
        exam_id=str(data.exam_id),
    )


def _load_graph(record: dict[str, object], device: torch.device) -> object:
    data = torch.load(Path(str(record["graph_path"])), map_location="cpu")
    return data.to(device)


def _forward(
    model: torch.nn.Module, window: ForecastGraph, uses_graph: bool
) -> torch.Tensor:
    if uses_graph:
        return model(
            window.x_seq,
            window.edge_index,
            window.acquisition_times,
            window.baseline_observed,
            window.target_times,
        )
    return model(
        window.x_seq,
        window.acquisition_times,
        window.baseline_observed,
        window.target_times,
    )


@torch.no_grad()
def evaluate_records(
    model: torch.nn.Module,
    records: list[dict[str, object]],
    *,
    uses_graph: bool,
    device: torch.device,
) -> tuple[float, float]:
    """Return learned-model and persistence MAE, equally weighted by exam."""
    model.eval()
    exam_maes = []
    persistence_maes = []
    for record in records:
        data = _load_graph(record, device)
        learned_windows = []
        persistence_windows = []
        for start in eligible_starts(data):
            window = make_window(data, start)
            learned_windows.append(
                masked_mae(
                    _forward(model, window, uses_graph),
                    window.target,
                    window.target_mask,
                )
            )
            persistence_windows.append(
                masked_mae(
                    last_frame_forecast(window.x_seq[:, :, 0], HORIZON.target_len),
                    window.target,
                    window.target_mask,
                )
            )
        exam_maes.append(torch.stack(learned_windows).mean().item())
        persistence_maes.append(torch.stack(persistence_windows).mean().item())
    return (
        float(sum(exam_maes) / len(exam_maes)),
        float(sum(persistence_maes) / len(persistence_maes)),
    )


def train_one_encoder(
    model: torch.nn.Module,
    train_records: list[dict[str, object]],
    val_records: list[dict[str, object]],
    *,
    uses_graph: bool,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    progress_path: Path | None = None,
    resume_fingerprint: dict[str, object] | None = None,
) -> tuple[torch.nn.Module, list[dict[str, float]], float, int, float]:
    """Train with one random exam window per epoch and inner validation only."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    best_persistence = float("nan")
    history: list[dict[str, float]] = []
    start_epoch = 1
    if (progress_path is None) != (resume_fingerprint is None):
        raise ValueError(
            "progress_path and resume_fingerprint must be provided together"
        )
    if progress_path is not None and resume_fingerprint is not None:
        resumed = load_epoch_progress(
            progress_path,
            expected_fingerprint=resume_fingerprint,
            model=model,
            optimizer=optimizer,
            sampling_generator=generator,
            device=device,
        )
        if resumed is not None:
            start_epoch = resumed.next_epoch
            history = resumed.history
            best_state = resumed.best_state_dict
            best_val = resumed.best_metric
            best_epoch = resumed.best_epoch
            try:
                best_persistence = float(resumed.best_metadata["persistence_mae"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "saved epoch progress has no valid persistence MAE"
                ) from error
            logging.info(
                "resuming exact epoch state from %s at epoch=%d",
                progress_path,
                start_epoch,
            )
    if start_epoch > epochs + 1:
        raise ValueError("saved epoch progress exceeds the requested epoch budget")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = torch.randperm(len(train_records), generator=generator).tolist()
        losses = []
        for index in order:
            data = _load_graph(train_records[index], device)
            starts = eligible_starts(data)
            choice = int(torch.randint(len(starts), (1,), generator=generator))
            window = make_window(data, starts[choice])
            optimizer.zero_grad()
            loss = masked_mae(
                _forward(model, window, uses_graph),
                window.target,
                window.target_mask,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        val_mae, persistence_mae = evaluate_records(
            model, val_records, uses_graph=uses_graph, device=device
        )
        train_mae = float(sum(losses) / len(losses))
        history.append(
            {"epoch": float(epoch), "train_mae": train_mae, "val_mae": val_mae}
        )
        logging.info(
            "epoch=%d train_mae=%.6f val_mae=%.6f persistence=%.6f",
            epoch,
            train_mae,
            val_mae,
            persistence_mae,
        )
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_persistence = persistence_mae
            best_state = deepcopy(model.state_dict())
        if progress_path is not None and resume_fingerprint is not None:
            if best_state is None:
                raise RuntimeError("pretraining epoch produced no finite best state")
            save_epoch_progress(
                progress_path,
                fingerprint=resume_fingerprint,
                model=model,
                optimizer=optimizer,
                next_epoch=epoch + 1,
                history=history,
                best_state_dict=best_state,
                best_metric=best_val,
                best_epoch=best_epoch,
                best_metadata={"persistence_mae": best_persistence},
                sampling_generator=generator,
                device=device,
            )
    if best_state is None:
        raise RuntimeError("pretraining produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_val, best_epoch, best_persistence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse production UChicago pretraining settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-gcn-layers", type=int, default=2)
    parser.add_argument("--num-gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--arm",
        choices=("both", "gnn", "graph_free"),
        default="both",
        help="Train both matched arms sequentially or one arm in an isolated job.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Pretrain matched graph and graph-free encoders and save checkpoints."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.cache_dir) / "temporal_cache_manifest.json"
    manifest = load_temporal_cache_manifest(args.cache_dir)
    if manifest.get("excluded_labeled_manifest") is None:
        raise ValueError(
            "pretraining cache does not record exclusion of the labeled cohort; "
            "rebuild with --exclude-labeled-manifest"
        )
    records = list(manifest["records"])
    train_records, val_records = split_records_by_patient(
        records, val_fraction=args.val_fraction, seed=args.seed
    )
    device = torch.device(args.device)
    common = {
        "in_channels": 1,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_gcn_layers,
        "num_gru_layers": args.num_gru_layers,
        "dropout": args.dropout,
    }
    arm_specs = {
        "gnn": (ContrastForecastGNN, True),
        "graph_free": (PerNodeForecaster, False),
    }
    arms = arm_specs if args.arm == "both" else {args.arm: arm_specs[args.arm]}
    git_commit, git_status = _git_provenance()
    resolved_config = _resolved_run_config(
        args,
        manifest_path=manifest_path,
        git_commit=git_commit,
        git_status=git_status,
    )
    run_fingerprint = make_run_fingerprint(resolved_config)
    final_marker = args.outdir / "COMPLETED.json"
    if validate_run_completion(final_marker, expected_fingerprint=run_fingerprint):
        logging.info("validated already-complete pretraining run at %s", args.outdir)
        return

    results = {
        "requested_arm": args.arm,
        "resolved_config": resolved_config,
        "run_fingerprint": run_fingerprint,
    }
    final_artifacts: list[Path] = []
    for arm, (model_class, uses_graph) in arms.items():
        torch.manual_seed(args.seed)
        model = model_class(**common)
        arm_config = {
            **resolved_config,
            "current_arm": arm,
            "uses_graph": uses_graph,
            "model_class": f"{model_class.__module__}.{model_class.__name__}",
            "encoder_config": model.encoder.config(),
        }
        arm_fingerprint = make_run_fingerprint(arm_config)
        checkpoint_path = args.outdir / f"{arm}_encoder.pt"
        progress_path = args.outdir / f"{arm}_progress.pt"
        arm_marker = args.outdir / f"{arm}_COMPLETED.json"
        validated_checkpoints: list[dict[str, object]] = []
        expected_encoder_config = model.encoder.config()

        def validate_final_checkpoint(
            path: Path,
            *,
            expected_config: dict[str, object] = arm_config,
            expected_fingerprint: dict[str, object] = arm_fingerprint,
            encoder_config: dict[str, object] = expected_encoder_config,
            validated: list[dict[str, object]] = validated_checkpoints,
        ) -> None:
            checkpoint = load_checkpoint(
                path,
                expected_resolved_config=expected_config,
                expected_run_fingerprint=expected_fingerprint,
            )
            if checkpoint["encoder_config"] != encoder_config:
                raise ValueError("completed arm encoder configuration mismatch")
            if checkpoint["git_commit"] != git_commit:
                raise ValueError("completed arm checkpoint git commit mismatch")
            if Path(checkpoint["data_manifest"]) != manifest_path:
                raise ValueError("completed arm checkpoint manifest path mismatch")
            validated.append(checkpoint)

        completed_result = load_arm_completion(
            arm_marker,
            expected_fingerprint=arm_fingerprint,
            checkpoint_path=checkpoint_path,
            checkpoint_validator=validate_final_checkpoint,
        )
        if completed_result is not None:
            checkpoint = validated_checkpoints[0]
            if int(checkpoint["epoch"]) != int(completed_result["best_epoch"]):
                raise ValueError("completed arm best epoch does not match checkpoint")
            if float(checkpoint["validation_mae"]) != float(
                completed_result["validation_mae"]
            ):
                raise ValueError(
                    "completed arm validation metric does not match checkpoint"
                )
            logging.info("validated and skipped completed arm=%s", arm)
            results[arm] = completed_result
            progress_path.unlink(missing_ok=True)
            final_artifacts.extend([checkpoint_path, arm_marker])
            continue

        model, history, val_mae, epoch, persistence = train_one_encoder(
            model,
            train_records,
            val_records,
            uses_graph=uses_graph,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
            progress_path=progress_path,
            resume_fingerprint=arm_fingerprint,
        )
        save_encoder_checkpoint(
            checkpoint_path,
            model=model,
            preprocessing_contract=DEFAULT_PREPROCESSING_CONTRACT,
            data_manifest=Path(args.cache_dir) / "temporal_cache_manifest.json",
            epoch=epoch,
            validation_mae=val_mae,
            resolved_config=arm_config,
            run_fingerprint=arm_fingerprint,
        )
        arm_result = {
            "validation_mae": val_mae,
            "persistence_mae": persistence,
            "beats_persistence": val_mae < persistence,
            "best_epoch": epoch,
            "history": history,
            "checkpoint": str(checkpoint_path),
        }
        results[arm] = arm_result
        write_arm_completion(
            arm_marker,
            fingerprint=arm_fingerprint,
            checkpoint_path=checkpoint_path,
            result=arm_result,
        )
        progress_path.unlink(missing_ok=True)
        final_artifacts.extend([checkpoint_path, arm_marker])
    if args.arm == "both":
        results["graph_helps_forecasting"] = (
            results["gnn"]["validation_mae"] < results["graph_free"]["validation_mae"]
        )
    results["train_patient_count"] = len(
        {str(record["patient_key"]) for record in train_records}
    )
    results["val_patient_count"] = len(
        {str(record["patient_key"]) for record in val_records}
    )
    results_path = atomic_json_save(
        results,
        args.outdir / "pretraining_results.json",
    )
    readme_path = write_pretraining_readme(
        args=args, results=results, manifest=manifest
    )
    final_artifacts.extend([results_path, readme_path])
    write_run_completion(
        final_marker,
        fingerprint=run_fingerprint,
        artifacts=final_artifacts,
    )


if __name__ == "__main__":
    main()
