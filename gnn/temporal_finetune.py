"""Patient-level UChicago temporal fine-tuning and the required 2x2 comparison."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gnn.pretrain.checkpoint import load_checkpoint
from gnn.pretrain.uchicago import _git_provenance, _sha256
from gnn.temporal_model import (
    PerNodeTemporalEncoder,
    TemporalGraphClassifier,
    TemporalGraphEncoder,
    TemporalPerNodeClassifier,
    build_encoder_from_config,
    load_encoder_strict,
)
from gnn.uchicago_temporal_data import load_temporal_cache_manifest

_BINARY_CLASS_COUNT = 2
_DEFAULT_TABULAR_REFERENCE = Path(
    "experiments/uchicago_tabular_bar_floored/gate_auc_ci.csv"
)
_ARM_SPECS = {
    "graph_free_random": ("graph_free", False),
    "graph_free_pretrained": ("graph_free", True),
    "gnn_random": ("gnn", False),
    "gnn_pretrained": ("gnn", True),
}
_COMPARISONS = {
    "pretraining_without_graph": ("graph_free_pretrained", "graph_free_random"),
    "connectivity_without_pretraining": ("gnn_random", "graph_free_random"),
    "graph_pretraining": ("gnn_pretrained", "gnn_random"),
    "pretrained_graph_beyond_dynamics": (
        "gnn_pretrained",
        "graph_free_pretrained",
    ),
}


def _group_by_patient(
    records: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("pcr") is None or record.get("fold") is None:
            raise ValueError("downstream records require both pcr and fold")
        groups[str(record["patient_key"])].append(record)
    for patient, exams in groups.items():
        labels = {int(exam["pcr"]) for exam in exams}
        folds = {int(exam["fold"]) for exam in exams}
        if len(labels) != 1:
            raise ValueError(f"patient {patient} has inconsistent pCR labels: {labels}")
        if len(folds) != 1:
            raise ValueError(f"patient {patient} crosses outer folds: {folds}")
    return dict(groups)


def validate_cache_against_cohort(
    records: list[dict[str, object]], cohort_manifest: Path
) -> None:
    """Require every cached exam to retain the labeled cohort identity and split."""
    cohort = pd.read_csv(cohort_manifest)
    required = {"exam_id", "patient_key", "fold", "pcr"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"downstream cohort is missing columns: {sorted(missing)}")
    if cohort["exam_id"].duplicated().any():
        raise ValueError("downstream cohort contains duplicate exam_id values")
    expected = cohort.set_index("exam_id")
    for record in records:
        exam_id = str(record["exam_id"])
        if exam_id not in expected.index:
            raise ValueError(f"cached exam {exam_id} is absent from downstream cohort")
        row = expected.loc[exam_id]
        identity = (
            str(record["patient_key"]),
            int(record["fold"]),
            int(record["pcr"]),
        )
        expected_identity = (
            str(row["patient_key"]),
            int(row["fold"]),
            int(row["pcr"]),
        )
        if identity != expected_identity:
            raise ValueError(
                f"cached exam {exam_id} identity/split mismatch: "
                f"{identity} != {expected_identity}"
            )


def _full_exam_inputs(data: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline_count = int(data.baseline_frame_count)
    times = (data.times_seconds - data.times_seconds[baseline_count - 1]) / 60.0
    return torch.nan_to_num(data.node_series), times, data.node_valid.bool()


def _build_classifier(
    checkpoint: dict[str, object], *, pretrained: bool
) -> tuple[torch.nn.Module, bool]:
    encoder = build_encoder_from_config(checkpoint["encoder_config"])
    if pretrained:
        load_encoder_strict(encoder, checkpoint)
    if isinstance(encoder, TemporalGraphEncoder):
        return TemporalGraphClassifier(encoder), True
    if isinstance(encoder, PerNodeTemporalEncoder):
        return TemporalPerNodeClassifier(encoder), False
    raise TypeError(f"unsupported temporal encoder {type(encoder)}")


def _forward_exam(
    model: torch.nn.Module, data: object, uses_graph: bool
) -> torch.Tensor:
    series, times, observed = _full_exam_inputs(data)
    if uses_graph:
        return model(series, data.edge_index, times, observed)
    return model(series, times, observed)


def _load_record(record: dict[str, object], device: torch.device) -> object:
    return torch.load(Path(str(record["graph_path"])), map_location="cpu").to(device)


def _inner_patient_split(
    patient_groups: dict[str, list[dict[str, object]]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    patients = sorted(patient_groups)
    labels = [int(patient_groups[patient][0]["pcr"]) for patient in patients]
    train, val = train_test_split(
        patients,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    return list(train), list(val)


@torch.no_grad()
def predict_patients(
    model: torch.nn.Module,
    patient_groups: dict[str, list[dict[str, object]]],
    patients: list[str],
    *,
    uses_graph: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict every exam, then average probabilities within patient."""
    model.eval()
    labels = []
    probabilities = []
    for patient in patients:
        exam_probabilities = []
        exams = patient_groups[patient]
        for record in exams:
            data = _load_record(record, device)
            exam_probabilities.append(
                torch.sigmoid(_forward_exam(model, data, uses_graph)).item()
            )
        labels.append(int(exams[0]["pcr"]))
        probabilities.append(float(np.mean(exam_probabilities)))
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def _patient_loss(
    model: torch.nn.Module,
    patient_groups: dict[str, list[dict[str, object]]],
    patients: list[str],
    *,
    uses_graph: bool,
    device: torch.device,
) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for patient in patients:
            exam_logits = []
            exams = patient_groups[patient]
            for record in exams:
                exam_logits.append(
                    _forward_exam(model, _load_record(record, device), uses_graph)
                )
            patient_probability = torch.sigmoid(torch.stack(exam_logits)).mean()
            label = torch.tensor(float(exams[0]["pcr"]), device=device)
            losses.append(
                torch.nn.functional.binary_cross_entropy(
                    patient_probability, label
                ).item()
            )
    return float(np.mean(losses))


def validate_matched_encoder_configs(
    graph_checkpoint: dict[str, object], free_checkpoint: dict[str, object]
) -> None:
    """Require the two pretrained arms to differ only by connectivity."""
    graph_config = dict(graph_checkpoint["encoder_config"])
    free_config = dict(free_checkpoint["encoder_config"])
    if graph_config.pop("encoder_type", None) != "graph":
        raise ValueError("--gnn-checkpoint must contain a graph encoder")
    if free_config.pop("encoder_type", None) != "graph_free":
        raise ValueError("--graph-free-checkpoint must contain a graph-free encoder")
    if graph_config != free_config:
        raise ValueError(
            "graph and graph-free checkpoints must have matched encoder "
            "configurations apart from encoder_type"
        )


def protocol_only_oof(
    patient_groups: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    """Audit label prediction from acquisition timing without imaging content."""
    rows = []
    feature_names = ("n_frames", "duration_seconds", "median_dt_seconds")
    for patient, exams in sorted(patient_groups.items()):
        try:
            features = np.asarray(
                [[float(exam[name]) for name in feature_names] for exam in exams]
            ).mean(axis=0)
        except KeyError as error:
            raise ValueError(
                "temporal cache lacks protocol descriptors; rebuild it with the "
                "current UChicago cache builder"
            ) from error
        rows.append(
            {
                "patient_key": patient,
                "fold": int(exams[0]["fold"]),
                "y_true": int(exams[0]["pcr"]),
                **dict(zip(feature_names, features, strict=True)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["y_prob"] = np.nan
    for fold in sorted(frame["fold"].unique()):
        train = frame["fold"] != fold
        test = ~train
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)
        )
        model.fit(frame.loc[train, list(feature_names)], frame.loc[train, "y_true"])
        frame.loc[test, "y_prob"] = model.predict_proba(
            frame.loc[test, list(feature_names)]
        )[:, 1]
    if frame["y_prob"].isna().any():
        raise RuntimeError("protocol-only audit did not produce every OOF prediction")
    return frame[["patient_key", "fold", "y_true", "y_prob"]]


def load_tabular_reference(path: Path) -> dict[str, object]:
    """Load the strongest existing UChicago vessel-tabular OOF result."""
    frame = pd.read_csv(path)
    required = {"model", "pooled_oof_auc", "ci_lo", "ci_hi"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"tabular reference is missing columns: {sorted(missing)}")
    candidates = frame.loc[frame["model"].astype(str).str.startswith("tabular_")]
    if candidates.empty:
        raise ValueError("tabular reference contains no model starting with 'tabular_'")
    best = candidates.loc[candidates["pooled_oof_auc"].idxmax()]
    return {
        "model": str(best["model"]),
        "auc": float(best["pooled_oof_auc"]),
        "ci_low": float(best["ci_lo"]),
        "ci_high": float(best["ci_hi"]),
        "source": str(path),
        "role": "existing_exam_level_reference_not_paired_with_patient_level_arms",
    }


def fit_fold(
    model: torch.nn.Module,
    train_groups: dict[str, list[dict[str, object]]],
    *,
    uses_graph: bool,
    epochs: int,
    learning_rate: float,
    inner_val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    """Fine-tune the entire encoder and head from epoch one."""
    fit_patients, inner_patients = _inner_patient_split(
        train_groups, val_fraction=inner_val_fraction, seed=seed
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(fit_patients), generator=generator).tolist()
        epoch_losses = []
        for patient_index in order:
            patient = fit_patients[patient_index]
            exams = train_groups[patient]
            exam_index = int(torch.randint(len(exams), (1,), generator=generator))
            record = exams[exam_index]
            data = _load_record(record, device)
            optimizer.zero_grad()
            logit = _forward_exam(model, data, uses_graph).squeeze()
            label = torch.tensor(float(record["pcr"]), device=device)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        inner_loss = _patient_loss(
            model,
            train_groups,
            inner_patients,
            uses_graph=uses_graph,
            device=device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(epoch_losses)),
                "inner_val_loss": inner_loss,
            }
        )
        if inner_loss < best_loss:
            best_loss = inner_loss
            best_state = deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("fine-tuning produced no selected checkpoint")
    model.load_state_dict(best_state)
    return model, history


def _bootstrap_auc(
    y: np.ndarray, probability: np.ndarray, *, seed: int, n_bootstrap: int = 2000
) -> tuple[float, float, float]:
    auc = float(roc_auc_score(y, probability))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == _BINARY_CLASS_COUNT:
            values.append(roc_auc_score(y[index], probability[index]))
    if not values:
        raise ValueError("no valid patient-level bootstrap samples")
    low, high = np.quantile(values, [0.025, 0.975])
    return auc, float(low), float(high)


def _paired_auc_delta(
    predictions: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    *,
    seed: int,
    n_bootstrap: int = 2000,
) -> dict[str, float]:
    """Patient-resampled paired OOF AUC difference ``arm_a - arm_b``."""
    left = predictions.loc[
        predictions["arm"] == arm_a, ["patient_key", "y_true", "y_prob"]
    ].rename(columns={"y_prob": "prob_a"})
    right = predictions.loc[
        predictions["arm"] == arm_b, ["patient_key", "y_true", "y_prob"]
    ].rename(columns={"y_prob": "prob_b"})
    paired = left.merge(
        right,
        on=["patient_key", "y_true"],
        how="inner",
        validate="one_to_one",
    )
    expected = predictions.loc[predictions["arm"] == arm_a, "patient_key"].nunique()
    if len(paired) != expected:
        raise ValueError(f"arms {arm_a} and {arm_b} do not have identical patients")
    y = paired["y_true"].to_numpy()
    prob_a = paired["prob_a"].to_numpy()
    prob_b = paired["prob_b"].to_numpy()
    point = float(roc_auc_score(y, prob_a) - roc_auc_score(y, prob_b))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == _BINARY_CLASS_COUNT:
            values.append(
                roc_auc_score(y[index], prob_a[index])
                - roc_auc_score(y[index], prob_b[index])
            )
    low, high = np.quantile(values, [0.025, 0.975])
    return {"delta_auc": point, "ci_low": float(low), "ci_high": float(high)}


def _arm_metrics(predictions: pd.DataFrame, *, seed: int) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for arm_name, arm_frame in predictions.groupby("arm"):
        auc, low, high = _bootstrap_auc(
            arm_frame["y_true"].to_numpy(),
            arm_frame["y_prob"].to_numpy(),
            seed=seed,
        )
        metrics[str(arm_name)] = {"auc": auc, "ci_low": low, "ci_high": high}
    return metrics


def load_parallel_arm_predictions(outdir: Path) -> pd.DataFrame:
    """Load one complete patient-level prediction table from every 2x2 arm."""
    frames = []
    expected_patients = None
    for arm in _ARM_SPECS:
        path = outdir / arm / "patient_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed arm predictions: {path}")
        frame = pd.read_csv(path)
        required = {"arm", "fold", "patient_key", "y_true", "y_prob", "n_exams"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if set(frame["arm"].astype(str)) != {arm}:
            raise ValueError(f"{path} contains predictions for the wrong arm")
        patients = set(frame["patient_key"].astype(str))
        if expected_patients is None:
            expected_patients = patients
        elif patients != expected_patients:
            raise ValueError("parallel arms do not contain identical patient sets")
        if frame["patient_key"].duplicated().any():
            raise ValueError(f"{path} contains duplicate patient predictions")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _reproduction_command(args: argparse.Namespace) -> str:
    values = (
        ("cache-dir", args.cache_dir),
        ("cohort-manifest", args.cohort_manifest),
        ("gnn-checkpoint", args.gnn_checkpoint),
        ("graph-free-checkpoint", args.graph_free_checkpoint),
        ("outdir", args.outdir),
        ("tabular-reference", args.tabular_reference),
        ("epochs", args.epochs),
        ("learning-rate", args.learning_rate),
        ("inner-val-fraction", args.inner_val_fraction),
        ("seed", args.seed),
        ("arm", args.arm),
        ("device", args.device),
    )
    flags = " ".join(f"--{name} {value}" for name, value in values)
    if args.aggregate_only:
        flags += " --aggregate-only"
    return f"python -u -m gnn.temporal_finetune {flags}"


def _result_artifacts(outdir: Path, *, aggregate: bool) -> list[Path]:
    paths = [outdir / "patient_predictions.csv", outdir / "metrics.json"]
    if aggregate:
        paths.append(outdir / "protocol_only_patient_predictions.csv")
        for arm in _ARM_SPECS:
            paths.extend(
                [
                    outdir / arm / "patient_predictions.csv",
                    outdir / arm / "metrics.json",
                    outdir / arm / "history.json",
                    outdir / arm / "README.md",
                ]
            )
    else:
        paths.append(outdir / "history.json")
    return paths


def write_finetune_readme(
    *,
    args: argparse.Namespace,
    manifest: dict[str, object],
    metrics: dict[str, object],
    predictions: pd.DataFrame,
    aggregate: bool,
) -> Path:
    """Write the mandatory self-contained provenance note for a downstream result."""
    commit, status = _git_provenance()
    cache_manifest = args.cache_dir / "temporal_cache_manifest.json"
    source_manifest = Path(str(manifest["source_manifest"]))
    cohort_exam_count = pd.read_csv(
        args.cohort_manifest, usecols=["exam_id"]
    )["exam_id"].nunique()
    fold_counts = (
        predictions.drop_duplicates("patient_key")["fold"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    artifacts = [
        f"- `{path.relative_to(args.outdir)}` SHA-256 `{_sha256(path)}`"
        for path in _result_artifacts(args.outdir, aggregate=aggregate)
        if path.is_file()
    ]
    if aggregate:
        result_rows = [
            f"| {arm} | {float(metrics[arm]['auc']):.6f} | "
            f"{float(metrics[arm]['ci_low']):.6f} | "
            f"{float(metrics[arm]['ci_high']):.6f} |"
            for arm in _ARM_SPECS
        ]
        delta_rows = [
            f"| {name} | {arm_a} - {arm_b} | "
            f"{float(metrics['paired_oof_auc_differences'][name]['delta_auc']):+.6f} | "
            f"{float(metrics['paired_oof_auc_differences'][name]['ci_low']):+.6f} | "
            f"{float(metrics['paired_oof_auc_differences'][name]['ci_high']):+.6f} |"
            for name, (arm_a, arm_b) in _COMPARISONS.items()
        ]
        title = "UChicago temporal-transfer downstream 2x2"
        status_line = "Complete: all four matched arms and paired OOF comparisons."
        results_text = f"""| Arm | Patient-level pooled OOF AUC | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|
{chr(10).join(result_rows)}

| Question | Contrast | Delta AUC | 95% CI low | 95% CI high |
|---|---|---:|---:|---:|
{chr(10).join(delta_rows)}"""
        interpretation = (
            "The paired contrasts, rather than any arm in isolation, decide whether "
            "pretraining or connectivity helps. A confidence interval spanning zero "
            "does not support a resolved improvement on this cohort."
        )
    else:
        arm = str(predictions["arm"].iloc[0])
        arm_metric = metrics[arm]
        title = f"UChicago temporal-transfer arm: {arm}"
        status_line = "Complete as one arm; scientific interpretation awaits aggregation."
        results_text = (
            "| Arm | Patient-level pooled OOF AUC | 95% CI low | 95% CI high |\n"
            "|---|---:|---:|---:|---:|\n"
            f"| {arm} | {float(arm_metric['auc']):.6f} | "
            f"{float(arm_metric['ci_low']):.6f} | "
            f"{float(arm_metric['ci_high']):.6f} |"
        )
        interpretation = (
            "Do not interpret this arm alone. The decision requires paired predictions "
            "from all four matched connectivity-by-initialization arms."
        )
    status_text = status if status else "clean"
    readme = f"""# {title}

## Status and question

{status_line} The experiment tests whether future-frame pretraining and/or vascular
message passing improves patient-level pCR discrimination.

## Results

{results_text}

## Data and split

- Anna-confirmed larger downstream cohort: `{args.cohort_manifest}`
- Downstream cohort SHA-256: `{_sha256(args.cohort_manifest)}`
- Audited temporal cache: `{args.cache_dir}`
- Cache manifest SHA-256: `{_sha256(cache_manifest)}`
- Cache source manifest: `{source_manifest}`
- Cache source manifest SHA-256: `{_sha256(source_manifest)}`
- GNN pretraining checkpoint: `{args.gnn_checkpoint}`
- GNN checkpoint SHA-256: `{_sha256(args.gnn_checkpoint)}`
- Graph-free pretraining checkpoint: `{args.graph_free_checkpoint}`
- Graph-free checkpoint SHA-256: `{_sha256(args.graph_free_checkpoint)}`
- Cached usable exams: {len(manifest['records'])} of {cohort_exam_count} cohort exams
- Patients: {predictions['patient_key'].nunique()}
- Outer split: predefined patient-grouped five-fold OOF; patient counts {fold_counts}
- Inner selection split: pCR-stratified patient split, fraction {args.inner_val_fraction},
  seed `seed + outer_fold`; outer labels are used only for final prediction.

## Matched configuration

- Epochs: {args.epochs}
- Learning rate: {args.learning_rate}
- Seed: {args.seed}
- Encoder: hidden dimension 32, two frame layers, one GRU layer, dropout 0.2
- Fine-tuning: full encoder and linear classification head from epoch one
- Patient prediction: mean of exam probabilities
- Primary metric: pooled patient-level OOF AUC
- Uncertainty: 2,000 paired patient-resampling bootstrap replicates

## Scheduler and artifacts

- Slurm job ID: `{os.environ.get('SLURM_JOB_ID', 'not available')}`
- Parallel arm job IDs: `{os.environ.get('ARM_JOB_IDS', 'not available')}`
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

## Interpretation and limits

{interpretation} This is one fixed cohort and one training seed. The existing tabular
reference is exam-level and is therefore contextual, not a paired patient-level control.
"""
    path = args.outdir / "README.md"
    path.write_text(readme)
    return path


def write_aggregate_provenance(outdir: Path) -> None:
    """Record scheduler evidence and checksums after all downstream arms complete."""
    provenance = outdir / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    snapshot = provenance / "source_snapshot.tar.gz"
    snapshot_line = (
        f"- `source_snapshot.tar.gz` SHA-256 `{_sha256(snapshot)}`"
        if snapshot.is_file()
        else "- Source snapshot was not present when aggregation ran."
    )
    (provenance / "README.md").write_text(
        "# Downstream 2x2 provenance\n\n"
        f"- Parallel arm job IDs: `{os.environ.get('ARM_JOB_IDS', 'not available')}`\n"
        f"- Aggregation job ID: `{os.environ.get('SLURM_JOB_ID', 'not available')}`\n"
        "- Slurm stdout/stderr files in this directory are named by job and ID.\n"
        f"{snapshot_line}\n"
        "- `artifact_checksums.sha256` verifies completed result artifacts.\n"
    )


def write_aggregate_checksums(outdir: Path) -> Path:
    """Write checksums for the finalized downstream outputs and source snapshot."""
    candidates = _result_artifacts(outdir, aggregate=True) + [
        outdir / "README.md",
        outdir / "provenance" / "README.md",
        outdir / "provenance" / "source_snapshot.tar.gz",
    ]
    lines = [
        f"{_sha256(path)}  {path.relative_to(outdir)}"
        for path in candidates
        if path.is_file()
    ]
    checksum_path = outdir / "provenance" / "artifact_checksums.sha256"
    checksum_path.write_text("\n".join(lines) + "\n")
    return checksum_path


def aggregate_parallel_results(
    *,
    args: argparse.Namespace,
    manifest: dict[str, object],
    patient_groups: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    """Combine parallel arm outputs and compute the predeclared paired readouts."""
    predictions = load_parallel_arm_predictions(args.outdir)
    metrics = _arm_metrics(predictions, seed=args.seed)
    metrics["paired_oof_auc_differences"] = {
        name: _paired_auc_delta(predictions, arm_a, arm_b, seed=args.seed)
        for name, (arm_a, arm_b) in _COMPARISONS.items()
    }
    protocol_predictions = protocol_only_oof(patient_groups)
    protocol_auc, protocol_low, protocol_high = _bootstrap_auc(
        protocol_predictions["y_true"].to_numpy(),
        protocol_predictions["y_prob"].to_numpy(),
        seed=args.seed,
    )
    metrics["protocol_only_audit"] = {
        "auc": protocol_auc,
        "ci_low": protocol_low,
        "ci_high": protocol_high,
        "features": ["n_frames", "duration_seconds", "median_dt_seconds"],
        "primary_model_residualized": False,
    }
    metrics["existing_tabular_reference"] = load_tabular_reference(
        args.tabular_reference
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "patient_predictions.csv", index=False)
    protocol_predictions.to_csv(
        args.outdir / "protocol_only_patient_predictions.csv", index=False
    )
    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_aggregate_provenance(args.outdir)
    write_finetune_readme(
        args=args,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
        aggregate=True,
    )
    write_aggregate_checksums(args.outdir)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse labeled-cache, checkpoint, and fine-tuning settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--gnn-checkpoint", type=Path, required=True)
    parser.add_argument("--graph-free-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--tabular-reference", type=Path, default=_DEFAULT_TABULAR_REFERENCE
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=("all", *_ARM_SPECS), required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.aggregate_only and args.arm != "all":
        parser.error("--aggregate-only requires --arm all")
    if not args.aggregate_only and args.arm == "all":
        parser.error("--arm all is reserved for --aggregate-only")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run one temporal arm or aggregate all four arms on fixed outer folds."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    manifest = load_temporal_cache_manifest(args.cache_dir)
    records = list(manifest["records"])
    validate_cache_against_cohort(records, args.cohort_manifest)
    patient_groups = _group_by_patient(records)
    graph_checkpoint = load_checkpoint(args.gnn_checkpoint)
    free_checkpoint = load_checkpoint(args.graph_free_checkpoint)
    validate_matched_encoder_configs(graph_checkpoint, free_checkpoint)
    if args.aggregate_only:
        aggregate_parallel_results(
            args=args, manifest=manifest, patient_groups=patient_groups
        )
        return

    checkpoint_by_kind = {"gnn": graph_checkpoint, "graph_free": free_checkpoint}
    kind, pretrained = _ARM_SPECS[args.arm]
    outer_folds = sorted({int(record["fold"]) for record in records})
    rows = []
    history_rows = []
    device = torch.device(args.device)
    for fold in outer_folds:
        outer_patients = sorted(
            patient
            for patient, exams in patient_groups.items()
            if int(exams[0]["fold"]) == fold
        )
        outer_patient_set = set(outer_patients)
        train_groups = {
            patient: exams
            for patient, exams in patient_groups.items()
            if patient not in outer_patient_set
        }
        torch.manual_seed(args.seed + fold)
        model, uses_graph = _build_classifier(
            checkpoint_by_kind[kind], pretrained=pretrained
        )
        model, history = fit_fold(
            model,
            train_groups,
            uses_graph=uses_graph,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            inner_val_fraction=args.inner_val_fraction,
            seed=args.seed + fold,
            device=device,
        )
        y, probability = predict_patients(
            model,
            patient_groups,
            outer_patients,
            uses_graph=uses_graph,
            device=device,
        )
        rows.extend(
            {
                "arm": args.arm,
                "fold": fold,
                "patient_key": patient,
                "y_true": int(label),
                "y_prob": float(prob),
                "n_exams": len(patient_groups[patient]),
            }
            for patient, label, prob in zip(
                outer_patients, y, probability, strict=True
            )
        )
        history_rows.extend({"fold": fold, **item} for item in history)

    predictions = pd.DataFrame(rows)
    metrics = _arm_metrics(predictions, seed=args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "patient_predictions.csv", index=False)
    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.outdir / "history.json").write_text(
        json.dumps({args.arm: history_rows}, indent=2) + "\n"
    )
    write_finetune_readme(
        args=args,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
        aggregate=False,
    )


if __name__ == "__main__":
    main()
