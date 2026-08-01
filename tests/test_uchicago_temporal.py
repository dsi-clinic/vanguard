"""Synthetic tests for the UChicago temporal-transfer path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from gnn.pretrain.model import ContrastForecastGNN  # noqa: E402
from gnn.pretrain.uchicago import (  # noqa: E402
    HORIZON,
    eligible_starts,
    make_window,
    split_records_by_patient,
    write_pretraining_readme,
)
from gnn.pretrain.uchicago import (  # noqa: E402
    parse_args as parse_pretrain_args,
)
from gnn.temporal_finetune import (  # noqa: E402
    load_parallel_arm_predictions,
    load_tabular_reference,
    protocol_only_oof,
    validate_cache_against_cohort,
    validate_matched_encoder_configs,
    write_finetune_readme,
)
from gnn.temporal_finetune import (  # noqa: E402
    parse_args as parse_finetune_args,
)
from gnn.temporal_model import (  # noqa: E402
    PerNodeTemporalEncoder,
    TemporalGraphClassifier,
    TemporalGraphEncoder,
    load_encoder_strict,
    masked_mean_max_pool,
)
from gnn.uchicago_temporal_data import (  # noqa: E402
    build_temporal_cache,
    exclude_labeled_patients,
)


def _exam() -> Data:
    series = torch.arange(3 * 7, dtype=torch.float).reshape(3, 7, 1)
    series[1, 5, 0] = float("nan")
    return Data(
        node_series=series,
        node_valid=torch.tensor([True, False, True]),
        times_seconds=torch.tensor([0.0, 5.0, 40.0, 45.0, 50.0, 55.0, 60.0]),
        baseline_frame_count=2,
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        exam_id="exam-a",
    )


def test_labeled_patients_are_removed_once_before_pretraining() -> None:
    """All exams of a labeled patient are absent from the SSL cohort."""
    pretrain = pd.DataFrame(
        {
            "exam_id": ["e1", "e2", "e3"],
            "patient_key": ["p1", "p1", "p2"],
            "dataset": ["u", "u", "u"],
        }
    )
    labeled = pd.DataFrame({"exam_id": ["l1"], "patient_key": ["p1"], "dataset": ["u"]})
    filtered = exclude_labeled_patients(pretrain, labeled)
    assert filtered["exam_id"].tolist() == ["e3"]


def test_temporal_cache_resume_reuses_and_validates_existing_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed cache rebuilds its manifest without recomputing valid graphs."""
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"exam_id": ["e1"], "patient_key": ["p1"], "dataset": ["u"]}
    ).to_csv(manifest_path, index=False)
    cached = Data(
        times_seconds=torch.tensor([0.0, 5.0, 10.0]),
        exam_id="e1",
        patient_key="p1",
        dataset="u",
    )

    def build_once(*args: object, **kwargs: object) -> Data:
        return cached

    monkeypatch.setattr(
        "gnn.uchicago_temporal_data.build_temporal_exam", build_once
    )
    cache_dir = tmp_path / "cache"
    build_temporal_cache(
        manifest_path=manifest_path,
        centerline_root=tmp_path,
        dce_root=tmp_path,
        cache_dir=cache_dir,
        write_manifest=False,
    )
    assert not (cache_dir / "temporal_cache_manifest.json").exists()

    def fail_if_rebuilt(*args: object, **kwargs: object) -> Data:
        raise AssertionError("resume recomputed an existing graph")

    monkeypatch.setattr(
        "gnn.uchicago_temporal_data.build_temporal_exam", fail_if_rebuilt
    )
    records = build_temporal_cache(
        manifest_path=manifest_path,
        centerline_root=tmp_path,
        dce_root=tmp_path,
        cache_dir=cache_dir,
        resume=True,
    )
    assert [record.exam_id for record in records] == ["e1"]

    wrong = torch.load(cache_dir / "exams" / "e1.pt")
    wrong.patient_key = "wrong"
    torch.save(wrong, cache_dir / "exams" / "e1.pt")
    with pytest.raises(ValueError, match="cached graph identity mismatch"):
        build_temporal_cache(
            manifest_path=manifest_path,
            centerline_root=tmp_path,
            dce_root=tmp_path,
            cache_dir=cache_dir,
            resume=True,
        )


def test_patient_split_keeps_longitudinal_exams_together() -> None:
    """Random pretraining splits operate on patient_key, never exam_id."""
    records = [
        {"exam_id": "a1", "patient_key": "a"},
        {"exam_id": "a2", "patient_key": "a"},
        {"exam_id": "b1", "patient_key": "b"},
        {"exam_id": "c1", "patient_key": "c"},
    ]
    train, val = split_records_by_patient(records, val_fraction=0.34, seed=3)
    assert {r["patient_key"] for r in train}.isdisjoint({r["patient_key"] for r in val})
    location = {r["exam_id"]: "train" for r in train} | {
        r["exam_id"]: "val" for r in val
    }
    assert location["a1"] == location["a2"]


def test_pretraining_cli_selects_one_isolated_arm(tmp_path: Path) -> None:
    """GPU-QoS-compatible jobs can train matched arms independently."""
    args = parse_pretrain_args(
        [
            "--cache-dir",
            str(tmp_path / "cache"),
            "--outdir",
            str(tmp_path / "out"),
            "--arm",
            "graph_free",
        ]
    )
    assert args.arm == "graph_free"


def test_pretraining_result_directory_gets_readme(tmp_path: Path) -> None:
    """A completed pretraining outdir is not handoff-ready without provenance."""
    cache_dir = tmp_path / "cache"
    outdir = tmp_path / "out"
    cache_dir.mkdir()
    outdir.mkdir()
    manifest = {
        "source_manifest": "/source.csv",
        "excluded_labeled_manifest": "/labeled.csv",
        "records": [{"exam_id": "e1"}],
    }
    (cache_dir / "temporal_cache_manifest.json").write_text("{}\n")
    args = parse_pretrain_args(
        [
            "--cache-dir",
            str(cache_dir),
            "--outdir",
            str(outdir),
            "--arm",
            "graph_free",
        ]
    )
    results = {
        "requested_arm": "graph_free",
        "graph_free": {
            "validation_mae": 0.2,
            "persistence_mae": 0.25,
            "beats_persistence": True,
            "best_epoch": 3,
        },
        "train_patient_count": 8,
        "val_patient_count": 2,
    }
    torch.save({}, outdir / "graph_free_encoder.pt")
    (outdir / "pretraining_results.json").write_text("{}\n")
    path = write_pretraining_readme(args=args, results=results, manifest=manifest)
    text = path.read_text()
    assert "# UChicago temporal pretraining result" in text
    assert "graph_free" in text
    assert "Train patients: 8" in text
    assert "Dirty-worktree status" in text


def test_window_policy_uses_b_minus_one_and_minutes_from_last_baseline() -> None:
    """Windows and physical-time anchoring match the saved UChicago plan."""
    data = _exam()
    assert eligible_starts(data) == [1, 2]
    window = make_window(data, 1)
    assert window.x_seq.shape == (3, HORIZON.input_len, 1)
    assert window.target.shape == (3, HORIZON.target_len)
    torch.testing.assert_close(
        window.acquisition_times, torch.tensor([0.0, 35.0 / 60.0, 40.0 / 60.0])
    )
    assert not window.target_mask[1].any()
    assert window.target_mask[0].all()
    assert torch.isfinite(window.target).all()


def test_masked_pool_excludes_invalid_nodes_from_mean_and_max() -> None:
    """Originally invalid nodes influence neither downstream pooling operator."""
    embedding = torch.tensor([[1.0, 2.0], [100.0, 100.0], [3.0, 0.0]])
    pooled = masked_mean_max_pool(embedding, torch.tensor([True, False, True]))
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 1.0, 3.0, 2.0]]))


def test_forecaster_and_classifier_share_exact_encoder_state() -> None:
    """Forecast weights load strictly into the identical downstream encoder."""
    forecaster = ContrastForecastGNN(
        in_channels=1, hidden_dim=8, num_layers=2, dropout=0.0
    )
    classifier_encoder = TemporalGraphEncoder(
        signal_channels=1,
        hidden_dim=8,
        num_gcn_layers=2,
        num_gru_layers=1,
        dropout=0.0,
    )
    checkpoint = {
        "encoder_config": forecaster.encoder.config(),
        "encoder_state_dict": forecaster.encoder.state_dict(),
    }
    load_encoder_strict(classifier_encoder, checkpoint)
    classifier = TemporalGraphClassifier(classifier_encoder)
    for expected, actual in zip(
        forecaster.encoder.parameters(), classifier.encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual)


def test_strict_transfer_rejects_architecture_mismatch() -> None:
    """A different hidden width fails before any partial weight load."""
    source = TemporalGraphEncoder(hidden_dim=8, dropout=0.0)
    target = TemporalGraphEncoder(hidden_dim=16, dropout=0.0)
    checkpoint = {
        "encoder_config": source.config(),
        "encoder_state_dict": source.state_dict(),
    }
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_encoder_strict(target, checkpoint)


def test_graph_and_free_checkpoints_must_be_architecturally_matched() -> None:
    """The 2x2 comparison cannot silently vary width or depth by arm."""
    graph = {"encoder_config": TemporalGraphEncoder(hidden_dim=8).config()}
    free = {"encoder_config": PerNodeTemporalEncoder(hidden_dim=8).config()}
    validate_matched_encoder_configs(graph, free)
    free["encoder_config"]["hidden_dim"] = 16
    with pytest.raises(ValueError, match="matched encoder configurations"):
        validate_matched_encoder_configs(graph, free)


def test_protocol_only_audit_is_patient_level_and_oof() -> None:
    """Protocol audit aggregates exams and respects the provided patient fold."""
    groups = {}
    for index in range(8):
        patient = f"p{index}"
        groups[patient] = [
            {
                "patient_key": patient,
                "fold": index % 2,
                "pcr": (index // 2) % 2,
                "n_frames": 10 + index,
                "duration_seconds": 50.0 + index,
                "median_dt_seconds": 5.0,
            }
        ]
    predictions = protocol_only_oof(groups)
    assert predictions["patient_key"].nunique() == len(groups)
    assert predictions["y_prob"].between(0.0, 1.0).all()


def test_existing_tabular_reference_selects_best_tabular_row(tmp_path: Path) -> None:
    """Historical non-tabular rows cannot become the declared reference."""
    path = tmp_path / "reference.csv"
    pd.DataFrame(
        [
            {
                "model": "gnn_old",
                "pooled_oof_auc": 0.9,
                "ci_lo": 0.8,
                "ci_hi": 1.0,
            },
            {
                "model": "tabular_vessel",
                "pooled_oof_auc": 0.6,
                "ci_lo": 0.5,
                "ci_hi": 0.7,
            },
        ]
    ).to_csv(path, index=False)
    reference = load_tabular_reference(path)
    assert reference["model"] == "tabular_vessel"


def _finetune_cli_paths(tmp_path: Path) -> list[str]:
    return [
        "--cache-dir",
        str(tmp_path / "cache"),
        "--cohort-manifest",
        str(tmp_path / "cohort.csv"),
        "--gnn-checkpoint",
        str(tmp_path / "gnn.pt"),
        "--graph-free-checkpoint",
        str(tmp_path / "free.pt"),
        "--outdir",
        str(tmp_path / "out"),
    ]


def test_finetune_cli_separates_parallel_arms_from_aggregation(
    tmp_path: Path,
) -> None:
    """Each GPU job runs one arm and only the dependent CPU job aggregates."""
    arm_args = parse_finetune_args(
        [*_finetune_cli_paths(tmp_path), "--arm", "gnn_pretrained"]
    )
    assert arm_args.arm == "gnn_pretrained"
    assert not arm_args.aggregate_only
    aggregate_args = parse_finetune_args(
        [
            *_finetune_cli_paths(tmp_path),
            "--arm",
            "all",
            "--aggregate-only",
        ]
    )
    assert aggregate_args.aggregate_only


def test_labeled_cache_must_match_confirmed_downstream_cohort(
    tmp_path: Path,
) -> None:
    """Fine-tuning fails before training if identity, label, or fold drifted."""
    cohort_path = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "exam_id": ["e1", "e2"],
            "patient_key": ["p1", "p2"],
            "fold": [0, 1],
            "pcr": [0, 1],
        }
    ).to_csv(cohort_path, index=False)
    records = [
        {"exam_id": "e1", "patient_key": "p1", "fold": 0, "pcr": 0},
        {"exam_id": "e2", "patient_key": "p2", "fold": 1, "pcr": 1},
    ]
    validate_cache_against_cohort(records, cohort_path)
    records[1]["fold"] = 0
    with pytest.raises(ValueError, match="identity/split mismatch"):
        validate_cache_against_cohort(records, cohort_path)


def test_parallel_prediction_loader_requires_identical_complete_arms(
    tmp_path: Path,
) -> None:
    """Aggregation cannot silently accept a missing patient or wrong arm label."""
    arms = (
        "graph_free_random",
        "graph_free_pretrained",
        "gnn_random",
        "gnn_pretrained",
    )
    for arm in arms:
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        pd.DataFrame(
            {
                "arm": [arm, arm],
                "fold": [0, 1],
                "patient_key": ["p0", "p1"],
                "y_true": [0, 1],
                "y_prob": [0.2, 0.8],
                "n_exams": [1, 1],
            }
        ).to_csv(arm_dir / "patient_predictions.csv", index=False)
    predictions = load_parallel_arm_predictions(tmp_path)
    assert len(predictions) == 2 * len(arms)
    assert set(predictions["arm"]) == set(arms)

    broken = pd.read_csv(tmp_path / "gnn_pretrained" / "patient_predictions.csv")
    broken.iloc[:1].to_csv(
        tmp_path / "gnn_pretrained" / "patient_predictions.csv", index=False
    )
    with pytest.raises(ValueError, match="identical patient sets"):
        load_parallel_arm_predictions(tmp_path)


def test_finetune_arm_result_gets_provenance_readme(tmp_path: Path) -> None:
    """A completed fine-tuning arm writes its required result README."""
    cache_dir = tmp_path / "cache"
    outdir = tmp_path / "out"
    cache_dir.mkdir()
    outdir.mkdir()
    source_manifest = tmp_path / "usable.csv"
    cohort_manifest = tmp_path / "cohort.csv"
    pd.DataFrame({"exam_id": ["e0", "e1"]}).to_csv(
        source_manifest, index=False
    )
    pd.DataFrame({"exam_id": ["e0", "e1"]}).to_csv(
        cohort_manifest, index=False
    )
    (cache_dir / "temporal_cache_manifest.json").write_text("{}\n")
    (tmp_path / "gnn.pt").write_bytes(b"gnn")
    (tmp_path / "free.pt").write_bytes(b"free")
    args = parse_finetune_args(
        [
            *_finetune_cli_paths(tmp_path),
            "--arm",
            "graph_free_random",
            "--device",
            "cpu",
        ]
    )
    predictions = pd.DataFrame(
        {
            "arm": ["graph_free_random", "graph_free_random"],
            "fold": [0, 1],
            "patient_key": ["p0", "p1"],
            "y_true": [0, 1],
            "y_prob": [0.25, 0.75],
            "n_exams": [1, 1],
        }
    )
    predictions.to_csv(outdir / "patient_predictions.csv", index=False)
    metrics = {
        "graph_free_random": {"auc": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    }
    (outdir / "metrics.json").write_text("{}\n")
    (outdir / "history.json").write_text("{}\n")
    manifest = {
        "source_manifest": str(source_manifest),
        "records": [{"exam_id": "e0"}, {"exam_id": "e1"}],
    }
    readme = write_finetune_readme(
        args=args,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
        aggregate=False,
    )
    text = readme.read_text()
    assert "graph_free_random" in text
    assert "Anna-confirmed larger downstream cohort" in text
    assert "Dirty-worktree status" in text
