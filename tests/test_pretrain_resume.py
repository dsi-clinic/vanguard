"""Deterministic checks for exact pretraining resume primitives."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from gnn.pretrain.resume import (  # noqa: E402
    load_arm_completion,
    load_epoch_progress,
    make_run_fingerprint,
    save_epoch_progress,
    validate_run_completion,
    write_arm_completion,
    write_run_completion,
)


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(5, 1),
    )


def _train_epochs(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    fingerprint: dict[str, Any],
    progress_path: Path,
    final_epoch: int,
) -> tuple[
    list[dict[str, float]],
    dict[str, torch.Tensor],
    float,
    int,
    dict[str, Any],
]:
    device = torch.device("cpu")
    resumed = load_epoch_progress(
        progress_path,
        expected_fingerprint=fingerprint,
        model=model,
        optimizer=optimizer,
        sampling_generator=generator,
        device=device,
    )
    if resumed is None:
        start_epoch = 1
        history: list[dict[str, float]] = []
        best_state = deepcopy(model.state_dict())
        best_metric = float("inf")
        best_epoch = 0
        best_metadata: dict[str, Any] = {}
    else:
        start_epoch = resumed.next_epoch
        history = resumed.history
        best_state = resumed.best_state_dict
        best_metric = resumed.best_metric
        best_epoch = resumed.best_epoch
        best_metadata = resumed.best_metadata

    features = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, -1.0, 0.5], [2.0, 0.0, -0.5], [-1.0, 1.5, 0.0]]
    )
    targets = torch.tensor([[0.5], [-0.25], [0.75], [0.0]])
    for epoch in range(start_epoch, final_epoch + 1):
        model.train()
        losses = []
        for index in torch.randperm(len(features), generator=generator).tolist():
            optimizer.zero_grad()
            loss = torch.square(model(features[index]) - targets[index]).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            metric = float(torch.square(model(features) - targets).mean())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": sum(losses) / len(losses),
                "metric": metric,
            }
        )
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_metadata = {"tag": f"epoch-{epoch}"}
        save_epoch_progress(
            progress_path,
            fingerprint=fingerprint,
            model=model,
            optimizer=optimizer,
            next_epoch=epoch + 1,
            history=history,
            best_state_dict=best_state,
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_metadata=best_metadata,
            sampling_generator=generator,
            device=device,
        )
    return history, best_state, best_metric, best_epoch, best_metadata


def test_interrupted_resume_exactly_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    """Model, history, best state, optimizer path, and RNG stream resume exactly."""
    fingerprint = make_run_fingerprint(
        {
            "model": {"hidden_dim": 5, "dropout": 0.25},
            "objective": "synthetic",
            "seed": 17,
            "manifest_sha256": "abc",
            "git_commit": "deadbeef",
        }
    )

    torch.manual_seed(17)
    uninterrupted_model = _model()
    uninterrupted_optimizer = torch.optim.Adam(
        uninterrupted_model.parameters(), lr=0.01
    )
    uninterrupted = _train_epochs(
        model=uninterrupted_model,
        optimizer=uninterrupted_optimizer,
        generator=torch.Generator().manual_seed(29),
        fingerprint=fingerprint,
        progress_path=tmp_path / "uninterrupted.pt",
        final_epoch=4,
    )

    torch.manual_seed(17)
    interrupted_model = _model()
    interrupted_optimizer = torch.optim.Adam(interrupted_model.parameters(), lr=0.01)
    interrupted_generator = torch.Generator().manual_seed(29)
    _train_epochs(
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        generator=interrupted_generator,
        fingerprint=fingerprint,
        progress_path=tmp_path / "resumed.pt",
        final_epoch=2,
    )

    torch.manual_seed(999)
    resumed_model = _model()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    resumed = _train_epochs(
        model=resumed_model,
        optimizer=resumed_optimizer,
        generator=torch.Generator().manual_seed(999),
        fingerprint=fingerprint,
        progress_path=tmp_path / "resumed.pt",
        final_epoch=4,
    )

    assert resumed[0] == uninterrupted[0]
    assert resumed[2:] == uninterrupted[2:]
    for name, value in uninterrupted_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name])
    for name, value in uninterrupted[1].items():
        assert torch.equal(value, resumed[1][name])
    assert not (tmp_path / ".resumed.pt.tmp").exists()


def test_resume_fails_before_loading_on_fingerprint_mismatch(tmp_path: Path) -> None:
    """A changed run contract cannot silently consume prior epoch state."""
    saved_fingerprint = make_run_fingerprint({"seed": 1, "objective": "a"})
    torch.manual_seed(1)
    source_model = _model()
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    _train_epochs(
        model=source_model,
        optimizer=source_optimizer,
        generator=torch.Generator().manual_seed(2),
        fingerprint=saved_fingerprint,
        progress_path=tmp_path / "progress.pt",
        final_epoch=1,
    )

    target_model = _model()
    before = deepcopy(target_model.state_dict())
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_epoch_progress(
            tmp_path / "progress.pt",
            expected_fingerprint=make_run_fingerprint({"seed": 2, "objective": "a"}),
            model=target_model,
            optimizer=torch.optim.Adam(target_model.parameters(), lr=0.01),
            sampling_generator=torch.Generator(),
            device=torch.device("cpu"),
        )
    for name, value in before.items():
        assert torch.equal(value, target_model.state_dict()[name])


def test_cuda_rng_states_are_cpu_byte_tensors_before_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUDA-mapped RNG states are restored through CPU uint8 ByteTensors."""
    fingerprint = make_run_fingerprint({"seed": 5, "device": "cuda"})
    source_model = _model()
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    _train_epochs(
        model=source_model,
        optimizer=source_optimizer,
        generator=torch.Generator().manual_seed(7),
        fingerprint=fingerprint,
        progress_path=tmp_path / "progress.pt",
        final_epoch=1,
    )
    payload = torch.load(tmp_path / "progress.pt", map_location="cpu")
    payload["rng_device_type"] = "cuda"
    payload["torch_cuda_rng_states"] = [
        torch.tensor([0, 1, 127, 255], dtype=torch.int64)
    ]
    restored: list[list[torch.Tensor]] = []
    monkeypatch.setattr(
        "gnn.pretrain.resume.torch.load", lambda *args, **kwargs: payload
    )
    monkeypatch.setattr("gnn.pretrain.resume.torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr(
        "gnn.pretrain.resume.torch.cuda.set_rng_state_all",
        lambda states: restored.append(states),
    )

    target_model = _model()
    resumed = load_epoch_progress(
        tmp_path / "progress.pt",
        expected_fingerprint=fingerprint,
        model=target_model,
        optimizer=torch.optim.Adam(target_model.parameters(), lr=0.01),
        sampling_generator=torch.Generator(),
        device=torch.device("cuda"),
    )

    assert resumed is not None
    assert len(restored) == 1
    assert len(restored[0]) == 1
    restored_state = restored[0][0]
    assert restored_state.device.type == "cpu"
    assert restored_state.dtype == torch.uint8
    assert isinstance(restored_state, torch.ByteTensor)
    assert restored_state.is_contiguous()
    assert restored_state.tolist() == [0, 1, 127, 255]


def test_completed_arm_requires_unchanged_validated_checkpoint(tmp_path: Path) -> None:
    """An arm marker never skips training around a missing or altered checkpoint."""
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"checkpoint")
    fingerprint = make_run_fingerprint({"arm": "gnn", "seed": 3})
    marker = write_arm_completion(
        tmp_path / "gnn_complete.json",
        fingerprint=fingerprint,
        checkpoint_path=checkpoint,
        result={"validation_mae": 0.25},
    )
    validated = []
    result = load_arm_completion(
        marker,
        expected_fingerprint=fingerprint,
        checkpoint_path=checkpoint,
        checkpoint_validator=lambda path: validated.append(path),
    )
    assert result == {"validation_mae": 0.25}
    assert validated == [checkpoint]

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        load_arm_completion(
            marker,
            expected_fingerprint=fingerprint,
            checkpoint_path=checkpoint,
        )


def test_run_completion_binds_every_final_artifact(tmp_path: Path) -> None:
    """The final sentinel is valid only while every audited output is unchanged."""
    first = tmp_path / "results.json"
    second = tmp_path / "README.md"
    first.write_text("{}\n")
    second.write_text("complete\n")
    fingerprint = make_run_fingerprint({"seed": 4})
    marker = write_run_completion(
        tmp_path / "COMPLETED.json",
        fingerprint=fingerprint,
        artifacts=[first, second],
    )
    assert validate_run_completion(marker, expected_fingerprint=fingerprint)
    second.write_text("changed\n")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_run_completion(marker, expected_fingerprint=fingerprint)
