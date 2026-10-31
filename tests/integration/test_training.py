"""Actual CPU training, checkpoint replay and corruption rejection."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from qvelt.config.training import TrainingJob, load_training_job
from qvelt.model.decoder import Decoder
from qvelt.train.optim import make_optimizer
from qvelt.train.runner import train_synthetic
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore
from qvelt.utils.checkpoints import latest_checkpoint, load_checkpoint
from qvelt.utils.distributed import DistributedContext

ROOT = Path(__file__).resolve().parents[2]


def short_job() -> TrainingJob:
    job = load_training_job(ROOT / "configs/train/smoke.yaml")
    return replace(
        job,
        training=replace(
            job.training, steps=10, warmup_steps=1, decay_steps=2, checkpoint_every=5
        ),
    )


def test_cpu_training_replay_and_rng(tmp_path: Path) -> None:
    job = short_job()
    full, replay = tmp_path / "full", tmp_path / "replay"
    train_synthetic(job, full, ROOT)
    train_synthetic(job, replay, ROOT, stop_after=4)
    python_next = random.random()
    numpy_next = float(np.random.rand())
    torch_next = torch.rand(3)
    torch.set_num_threads(job.training.num_threads)
    ctx = DistributedContext(0, 1, 0, torch.device("cpu"))
    model = Decoder(job.experiment.model)
    optimizer = make_optimizer(model, job.training)
    # Simulate random work between failure and restart, then load saved RNG.
    random.random()
    np.random.rand()
    torch.rand(3)
    assert load_checkpoint(model, optimizer, job, replay, ctx) == 4
    assert random.random() == python_next and float(np.random.rand()) == numpy_next
    torch.testing.assert_close(torch.rand(3), torch_next, rtol=0, atol=0)
    train_synthetic(job, replay, ROOT, resume=True)
    a = load_file(str(latest_checkpoint(full) / "model.safetensors"))
    b = load_file(str(latest_checkpoint(replay) / "model.safetensors"))
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert (replay / "pip-freeze.txt").exists() and (replay / "job.yaml").exists()
    for path in [latest_checkpoint(full), latest_checkpoint(replay)]:
        ArtifactStore(path).read_manifest()


def test_checkpoint_corruption_no_fallback(tmp_path: Path) -> None:
    job = short_job()
    directory = tmp_path / "run"
    train_synthetic(job, directory, ROOT)
    latest = latest_checkpoint(directory)
    (latest / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ArtifactIntegrityError, match="size/hash"):
        train_synthetic(job, directory, ROOT, resume=True)


def test_incomplete_checkpoint_ignored_and_config_mismatch(tmp_path: Path) -> None:
    job = short_job()
    directory = tmp_path / "run"
    train_synthetic(job, directory, ROOT, stop_after=4)
    (directory / "checkpoints/step-99999999").mkdir()
    assert latest_checkpoint(directory).name == "step-00000004"
    changed = replace(job, training=replace(job.training, learning_rate=0.002))
    with pytest.raises(ArtifactIntegrityError, match="compatibility"):
        train_synthetic(changed, directory, ROOT, resume=True)


def test_existing_training_output_is_not_overwritten(tmp_path: Path) -> None:
    job = short_job()
    directory = tmp_path / "run"
    directory.mkdir()
    sentinel = directory / "user-file"
    sentinel.write_text("preserve")
    with pytest.raises(RuntimeError, match="FileExistsError"):
        train_synthetic(job, directory, ROOT)
    assert sentinel.read_text() == "preserve"


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires actual CUDA hardware"
)
def test_single_gpu_training(tmp_path: Path) -> None:
    job = short_job()
    job = replace(
        job,
        experiment=replace(
            job.experiment,
            runtime=replace(job.experiment.runtime, device="cuda", deterministic=False),
        ),
    )
    result = train_synthetic(job, tmp_path / "gpu", ROOT)
    assert result["optimizer_step"] == 10


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires two actual CUDA GPUs"
)
def test_fsdp2_gpu_checkpoint_resume(tmp_path: Path) -> None:
    import subprocess
    import sys

    base = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "qvelt",
        "train",
        "configs/train/smoke.yaml",
        "--directory",
        str(tmp_path / "fsdp"),
        "--repository",
        str(ROOT),
        "--set",
        "training.strategy=fsdp",
        "--set",
        "training.steps=4",
        "--set",
        "training.warmup_steps=1",
        "--set",
        "training.decay_steps=1",
        "--set",
        "training.checkpoint_every=2",
        "--set",
        "experiment.runtime.device=cuda",
        "--set",
        "experiment.runtime.deterministic=false",
    ]
    for extra in [["--stop-after", "2"], ["--resume", "latest"]]:
        result = subprocess.run(
            base + extra, cwd=ROOT, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert latest_checkpoint(tmp_path / "fsdp").name == "step-00000004"


def test_nonfinite_loss_stops_before_optimizer_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any
    from unittest.mock import patch

    from torch import Tensor

    def poisoned(self: Decoder, token_ids: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return torch.full(
            (*token_ids.shape, self.config.vocab), float("nan"), requires_grad=True
        )

    monkeypatch.setattr(Decoder, "forward", poisoned)
    output = tmp_path / "nonfinite"
    with patch.object(torch.optim.AdamW, "step") as step:
        with pytest.raises(FloatingPointError, match="nonfinite"):
            train_synthetic(short_job(), output, ROOT)
        step.assert_not_called()
    assert not list((output / "checkpoints").glob("step-*/manifest.json"))
