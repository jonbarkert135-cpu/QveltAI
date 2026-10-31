"""Bounded 100-update acceptance run plus interruption/replay comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qvelt.config.training import load_training_job
from qvelt.train.runner import train_synthetic
from qvelt.utils.checkpoints import latest_checkpoint


def _equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


def run_smoke(directory: Path, repository: Path) -> dict[str, object]:
    """Assert equal weights AND Adam states after an interrupted fixed-data run."""
    job = load_training_job(repository / "configs/train/smoke.yaml")
    full = directory / "continuous"
    replay = directory / "replayed"
    train_synthetic(job, full, repository)
    train_synthetic(job, replay, repository, stop_after=37)
    train_synthetic(job, replay, repository, resume=True)
    full_checkpoint = latest_checkpoint(full)
    replay_checkpoint = latest_checkpoint(replay)
    a, b = (
        load_file(str(full_checkpoint / "model.safetensors")),
        load_file(str(replay_checkpoint / "model.safetensors")),
    )
    _equal(a, b)
    _equal(
        torch.load(full_checkpoint / "optimizer.pt", weights_only=True),
        torch.load(replay_checkpoint / "optimizer.pt", weights_only=True),
    )
    logs = sorted(full.glob("metrics-*.jsonl"))
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert len(records) == 100 and records[-1]["step"] == 100
    assert records[-1]["loss"] < records[0]["loss"], (
        "Synthetic learning smoke did not improve its own fixture loss"
    )
    result = {
        "steps": 100,
        "parameters": job.experiment.model.parameter_count,
        "processed_tokens": records[-1]["processed_tokens"],
        "first_loss": records[0]["loss"],
        "last_loss": records[-1]["loss"],
        "replay_weights_equal": True,
        "replay_optimizer_equal": True,
    }
    (directory / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
