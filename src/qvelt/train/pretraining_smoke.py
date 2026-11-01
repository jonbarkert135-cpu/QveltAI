"""Original offline BPE-data optimizer smoke with atomic consumer-state replay."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qvelt.config.pretraining import PretrainingJob, TokenSchedule, dump_pretraining_job
from qvelt.config.training import load_training_job
from qvelt.data.demo import create_demo
from qvelt.data.loader import PackedDataset
from qvelt.train.pretraining import train_pretraining
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.checkpoints import latest_checkpoint


def demo_recipe(
    path: Path,
    repository: Path,
    *,
    steps: int = 20,
    accumulation: int = 2,
    workers: int = 0,
    world_size: int = 1,
) -> PretrainingJob:
    dataset = PackedDataset(path)
    base = load_training_job(repository / "configs/train/smoke.yaml")
    base = replace(
        base,
        experiment=replace(
            base.experiment,
            model=replace(
                base.experiment.model, vocab=dataset.vocab, context=dataset.context
            ),
        ),
        training=replace(
            base.training,
            steps=steps,
            sequence_length=dataset.context,
            warmup_steps=0,
            decay_steps=0,
            checkpoint_every=max(1, steps // 2),
        ),
    )
    capacity = (
        steps * accumulation * base.training.batch_size * dataset.context * world_size
    )
    warmup = min(
        capacity // 8,
        base.training.batch_size * dataset.context * accumulation * world_size * 2,
    )
    tail = max(1, capacity // 3)
    return PretrainingJob(
        base,
        str(path.resolve()),
        dataset.identity,
        TokenSchedule(warmup, capacity - tail, tail),
        accumulation_steps=accumulation,
        workers=workers,
        allow_demo=True,
        validation_batches=2,
        validation_every=max(1, steps // 2),
    )


def run_pretraining_smoke(
    directory: Path, repository: Path, *, steps: int = 20
) -> dict[str, Any]:
    if type(steps) is not int or steps < 3:
        raise ValueError("Replay smoke needs at least3 optimizer steps")
    root = create_directory(directory)
    create_demo(root / "data", repository, documents=96)
    recipe = demo_recipe(root / "data/prepared/manifest.json", repository, steps=steps)
    (root / "pretraining.yaml").write_text(
        dump_pretraining_job(recipe), encoding="utf-8"
    )
    full = train_pretraining(recipe, root / "continuous", repository)
    stop = max(1, steps // 3)
    train_pretraining(recipe, root / "replayed", repository, stop_after=stop)
    replay = train_pretraining(recipe, root / "replayed", repository, resume=True)
    parameter_count = 0
    for filename in ("model.safetensors", "optimizer.pt"):
        a, b = (
            latest_checkpoint(root / "continuous") / filename,
            latest_checkpoint(root / "replayed") / filename,
        )
        left = (
            load_file(str(a))
            if filename.endswith("safetensors")
            else torch.load(a, weights_only=True)
        )
        right = (
            load_file(str(b))
            if filename.endswith("safetensors")
            else torch.load(b, weights_only=True)
        )
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if filename.endswith("safetensors"):
            parameter_count = sum(value.numel() for value in left.values())
            assert parameter_count == recipe.job.experiment.model.parameter_count
    applications = [
        json.loads(
            (
                latest_checkpoint(root / name) / "application-rank-00000.json"
            ).read_bytes()
        )
        for name in ("continuous", "replayed")
    ]
    assert applications[0]["loader"] == applications[1]["loader"]
    assert full["processed_tokens"] == replay["processed_tokens"]
    result = {
        "scope": "original offline BPE-data engineering smoke, not reasoning quality",
        "parameters": parameter_count,
        "steps": steps,
        "stop_after": stop,
        "continuous": full,
        "replayed": replay,
        "weights_bit_exact": True,
        "optimizer_bit_exact": True,
        "consumer_state_exact": True,
    }
    ArtifactStore(root).add_bytes(
        "smoke-result.json",
        (json.dumps(result, sort_keys=True) + "\n").encode(),
        role="smoke_result",
    )
    return result
