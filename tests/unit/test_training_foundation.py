"""Configuration, fixtures, optimizer/schedule and compatibility boundaries."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from qvelt.config import ConfigError
from qvelt.config.training import (
    TrainingConfig,
    dump_training_job,
    load_training_job,
    parse_training_job,
    training_job_digest,
)
from qvelt.data.synthetic import synthetic_batch
from qvelt.model.decoder import Decoder
from qvelt.train.optim import flops_per_token, learning_rate, make_optimizer

ROOT = Path(__file__).resolve().parents[2]


def test_resolved_job_and_cli_overrides() -> None:
    path = ROOT / "configs/train/smoke.yaml"
    job = load_training_job(path)
    assert parse_training_job(dump_training_job(job)) == job
    changed = load_training_job(
        path, ["training.learning_rate=0.002", "experiment.runtime.seed=17"]
    )
    assert (
        changed.training.learning_rate == 0.002
        and changed.experiment.runtime.seed == 17
    )
    assert training_job_digest(changed) != training_job_digest(job)
    for values in [
        ["training.missing=1"],
        ["training.steps=true"],
        ["training.steps=3", "training.steps=4"],
        ["training={steps: 3}"],
        ["training.steps=2\nextra: 3"],
    ]:
        with pytest.raises(ConfigError):
            load_training_job(path, values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", 0),
        ("steps", True),
        ("learning_rate", float("nan")),
        ("weight_decay", float("inf")),
        ("warmup_steps", 101),
        ("decay_steps", -1),
        ("strategy", "tp"),
        ("metrics_backend", "unknown"),
        ("num_threads", 0),
    ],
)
def test_invalid_training_config(field: str, value: object) -> None:
    config = TrainingConfig()
    with pytest.raises(ConfigError):
        parse_training_job(
            yaml.safe_dump({"training": {**asdict(config), field: value}})
        )


def test_stateless_data_order_shift_and_rank_disjointness() -> None:
    a = synthetic_batch(2, 2, 3, 288, rank=0, world_size=2)
    b = synthetic_batch(2, 2, 3, 288, rank=1, world_size=2)
    assert a.sample_ids.tolist() == [8, 9] and b.sample_ids.tolist() == [10, 11]
    assert a.inputs.tolist() == [[40, 41, 42], [41, 42, 43]]
    assert a.targets.tolist() == [[41, 42, 43], [42, 43, 44]]
    torch.testing.assert_close(
        a.inputs,
        synthetic_batch(2, 2, 3, 288, rank=0, world_size=2).inputs,
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError):
        synthetic_batch(0, 1, 2, 288, rank=2, world_size=2)


def test_adamw_groups_count_tied_embedding_once() -> None:
    job = load_training_job(ROOT / "configs/train/smoke.yaml")
    model = Decoder(job.experiment.model)
    optimizer = make_optimizer(model, job.training)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in parameters}) == len(list(model.parameters()))
    assert sum(p.numel() for p in parameters) == job.experiment.model.parameter_count
    excluded = optimizer.param_groups[1]["params"]
    assert any(p is model.embedding.weight for p in excluded)
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0


def test_rfc_token_schedule_and_flops() -> None:
    job = load_training_job(ROOT / "configs/train/smoke.yaml")
    config = job.training
    assert learning_rate(config, 0, 64, 6400) == pytest.approx(0.0002)
    assert learning_rate(config, 4 * 64, 64, 6400) == pytest.approx(0.001)
    assert learning_rate(config, 50 * 64, 64, 6400) == pytest.approx(0.001)
    assert learning_rate(config, 99 * 64, 64, 6400) == pytest.approx(0.0001)
    with pytest.raises(ValueError):
        learning_rate(config, 6400, 64, 6400)
    spec = job.experiment.model
    actual = flops_per_token(spec, 32)
    # Count projection multiply-adds independently from listed tensor dimensions.
    shapes = [
        (288, 64),
        *[(64, 64), (32, 64), (32, 64), (64, 64), (192, 64), (192, 64), (64, 192)] * 2,
    ]
    reference = 6 * sum(a * b for a, b in shapes) + 12 * 2 * 32 * 64
    assert actual == reference


def test_legacy_import_identity() -> None:
    from qvelt.artifacts import ArtifactStore as OldStore
    from qvelt.tokenization import QveltTokenizer as OldTokenizer
    from qvelt.tokenizer.core import QveltTokenizer
    from qvelt.utils.artifacts import ArtifactStore

    assert OldStore is ArtifactStore and OldTokenizer is QveltTokenizer
