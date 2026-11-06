"""Resolved synthetic training jobs embedding the single experiment schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import cast

import yaml

from qvelt.config import (
    MAX_CONFIG_BYTES,
    ConfigError,
    ExperimentConfig,
    _integer,
    _mapping,
    _number,
    apply_overrides,
    parse_config,
    parse_yaml_mapping,
)
from qvelt.config.execution import ModelExecution
from qvelt.utils.manifest_types import digest


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed-size synthetic updates align warmup/tail boundaries exactly."""

    steps: int = 100
    batch_size: int = 2
    sequence_length: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 5
    decay_steps: int = 10
    checkpoint_every: int = 50
    num_threads: int = 1
    strategy: str = "single"
    metrics_backend: str = "json"
    peak_tflops_per_gpu: float | None = None
    fused_optimizer: bool = False
    compile: bool = False
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    regression_recipe: str | None = None
    regression_sha256: str | None = None
    execution: ModelExecution = field(default_factory=ModelExecution, kw_only=True)

    def __post_init__(self) -> None:
        if (self.regression_recipe is None) != (self.regression_sha256 is None):
            raise ConfigError(
                "Regression evaluation requires recipe and digest together"
            )
        if self.regression_recipe is not None:
            if (
                not isinstance(self.regression_recipe, str)
                or not 1 <= len(self.regression_recipe) <= 2048
                or any(ord(c) < 32 for c in self.regression_recipe)
            ):
                raise ConfigError("Invalid regression recipe path")
            digest(self.regression_sha256, "training.regression_sha256")
            if self.strategy != "single":
                raise ConfigError(
                    "Checkpoint regression currently requires single-rank CPU checkpoints"
                )
        if not isinstance(self.execution, ModelExecution):
            raise ConfigError("training.execution requires ModelExecution")
        for name in [
            "steps",
            "batch_size",
            "sequence_length",
            "checkpoint_every",
            "num_threads",
        ]:
            _integer(getattr(self, name), "training." + name)
        for name in ["warmup_steps", "decay_steps"]:
            _integer(getattr(self, name), "training." + name, minimum=0)
        if self.warmup_steps + self.decay_steps > self.steps:
            raise ConfigError("Warmup and decay exceed the training horizon")
        for name in ["learning_rate", "grad_clip"]:
            object.__setattr__(
                self, name, _number(getattr(self, name), "training." + name)
            )
        if (
            type(self.weight_decay) not in (int, float)
            or not 0 <= self.weight_decay < 1
        ):
            raise ConfigError("training.weight_decay must be finite in [0,1)")
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        if self.strategy not in (
            "single",
            "ddp",
            "fsdp",
        ) or self.metrics_backend not in ("json", "tensorboard", "wandb"):
            raise ConfigError("Unsupported training strategy or metrics backend")
        if type(self.fused_optimizer) is not bool or type(self.compile) is not bool:
            raise ConfigError("training.fused_optimizer/compile must be bool")
        for name in ("tensor_parallel", "pipeline_parallel"):
            _integer(getattr(self, name), "training." + name)
            if getattr(self, name) != 1:
                raise ConfigError(
                    f"training.{name} is a reserved hook: tensor/pipeline "
                    "parallelism is not implemented (ADR-0042); use 1"
                )
        if self.peak_tflops_per_gpu is not None:
            object.__setattr__(
                self,
                "peak_tflops_per_gpu",
                _number(self.peak_tflops_per_gpu, "training.peak_tflops_per_gpu"),
            )


@dataclass(frozen=True)
class TrainingJob:
    """Schema-1 training recipe; the experiment schema is reused, not duplicated."""

    experiment: ExperimentConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigError("Only training-job schema1 is supported")
        if not isinstance(self.experiment, ExperimentConfig) or not isinstance(
            self.training, TrainingConfig
        ):
            raise ConfigError("TrainingJob requires validated dataclasses")
        if self.training.sequence_length > self.experiment.model.context:
            raise ConfigError("Training sequence exceeds model context")
        if self.training.regression_recipe is not None and (
            self.experiment.runtime.device != "cpu"
            or self.experiment.runtime.precision != "fp32"
            or self.experiment.model.parameter_count > 20_000_000
        ):
            raise ConfigError(
                "Checkpoint regression currently supports bounded CPU FP32 models"
            )


def parse_training_job(text: str) -> TrainingJob:
    root = _mapping(
        parse_yaml_mapping(text), "job", {"schema_version", "experiment", "training"}
    )
    experiment = parse_config(yaml.safe_dump(root.get("experiment", {})))
    m = _mapping(
        root.get("training", {}), "training", {f.name for f in fields(TrainingConfig)}
    )
    d = asdict(TrainingConfig())
    d.update(m)
    e = _mapping(
        d["execution"],
        "training.execution",
        {"attention_backend", "activation_checkpointing"},
    )
    execution = ModelExecution(
        attention_backend=cast(str, e.get("attention_backend", "reference")),
        activation_checkpointing=cast(bool, e.get("activation_checkpointing", False)),
    )
    return TrainingJob(
        experiment,
        TrainingConfig(
            execution=execution,
            steps=cast(int, d["steps"]),
            batch_size=cast(int, d["batch_size"]),
            sequence_length=cast(int, d["sequence_length"]),
            learning_rate=cast(float, d["learning_rate"]),
            weight_decay=cast(float, d["weight_decay"]),
            grad_clip=cast(float, d["grad_clip"]),
            warmup_steps=cast(int, d["warmup_steps"]),
            decay_steps=cast(int, d["decay_steps"]),
            checkpoint_every=cast(int, d["checkpoint_every"]),
            num_threads=cast(int, d["num_threads"]),
            strategy=cast(str, d["strategy"]),
            metrics_backend=cast(str, d["metrics_backend"]),
            peak_tflops_per_gpu=cast(float | None, d["peak_tflops_per_gpu"]),
            fused_optimizer=cast(bool, d["fused_optimizer"]),
            compile=cast(bool, d["compile"]),
            tensor_parallel=cast(int, d["tensor_parallel"]),
            pipeline_parallel=cast(int, d["pipeline_parallel"]),
            regression_recipe=cast(str | None, d["regression_recipe"]),
            regression_sha256=cast(str | None, d["regression_sha256"]),
        ),
        schema_version=cast(int, root.get("schema_version", 1)),
    )


def load_training_job(path: str | Path, overrides: Sequence[str] = ()) -> TrainingJob:
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Training job exceeds config byte limit")
    try:
        job = parse_training_job(data.decode("utf-8-sig"))
    except UnicodeError as error:
        raise ConfigError("Training job must be UTF-8") from error
    return parse_training_job(yaml.safe_dump(apply_overrides(asdict(job), overrides)))


def dump_training_job(job: TrainingJob) -> str:
    return yaml.safe_dump(asdict(job), sort_keys=True, allow_unicode=True)


def training_job_digest(job: TrainingJob) -> str:
    return hashlib.sha256(
        json.dumps(
            asdict(job), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
