"""Strict packed-data training recipe with explicit valid-token LR boundaries."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import yaml

from qvelt.config import (
    MAX_CONFIG_BYTES,
    ConfigError,
    _integer,
    _mapping,
    apply_overrides,
    parse_yaml_mapping,
)
from qvelt.config.training import TrainingJob, parse_training_job
from qvelt.utils.manifest_types import digest


@dataclass(frozen=True)
class TokenSchedule:
    warmup_tokens: int
    decay_start_tokens: int
    decay_tokens: int
    shape: str = "wsd"
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.shape not in ("wsd", "cosine"):
            raise ConfigError("schedule.shape must be wsd or cosine")
        if (
            type(self.min_lr_ratio) not in (int, float)
            or not math.isfinite(self.min_lr_ratio)
            or not 0 <= self.min_lr_ratio < 1
        ):
            raise ConfigError("schedule.min_lr_ratio must be finite in [0,1)")
        object.__setattr__(self, "min_lr_ratio", float(self.min_lr_ratio))
        _integer(
            self.warmup_tokens, "schedule.warmup_tokens", minimum=0, maximum=2**63 - 1
        )
        _integer(
            self.decay_start_tokens,
            "schedule.decay_start_tokens",
            minimum=0,
            maximum=2**63 - 1,
        )
        _integer(self.decay_tokens, "schedule.decay_tokens", maximum=2**63 - 1)
        if self.decay_start_tokens < self.warmup_tokens:
            raise ConfigError("Decay cannot start before warmup ends")


@dataclass(frozen=True)
class PretrainingJob:
    job: TrainingJob
    data_manifest: str
    data_sha256: str
    schedule: TokenSchedule
    accumulation_steps: int = 1
    total_batches: int | None = None
    workers: int = 0
    prefetch: int = 2
    attention_mode: str = "document"
    validation_batches: int = 4
    validation_every: int = 10
    allow_demo: bool = False
    nonfinite_policy: str = "abort"
    max_nonfinite_windows: int = 8
    spike_threshold: float = 6.0
    spike_warmup_windows: int = 20
    z_loss: float = 0.0
    layer_stats_every: int = 0
    domain_validation_batches: int = 0
    sample_every: int = 0
    sample_prompts: tuple[str, ...] = ()
    sample_tokens: int = 32
    keep_checkpoints: int = 0
    schema_version: int = 1
    objective: str = "pretraining"
    initial_checkpoint: str | None = None
    initial_sha256: str | None = None
    lora_rank: int = 0
    lora_alpha: float = 16.0

    def __post_init__(self) -> None:
        _integer(self.lora_rank, "sft.lora_rank", minimum=0, maximum=64)
        if (
            type(self.lora_alpha) not in (int, float)
            or not math.isfinite(self.lora_alpha)
            or not 0 < self.lora_alpha <= 128
        ):
            raise ConfigError("LoRA alpha must be finite in (0,128]")
        if self.lora_rank and (
            self.objective != "sft"
            or self.job.training.strategy != "single"
            or self.job.experiment.runtime.device != "cpu"
            or self.job.experiment.runtime.precision != "fp32"
            or self.job.training.compile
            or self.job.training.regression_recipe is not None
        ):
            raise ConfigError(
                "LoRA currently requires single-process CPU FP32 SFT without compile"
            )
        if self.objective not in ("pretraining", "sft"):
            raise ConfigError("Unknown packed training objective")
        if (self.initial_checkpoint is None) != (self.initial_sha256 is None):
            raise ConfigError(
                "Initial checkpoint path and digest must be supplied together"
            )
        if self.initial_checkpoint is not None:
            if (
                not isinstance(self.initial_checkpoint, str)
                or not self.initial_checkpoint
                or len(self.initial_checkpoint) > 2048
                or any(ord(c) < 32 for c in self.initial_checkpoint)
            ):
                raise ConfigError("Invalid initial checkpoint path")
            digest(self.initial_sha256, "initial_sha256")
        if self.objective == "sft" and (
            self.initial_checkpoint is None or self.attention_mode != "document"
        ):
            raise ConfigError("SFT requires pinned base weights and document attention")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigError("Unsupported pretraining schema")
        if not isinstance(self.job, TrainingJob) or not isinstance(
            self.schedule, TokenSchedule
        ):
            raise ConfigError("Pretraining requires validated job/schedule dataclasses")
        if (
            not isinstance(self.data_manifest, str)
            or not self.data_manifest
            or len(self.data_manifest) > 2048
            or any(ord(c) < 32 for c in self.data_manifest)
        ):
            raise ConfigError("Invalid data manifest path")
        digest(self.data_sha256, "pretrain.data_sha256")
        for name in (
            "accumulation_steps",
            "prefetch",
            "validation_batches",
            "validation_every",
        ):
            _integer(getattr(self, name), "pretrain." + name)
        _integer(self.workers, "pretrain.workers", minimum=0)
        if (
            self.workers > 16
            or self.prefetch > 64
            or self.attention_mode not in ("document", "causal")
        ):
            raise ConfigError("Invalid pretraining loader policy")
        if type(self.allow_demo) is not bool:
            raise ConfigError("allow_demo must be bool")
        if self.nonfinite_policy not in ("abort", "skip", "rollback"):
            raise ConfigError("nonfinite_policy must be abort, skip or rollback")
        for name in (
            "max_nonfinite_windows",
            "spike_warmup_windows",
            "layer_stats_every",
            "domain_validation_batches",
            "sample_every",
            "keep_checkpoints",
        ):
            _integer(getattr(self, name), "pretrain." + name, minimum=0)
        _integer(self.sample_tokens, "pretrain.sample_tokens", maximum=4096)
        for name, upper in (("spike_threshold", 1e6), ("z_loss", 1.0)):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value < upper
            ):
                raise ConfigError(f"pretrain.{name} must be finite in [0,{upper})")
            object.__setattr__(self, name, float(value))
        if (
            not isinstance(self.sample_prompts, list | tuple)
            or len(self.sample_prompts) > 16
            or any(
                not isinstance(prompt, str) or not 0 < len(prompt) <= 1024
                for prompt in self.sample_prompts
            )
        ):
            raise ConfigError("sample_prompts must be at most16 nonempty strings")
        object.__setattr__(self, "sample_prompts", tuple(self.sample_prompts))
        if self.sample_every and not self.sample_prompts:
            raise ConfigError("sample_every requires sample_prompts")
        if self.total_batches is not None:
            _integer(self.total_batches, "pretrain.total_batches")
            if (
                self.total_batches + self.accumulation_steps - 1
            ) // self.accumulation_steps != self.job.training.steps:
                raise ConfigError(
                    "total_batches/accumulation must match optimizer-step horizon"
                )
        if (
            self.batches
            * self.job.training.batch_size
            * self.job.training.sequence_length
            > 2**63 - 1
        ):
            raise ConfigError("Token capacity exceeds int64 counters")
        if self.job.training.warmup_steps or self.job.training.decay_steps:
            raise ConfigError(
                "Packed jobs require legacy warmup_steps=decay_steps=0; use TokenSchedule"
            )

    @property
    def batches(self) -> int:
        return (
            self.total_batches
            if self.total_batches is not None
            else self.job.training.steps * self.accumulation_steps
        )


def parse_pretraining_job(text: str) -> PretrainingJob:
    raw = _mapping(
        parse_yaml_mapping(text),
        "pretraining",
        {f.name for f in fields(PretrainingJob)},
    )
    for required in ("job", "data_manifest", "data_sha256", "schedule"):
        if required not in raw:
            raise ConfigError("Pretraining recipe missing required fields")
    job = parse_training_job(yaml.safe_dump(raw.pop("job")))
    schedule = _mapping(
        raw.pop("schedule"), "TokenSchedule", {f.name for f in fields(TokenSchedule)}
    )
    if not {"warmup_tokens", "decay_start_tokens", "decay_tokens"} <= set(schedule):
        raise ConfigError("All token-schedule boundaries must be explicit")
    return PretrainingJob(
        job, schedule=TokenSchedule(**cast(Any, schedule)), **cast(Any, raw)
    )


def load_pretraining_job(path: Path, overrides: Sequence[str] = ()) -> PretrainingJob:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Pretraining recipe exceeds128KiB")
    parsed = parse_pretraining_job(data.decode("utf-8-sig"))
    return parse_pretraining_job(
        yaml.safe_dump(apply_overrides(asdict(parsed), overrides))
    )


def dump_pretraining_job(recipe: PretrainingJob) -> str:
    return yaml.safe_dump(asdict(recipe), sort_keys=True, allow_unicode=True)


def pretraining_digest(recipe: PretrainingJob) -> str:
    values = asdict(recipe)
    for key in ("data_manifest", "workers", "prefetch", "initial_checkpoint"):
        values.pop(key)
    return hashlib.sha256(
        json.dumps(
            values, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
