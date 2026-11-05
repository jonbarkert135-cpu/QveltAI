"""Turn a size preset plus an actual prepared dataset into a runnable recipe."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from qvelt.config import ConfigError, apply_overrides, parse_yaml_mapping
from qvelt.config.pretraining import PretrainingJob, parse_pretraining_job
from qvelt.data.loader import PackedDataset


def build_recipe(
    preset: Path,
    data_manifest: Path,
    *,
    world_size: int,
    overrides: Sequence[str] = (),
) -> PretrainingJob:
    """Fill data identity, accumulation and strategy; never invent a checksum.

    The preset fixes the global sequences per update; accumulation is derived as
    global / (batch_size * world_size) and must divide exactly.
    """
    raw = parse_yaml_mapping(preset.read_text(encoding="utf-8"))
    if set(raw) != {"preset", "global_batch_sequences", "recipe"}:
        raise ConfigError("Preset needs preset/global_batch_sequences/recipe")
    recipe = copy.deepcopy(raw["recipe"])
    if not isinstance(recipe, dict) or {"data_manifest", "data_sha256"} & set(recipe):
        raise ConfigError("Presets must not contain data identity fields")
    if type(world_size) is not int or world_size < 1:
        raise ConfigError("world_size must be a positive integer")
    dataset = PackedDataset(data_manifest)
    job: Any = recipe["job"]
    total = raw["global_batch_sequences"]
    if type(total) is not int or total < 1:
        raise ConfigError("global_batch_sequences must be positive")
    recipe["accumulation_steps"] = 1
    strategy = job["training"].get("strategy", "ddp")
    job["training"]["strategy"] = "single" if world_size == 1 else strategy
    recipe["data_manifest"] = str(data_manifest.resolve())
    recipe["data_sha256"] = dataset.identity
    resolved = parse_pretraining_job(yaml.safe_dump(recipe))
    resolved = parse_pretraining_job(
        yaml.safe_dump(apply_overrides(asdict(resolved), overrides))
    )
    if not any(value.split("=", 1)[0] == "accumulation_steps" for value in overrides):
        per_update = resolved.job.training.batch_size * world_size
        if total % per_update:
            raise ConfigError(
                "global_batch_sequences must be divisible by batch_size*world_size"
            )
        values = asdict(resolved)
        values["accumulation_steps"] = total // per_update
        resolved = parse_pretraining_job(yaml.safe_dump(values))
    model = resolved.job.experiment.model
    if (
        dataset.vocab != model.vocab
        or dataset.context != resolved.job.training.sequence_length
    ):
        raise ConfigError(
            f"Data vocab/context {dataset.vocab}/{dataset.context} disagree with "
            f"preset {model.vocab}/{resolved.job.training.sequence_length}"
        )
    return resolved
