"""Bounded named checkpoints and exact recipe bytes for matched-seed studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import yaml

from qvelt.config import (
    MAX_CONFIG_BYTES,
    ConfigError,
    _integer,
    _mapping,
    parse_yaml_mapping,
)
from qvelt.eval.config import ArtifactRef

ROLES = {"base", "sft", "dpo", "grpo", "dpo-half-lr"}


@dataclass(frozen=True)
class StudyEntry:
    role: str
    seed: int
    checkpoint: ArtifactRef

    def __post_init__(self) -> None:
        if (
            not isinstance(self.role, str)
            or self.role not in ROLES
            or not isinstance(self.checkpoint, ArtifactRef)
        ):
            raise ConfigError("Study entry needs a known role and checkpoint reference")
        _integer(self.seed, "study.seed", minimum=0, maximum=2**32 - 1)


@dataclass(frozen=True)
class StudyJob:
    evaluation_recipe: ArtifactRef
    entries: tuple[StudyEntry, ...]
    protocol: str = "qvelt-promotion-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.protocol != "qvelt-promotion-v1"
        ):
            raise ConfigError("Only qvelt-promotion-v1/schema1 is supported")
        if (
            not isinstance(self.evaluation_recipe, ArtifactRef)
            or type(self.entries) is not tuple
            or not 4 <= len(self.entries) <= 15
            or not all(isinstance(e, StudyEntry) for e in self.entries)
        ):
            raise ConfigError(
                "Study needs a recipe identity and4..15 immutable entries"
            )
        keys = {(e.role, e.seed) for e in self.entries}
        seeds = {e.seed for e in self.entries}
        roles = {e.role for e in self.entries}
        if (
            len(keys) != len(self.entries)
            or len({e.checkpoint.sha256 for e in self.entries}) != len(self.entries)
            or len(seeds) > 3
            or not {"base", "sft", "dpo", "grpo"} <= roles
            or any(
                {e.role for e in self.entries if e.seed == seed} != roles
                for seed in seeds
            )
        ):
            raise ConfigError(
                "Each of at most3 seeds needs every role once and unique checkpoints"
            )


def dump_study(job: StudyJob) -> str:
    return yaml.safe_dump(asdict(job), sort_keys=True)


def load_study(path: Path) -> StudyJob:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Study recipe exceeds128KiB")
    root = _mapping(
        parse_yaml_mapping(data.decode("utf-8-sig")),
        "study",
        {f.name for f in fields(StudyJob)},
    )
    if not {"evaluation_recipe", "entries"} <= set(root) or not isinstance(
        root["entries"], list
    ):
        raise ConfigError("Study requires evaluation recipe and entry list")

    def ref(value: object) -> ArtifactRef:
        body = _mapping(value, "study.reference", {"path", "sha256"})
        if set(body) != {"path", "sha256"}:
            raise ConfigError("Study reference requires path and sha256")
        return ArtifactRef(**cast(dict[str, Any], body))

    root["evaluation_recipe"] = ref(root["evaluation_recipe"])
    entries = []
    for value in root["entries"]:
        entry = _mapping(value, "study.entry", {"role", "seed", "checkpoint"})
        if set(entry) != {"role", "seed", "checkpoint"}:
            raise ConfigError("Study entry requires role,seed,checkpoint")
        entry["checkpoint"] = ref(entry["checkpoint"])
        entries.append(StudyEntry(**cast(dict[str, Any], entry)))
    root["entries"] = tuple(entries)
    return StudyJob(**cast(dict[str, Any], root))
