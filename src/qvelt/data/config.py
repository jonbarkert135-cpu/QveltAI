"""Strict immutable recipe for approved local raw inputs and staged data loading."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

from qvelt.config import MAX_CONFIG_BYTES, ConfigError, _mapping, parse_yaml_mapping
from qvelt.data.cleaning import CleaningConfig
from qvelt.data.dedup import DedupConfig
from qvelt.tokenizer.core import TokenizerSettings
from qvelt.utils.manifest_types import digest, identifier

T = TypeVar("T")


def decode(cls: type[T], value: object) -> T:
    if not is_dataclass(cls):
        raise TypeError("Expected a config dataclass")
    mapping = _mapping(value, cls.__name__, {f.name for f in fields(cls)})
    try:
        return cast(T, cls(**cast(Any, mapping)))
    except TypeError as error:
        raise ConfigError("Missing or invalid config fields") from error


@dataclass(frozen=True)
class SourceSpec:
    name: str
    manifest: str
    sha256: str
    approved_for_training: bool
    fixture: bool = False

    def __post_init__(self) -> None:
        identifier(self.name, "source.name")
        digest(self.sha256, "source.sha256")
        if (
            not isinstance(self.manifest, str)
            or not self.manifest
            or len(self.manifest) > 2048
            or "\x00" in self.manifest
        ):
            raise ConfigError("Invalid source manifest path")
        if (
            type(self.approved_for_training) is not bool
            or type(self.fixture) is not bool
            or not self.approved_for_training
        ):
            raise ConfigError("Source is not approved for training use")


@dataclass(frozen=True)
class MixStage:
    start_step: int
    weights: tuple[tuple[str, float], ...]
    name: str = "pretrain"

    def __post_init__(self) -> None:
        if type(self.start_step) is not int or self.start_step < 0 or not self.weights:
            raise ConfigError("Invalid mixing stage")
        identifier(self.name, "stage.name")
        seen = set()
        for name, weight in self.weights:
            identifier(name, "stage.source")
            if (
                name in seen
                or type(weight) not in (int, float)
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ConfigError("Invalid mixing weight")
            seen.add(name)
        if sum(weight for _, weight in self.weights) <= 0:
            raise ConfigError("Mixing weights need positive mass")
        object.__setattr__(
            self,
            "weights",
            tuple(sorted((name, float(weight)) for name, weight in self.weights)),
        )


@dataclass(frozen=True)
class DataRecipe:
    sources: tuple[SourceSpec, ...]
    benchmark_registry: str
    benchmark_sha256: str
    mixing: tuple[MixStage, ...]
    purpose: str = "demo"
    schema_version: int = 1
    seed: int = 20261031
    context: int = 128
    shard_tokens: int = 1048576
    validation_fraction: float = 0.005
    test_fraction: float = 0.005
    ngram_size: int = 13
    tokenizer: TokenizerSettings = TokenizerSettings()
    cleaning: CleaningConfig = CleaningConfig()
    dedup: DedupConfig = DedupConfig()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tokenizer, TokenizerSettings)
            or not isinstance(self.cleaning, CleaningConfig)
            or not isinstance(self.dedup, DedupConfig)
        ):
            raise ConfigError("Nested policies must be validated dataclasses")
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.purpose not in ("demo", "production")
        ):
            raise ConfigError("Unsupported data recipe version/purpose")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ConfigError("Data seed must be uint32")
        if (
            type(self.sources) is not tuple
            or not all(isinstance(s, SourceSpec) for s in self.sources)
            or not self.sources
            or len(self.sources) > 64
            or len({s.name for s in self.sources}) != len(self.sources)
        ):
            raise ConfigError("Sources must be a bounded unique list")
        if self.purpose == "production" and any(s.fixture for s in self.sources):
            raise ConfigError("Fixture sources cannot pass production preparation")
        digest(self.benchmark_sha256, "benchmark registry identity")
        if (
            not isinstance(self.benchmark_registry, str)
            or not self.benchmark_registry
            or "\x00" in self.benchmark_registry
        ):
            raise ConfigError("Invalid benchmark registry path")
        for value in [self.context, self.shard_tokens, self.ngram_size]:
            if type(value) is not int or not 1 <= value <= 2**31 - 1:
                raise ConfigError("Invalid data shape/budget")
        for fraction in [self.validation_fraction, self.test_fraction]:
            if type(fraction) not in (int, float) or not 0 <= fraction < 1:
                raise ConfigError("Invalid split fraction")
        if self.ngram_size > 64:
            raise ConfigError("N-gram size cannot exceed 64")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ConfigError("Train split must have positive mass")
        if (
            type(self.mixing) is not tuple
            or not all(isinstance(s, MixStage) for s in self.mixing)
            or not self.mixing
            or self.mixing[0].start_step != 0
            or any(
                a.start_step >= b.start_step
                for a, b in zip(self.mixing, self.mixing[1:], strict=False)
            )
        ):
            raise ConfigError("Mix stages must start at zero and increase strictly")
        names = {s.name for s in self.sources}
        if any(
            set(name for name, _ in stage.weights) != names for stage in self.mixing
        ):
            raise ConfigError("Each stage must explicitly weight every source")


def parse_data_recipe(text: str) -> DataRecipe:
    root = _mapping(
        parse_yaml_mapping(text), "data recipe", {f.name for f in fields(DataRecipe)}
    )
    if not {"sources", "mixing", "benchmark_registry", "benchmark_sha256"} <= set(root):
        raise ConfigError("Data recipe is missing required fields")
    sources = root.pop("sources", None)
    stages = root.pop("mixing", None)
    if not isinstance(sources, list) or not isinstance(stages, list):
        raise ConfigError("Data sources and mixing must be lists")
    parsed_stages = []
    for stage in stages:
        m = _mapping(stage, "mix stage", {"start_step", "name", "weights"})
        weights = m.get("weights")
        if not isinstance(weights, dict):
            raise ConfigError("Stage weights must map source names to weights")
        parsed_stages.append(
            MixStage(
                cast(int, m.get("start_step")),
                tuple(weights.items()),
                cast(str, m.get("name", "pretrain")),
            )
        )
    tokenizer = decode(TokenizerSettings, root.pop("tokenizer", {}))
    cleaning = decode(CleaningConfig, root.pop("cleaning", {}))
    dedup = decode(DedupConfig, root.pop("dedup", {}))
    return DataRecipe(
        sources=tuple(decode(SourceSpec, source) for source in sources),
        mixing=tuple(parsed_stages),
        tokenizer=tokenizer,
        cleaning=cleaning,
        dedup=dedup,
        **cast(Any, root),
    )


def load_data_recipe(path: Path) -> DataRecipe:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Data recipe exceeds128 KiB")
    try:
        return parse_data_recipe(data.decode("utf-8-sig"))
    except UnicodeError as error:
        raise ConfigError("Data recipe must be UTF-8") from error


def dump_data_recipe(recipe: DataRecipe) -> str:
    value = asdict(recipe)
    value["mixing"] = [
        {"name": s.name, "start_step": s.start_step, "weights": dict(s.weights)}
        for s in recipe.mixing
    ]
    return yaml.safe_dump(value, sort_keys=True, allow_unicode=True)


def recipe_digest(recipe: DataRecipe) -> str:
    return hashlib.sha256(
        json.dumps(
            asdict(recipe), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
