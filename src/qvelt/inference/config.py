"""Strict immutable CPU request limits and versioned native generation recipes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
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
from qvelt.tokenizer.core import _validate_text
from qvelt.utils.manifest_types import identifier


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 32
    strategy: str = "greedy"
    temperature: float = 0.0
    seed: int = 20261106
    stop_tokens: tuple[int, ...] = (2, 6)
    stop_strings: tuple[str, ...] = ()
    context_limit: int = 2048
    max_cache_bytes: int = 64 * 1024**2
    max_seconds: float = 30.0
    max_output_chars: int = 8192
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self) -> None:
        for key, low, high in (
            ("top_k", 0, 262144),
            ("max_new_tokens", 1, 256),
            ("seed", 0, 2**32 - 1),
            ("context_limit", 4, 2048),
            ("max_cache_bytes", 1, 128 * 1024**2),
            ("max_output_chars", 1, 16384),
        ):
            _integer(getattr(self, key), "generation." + key, minimum=low, maximum=high)
        for key, scalar_low, scalar_high in (
            ("top_p", 1e-6, 1),
            ("min_p", 0, 1),
            ("repetition_penalty", 0.01, 100),
            ("frequency_penalty", -2, 2),
            ("presence_penalty", -2, 2),
        ):
            value = getattr(self, key)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or not scalar_low <= value <= scalar_high
            ):
                raise ConfigError("Invalid generation sampling control:" + key)
            object.__setattr__(self, key, float(value))
        if self.strategy not in ("greedy", "sample"):
            raise ConfigError("Generation strategy must be greedy or sample")
        for key in ("temperature", "max_seconds"):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ConfigError("Generation scalars must be finite numbers")
            object.__setattr__(self, key, float(value))
        if (
            not 0 < self.max_seconds <= 300
            or self.strategy == "greedy"
            and self.temperature != 0
            or self.strategy == "sample"
            and not 1e-4 <= self.temperature <= 2
        ):
            raise ConfigError("Invalid generation temperature/time budget")
        if (
            type(self.stop_tokens) is not tuple
            or len(self.stop_tokens) > 8
            or any(type(t) is not int or not 0 <= t < 262144 for t in self.stop_tokens)
            or len(set(self.stop_tokens)) != len(self.stop_tokens)
        ):
            raise ConfigError("Stop tokens must be at most8 unique nonnegative IDs")
        if (
            type(self.stop_strings) is not tuple
            or len(self.stop_strings) > 4
            or any(
                not isinstance(s, str) or not 1 <= len(s) <= 64
                for s in self.stop_strings
            )
            or len(set(self.stop_strings)) != len(self.stop_strings)
        ):
            raise ConfigError("Stop strings must be at most4 unique bounded strings")
        for value in self.stop_strings:
            _validate_text(value)


@dataclass(frozen=True)
class EngineLimits:
    max_batch: int = 8
    max_context: int = 2048
    max_total_tokens: int = 8192
    max_batch_cache_bytes: int = 128 * 1024**2

    def __post_init__(self) -> None:
        for key, high in (
            ("max_batch", 8),
            ("max_context", 2048),
            ("max_total_tokens", 8192),
            ("max_batch_cache_bytes", 128 * 1024**2),
        ):
            _integer(getattr(self, key), "inference." + key, minimum=1, maximum=high)


@dataclass(frozen=True)
class TokenRequest:
    identifier: str
    tokens: tuple[int, ...]
    options: GenerationOptions = field(default_factory=GenerationOptions)

    def __post_init__(self) -> None:
        identifier(self.identifier, "generation.request")
        if (
            not isinstance(self.options, GenerationOptions)
            or type(self.tokens) is not tuple
            or not 1 <= len(self.tokens) <= 2048
            or any(type(t) is not int or t < 0 for t in self.tokens)
        ):
            raise ConfigError(
                "Token request requires validated options and bounded immutable IDs"
            )


@dataclass(frozen=True)
class TextRequest:
    identifier: str
    prompt: str
    options: GenerationOptions = field(default_factory=GenerationOptions)

    def __post_init__(self) -> None:
        identifier(self.identifier, "generation.request")
        _validate_text(self.prompt)
        if len(self.prompt) > 8192 or not isinstance(self.options, GenerationOptions):
            raise ConfigError(
                "Text request requires bounded prompt and validated options"
            )


@dataclass(frozen=True)
class GenerationJob:
    snapshot: ArtifactRef
    requests: tuple[TextRequest, ...]
    limits: EngineLimits = field(default_factory=EngineLimits)
    prompt_format: str = "chat-v1"
    use_cache: bool = True
    schema_version: int = 1
    protocol: str = "qvelt-generate-v1"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.protocol != "qvelt-generate-v1"
        ):
            raise ConfigError("Only qvelt-generate-v1/schema1 is supported")
        if (
            not isinstance(self.snapshot, ArtifactRef)
            or not isinstance(self.limits, EngineLimits)
            or type(self.requests) is not tuple
            or not 1 <= len(self.requests) <= self.limits.max_batch
            or not all(isinstance(r, TextRequest) for r in self.requests)
        ):
            raise ConfigError(
                "Generation needs a snapshot,limits and bounded immutable requests"
            )
        if (
            len({r.identifier for r in self.requests}) != len(self.requests)
            or self.prompt_format not in ("chat-v1", "plain-v1")
            or type(self.use_cache) is not bool
        ):
            raise ConfigError("Invalid generation identities/prompt/cache policy")


def dump_generation(job: GenerationJob) -> str:
    return yaml.safe_dump(asdict(job), sort_keys=True, allow_unicode=True)


def load_generation(path: Path) -> GenerationJob:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Generation recipe exceeds128KiB")
    body = _mapping(
        parse_yaml_mapping(data.decode("utf-8-sig")),
        "generation",
        {f.name for f in fields(GenerationJob)},
    )
    if not {"snapshot", "requests"} <= set(body):
        raise ConfigError("Generation recipe requires snapshot and requests")
    ref = _mapping(body["snapshot"], "snapshot", {"path", "sha256"})
    if set(ref) != {"path", "sha256"}:
        raise ConfigError("Snapshot needs path and sha256")
    body["snapshot"] = ArtifactRef(**cast(dict[str, Any], ref))
    if "limits" in body:
        body["limits"] = EngineLimits(
            **cast(
                dict[str, Any],
                _mapping(
                    body["limits"], "limits", {f.name for f in fields(EngineLimits)}
                ),
            )
        )
    if not isinstance(body["requests"], list):
        raise ConfigError("Generation requests must be a YAML list")
    requests = []
    for value in body["requests"]:
        row = _mapping(value, "request", {f.name for f in fields(TextRequest)})
        if not {"identifier", "prompt"} <= set(row):
            raise ConfigError("Text request needs identifier and prompt")
        options = _mapping(
            row.get("options", {}),
            "options",
            {f.name for f in fields(GenerationOptions)},
        )
        for key in ("stop_tokens", "stop_strings"):
            if key in options:
                value = options[key]
                if not isinstance(value, list):
                    raise ConfigError("Stop lists must be YAML sequences")
                options[key] = tuple(value)
        row["options"] = GenerationOptions(**cast(dict[str, Any], options))
        requests.append(TextRequest(**cast(dict[str, Any], row)))
    body["requests"] = tuple(requests)
    return GenerationJob(**cast(dict[str, Any], body))
