"""Strict versioned experiment configuration without runtime initialization."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Hashable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal, cast

import yaml
from yaml.events import (
    AliasEvent,
    CollectionEndEvent,
    CollectionStartEvent,
    DocumentStartEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceStartEvent,
)
from yaml.nodes import MappingNode, Node, ScalarNode

Device = Literal["cpu", "cuda"]
Precision = Literal["fp32", "bf16"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
MAX_CONFIG_BYTES = 128 * 1024
MAX_DEPTH = 8
MAX_EVENTS = 2048


class ConfigError(ValueError):
    """A configuration violates the supported schema or YAML subset."""


def _integer(
    value: object, name: str, minimum: int = 1, maximum: int = 2**31 - 1
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConfigError(f"{name}: expected integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{name}: expected a finite positive number")
    try:
        number = float(cast("int | float", value))
    except OverflowError as error:
        raise ConfigError(f"{name}: number is too large") from error
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{name}: expected a finite positive number")
    return number


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{name}: expected true or false")
    return value


def _choice(value: object, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{name}: expected one of {', '.join(choices)}")
    return value


def _seed(value: object) -> int:
    seed = _integer(value, "runtime.seed", minimum=0, maximum=2**32 - 1)
    if seed >= 2**32:
        raise ConfigError("runtime.seed: expected integer < 2**32")
    return seed


@dataclass(frozen=True)
class ModelConfig:
    """Bias-free, tied GQA decoder dimensions selected in RFC-001.

    Unique parameters: V*d + L*(2*d*d + 2*d*k + 3*d*f + 2*d) + d,
    with k=kv_heads*hidden/heads. This counts shapes, not an existing model.
    """

    layers: int = 8
    hidden: int = 320
    heads: int = 5
    kv_heads: int = 1
    ffn: int = 896
    vocab: int = 32768
    context: int = 512
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        for name in (
            "layers",
            "hidden",
            "heads",
            "kv_heads",
            "ffn",
            "vocab",
            "context",
        ):
            _integer(getattr(self, name), f"model.{name}")
        if self.hidden % self.heads or self.heads % self.kv_heads:
            raise ConfigError(
                "model: hidden/heads and heads/kv_heads must divide exactly"
            )
        if (self.hidden // self.heads) % 2:
            raise ConfigError("model: RoPE requires an even head dimension")
        if self.vocab < 288:
            raise ConfigError("model.vocab: must include 256 bytes and 32 controls")
        object.__setattr__(self, "norm_eps", _number(self.norm_eps, "model.norm_eps"))
        theta = _number(self.rope_theta, "model.rope_theta")
        if theta <= 1:
            raise ConfigError("model.rope_theta: must exceed one")
        object.__setattr__(self, "rope_theta", theta)

    @property
    def parameter_count(self) -> int:
        """Analytical unique-parameter count, including tied E and norm gains."""
        d, f, k = self.hidden, self.ffn, self.kv_heads * (self.hidden // self.heads)
        return (
            self.vocab * d
            + self.layers * (2 * d * d + 2 * d * k + 3 * d * f + 2 * d)
            + d
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Requested execution policy; validation does not claim hardware support."""

    seed: int = 20261031
    device: Device = "cpu"
    precision: Precision = "fp32"
    deterministic: bool = True
    log_level: LogLevel = "INFO"

    def __post_init__(self) -> None:
        _seed(self.seed)
        _choice(self.device, "runtime.device", ("cpu", "cuda"))
        _choice(self.precision, "runtime.precision", ("fp32", "bf16"))
        _boolean(self.deterministic, "runtime.deterministic")
        _choice(
            self.log_level, "runtime.log_level", ("DEBUG", "INFO", "WARNING", "ERROR")
        )
        if self.device == "cpu" and self.precision != "fp32":
            raise ConfigError("runtime: CPU reference policy requires fp32")


@dataclass(frozen=True)
class ExperimentConfig:
    """Version 1 model/runtime contract; training and artifact schemas follow later."""

    schema_version: int = 1
    name: str = "tiny-debug"
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigError("schema_version: only integer version 1 is supported")
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.name
        ):
            raise ConfigError(
                "name: expected 1–64 ASCII letters/digits/dot/underscore/hyphen"
            )
        if not isinstance(self.model, ModelConfig) or not isinstance(
            self.runtime, RuntimeConfig
        ):
            raise ConfigError("model and runtime must be validated config dataclasses")


class _StrictLoader(yaml.SafeLoader):
    """Safe construction with unique string keys and no implicit merge behavior."""

    def construct_mapping(
        self, node: Node, deep: bool = False
    ) -> dict[Hashable, object]:
        if not isinstance(node, MappingNode):
            raise ConfigError("YAML: expected mapping")
        result: dict[Hashable, object] = {}
        for key_node, value_node in node.value:
            if (
                not isinstance(key_node, ScalarNode)
                or key_node.tag != "tag:yaml.org,2002:str"
            ):
                raise ConfigError("YAML: mapping keys must be strings")
            key = key_node.value
            if key == "<<":
                raise ConfigError("YAML: merge keys are not supported")
            if key in result:
                raise ConfigError("YAML: duplicate mapping key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# Do not mutate PyYAML's global SafeLoader or retain YAML 1.1 octal/bool/date rules.
_REPLACED = {
    f"tag:yaml.org,2002:{kind}" for kind in ("bool", "int", "float", "timestamp")
}
_StrictLoader.yaml_implicit_resolvers = {
    key: [entry for entry in entries if entry[0] not in _REPLACED]
    for key, entries in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(true|false)$"), list("tf")
)
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int", re.compile(r"^[-+]?(0|[1-9][0-9]*)$"), list("-+0123456789")
)
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^[-+]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)$"
    ),
    list("-+.0123456789"),
)


def _inspect_yaml(text: str) -> None:
    depth, documents = 0, 0
    for count, event in enumerate(yaml.parse(text, Loader=_StrictLoader), 1):
        if count > MAX_EVENTS:
            raise ConfigError("YAML: too many nodes")
        if isinstance(event, DocumentStartEvent):
            documents += 1
            if documents > 1:
                raise ConfigError("YAML: exactly one document is supported")
        if isinstance(event, AliasEvent):
            raise ConfigError("YAML: aliases are not supported")
        if isinstance(event, ScalarEvent | MappingStartEvent | SequenceStartEvent):
            if event.anchor is not None or event.tag is not None:
                raise ConfigError("YAML: anchors and explicit tags are not supported")
        if isinstance(event, CollectionStartEvent):
            depth += 1
            if depth > MAX_DEPTH:
                raise ConfigError("YAML: nesting limit exceeded")
        elif isinstance(event, CollectionEndEvent):
            depth -= 1


def _mapping(value: object, name: str, allowed: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ConfigError(f"{name}: expected a string-keyed mapping")
    mapping = cast("dict[str, object]", value)
    if set(mapping) - allowed:
        raise ConfigError(
            f"{name}: unknown fields: {', '.join(sorted(set(mapping) - allowed))}"
        )
    return mapping


def parse_config(text: str) -> ExperimentConfig:
    """Parse bounded YAML, apply explicit defaults, and reject invalid schemas."""
    try:
        encoded = text.encode("utf-8")
    except UnicodeError as error:
        raise ConfigError("Config must be valid UTF-8") from error
    if len(encoded) > MAX_CONFIG_BYTES:
        raise ConfigError("Config exceeds the 128 KiB size limit")
    try:
        _inspect_yaml(text)
        raw: object = yaml.load(text, Loader=_StrictLoader)
    except ConfigError:
        raise
    except (yaml.YAMLError, ValueError) as error:
        raise ConfigError("Invalid YAML syntax; no config values are echoed") from error
    root = _mapping(raw, "config", {f.name for f in fields(ExperimentConfig)})
    m = _mapping(root.get("model", {}), "model", {f.name for f in fields(ModelConfig)})
    r = _mapping(
        root.get("runtime", {}), "runtime", {f.name for f in fields(RuntimeConfig)}
    )
    default, rt = ModelConfig(), RuntimeConfig()
    model = ModelConfig(
        layers=_integer(m.get("layers", default.layers), "model.layers"),
        hidden=_integer(m.get("hidden", default.hidden), "model.hidden"),
        heads=_integer(m.get("heads", default.heads), "model.heads"),
        kv_heads=_integer(m.get("kv_heads", default.kv_heads), "model.kv_heads"),
        ffn=_integer(m.get("ffn", default.ffn), "model.ffn"),
        vocab=_integer(m.get("vocab", default.vocab), "model.vocab"),
        context=_integer(m.get("context", default.context), "model.context"),
        norm_eps=_number(m.get("norm_eps", default.norm_eps), "model.norm_eps"),
        rope_theta=_number(m.get("rope_theta", default.rope_theta), "model.rope_theta"),
    )
    runtime = RuntimeConfig(
        seed=_seed(r.get("seed", rt.seed)),
        device=cast(
            Device,
            _choice(r.get("device", rt.device), "runtime.device", ("cpu", "cuda")),
        ),
        precision=cast(
            Precision,
            _choice(
                r.get("precision", rt.precision), "runtime.precision", ("fp32", "bf16")
            ),
        ),
        deterministic=_boolean(
            r.get("deterministic", rt.deterministic), "runtime.deterministic"
        ),
        log_level=cast(
            LogLevel,
            _choice(
                r.get("log_level", rt.log_level),
                "runtime.log_level",
                ("DEBUG", "INFO", "WARNING", "ERROR"),
            ),
        ),
    )
    name = root.get("name", "tiny-debug")
    if not isinstance(name, str):
        raise ConfigError("name: expected a string")
    return ExperimentConfig(
        schema_version=_integer(root.get("schema_version", 1), "schema_version"),
        name=name,
        model=model,
        runtime=runtime,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read at most the size limit before parsing a UTF-8 configuration file."""
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Config exceeds the 128 KiB size limit")
    try:
        return parse_config(data.decode("utf-8-sig"))
    except UnicodeError as error:
        raise ConfigError("Config must be valid UTF-8") from error


def dump_config(config: ExperimentConfig) -> str:
    """Emit resolved defaults as deterministic safe YAML; do not write files."""
    return yaml.safe_dump(asdict(config), sort_keys=True, allow_unicode=True)


def config_digest(config: ExperimentConfig) -> str:
    """SHA-256 over canonical UTF-8 JSON of the complete resolved dataclasses."""
    content = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
