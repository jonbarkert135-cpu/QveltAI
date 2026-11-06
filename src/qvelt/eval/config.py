"""Strict version1 local evaluation inputs and reproducible generation/scoring budgets."""

from __future__ import annotations

import hashlib
import json
import math
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
from qvelt.utils.manifest_types import digest


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not 1 <= len(self.path) <= 2048
            or any(ord(c) < 32 for c in self.path)
        ):
            raise ConfigError("Invalid evaluation artifact path")
        digest(self.sha256, "evaluation.artifact.sha256")


@dataclass(frozen=True)
class EvalJob:
    checkpoint: ArtifactRef
    tokenizer: ArtifactRef
    dataset: ArtifactRef
    training_sources: tuple[ArtifactRef, ...] = ()
    approved_for_evaluation: bool = False
    allow_final: bool = False
    prompt_format: str = "chat-v1"
    extraction: str = "whole-v1"
    sampling: str = "greedy"
    samples: int = 1
    pass_k: tuple[int, ...] = (1,)
    temperature: float = 0.0
    max_new_tokens: int = 16
    context_limit: int = 512
    lm_window: int = 128
    max_cases: int = 256
    seed: int = 20261105
    numeric_atol: float = 1e-8
    numeric_rtol: float = 1e-6
    audit_ngram: int = 13
    audit_jaccard: float = 0.85
    schema_version: int = 1
    protocol: str = "qvelt-eval-v1"
    choice_normalization: str = "token"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.protocol not in ("qvelt-eval-v1", "qvelt-eval-v2")
        ):
            raise ConfigError("Only qvelt-eval-v1/v2 with schema1 is supported")
        if self.choice_normalization not in (
            "token",
            "character",
            "character-without-delimiter",
        ):
            raise ConfigError("Unknown choice normalization")
        if (
            not all(
                isinstance(getattr(self, k), ArtifactRef)
                for k in ("checkpoint", "tokenizer", "dataset")
            )
            or type(self.training_sources) is not tuple
            or len(self.training_sources) > 16
            or not all(isinstance(x, ArtifactRef) for x in self.training_sources)
        ):
            raise ConfigError(
                "Evaluation inputs require bounded immutable artifact references"
            )
        if len({r.sha256 for r in self.training_sources}) != len(self.training_sources):
            raise ConfigError("Training audit sources must be unique")
        if (
            type(self.approved_for_evaluation) is not bool
            or type(self.allow_final) is not bool
        ):
            raise ConfigError("Evaluation approval/final access must be boolean")
        if (
            self.prompt_format not in ("chat-v1", "plain-v1")
            or self.extraction
            not in (
                "whole-v1",
                "final-line-v1",
                "gsm8k-strict-v1",
                "reasoning-final-v2",
            )
            or self.sampling not in ("greedy", "sample")
        ):
            raise ConfigError("Unknown evaluation prompt/extraction/sampler protocol")
        for key, low, high in [
            ("samples", 1, 32),
            ("max_new_tokens", 1, 128),
            ("context_limit", 4, 2048),
            ("lm_window", 1, 2048),
            ("max_cases", 1, 10000),
            ("seed", 0, 2**32 - 1),
            ("audit_ngram", 1, 64),
        ]:
            _integer(getattr(self, key), "evaluation." + key, minimum=low, maximum=high)
        if self.lm_window > self.context_limit:
            raise ConfigError("LM window exceeds evaluation context limit")
        if (
            type(self.pass_k) is not tuple
            or not self.pass_k
            or any(
                type(k) is not int or not 1 <= k <= self.samples for k in self.pass_k
            )
            or len(set(self.pass_k)) != len(self.pass_k)
        ):
            raise ConfigError("pass_k requires unique integers1..samples")
        for key in ("temperature", "numeric_atol", "numeric_rtol", "audit_jaccard"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ConfigError("Evaluation scalars must be finite numbers")
            object.__setattr__(self, key, float(value))
        if (
            not 0 <= self.numeric_atol <= 1
            or not 0 <= self.numeric_rtol <= 1
            or not 0 < self.audit_jaccard <= 1
        ):
            raise ConfigError("Invalid tolerance/audit threshold")
        if self.sampling == "greedy" and (
            self.samples != 1 or self.pass_k != (1,) or self.temperature != 0
        ):
            raise ConfigError("Greedy evaluation has one attempt/k1/temperature0")
        if self.sampling == "sample" and not 0 < self.temperature <= 2:
            raise ConfigError("Categorical sampling needs temperature in (0,2]")
        if self.max_cases * self.samples * self.max_new_tokens > 1_000_000:
            raise ConfigError("Evaluation exceeds bounded generation capacity")


def dump_eval(job: EvalJob) -> str:
    return yaml.safe_dump(asdict(job), sort_keys=True)


def load_eval(path: Path) -> EvalJob:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Evaluation recipe exceeds its byte limit")
    return parse_eval(data.decode("utf-8-sig"))


def parse_eval(text: str) -> EvalJob:
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ConfigError("Evaluation recipe exceeds its byte limit")
    root = cast(
        dict[str, Any],
        _mapping(
            parse_yaml_mapping(text),
            "evaluation",
            {f.name for f in fields(EvalJob)},
        ),
    )
    if not {"checkpoint", "tokenizer", "dataset"} <= set(root):
        raise ConfigError("Evaluation requires checkpoint/tokenizer/dataset identities")
    for key in ("checkpoint", "tokenizer", "dataset"):
        value = _mapping(root[key], key, {"path", "sha256"})
        if set(value) != {"path", "sha256"}:
            raise ConfigError("Artifact reference requires path and sha256")
        root[key] = ArtifactRef(**cast(dict[str, Any], value))
    refs = root.get("training_sources", [])
    if not isinstance(refs, list):
        raise ConfigError("training_sources must be a list")
    sources = []
    for ref in refs:
        value = _mapping(ref, "training_source", {"path", "sha256"})
        if set(value) != {"path", "sha256"}:
            raise ConfigError("Artifact reference requires path and sha256")
        sources.append(ArtifactRef(**cast(dict[str, Any], value)))
    root["training_sources"] = tuple(sources)
    if "pass_k" in root:
        if not isinstance(root["pass_k"], list):
            raise ConfigError("pass_k must be a YAML sequence")
        root["pass_k"] = tuple(root["pass_k"])
    return EvalJob(**root)


def eval_digest(job: EvalJob) -> str:
    value = asdict(job)
    if job.protocol == "qvelt-eval-v1" and job.choice_normalization == "token":
        value.pop("choice_normalization")
    for key in ("checkpoint", "tokenizer", "dataset"):
        value[key].pop("path")
    for source in value["training_sources"]:
        source.pop("path")
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
