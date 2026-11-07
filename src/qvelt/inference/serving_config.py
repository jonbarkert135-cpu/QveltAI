"""Strict local serving policy;secrets are named environment bindings,never YAML values."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from qvelt.config import (
    MAX_CONFIG_BYTES,
    ConfigError,
    _integer,
    _mapping,
    parse_yaml_mapping,
)
from qvelt.eval.config import ArtifactRef
from qvelt.utils.manifest_types import identifier


@dataclass(frozen=True)
class ServingConfig:
    snapshot: ArtifactRef
    model_name: str = "qvelt"
    snapshot_format: str = "native"
    host: str = "127.0.0.1"
    port: int = 8000
    max_pending: int = 16
    batch_wait_ms: int = 5
    request_timeout: float = 30.0
    max_body_bytes: int = 65536
    api_key_env: str = "QVELT_API_KEY"
    allow_unauthenticated: bool = False
    num_threads: int = 1

    def __post_init__(self) -> None:
        import math
        import re

        identifier(self.model_name, "serving.model_name")
        if not isinstance(self.snapshot, ArtifactRef) or self.snapshot_format not in (
            "native",
            "hqq",
        ):
            raise ConfigError("Serving requires a checked native or HQQ snapshot")
        if (
            self.host not in ("127.0.0.1", "::1", "0.0.0.0")
            or type(self.allow_unauthenticated) is not bool
            or self.allow_unauthenticated
            and self.host == "0.0.0.0"
        ):
            raise ConfigError("Unauthenticated serving must bind loopback")
        if (
            not isinstance(self.api_key_env, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", self.api_key_env) is None
        ):
            raise ConfigError("Invalid serving auth environment binding")
        if (
            type(self.request_timeout) not in (int, float)
            or not math.isfinite(self.request_timeout)
            or not 0 < self.request_timeout <= 300
        ):
            raise ConfigError("Invalid end-to-end request timeout")
        for key, low, high in (
            ("port", 1, 65535),
            ("max_pending", 1, 64),
            ("batch_wait_ms", 0, 100),
            ("max_body_bytes", 1024, 262144),
            ("num_threads", 1, 16),
        ):
            _integer(getattr(self, key), "serving." + key, minimum=low, maximum=high)


def load_serving(path: Path) -> ServingConfig:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Serving recipe exceeds size limit")
    body = _mapping(
        parse_yaml_mapping(data.decode()),
        "serving",
        {f.name for f in fields(ServingConfig)},
    )
    ref = _mapping(body.get("snapshot"), "snapshot", {"path", "sha256"})
    if set(ref) != {"path", "sha256"}:
        raise ConfigError("Snapshot needs path and SHA256")
    body["snapshot"] = ArtifactRef(**cast(dict[str, Any], ref))
    return ServingConfig(**cast(dict[str, Any], body))
