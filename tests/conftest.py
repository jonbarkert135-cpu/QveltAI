"""Small opaque artifact fixtures; none are trained models or approved corpora."""

from __future__ import annotations

import hashlib

import pytest

from qvelt import __version__
from qvelt.config import ExperimentConfig, config_digest, dump_config
from qvelt.utils.manifest_types import (
    ArtifactFile,
    CheckpointInfo,
    CodeIdentity,
    DataInfo,
    EnvironmentInfo,
    Manifest,
    PackageVersion,
    RunInfo,
    SourceIdentity,
    TokenizerInfo,
)


@pytest.fixture
def fixture_payloads() -> dict[str, bytes]:
    return {
        "config.yaml": dump_config(ExperimentConfig()).encode(),
        "uv.lock": b"version = 1\n",
        "data.bin": b"opaque fixture data",
        "tokenizer.json": b"{}\n",
        **{
            name + ".bin": ("opaque " + name).encode()
            for name in ("model", "optimizer", "scheduler", "rng", "sampler")
        },
    }


@pytest.fixture
def manifest_cases(fixture_payloads: dict[str, bytes]) -> tuple[Manifest, ...]:
    def record(path: str, role: str) -> ArtifactFile:
        data = fixture_payloads[path]
        return ArtifactFile(path, role, len(data), hashlib.sha256(data).hexdigest())

    config = ExperimentConfig()
    config_record, lock = (
        record("config.yaml", "config"),
        record("uv.lock", "dependency_lock"),
    )
    environment = EnvironmentInfo(
        "3.12.14",
        "CPython",
        "Linux",
        "x86_64",
        (PackageVersion("qvelt", __version__), PackageVersion("numpy", "2.2.6")),
    )
    run = RunInfo(
        "fixture-run",
        config_digest(config),
        CodeIdentity("1" * 40, False, lock.sha256),
        environment,
        config.runtime.seed,
        1,
    )
    source = SourceIdentity(
        "authored-test",
        "fixture-v1",
        "LicenseRef-TestFixture",
        "https://example.invalid/license",
    )
    files = (record("data.bin", "data"),)
    # Explicit construction keeps the discriminated metadata union type-checked.
    return (
        Manifest(
            "run", "fixture-run", "2026-10-31T12:00:00Z", run, (config_record, lock)
        ),
        Manifest(
            "data",
            "fixture",
            "2026-10-31T12:00:00Z",
            DataInfo("raw", (source,), 1, None, None),
            files,
        ),
        Manifest(
            "data",
            "fixture",
            "2026-10-31T12:00:00Z",
            DataInfo("token_ids", (source,), 1, 12, "a" * 64),
            files,
        ),
        Manifest(
            "tokenizer",
            "fixture",
            "2026-10-31T12:00:00Z",
            TokenizerInfo("byte_bpe", 32768, 32, "a" * 64),
            (record("tokenizer.json", "tokenizer"),),
        ),
        Manifest(
            "checkpoint",
            "fixture",
            "2026-10-31T12:00:00Z",
            CheckpointInfo(
                "a" * 64, config_digest(config), "b" * 64, "a" * 64, 2, 128, 1, "fp32"
            ),
            (
                config_record,
                *(
                    record(name + ".bin", name)
                    for name in ("model", "optimizer", "scheduler", "rng", "sampler")
                ),
            ),
        ),
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        marker = "integration" if "/integration/" in str(item.path) else "unit"
        item.add_marker(getattr(pytest.mark, marker))
