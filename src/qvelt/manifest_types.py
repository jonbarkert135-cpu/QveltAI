"""Immutable schema-1 provenance records; no model or corpus deserialization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit

from qvelt import __version__

ManifestKind = Literal["run", "data", "tokenizer", "checkpoint"]
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILES = 4096


class ManifestError(ValueError):
    """A manifest or declared identity violates schema 1."""


def text(value: object, field: str, maximum: int = 1024) -> str:
    """Require bounded text without controls; never echo rejected values."""
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ManifestError(f"{field}: expected nonempty bounded text")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ManifestError(f"{field}: control characters are not allowed")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ManifestError(f"{field}: expected UTF-8 text") from error
    return value


def uint(value: object, field: str, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    """Require a bounded integer, excluding bool despite its int subclass."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ManifestError(f"{field}: expected integer in [{minimum}, {maximum}]")
    return value


def digest(value: object, field: str) -> str:
    """Require a canonical lowercase SHA-256 identity."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ManifestError(f"{field}: expected lowercase SHA-256")
    return value


def identifier(value: object, field: str) -> str:
    """Portable role/owner/run identifier, distinct from a filesystem path."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None
    ):
        raise ManifestError(f"{field}: invalid identifier")
    return value


def artifact_path(value: object) -> str:
    """Canonical relative POSIX path, with no drive, traversal or empty parts."""
    path = text(value, "artifact.path", 1024)
    if (
        "\\" in path
        or ":" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ManifestError("artifact.path: expected canonical relative POSIX path")
    return path


@dataclass(frozen=True)
class ArtifactFile:
    """Raw-byte integrity and logical ownership role for a regular local file."""

    path: str
    role: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        artifact_path(self.path)
        identifier(self.role, "artifact.role")
        uint(self.size_bytes, "artifact.size_bytes")
        digest(self.sha256, "artifact.sha256")


@dataclass(frozen=True)
class PackageVersion:
    """Installed distribution identity without URLs, paths or credentials."""

    name: str
    version: str

    def __post_init__(self) -> None:
        name = text(self.name, "package.name", 128)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
            raise ManifestError("package.name: invalid distribution name")
        object.__setattr__(self, "name", re.sub(r"[-_.]+", "-", name).lower())
        text(self.version, "package.version", 128)


@dataclass(frozen=True)
class EnvironmentInfo:
    """Observed interpreter/platform/package versions; not a hardware profile."""

    python: str
    implementation: str
    system: str
    machine: str
    packages: tuple[PackageVersion, ...]

    def __post_init__(self) -> None:
        for name in ("python", "implementation", "system", "machine"):
            text(getattr(self, name), f"environment.{name}", 256)
        if type(self.packages) is not tuple or not 1 <= len(self.packages) <= 2048:
            raise ManifestError(
                "environment.packages: expected 1–2048 immutable entries"
            )
        if not all(isinstance(p, PackageVersion) for p in self.packages):
            raise ManifestError("environment.packages: invalid entry")
        if len({p.name for p in self.packages}) != len(self.packages):
            raise ManifestError("environment.packages: duplicate distribution")
        object.__setattr__(
            self, "packages", tuple(sorted(self.packages, key=lambda p: p.name))
        )


@dataclass(frozen=True)
class CodeIdentity:
    """Full local Git commit, observed dirty status and raw dependency-lock hash."""

    commit: str
    dirty: bool
    lock_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.commit, str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.commit) is None
        ):
            raise ManifestError("code.commit: expected full Git commit identity")
        if type(self.dirty) is not bool:
            raise ManifestError("code.dirty: expected boolean")
        digest(self.lock_sha256, "code.lock_sha256")


@dataclass(frozen=True)
class RunInfo:
    """Declared run policy plus observed code/environment; no training result."""

    run_id: str
    config_sha256: str
    code: CodeIdentity
    environment: EnvironmentInfo
    seed: int
    world_size: int

    def __post_init__(self) -> None:
        identifier(self.run_id, "run.run_id")
        digest(self.config_sha256, "run.config_sha256")
        if not isinstance(self.code, CodeIdentity) or not isinstance(
            self.environment, EnvironmentInfo
        ):
            raise ManifestError("run: invalid code/environment record")
        uint(self.seed, "run.seed", maximum=2**32 - 1)
        uint(self.world_size, "run.world_size", minimum=1, maximum=2**31 - 1)


@dataclass(frozen=True)
class SourceIdentity:
    """Declared data origin/license evidence, not ingestion or legal approval."""

    name: str
    revision: str
    license_id: str
    license_reference: str

    def __post_init__(self) -> None:
        for name in ("name", "revision", "license_id"):
            text(getattr(self, name), f"source.{name}", 512)
        reference = text(self.license_reference, "source.license_reference", 2048)
        try:
            parsed = urlsplit(reference)
        except ValueError as error:
            raise ManifestError(
                "source.license_reference: invalid HTTPS URL"
            ) from error
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
        ):
            raise ManifestError(
                "source.license_reference: expected public HTTPS URL without credentials/query"
            )


@dataclass(frozen=True)
class DataInfo:
    """Raw or tokenized dataset identity and declared post-processing counts."""

    representation: Literal["raw", "token_ids"]
    sources: tuple[SourceIdentity, ...]
    documents: int
    tokens: int | None
    tokenizer_sha256: str | None

    def __post_init__(self) -> None:
        if self.representation not in ("raw", "token_ids"):
            raise ManifestError("data.representation: unsupported value")
        if (
            type(self.sources) is not tuple
            or not 1 <= len(self.sources) <= 1024
            or not all(isinstance(s, SourceIdentity) for s in self.sources)
        ):
            raise ManifestError(
                "data.sources: expected 1–1024 immutable source records"
            )
        uint(self.documents, "data.documents")
        if self.representation == "raw":
            if self.tokens is not None or self.tokenizer_sha256 is not None:
                raise ManifestError(
                    "data: raw data must not claim tokenizer/token counts"
                )
        else:
            uint(self.tokens, "data.tokens")
            digest(self.tokenizer_sha256, "data.tokenizer_sha256")


@dataclass(frozen=True)
class TokenizerInfo:
    """Byte-BPE artifact metadata; actual token semantics are validated in M3."""

    algorithm: Literal["byte_bpe"]
    vocab_size: int
    reserved_tokens: int
    training_data_sha256: str

    def __post_init__(self) -> None:
        if self.algorithm != "byte_bpe":
            raise ManifestError("tokenizer.algorithm: unsupported value")
        uint(self.reserved_tokens, "tokenizer.reserved_tokens", minimum=1)
        uint(
            self.vocab_size, "tokenizer.vocab_size", minimum=256 + self.reserved_tokens
        )
        digest(self.training_data_sha256, "tokenizer.training_data_sha256")


@dataclass(frozen=True)
class CheckpointInfo:
    """Checkpoint lineage/counters, not validation of opaque training state."""

    run_sha256: str
    config_sha256: str
    data_sha256: str
    tokenizer_sha256: str
    optimizer_step: int
    processed_tokens: int
    world_size: int
    precision: Literal["fp32", "bf16"]

    def __post_init__(self) -> None:
        for name in ("run_sha256", "config_sha256", "data_sha256", "tokenizer_sha256"):
            digest(getattr(self, name), f"checkpoint.{name}")
        uint(self.optimizer_step, "checkpoint.optimizer_step")
        uint(self.processed_tokens, "checkpoint.processed_tokens")
        uint(self.world_size, "checkpoint.world_size", minimum=1, maximum=2**31 - 1)
        if self.precision not in ("fp32", "bf16"):
            raise ManifestError("checkpoint.precision: unsupported value")


Metadata = RunInfo | DataInfo | TokenizerInfo | CheckpointInfo


@dataclass(frozen=True)
class Manifest:
    """Versioned commit record for a bundle of integrity-checked files."""

    kind: ManifestKind
    owner: str
    created_at: str
    metadata: Metadata
    files: tuple[ArtifactFile, ...]
    schema_version: int = 1
    producer_version: str = __version__

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ManifestError("schema_version: only integer version 1 is supported")
        expected = {
            "run": RunInfo,
            "data": DataInfo,
            "tokenizer": TokenizerInfo,
            "checkpoint": CheckpointInfo,
        }
        if (
            not isinstance(self.kind, str)
            or self.kind not in expected
            or not isinstance(self.metadata, expected[self.kind])
        ):
            raise ManifestError("manifest.kind: unknown kind or mismatched metadata")
        identifier(self.owner, "manifest.owner")
        text(self.producer_version, "manifest.producer_version", 128)
        try:
            date = datetime.fromisoformat(
                text(self.created_at, "manifest.created_at", 64)
            )
        except ValueError as error:
            raise ManifestError("manifest.created_at: invalid ISO timestamp") from error
        if date.tzinfo is None or date.utcoffset() is None:
            raise ManifestError("manifest.created_at: timezone is required")
        try:
            object.__setattr__(self, "created_at", date.astimezone(UTC).isoformat())
        except (ValueError, OverflowError) as error:
            raise ManifestError(
                "manifest.created_at: timestamp is outside range"
            ) from error
        if (
            type(self.files) is not tuple
            or not 1 <= len(self.files) <= MAX_FILES
            or not all(isinstance(f, ArtifactFile) for f in self.files)
        ):
            raise ManifestError(
                "manifest.files: expected 1–4096 immutable file records"
            )
        if len({f.path for f in self.files}) != len(self.files):
            raise ManifestError("manifest.files: duplicate artifact path")
        roles = [f.role for f in self.files]
        required = {
            "run": {"config", "dependency_lock"},
            "data": {"data"},
            "tokenizer": {"tokenizer"},
            "checkpoint": {
                "config",
                "model",
                "optimizer",
                "scheduler",
                "rng",
                "sampler",
            },
        }
        if not required[self.kind].issubset(roles):
            raise ManifestError("manifest.files: missing required artifact roles")
        if self.kind in ("run", "checkpoint") and roles.count("config") != 1:
            raise ManifestError("manifest.files: exactly one config is required")
        if self.kind == "run":
            if roles.count("dependency_lock") != 1:
                raise ManifestError(
                    "manifest.files: exactly one dependency lock is required"
                )
            if (
                isinstance(self.metadata, RunInfo)
                and self.owner != self.metadata.run_id
            ):
                raise ManifestError("manifest.owner must match run_id")
        object.__setattr__(
            self, "files", tuple(sorted(self.files, key=lambda f: f.path))
        )
