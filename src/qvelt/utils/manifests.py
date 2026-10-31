"""Bounded, strict JSON serialization and canonical manifest identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields, is_dataclass
from typing import Literal, cast

from qvelt.utils.manifest_types import (
    MAX_FILES,
    MAX_MANIFEST_BYTES,
    ArtifactFile,
    CheckpointInfo,
    CodeIdentity,
    DataInfo,
    EnvironmentInfo,
    Manifest,
    ManifestError,
    ManifestKind,
    Metadata,
    PackageVersion,
    RunInfo,
    SourceIdentity,
    TokenizerInfo,
    digest,
    text,
    uint,
)


def _unique_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError("JSON: duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ManifestError("JSON: non-finite constants are not allowed")


def _mapping(value: object, schema: type[object]) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestError(f"{schema.__name__}: expected mapping")
    mapping = cast("dict[str, object]", value)
    # Every serialized field is required, including version/producer defaults.
    if not is_dataclass(schema):
        raise TypeError("Schema must be a dataclass type")
    if set(mapping) != {item.name for item in fields(schema)}:
        raise ManifestError(f"{schema.__name__}: missing or unknown fields")
    return mapping


def _array(value: object, field: str, limit: int) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= limit:
        raise ManifestError(f"{field}: expected bounded nonempty array")
    return cast("list[object]", value)


def _bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ManifestError(f"{field}: expected boolean")
    return value


def _optional_digest(value: object, field: str) -> str | None:
    return None if value is None else digest(value, field)


def _file(value: object) -> ArtifactFile:
    item = _mapping(value, ArtifactFile)
    return ArtifactFile(
        path=text(item["path"], "file.path"),
        role=text(item["role"], "file.role"),
        size_bytes=uint(item["size_bytes"], "file.size_bytes"),
        sha256=digest(item["sha256"], "file.sha256"),
    )


def _code(value: object) -> CodeIdentity:
    item = _mapping(value, CodeIdentity)
    return CodeIdentity(
        commit=text(item["commit"], "code.commit"),
        dirty=_bool(item["dirty"], "code.dirty"),
        lock_sha256=digest(item["lock_sha256"], "code.lock_sha256"),
    )


def _package(value: object) -> PackageVersion:
    item = _mapping(value, PackageVersion)
    return PackageVersion(
        text(item["name"], "package.name"), text(item["version"], "package.version")
    )


def _environment(value: object) -> EnvironmentInfo:
    item = _mapping(value, EnvironmentInfo)
    return EnvironmentInfo(
        python=text(item["python"], "environment.python"),
        implementation=text(item["implementation"], "environment.implementation"),
        system=text(item["system"], "environment.system"),
        machine=text(item["machine"], "environment.machine"),
        packages=tuple(
            _package(p) for p in _array(item["packages"], "environment.packages", 2048)
        ),
    )


def _source(value: object) -> SourceIdentity:
    item = _mapping(value, SourceIdentity)
    return SourceIdentity(
        name=text(item["name"], "source.name"),
        revision=text(item["revision"], "source.revision"),
        license_id=text(item["license_id"], "source.license_id"),
        license_reference=text(
            item["license_reference"], "source.license_reference", 2048
        ),
    )


def _metadata(kind: ManifestKind, value: object) -> Metadata:
    if kind == "run":
        item = _mapping(value, RunInfo)
        return RunInfo(
            run_id=text(item["run_id"], "run.run_id"),
            config_sha256=digest(item["config_sha256"], "run.config_sha256"),
            code=_code(item["code"]),
            environment=_environment(item["environment"]),
            seed=uint(item["seed"], "run.seed", maximum=2**32 - 1),
            world_size=uint(item["world_size"], "run.world_size", minimum=1),
        )
    if kind == "data":
        item = _mapping(value, DataInfo)
        return DataInfo(
            representation=cast(
                Literal["raw", "token_ids"],
                text(item["representation"], "data.representation"),
            ),
            sources=tuple(
                _source(s) for s in _array(item["sources"], "data.sources", 1024)
            ),
            documents=uint(item["documents"], "data.documents"),
            tokens=None
            if item["tokens"] is None
            else uint(item["tokens"], "data.tokens"),
            tokenizer_sha256=_optional_digest(
                item["tokenizer_sha256"], "data.tokenizer_sha256"
            ),
        )
    if kind == "tokenizer":
        item = _mapping(value, TokenizerInfo)
        return TokenizerInfo(
            algorithm=cast(
                Literal["byte_bpe"], text(item["algorithm"], "tokenizer.algorithm")
            ),
            vocab_size=uint(item["vocab_size"], "tokenizer.vocab_size", minimum=1),
            reserved_tokens=uint(
                item["reserved_tokens"], "tokenizer.reserved_tokens", minimum=1
            ),
            training_data_sha256=digest(
                item["training_data_sha256"], "tokenizer.training_data_sha256"
            ),
        )
    item = _mapping(value, CheckpointInfo)
    return CheckpointInfo(
        run_sha256=digest(item["run_sha256"], "checkpoint.run_sha256"),
        config_sha256=digest(item["config_sha256"], "checkpoint.config_sha256"),
        data_sha256=digest(item["data_sha256"], "checkpoint.data_sha256"),
        tokenizer_sha256=digest(
            item["tokenizer_sha256"], "checkpoint.tokenizer_sha256"
        ),
        optimizer_step=uint(item["optimizer_step"], "checkpoint.optimizer_step"),
        processed_tokens=uint(item["processed_tokens"], "checkpoint.processed_tokens"),
        world_size=uint(item["world_size"], "checkpoint.world_size", minimum=1),
        precision=cast(
            Literal["fp32", "bf16"], text(item["precision"], "checkpoint.precision")
        ),
    )


def manifest_bytes(manifest: Manifest) -> bytes:
    """Canonical sorted compact UTF-8 JSON including one terminating newline."""
    data = (
        json.dumps(
            asdict(manifest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(data) > MAX_MANIFEST_BYTES:
        raise ManifestError("Manifest exceeds the 1 MiB size limit")
    return data


def manifest_digest(manifest: Manifest) -> str:
    """Content identity of canonical metadata, not of arbitrary input formatting."""
    return hashlib.sha256(manifest_bytes(manifest)).hexdigest()


def parse_manifest(data: bytes, *, expected_sha256: str | None = None) -> Manifest:
    """Strictly parse schema 1 and optionally enforce an externally trusted ID."""
    if len(data) > MAX_MANIFEST_BYTES:
        raise ManifestError("Manifest exceeds the 1 MiB size limit")
    try:
        value: object = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_mapping,
            parse_constant=_reject_constant,
        )
    except ManifestError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ManifestError("Invalid manifest JSON") from error
    item = _mapping(value, Manifest)
    if type(item["schema_version"]) is not int or item["schema_version"] != 1:
        raise ManifestError("schema_version: only integer version 1 is supported")
    kind = item["kind"]
    if not isinstance(kind, str) or kind not in (
        "run",
        "data",
        "tokenizer",
        "checkpoint",
    ):
        raise ManifestError("manifest.kind: unknown kind")
    kind = cast(ManifestKind, kind)
    manifest = Manifest(
        kind=kind,
        owner=text(item["owner"], "manifest.owner"),
        created_at=text(item["created_at"], "manifest.created_at"),
        metadata=_metadata(kind, item["metadata"]),
        files=tuple(
            _file(f) for f in _array(item["files"], "manifest.files", MAX_FILES)
        ),
        schema_version=1,
        producer_version=text(item["producer_version"], "manifest.producer_version"),
    )
    if expected_sha256 is not None and manifest_digest(manifest) != digest(
        expected_sha256, "expected_sha256"
    ):
        raise ManifestError("Manifest identity does not match expected_sha256")
    return manifest
