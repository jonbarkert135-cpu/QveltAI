"""Canonical identities, strict schema parsing and malformed metadata handling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from qvelt.utils.manifest_types import (
    MAX_MANIFEST_BYTES,
    ArtifactFile,
    Manifest,
    ManifestError,
)
from qvelt.utils.manifests import manifest_bytes, manifest_digest, parse_manifest


@pytest.mark.parametrize("index", range(5))
def test_all_metadata_kinds_round_trip(
    index: int, manifest_cases: tuple[Manifest, ...]
) -> None:
    manifest = manifest_cases[index]
    data = manifest_bytes(manifest)
    assert data.endswith(b"\n")
    assert parse_manifest(data) == manifest
    assert manifest_digest(manifest) == hashlib.sha256(data).hexdigest()
    assert parse_manifest(data, expected_sha256=manifest_digest(manifest)) == manifest


def test_canonical_order_and_external_expected_identity(
    manifest_cases: tuple[Manifest, ...],
) -> None:
    manifest = manifest_cases[0]
    pretty = json.dumps(asdict(manifest), indent=4, sort_keys=False).encode()
    assert manifest_digest(parse_manifest(pretty)) == manifest_digest(manifest)
    assert replace(manifest, files=tuple(reversed(manifest.files))) == manifest
    with pytest.raises(ManifestError, match="identity"):
        parse_manifest(pretty, expected_sha256="0" * 64)
    altered = replace(manifest, producer_version="untrusted-replacement")
    with pytest.raises(ManifestError, match="identity"):
        parse_manifest(
            manifest_bytes(altered), expected_sha256=manifest_digest(manifest)
        )


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\xff",
        b"[]",
        b"null",
        b"{}",
        b'{"schema_version":1,"schema_version":1}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b"[{",
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_invalid_json_and_duplicate_keys_fail_cleanly(data: bytes) -> None:
    with pytest.raises(ManifestError):
        parse_manifest(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("kind", "future"),
        ("files", []),
        ("files", {}),
        ("created_at", "bad"),
        ("owner", False),
    ],
)
def test_invalid_root_schema(
    field: str, value: object, manifest_cases: tuple[Manifest, ...]
) -> None:
    document = asdict(manifest_cases[0])
    document[field] = value
    with pytest.raises(ManifestError):
        parse_manifest(json.dumps(document).encode())


def test_missing_unknown_and_nested_fields_are_rejected(
    manifest_cases: tuple[Manifest, ...],
) -> None:
    encoded = manifest_bytes(manifest_cases[0])
    for before, after in [
        (b'"schema_version":1', b'"schema_version":1,"other":0'),
        (b'"seed":20261031', b'"seed":true'),
        (b'"dirty":false', b'"dirty":0'),
        (b'"world_size":1', b'"world_size":0'),
        (b'"size_bytes":', b'"unknown_size":'),
    ]:
        assert before in encoded
        with pytest.raises(ManifestError):
            parse_manifest(encoded.replace(before, after, 1))


def test_metadata_size_and_count_bounds(manifest_cases: tuple[Manifest, ...]) -> None:
    with pytest.raises(ManifestError, match="1 MiB"):
        parse_manifest(b" " * (MAX_MANIFEST_BYTES + 1))
    document = asdict(manifest_cases[0])
    document["files"] = [document["files"][0]] * 4097
    with pytest.raises(ManifestError):
        parse_manifest(json.dumps(document).encode())


def test_serialized_manifest_size_is_bounded(
    manifest_cases: tuple[Manifest, ...],
) -> None:
    files = tuple(
        ArtifactFile(str(index) + "x" * 990, "data", 0, "a" * 64)
        for index in range(2048)
    )
    large = replace(manifest_cases[1], files=files)
    with pytest.raises(ManifestError, match="1 MiB"):
        manifest_bytes(large)
