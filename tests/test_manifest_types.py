"""Schema invariants independent of serialized JSON or physical artifact files."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from typing import cast

import pytest

from qvelt.manifest_types import (
    ArtifactFile,
    CheckpointInfo,
    CodeIdentity,
    DataInfo,
    EnvironmentInfo,
    Manifest,
    ManifestError,
    PackageVersion,
    RunInfo,
    SourceIdentity,
    TokenizerInfo,
    artifact_path,
)


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "..",
        "/absolute",
        "a//b",
        "a/./b",
        "a/../b",
        "a/",
        "C:/file",
        "a\\file",
        "a\x00file",
        "a\nfile",
    ],
)
def test_paths_cannot_escape_or_hide_components(path: str) -> None:
    with pytest.raises(ManifestError):
        artifact_path(path)


def test_regular_unicode_relative_path() -> None:
    assert artifact_path("shards/данные.bin") == "shards/данные.bin"


@pytest.mark.parametrize("value", [-1, True, 2**63])
def test_file_size_range(value: int) -> None:
    with pytest.raises(ManifestError):
        ArtifactFile("data.bin", "data", value, "a" * 64)


@pytest.mark.parametrize("sha", ["", "abc", "A" * 64, "g" * 64, "a" * 63])
def test_digest_format(sha: str) -> None:
    with pytest.raises(ManifestError):
        ArtifactFile("data.bin", "data", 0, sha)


def test_tuples_frozen_fields_normalized_packages_and_utc(
    manifest_cases: tuple[Manifest, ...],
) -> None:
    manifest = manifest_cases[0]
    assert manifest.created_at == "2026-10-31T12:00:00+00:00"
    assert replace(manifest, created_at="2026-10-31T14:00:00+02:00") == manifest
    for item in fields(manifest):
        with pytest.raises(FrozenInstanceError):
            setattr(manifest, item.name, getattr(manifest, item.name))
    with pytest.raises(ManifestError):
        replace(manifest, files=cast(tuple[ArtifactFile, ...], list(manifest.files)))
    assert PackageVersion("Py_YAML", "6.0.3").name == "py-yaml"


@pytest.mark.parametrize(
    "date",
    [
        "",
        "not-a-date",
        "2026-10-31",
        "2026-10-31T12:00:00",
        "9999-12-31T23:00:00-23:59",
    ],
)
def test_timestamp_rejection(date: str, manifest_cases: tuple[Manifest, ...]) -> None:
    with pytest.raises(ManifestError):
        replace(manifest_cases[0], created_at=date)


def test_schema_kind_owner_and_roles(manifest_cases: tuple[Manifest, ...]) -> None:
    run, raw, _, tokenizer, checkpoint = manifest_cases
    with pytest.raises(ManifestError):
        replace(run, schema_version=True)
    with pytest.raises(ManifestError):
        replace(run, schema_version=2)
    with pytest.raises(ManifestError):
        replace(run, metadata=tokenizer.metadata)
    with pytest.raises(ManifestError):
        replace(run, owner="another-owner")
    with pytest.raises(ManifestError):
        replace(run, files=(run.files[0], run.files[0]))
    with pytest.raises(ManifestError):
        replace(
            checkpoint,
            files=tuple(file for file in checkpoint.files if file.role != "rng"),
        )
    with pytest.raises(ManifestError):
        replace(raw, files=(replace(raw.files[0], role="not-data"),))
    with pytest.raises(ManifestError):
        replace(
            run, files=(*run.files, replace(run.files[0], path="extra-config.yaml"))
        )


def test_code_identity_rejects_partial_or_unknown_commits() -> None:
    for commit in ("HEAD", "abc123", "", "x" * 40):
        with pytest.raises(ManifestError):
            CodeIdentity(commit, False, "a" * 64)
    assert CodeIdentity("a" * 64, False, "b" * 64).commit == "a" * 64
    with pytest.raises(ManifestError):
        CodeIdentity("a" * 40, cast(bool, 1), "b" * 64)


def test_environment_names_immutable_packages_and_duplicates() -> None:
    args = ("3.12", "CPython", "Linux", "x86_64")
    with pytest.raises(ManifestError):
        EnvironmentInfo(*args, ())
    with pytest.raises(ManifestError):
        EnvironmentInfo(
            *args, (PackageVersion("Qvelt", "1"), PackageVersion("qvelt", "2"))
        )
    with pytest.raises(ManifestError):
        PackageVersion("bad/name", "1")
    with pytest.raises(ManifestError):
        PackageVersion("qvelt", "\ud800")
    assert [
        p.name
        for p in EnvironmentInfo(
            *args, (PackageVersion("z", "1"), PackageVersion("a", "1"))
        ).packages
    ] == ["a", "z"]


@pytest.mark.parametrize(
    "reference",
    [
        "http://example.org/license",
        "https://user:secret@example.org/license",
        "https://example.org/license?token=value",
        "https://[broken",
    ],
)
def test_license_reference_never_accepts_credentials_or_query(reference: str) -> None:
    with pytest.raises(ManifestError):
        SourceIdentity("fixture", "v1", "LicenseRef-Fixture", reference)


def test_data_and_tokenizer_linkage_rules(manifest_cases: tuple[Manifest, ...]) -> None:
    raw = cast(DataInfo, manifest_cases[1].metadata)
    tokenized = cast(DataInfo, manifest_cases[2].metadata)
    with pytest.raises(ManifestError):
        replace(raw, tokens=1)
    with pytest.raises(ManifestError):
        replace(tokenized, tokenizer_sha256=None)
    with pytest.raises(ManifestError):
        replace(raw, sources=())
    with pytest.raises(ManifestError):
        TokenizerInfo("byte_bpe", 287, 32, "a" * 64)
    with pytest.raises(ManifestError):
        replace(cast(CheckpointInfo, manifest_cases[4].metadata), world_size=0)
    with pytest.raises(ManifestError):
        replace(cast(RunInfo, manifest_cases[0].metadata), seed=2**32)
