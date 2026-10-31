"""Real local IO, interruption, races and injected fsync/integrity failures."""

from __future__ import annotations

import hashlib
import io
import os
import select
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, cast
from unittest.mock import patch

import pytest

import qvelt.utils.artifacts as artifacts
from qvelt.utils.artifacts import (
    CHUNK_BYTES,
    ArtifactDurabilityError,
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactStore,
    atomic_write,
    create_directory,
    hash_file,
)
from qvelt.utils.manifest_types import (
    ArtifactFile,
    CheckpointInfo,
    Manifest,
    ManifestError,
    RunInfo,
)
from qvelt.utils.manifests import manifest_bytes, manifest_digest


def populate(
    root: Path, manifest: Manifest, payloads: dict[str, bytes]
) -> ArtifactStore:
    store = ArtifactStore(root)
    for file in manifest.files:
        assert store.add_bytes(file.path, payloads[file.path], role=file.role) == file
    return store


@pytest.mark.parametrize("index", range(5))
def test_commit_all_kinds_and_inspect_from_a_new_store(
    index: int,
    tmp_path: Path,
    manifest_cases: tuple[Manifest, ...],
    fixture_payloads: dict[str, bytes],
) -> None:
    manifest = manifest_cases[index]
    store = populate(tmp_path, manifest, fixture_payloads)
    store.publish(manifest)
    assert (
        ArtifactStore(tmp_path).read_manifest(expected_sha256=manifest_digest(manifest))
        == manifest
    )
    assert not list(tmp_path.glob(".qvelt-*.tmp"))
    with pytest.raises(FileExistsError):
        store.publish(manifest)


def test_streaming_copy_and_known_sha256_vector(tmp_path: Path) -> None:
    path = tmp_path / "abc"
    path.write_bytes(b"abc")
    assert (
        hash_file(path).sha256
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

    class BoundedSource(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 < size <= CHUNK_BYTES
            return super().read(size)

    data = b"x" * (3 * CHUNK_BYTES + 17)
    record = ArtifactStore(tmp_path).add_stream(
        "parts/shard.bin", BoundedSource(data), role="data"
    )
    assert record.size_bytes == len(data)
    assert record.sha256 == hashlib.sha256(data).hexdigest()
    assert (tmp_path / record.path).read_bytes() == data


def test_copy_does_not_share_source_inode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"original")
    store = ArtifactStore(tmp_path)
    store.add_file("copied", source, role="data")
    source.write_bytes(b"modified")
    assert (tmp_path / "copied").read_bytes() == b"original"


def test_declared_stream_size_rejects_growth_without_publishing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="snapshot"):
        store.add_stream(
            "data", io.BytesIO(b"more-than-declared"), role="data", max_bytes=3
        )
    assert not (tmp_path / "data").exists()
    assert not list(tmp_path.glob(".qvelt-*"))
    with pytest.raises(ManifestError):
        store.add_stream("data", io.BytesIO(b""), role="data", max_bytes=-1)


def test_new_parent_directory_entries_are_synced(tmp_path: Path) -> None:
    synchronize = artifacts._sync_directory
    with patch.object(artifacts, "_sync_directory", wraps=synchronize) as sync:
        ArtifactStore(tmp_path).add_bytes("a/b/file", b"bytes", role="data")
    assert [call.args[0] for call in sync.call_args_list] == [
        tmp_path,
        tmp_path / "a",
        tmp_path / "a/b",
    ]


def test_source_edit_after_copy_is_rejected_as_uncommitted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"snapshot")
    original = ArtifactStore.add_stream

    def after_copy(
        store: ArtifactStore,
        relative: str,
        stream: BinaryIO,
        *,
        role: str,
        max_bytes: int | None = None,
    ) -> ArtifactFile:
        result = original(store, relative, stream, role=role, max_bytes=max_bytes)
        source.write_bytes(b"changed-after-copy")
        return result

    with patch.object(ArtifactStore, "add_stream", after_copy):
        with pytest.raises(ArtifactIntegrityError, match="Source changed"):
            ArtifactStore(tmp_path).add_file("copied", source, role="data")
    assert (tmp_path / "copied").read_bytes() == b"snapshot"
    assert not (tmp_path / "manifest.json").exists()


def test_callback_and_file_fsync_failures_never_publish(tmp_path: Path) -> None:
    destination = tmp_path / "target"

    def broken(stream: BinaryIO) -> None:
        stream.write(b"partial")
        assert not destination.exists()
        raise OSError("injected callback failure")

    with pytest.raises(OSError, match="callback"):
        atomic_write(destination, broken)
    assert not destination.exists() and not list(tmp_path.glob(".qvelt-*"))
    with patch.object(os, "fsync", side_effect=OSError("injected fsync failure")):
        with pytest.raises(OSError, match="fsync"):
            ArtifactStore(tmp_path).add_bytes("target", b"complete", role="data")
    assert not destination.exists() and not list(tmp_path.glob(".qvelt-*"))


def test_postpublication_directory_fsync_error_is_not_reported_as_rollback(
    tmp_path: Path,
) -> None:
    with patch.object(
        artifacts, "_sync_directory", side_effect=OSError("injected directory failure")
    ):
        with pytest.raises(ArtifactDurabilityError) as error:
            ArtifactStore(tmp_path).add_bytes("target", b"complete", role="data")
    assert error.value.published is True
    assert (tmp_path / "target").read_bytes() == b"complete"
    assert not list(tmp_path.glob(".qvelt-*"))


def test_existing_destination_is_preserved_before_callback(tmp_path: Path) -> None:
    destination = tmp_path / "target"
    destination.write_bytes(b"old")

    def must_not_run(stream: BinaryIO) -> None:
        raise AssertionError("Existing target should fail before writer invocation")

    with pytest.raises(FileExistsError):
        atomic_write(destination, must_not_run)
    assert destination.read_bytes() == b"old"


def test_concurrent_writers_publish_exactly_one_complete_payload(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    payloads = [b"a" * 65536, b"b" * 65536]

    def attempt(data: bytes) -> bool:
        def write(stream: BinaryIO) -> None:
            stream.write(data)
            barrier.wait(timeout=10)

        try:
            atomic_write(tmp_path / "winner", write)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, payloads))
    assert results.count(True) == 1
    assert (tmp_path / "winner").read_bytes() in payloads
    assert not list(tmp_path.glob(".qvelt-*"))


def test_killed_writer_leaves_no_final_file_and_a_new_write_can_succeed(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "target"
    script = """
import signal,sys
from qvelt.utils.artifacts import atomic_write
def write(stream):
    stream.write(b'partial')
    stream.flush()
    print('READY', flush=True)
    signal.pause()
atomic_write(sys.argv[1],write)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(destination)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready and process.stdout.readline().strip() == "READY"
        assert not destination.exists()
    finally:
        process.kill()
        process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
    assert not destination.exists()
    assert list(tmp_path.glob(".qvelt-*.tmp"))  # SIGKILL cannot execute cleanup.
    record = ArtifactStore(tmp_path).add_bytes(
        "target", b"new complete data", role="data"
    )
    assert hash_file(destination).sha256 == record.sha256


def test_manifest_is_not_published_when_artifacts_are_bad(
    tmp_path: Path,
    manifest_cases: tuple[Manifest, ...],
    fixture_payloads: dict[str, bytes],
) -> None:
    manifest = manifest_cases[1]
    store = populate(tmp_path, manifest, fixture_payloads)
    (tmp_path / "data.bin").write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError):
        store.publish(manifest)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("mode", ["same_size", "truncated", "missing"])
def test_corrupted_committed_files_are_detected(
    mode: str,
    tmp_path: Path,
    manifest_cases: tuple[Manifest, ...],
    fixture_payloads: dict[str, bytes],
) -> None:
    manifest = manifest_cases[1]
    store = populate(tmp_path, manifest, fixture_payloads)
    store.publish(manifest)
    path = tmp_path / "data.bin"
    if mode == "missing":
        path.unlink()
    else:
        path.write_bytes(b"x" * path.stat().st_size if mode == "same_size" else b"")
    assert (
        store.read_manifest(verify=False) == manifest
    )  # Metadata-only means exactly that.
    with pytest.raises((ArtifactIntegrityError, OSError)):
        store.read_manifest()


def test_corrupted_marker_and_expected_identity_fail(
    tmp_path: Path,
    manifest_cases: tuple[Manifest, ...],
    fixture_payloads: dict[str, bytes],
) -> None:
    manifest = manifest_cases[1]
    store = populate(tmp_path, manifest, fixture_payloads)
    store.publish(manifest)
    with pytest.raises(ManifestError):
        store.read_manifest(expected_sha256="0" * 64)
    (tmp_path / "manifest.json").write_bytes(b"{partial")
    with pytest.raises(ManifestError):
        store.read_manifest()


def test_manifest_write_failure_preserves_data_but_no_commit_marker(
    tmp_path: Path,
    manifest_cases: tuple[Manifest, ...],
    fixture_payloads: dict[str, bytes],
) -> None:
    manifest = manifest_cases[1]
    store = populate(tmp_path, manifest, fixture_payloads)
    with patch.object(os, "link", side_effect=OSError("injected commit failure")):
        with pytest.raises(OSError):
            store.publish(manifest)
    assert (tmp_path / "data.bin").exists()
    assert not (tmp_path / "manifest.json").exists()
    store.publish(manifest)
    assert store.read_manifest() == manifest


def test_post_marker_fsync_failure_leaves_an_inspectable_commit(
    tmp_path: Path,
    manifest_cases: tuple[Manifest, ...],
    fixture_payloads: dict[str, bytes],
) -> None:
    manifest = manifest_cases[1]
    store = populate(tmp_path, manifest, fixture_payloads)
    with patch.object(artifacts, "_sync_directory", side_effect=OSError("injected")):
        with pytest.raises(ArtifactDurabilityError):
            store.publish(manifest)
    assert store.read_manifest() == manifest


def test_symlink_escapes_and_nonregular_files_are_rejected(tmp_path: Path) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    store = ArtifactStore(root)
    with pytest.raises(ArtifactIntegrityError):
        store.add_bytes("escape/payload", b"bad", role="data")
    assert not (outside / "payload").exists()
    (root / "dangling").symlink_to(outside / "absent")
    with pytest.raises(ArtifactIntegrityError):
        store.add_bytes("dangling", b"bad", role="data")
    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ArtifactIntegrityError):
        hash_file(fifo)
    with pytest.raises(ArtifactIntegrityError):
        hash_file(root)
    with pytest.raises((ArtifactError, OSError)):
        hash_file(root / "escape")


def test_edit_during_hashing_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "changing"
    path.write_bytes(b"abcdef")
    reader = artifacts._regular_reader

    @contextmanager
    def changing_reader(target: Path) -> Iterator[BinaryIO]:
        with reader(target) as stream:

            class ChangedReader:
                def fileno(self) -> int:
                    return stream.fileno()

                def read(self, size: int) -> bytes:
                    data = stream.read(size)
                    if data:
                        with path.open("ab") as out:
                            out.write(b"g")
                    return data

            yield cast(BinaryIO, ChangedReader())

    with patch.object(artifacts, "_regular_reader", changing_reader):
        with pytest.raises(ArtifactIntegrityError):
            hash_file(path)


def test_config_seed_lock_and_checkpoint_identity_consistency(
    tmp_path: Path,
    manifest_cases: tuple[Manifest, ...],
    fixture_payloads: dict[str, bytes],
) -> None:
    run = manifest_cases[0]
    store = populate(tmp_path, run, fixture_payloads)
    info = cast(RunInfo, run.metadata)
    for metadata in (
        replace(info, config_sha256="0" * 64),
        replace(info, seed=info.seed + 1),
        replace(info, code=replace(info.code, lock_sha256="0" * 64)),
    ):
        with pytest.raises(ArtifactIntegrityError):
            store.publish(replace(run, metadata=metadata))
    checkpoint = manifest_cases[4]
    for file in checkpoint.files:
        if file.role != "config":
            store.add_bytes(file.path, fixture_payloads[file.path], role=file.role)
    checkpoint_metadata = replace(
        cast(CheckpointInfo, checkpoint.metadata), precision="bf16"
    )
    with pytest.raises(ArtifactIntegrityError):
        store.verify(replace(checkpoint, metadata=checkpoint_metadata))


def test_self_reference_and_nested_commit_markers_are_not_allowed(
    tmp_path: Path, manifest_cases: tuple[Manifest, ...]
) -> None:
    store = ArtifactStore(tmp_path)
    manifest = replace(
        manifest_cases[1], files=(ArtifactFile("manifest.json", "data", 0, "a" * 64),)
    )
    with pytest.raises(ManifestError):
        store.publish(manifest)
    (tmp_path / "manifest.json").write_bytes(manifest_bytes(manifest))
    with pytest.raises(ManifestError):
        store.read_manifest(verify=False)
    with pytest.raises(ManifestError):
        store.read_manifest("nested/manifest.json")


def test_new_directory_creation_never_reuses_existing_work(tmp_path: Path) -> None:
    created = create_directory(tmp_path / "new")
    (created / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        create_directory(created)
    assert (created / "user.txt").read_text() == "keep"


def test_existing_parent_entries_are_synced_for_concurrent_creators(
    tmp_path: Path,
) -> None:
    (tmp_path / "a").mkdir()
    synchronize = artifacts._sync_directory
    with patch.object(artifacts, "_sync_directory", wraps=synchronize) as sync:
        ArtifactStore(tmp_path).add_bytes("a/file", b"complete", role="data")
    assert [call.args[0] for call in sync.call_args_list] == [tmp_path, tmp_path / "a"]


def test_public_reader_uses_safe_regular_file_policy(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.add_bytes("payload.bin", b"known bytes", role="data")
    with store.open_file("payload.bin") as stream:
        assert stream.read() == b"known bytes"
    with pytest.raises(ManifestError):
        with store.open_file("../outside"):
            pass
    (tmp_path / "link").symlink_to(tmp_path / "payload.bin")
    with pytest.raises(ArtifactIntegrityError, match="Symlink"):
        with store.open_file("link"):
            pass
    (tmp_path / "directory").mkdir()
    with pytest.raises(ArtifactIntegrityError, match="regular"):
        with store.open_file("directory"):
            pass
