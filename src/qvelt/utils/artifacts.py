"""Streaming integrity checks and write-once POSIX local artifact publication."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from qvelt.config import MAX_CONFIG_BYTES, ConfigError, config_digest, parse_config
from qvelt.utils.manifest_types import (
    MAX_MANIFEST_BYTES,
    ArtifactFile,
    CheckpointInfo,
    Manifest,
    ManifestError,
    RunInfo,
    artifact_path,
    identifier,
    uint,
)
from qvelt.utils.manifests import manifest_bytes, parse_manifest

CHUNK_BYTES = 1024 * 1024


class ArtifactError(RuntimeError):
    """A local artifact cannot be safely written, read or verified."""


class ArtifactIntegrityError(ArtifactError):
    """Registered bytes or identities do not match the manifest."""


class ArtifactDurabilityError(ArtifactError):
    """Complete bytes are published, but directory durability is unconfirmed."""

    published = True


@dataclass(frozen=True)
class FileDigest:
    """Observed raw-byte size and SHA-256, not semantic manifest identity."""

    size_bytes: int
    sha256: str


@contextmanager
def _regular_reader(path: Path) -> Iterator[BinaryIO]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ArtifactIntegrityError("Expected a regular artifact file")
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "rb") as stream:
        yield stream


def hash_file(path: str | Path) -> FileDigest:
    """Hash regular bytes in bounded memory and reject observed concurrent edits."""
    digest = hashlib.sha256()
    count = 0
    with _regular_reader(Path(path)) as stream:
        before = os.fstat(stream.fileno())
        remaining = before.st_size
        while remaining:
            block = stream.read(min(CHUNK_BYTES, remaining))
            if not block:
                break
            count += len(block)
            remaining -= len(block)
            digest.update(block)
        if stream.read(1):
            raise ArtifactIntegrityError("Artifact grew while being hashed")
        after = os.fstat(stream.fileno())
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if count != before.st_size or any(
        getattr(before, key) != getattr(after, key) for key in identity
    ):
        raise ArtifactIntegrityError("Artifact changed while being hashed")
    return FileDigest(count, digest.hexdigest())


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_directory(path: str | Path) -> Path:
    """Create one new private bundle root under an existing parent and sync it."""
    if os.name != "posix":
        raise ArtifactError("Artifact writes require the POSIX local backend")
    path = Path(path)
    parent = path.parent.resolve(strict=True)
    destination = parent / path.name
    destination.mkdir(mode=0o700)
    _sync_directory(parent)
    return destination


def atomic_write(path: str | Path, writer: Callable[[BinaryIO], None]) -> None:
    """Publish complete bytes once; never replace an existing destination.

    Callback/file-fsync/link failures do not publish. A directory-fsync failure
    after the hard link raises ArtifactDurabilityError and preserves final bytes.
    Only owned temporary names are removed. Process death may leave stale temp
    names; it cannot make them committed artifacts. Requires POSIX local storage.
    """
    if os.name != "posix":
        raise ArtifactError("Atomic artifact writes require the POSIX local backend")
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"Artifact already exists: {path}")
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".qvelt-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        try:
            _sync_directory(path.parent)
        except OSError as error:
            raise ArtifactDurabilityError(
                "Complete artifact published; directory durability could not be confirmed"
            ) from error
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_bytes(path: Path, limit: int) -> bytes:
    with _regular_reader(path) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ArtifactIntegrityError("Metadata file exceeds its size limit")
    return data


class ArtifactStore:
    """A trusted local root with relative-path, no-symlink artifact operations.

    Does not sandbox hostile concurrent filesystem writers or deserialize model
    state. Unreferenced files are not part of a committed manifest.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ArtifactError("Artifact root must be an existing directory")

    @contextmanager
    def open_file(self, relative: str) -> Iterator[BinaryIO]:
        """Read a regular contained file; callers validate consumed byte hashes."""
        with _regular_reader(self._path(relative)) as stream:
            yield stream

    def _path(self, relative: str, *, create_parents: bool = False) -> Path:
        relative = artifact_path(relative)
        current = self.root
        for part in relative.split("/"):
            current = current / part
            if current.is_symlink():
                raise ArtifactIntegrityError(
                    "Symlink traversal is not supported in artifacts"
                )
        if not current.resolve().is_relative_to(self.root):
            raise ArtifactIntegrityError("Artifact path escapes the store")
        if create_parents:
            if os.name != "posix":
                raise ArtifactError("Artifact writes require the POSIX local backend")
            directory = self.root
            for part in relative.split("/")[:-1]:
                parent = directory
                directory = parent / part
                try:
                    directory.mkdir(mode=0o700)
                except FileExistsError:
                    if directory.is_symlink() or not directory.is_dir():
                        raise ArtifactIntegrityError(
                            "Invalid artifact parent directory"
                        ) from None
                # Another writer may have just created this directory but not
                # synced its parent yet. Each writer confirms the path entries.
                _sync_directory(parent)
        return current

    def add_stream(
        self,
        relative: str,
        source: BinaryIO,
        *,
        role: str,
        max_bytes: int | None = None,
    ) -> ArtifactFile:
        """Copy from the stream's current position in bounded chunks, then commit."""
        identifier(role, "artifact.role")
        if max_bytes is not None:
            uint(max_bytes, "max_bytes")
        destination = self._path(relative, create_parents=True)
        digest = hashlib.sha256()
        count = 0

        def write(stream: BinaryIO) -> None:
            nonlocal count
            while block := source.read(CHUNK_BYTES):
                if max_bytes is not None and count + len(block) > max_bytes:
                    raise ArtifactIntegrityError(
                        "Source exceeds its declared snapshot size"
                    )
                stream.write(block)
                count += len(block)
                digest.update(block)

        atomic_write(destination, write)
        return ArtifactFile(relative, role, count, digest.hexdigest())

    def add_bytes(self, relative: str, data: bytes, *, role: str) -> ArtifactFile:
        """Write small metadata bytes; use add_stream/add_file for large artifacts."""
        import io

        return self.add_stream(relative, io.BytesIO(data), role=role)

    def add_file(self, relative: str, source: str | Path, *, role: str) -> ArtifactFile:
        """Copy a regular source file; do not hard-link mutable source contents."""
        with _regular_reader(Path(source)) as stream:
            before = os.fstat(stream.fileno())
            record = self.add_stream(
                relative, stream, role=role, max_bytes=before.st_size
            )
            after = os.fstat(stream.fileno())
        if (
            record.size_bytes != before.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ArtifactIntegrityError(
                "Source changed during copy; no manifest was committed"
            )
        return record

    def verify(self, manifest: Manifest) -> None:
        """Check every registered file and the declared config/lock consistency."""
        for record in manifest.files:
            observed = hash_file(self._path(record.path))
            if (
                observed.size_bytes != record.size_bytes
                or observed.sha256 != record.sha256
            ):
                raise ArtifactIntegrityError(
                    f"Artifact size/hash mismatch: {record.path}"
                )
        metadata = manifest.metadata
        if isinstance(metadata, RunInfo | CheckpointInfo):
            record = next(file for file in manifest.files if file.role == "config")
            data = _read_bytes(self._path(record.path), MAX_CONFIG_BYTES)
            if hashlib.sha256(data).hexdigest() != record.sha256:
                raise ArtifactIntegrityError("Config changed while being verified")
            try:
                config = parse_config(data.decode("utf-8-sig"))
            except (ConfigError, UnicodeError) as error:
                raise ArtifactIntegrityError(
                    "Config artifact is not valid schema-1 YAML"
                ) from error
            if config_digest(config) != metadata.config_sha256:
                raise ArtifactIntegrityError(
                    "Config identity disagrees with manifest metadata"
                )
            if isinstance(metadata, RunInfo):
                if config.runtime.seed != metadata.seed:
                    raise ArtifactIntegrityError(
                        "Run seed disagrees with resolved config"
                    )
                lock = next(
                    file for file in manifest.files if file.role == "dependency_lock"
                )
                if lock.sha256 != metadata.code.lock_sha256:
                    raise ArtifactIntegrityError(
                        "Dependency lock identity disagrees with code metadata"
                    )
            elif config.runtime.precision != metadata.precision:
                raise ArtifactIntegrityError(
                    "Checkpoint precision disagrees with resolved config"
                )

    def publish(self, manifest: Manifest, name: str = "manifest.json") -> None:
        """Verify existing artifacts, then publish the top-level commit marker."""
        artifact_path(name)
        if "/" in name or name in {file.path for file in manifest.files}:
            raise ManifestError(
                "Manifest must be a top-level file, not its own artifact"
            )
        data = manifest_bytes(manifest)
        self.verify(manifest)

        def write(stream: BinaryIO) -> None:
            stream.write(data)

        atomic_write(self._path(name), write)

    def read_manifest(
        self,
        name: str = "manifest.json",
        *,
        expected_sha256: str | None = None,
        verify: bool = True,
    ) -> Manifest:
        """Load bounded metadata, optionally enforcing trusted ID and file checks."""
        artifact_path(name)
        if "/" in name:
            raise ManifestError("Manifest must be a top-level commit marker")
        manifest = parse_manifest(
            _read_bytes(self._path(name), MAX_MANIFEST_BYTES),
            expected_sha256=expected_sha256,
        )
        if name in {file.path for file in manifest.files}:
            raise ManifestError("Manifest cannot reference itself")
        if verify:
            self.verify(manifest)
        return manifest
