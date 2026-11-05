"""Capture actual code/environment identities and initialize provenance bundles."""

from __future__ import annotations

import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from qvelt.config import ExperimentConfig, config_digest, dump_config
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.manifest_types import (
    CodeIdentity,
    EnvironmentInfo,
    Manifest,
    PackageVersion,
    RunInfo,
)


class ProvenanceError(RuntimeError):
    """Actual code/environment identity could not be established."""


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repository), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProvenanceError("Cannot query local Git identity") from error
    if result.returncode:
        # Git diagnostics/remote config are not copied into saved provenance.
        raise ProvenanceError(
            "A readable Git checkout with an existing commit is required"
        )
    return result.stdout.strip()


def capture_code(repository: str | Path) -> tuple[Path, CodeIdentity]:
    """Read local HEAD, dirty status and lock hash without changing the checkout."""
    repository = Path(repository).resolve(strict=True)
    root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve(strict=True)
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    dirty = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=normal"))
    lock = hash_file(root / "uv.lock")
    return root, CodeIdentity(commit, dirty, lock.sha256)


def capture_environment() -> EnvironmentInfo:
    """Capture installed distribution names/versions, never URLs or env values.

    This is an observed package inventory, not a CUDA/hardware diagnostic or a
    guarantee that every installed dependency agrees with the lock.
    """
    packages: dict[str, PackageVersion] = {}
    paths = list(sys.path)
    try:
        vendor = Path(
            str(metadata.distribution("setuptools").locate_file("setuptools/_vendor"))
        ).resolve()
    except metadata.PackageNotFoundError:
        pass
    else:
        # cpp_extension imports setuptools, which appends bundled metadata to sys.path.
        paths = [entry for entry in paths if Path(entry).resolve() != vendor]
    for distribution in metadata.distributions(path=paths):
        try:
            name = distribution.metadata["Name"]
            version = distribution.version
        except (KeyError, OSError) as error:
            raise ProvenanceError(
                "Cannot read installed distribution identity"
            ) from error
        if not name:
            raise ProvenanceError("Installed distribution has no readable name")
        record = PackageVersion(name, version)
        previous = packages.get(record.name)
        if previous is not None and previous.version != record.version:
            raise ProvenanceError("Conflicting installed versions of one distribution")
        packages[record.name] = record
    return EnvironmentInfo(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        system=platform.system(),
        machine=platform.machine(),
        packages=tuple(packages.values()),
    )


def initialize_run(
    config: ExperimentConfig, directory: str | Path, repository: str | Path
) -> Manifest:
    """Create a new config/lock/provenance bundle; do not start any training.

    Parent directory must exist. Existing destinations are never reused/deleted.
    Failed writes can leave an uncommitted directory; only manifest publication
    commits a bundle. Dirty=True records incomplete code reproducibility honestly.
    """
    root, code = capture_code(repository)
    environment = capture_environment()
    run_id = uuid.uuid4().hex
    info = RunInfo(
        run_id, config_digest(config), code, environment, config.runtime.seed, 1
    )
    store = ArtifactStore(create_directory(directory))
    config_record = store.add_bytes(
        "config.yaml", dump_config(config).encode("utf-8"), role="config"
    )
    lock_record = store.add_file("uv.lock", root / "uv.lock", role="dependency_lock")
    manifest = Manifest(
        kind="run",
        owner=run_id,
        created_at=datetime.now(UTC).isoformat(),
        metadata=info,
        files=(config_record, lock_record),
    )
    store.publish(manifest)
    return manifest
