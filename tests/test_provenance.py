"""Actual local Git, environment inventories and provenance-only run bundles."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from qvelt import __version__
from qvelt.artifacts import ArtifactStore, hash_file
from qvelt.config import ExperimentConfig, config_digest, dump_config
from qvelt.manifest_types import RunInfo
from qvelt.manifests import manifest_digest
from qvelt.provenance import (
    ProvenanceError,
    capture_code,
    capture_environment,
    initialize_run,
)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def code_repo(tmp_path: Path) -> Path:
    root = tmp_path / "code repo"
    root.mkdir()
    git(root, "init", "--initial-branch=main")
    (root / "uv.lock").write_text("version = 1\n")
    (root / "code.py").write_text("VALUE = 1\n")
    git(root, "add", "uv.lock", "code.py")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "local fixture",
    )
    return root


def test_actual_committed_dirty_and_untracked_code_identity(code_repo: Path) -> None:
    root, code = capture_code(code_repo)
    assert root == code_repo
    assert code.commit == git(code_repo, "rev-parse", "HEAD")
    assert code.dirty is False
    assert code.lock_sha256 == hash_file(code_repo / "uv.lock").sha256
    (code_repo / "code.py").write_text("VALUE = 2\n")
    assert capture_code(code_repo)[1].dirty is True
    git(code_repo, "restore", "code.py")
    (code_repo / "untracked.txt").write_text("fixture")
    assert capture_code(code_repo)[1].dirty is True


def test_unborn_non_git_and_missing_lock_are_not_invented(
    tmp_path: Path, code_repo: Path
) -> None:
    with pytest.raises(ProvenanceError):
        capture_code(tmp_path)
    unborn = tmp_path / "unborn"
    unborn.mkdir()
    git(unborn, "init", "--initial-branch=main")
    with pytest.raises(ProvenanceError):
        capture_code(unborn)
    (code_repo / "uv.lock").unlink()
    with pytest.raises(OSError):
        capture_code(code_repo)


def test_git_timeout_has_no_untrusted_stderr_in_public_error(code_repo: Path) -> None:
    with patch(
        "qvelt.provenance.subprocess.run",
        side_effect=subprocess.TimeoutExpired("git", 30),
    ):
        with pytest.raises(ProvenanceError, match="Cannot query"):
            capture_code(code_repo)


def test_actual_environment_inventory_contains_versions_not_env_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "fixture-private-value-never-persist"
    monkeypatch.setenv("QVELT_TEST_PRIVATE", marker)
    environment = capture_environment()
    packages = {p.name: p.version for p in environment.packages}
    assert packages["qvelt"] == __version__
    assert packages["numpy"] == "2.2.6"
    assert environment.python == sys.version.split()[0]
    assert marker not in repr(environment)
    assert "QVELT_TEST_PRIVATE" not in repr(environment)


def test_conflicting_and_missing_distribution_metadata_are_rejected() -> None:
    first = SimpleNamespace(metadata={"Name": "qvelt"}, version="1")
    second = SimpleNamespace(metadata={"Name": "Qvelt"}, version="2")
    with patch(
        "qvelt.provenance.metadata.distributions", return_value=iter([first, second])
    ):
        with pytest.raises(ProvenanceError, match="Conflicting"):
            capture_environment()
    with patch(
        "qvelt.provenance.metadata.distributions",
        return_value=iter([SimpleNamespace(metadata={}, version="1")]),
    ):
        with pytest.raises(ProvenanceError, match="Cannot read"):
            capture_environment()


def test_initialize_run_copies_actual_config_and_lock_without_training(
    code_repo: Path, tmp_path: Path
) -> None:
    config = replace(ExperimentConfig(), name="actual-test-run")
    directory = tmp_path / "run"
    manifest = initialize_run(config, directory, code_repo)
    info = cast(RunInfo, manifest.metadata)
    assert info.config_sha256 == config_digest(config)
    assert info.code.dirty is False and info.world_size == 1
    assert info.code.commit == git(code_repo, "rev-parse", "HEAD")
    assert manifest.owner == info.run_id
    assert (
        ArtifactStore(directory).read_manifest(
            expected_sha256=manifest_digest(manifest)
        )
        == manifest
    )
    assert (directory / "config.yaml").read_text() == dump_config(config)
    assert (directory / "uv.lock").read_bytes() == (code_repo / "uv.lock").read_bytes()
    assert {p.name for p in directory.iterdir()} == {
        "config.yaml",
        "uv.lock",
        "manifest.json",
    }
    assert "optimizer" not in {file.role for file in manifest.files}
    assert not git(code_repo, "status", "--porcelain")


def test_existing_run_directory_is_never_reused(
    code_repo: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        initialize_run(ExperimentConfig(), directory, code_repo)
    assert list(directory.iterdir()) == [directory / "user.txt"]
    assert (directory / "user.txt").read_text() == "keep"


def test_failure_before_marker_is_not_a_committed_run(
    code_repo: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "partial-run"
    with patch.object(
        ArtifactStore, "publish", side_effect=OSError("injected publication failure")
    ):
        with pytest.raises(OSError):
            initialize_run(ExperimentConfig(), directory, code_repo)
    assert (directory / "config.yaml").exists()
    assert (directory / "uv.lock").exists()
    assert not (directory / "manifest.json").exists()
    with pytest.raises(OSError):
        ArtifactStore(directory).read_manifest()


def test_cli_separate_process_init_inspect_verify_and_corruption(
    code_repo: Path, tmp_path: Path
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(dump_config(ExperimentConfig()))
    directory = tmp_path / "cli-run"

    def command(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "qvelt", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

    result = command(
        "run",
        "init",
        str(config_path),
        "--directory",
        str(directory),
        "--repository",
        str(code_repo),
    )
    assert result.returncode == 0, result.stderr
    identity = json.loads(result.stdout)["sha256"]
    path = str(directory / "manifest.json")
    result = command("manifest", "inspect", path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["files_verified"] is False
    result = command("manifest", "verify", path, "--expected-sha256", identity)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["files"] == 2
    (directory / "config.yaml").write_bytes(b"corrupt")
    result = command("manifest", "verify", path)
    assert result.returncode == 1 and "mismatch" in json.loads(result.stderr)["message"]
    result = command("manifest", "inspect", path, "--expected-sha256", "0" * 64)
    assert result.returncode == 1


def test_cli_corrupt_metadata_rejects_schema_without_dumping_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    marker = "fixture-sensitive-payload"
    path.write_bytes(("{broken " + marker).encode())
    result = subprocess.run(
        [sys.executable, "-m", "qvelt", "manifest", "inspect", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert marker not in result.stderr and "Traceback" not in result.stderr
