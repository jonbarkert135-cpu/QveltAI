"""Canonical document publication boundaries and exact secret baseline identities."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_preserves_canonical_links_and_excludes_private_paths(
    tmp_path: Path,
) -> None:
    docs = script("docs_build")
    (tmp_path / "README.md").write_text("hello")
    (tmp_path / "private.txt").write_text("example")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/state.md").write_text("state")
    files = docs.reference_files(tmp_path)
    assert files == {Path("README.md")}
    assert (
        docs.rewrite_links(
            "[home](README.md#start)", Path("README.md"), files, tmp_path
        )
        == "[home](README.md#start)"
    )
    assert "blob/main/private.txt" in docs.rewrite_links(
        "[source](private.txt)", Path("README.md"), files, tmp_path
    )
    with pytest.raises(ValueError, match="Broken"):
        docs.rewrite_links("[missing](gone.md)", Path("README.md"), files, tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        docs.rewrite_links("[escape](../outside)", Path("README.md"), files, tmp_path)


def test_credential_baseline_binds_path_detector_and_hash() -> None:
    secrets = script("check_secrets")
    row = {"type": "Secret Keyword", "hashed_secret": "one", "line_number": 1}
    original = secrets.identities({"results": {"example.py": [row]}})
    assert (
        secrets.identities({"results": {"example.py": [{**row, "line_number": 99}]}})
        == original
    )
    for path, changed in [
        ("other.py", row),
        ("example.py", {**row, "hashed_secret": "two"}),
        ("example.py", {**row, "type": "Other"}),
    ]:
        assert secrets.identities({"results": {path: [changed]}}) != original
