"""Framework version/tag and distribution-content validation;no publication side effects."""

from __future__ import annotations

import hashlib
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

from qvelt import __version__


def validate_version(repository: Path, tag: str) -> str:
    match = re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag)
    if match is None:
        raise ValueError("Release tag must be stable semantic vMAJOR.MINOR.PATCH")
    metadata = tomllib.loads((repository / "pyproject.toml").read_text())
    version = metadata["project"]["version"]
    if not isinstance(version, str) or version != tag[1:] or version != __version__:
        raise ValueError("Tag,project and package versions disagree")
    return version


def validate_distribution(path: Path) -> str:
    """Reject accidental model/private/vendor material in wheel/sdist;return SHA256."""
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            if any(member.issym() or member.islnk() for member in members):
                raise ValueError("Release archives must contain regular source files")
            names = [member.name for member in members]
    else:
        raise ValueError("Only wheel/sdist framework archives are supported")
    for name in names:
        value = PurePosixPath(name)
        if (
            value.is_absolute()
            or ".." in value.parts
            or set(value.parts)
            & {
                ".git",
                ".agents",
                ".cache",
                ".venv",
                ".bootstrap",
                "runs",
                "checkpoints",
                "artifacts",
            }
        ):
            raise ValueError("Release archive contains excluded/private paths")
        if any(part.startswith(".env") for part in value.parts) or value.suffix in {
            ".pt",
            ".pth",
            ".pkl",
            ".pickle",
            ".safetensors",
            ".key",
        }:
            raise ValueError(
                "Framework archive must not contain credentials or model/training state"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()
