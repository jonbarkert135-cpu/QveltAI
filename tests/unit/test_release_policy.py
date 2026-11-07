"""Version synchronization and exclusion of model/private files from release archives."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from qvelt import __version__
from qvelt.utils.release import validate_distribution, validate_version

ROOT = Path(__file__).resolve().parents[2]


def test_current_version_and_stable_tag() -> None:
    assert validate_version(ROOT, "v" + __version__) == __version__
    for tag in ("main", "v01.0.0", "v0.0.0", "v0.1.0;echo bad", "v0.1.0-rc.1"):
        with pytest.raises(ValueError):
            validate_version(ROOT, tag)


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".git/config",
        ".agents/helper.py",
        "runs/model.pt",
        "../escape",
        "/absolute",
        "qvelt/weights.safetensors",
    ],
)
def test_release_forbidden_files(tmp_path: Path, name: str) -> None:
    path = tmp_path / "qvelt.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, "not a real secret or weight")
    with pytest.raises(ValueError):
        validate_distribution(path)


def test_release_regular_archives_and_symlink_rejection(tmp_path: Path) -> None:
    path = tmp_path / "qvelt.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("qvelt/src/qvelt/__init__.py")
        entry.size = 4
        archive.addfile(entry, io.BytesIO(b"pass"))
    assert len(validate_distribution(path)) == 64
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("qvelt/link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../../outside"
        archive.addfile(entry)
    with pytest.raises(ValueError, match="regular"):
        validate_distribution(path)
