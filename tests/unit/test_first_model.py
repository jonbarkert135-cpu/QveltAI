"""The documented tutorial fails before allocation or overwrite on invalid budgets."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from qvelt.train.first_model import first_model


def test_tiny_tutorial_budget_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="4..20"):
        first_model(tmp_path / "invalid", tmp_path, steps=1)
    monkeypatch.setattr(
        "qvelt.train.first_model.shutil.disk_usage",
        lambda path: SimpleNamespace(free=0),
    )
    with pytest.raises(ValueError, match="free"):
        first_model(tmp_path / "full-disk", tmp_path)
    assert not (tmp_path / "full-disk").exists() and not (tmp_path / "invalid").exists()
