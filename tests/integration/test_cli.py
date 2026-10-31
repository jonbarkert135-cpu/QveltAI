"""Real CLI subprocesses and public import/metadata contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import qvelt
from qvelt.cli import main
from qvelt.config import config_digest, load_config, parse_config

ROOT = Path(__file__).resolve().parents[2]


def test_package_version_and_import_have_no_torch_side_effect() -> None:
    assert qvelt.__version__ == version("qvelt")
    result = subprocess.run(
        [sys.executable, "-c", "import sys, qvelt; assert 'torch' not in sys.modules"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "" and result.stderr == ""


def test_validate_resolve_and_missing_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = ROOT / "configs/tiny.yaml"
    assert main(["config", "validate", str(path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["sha256"] == config_digest(load_config(path))
    assert data["analytical_parameters"] == 19_338_560
    assert main(["config", "resolve", str(path)]) == 0
    assert parse_config(capsys.readouterr().out) == load_config(path)
    assert main(["config", "validate", str(tmp_path / "absent")]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["level"] == "ERROR"


def test_invalid_yaml_exit_status_and_no_value_leak(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model: {layers: confidential-value}")
    result = subprocess.run(
        [sys.executable, "-m", "qvelt", "config", "validate", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "confidential-value" not in result.stderr
    assert json.loads(result.stderr)["level"] == "ERROR"


def test_cpu_doctor_runs_real_tensor_probe(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--device", "cpu"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["requested_device"] == "cpu" and data["test_checksum"] == 8.0


def test_train_requires_a_real_recipe_and_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "qvelt", "train"], capture_output=True, text=True
    )
    assert result.returncode == 2


def test_module_entrypoint_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "qvelt", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"qvelt {qvelt.__version__}"
