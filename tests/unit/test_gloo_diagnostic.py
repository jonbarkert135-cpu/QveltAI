"""Fast subprocess checks for the opt-in diagnostic, without distributed training."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def diagnostic() -> Any:
    spec = importlib.util.spec_from_file_location(
        "gloo_diagnostic", ROOT / "scripts/gloo_diagnostic.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("exit_code", [0, 7])
def test_child_outcome_and_evidence(tmp_path: Path, exit_code: int) -> None:
    module = diagnostic()
    report = module.diagnose(
        tmp_path,
        iterations=2,
        iteration_timeout=5,
        total_timeout=15,
        command=[sys.executable, "-c", f"print('child evidence');exit({exit_code})"],
        environment={"fixture": True},
    )
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved == report
    assert report["completed_iterations"] == 2
    assert report["failures"] == (0 if exit_code == 0 else 2)
    ports = []
    for number, row in enumerate(report["iterations"], 1):
        assert row["iteration"] == number
        assert row["outcome"] == ("passed" if exit_code == 0 else "failed")
        assert row["returncode"] == exit_code
        assert row["duration_seconds"] > 0
        assert "child evidence" in row["last_log_lines"]
        assert row["environment"]["fixture"] is True
        assert Path(row["directory"]).is_dir()
        ports.append(row["selected_port"])
    assert len(set(ports)) == 2


def test_timeout_collects_rank_stacks_and_kills_descendants(tmp_path: Path) -> None:
    module = diagnostic()
    code = (
        "import os,subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "env={**os.environ,'RANK':'1'}); "
        "print('before hang',flush=True); time.sleep(60)"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RANK", "0")
        report = module.diagnose(
            tmp_path,
            iterations=1,
            iteration_timeout=1,
            total_timeout=5,
            command=[sys.executable, "-c", code],
            environment={"fixture": True},
        )
    row = report["iterations"][0]
    assert row["outcome"] == "timeout" and row["returncode"] is not None
    assert "before hang" in row["last_log_lines"]
    assert {item["rank"] for item in row["processes"]} == {"0", "1"}
    assert len(row["stack_dumps"]) == 2
    assert all("<string>" in text for text in row["stack_dumps"].values())
    for item in row["processes"]:
        # A killed orphan may remain a zombie until the container's init reaps it.
        stat = Path(f"/proc/{item['pid']}/stat")
        assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_total_deadline_stops_repetition(tmp_path: Path) -> None:
    module = diagnostic()
    start = time.monotonic()
    report = module.diagnose(
        tmp_path,
        iterations=10,
        iteration_timeout=5,
        total_timeout=0.5,
        command=[sys.executable, "-c", "import time;time.sleep(60)"],
        environment={"fixture": True},
    )
    assert report["completed_iterations"] == 1
    assert report["stop_reason"] == "total_wall_time_limit"
    assert report["iterations"][0]["outcome"] == "timeout"
    assert time.monotonic() - start < 2


def test_timeout_handles_a_rank_in_a_separate_session(tmp_path: Path) -> None:
    module = diagnostic()
    code = (
        "import os,subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "env={**os.environ,'RANK':'1'},start_new_session=True); "
        "time.sleep(60)"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RANK", "0")
        report = module.diagnose(
            tmp_path,
            iterations=1,
            iteration_timeout=1,
            total_timeout=5,
            command=[sys.executable, "-c", code],
            environment={"fixture": True},
        )
    row = report["iterations"][0]
    try:
        assert len(row["processes"]) == 2
        assert all("<string>" in text for text in row["stack_dumps"].values())
        for item in row["processes"]:
            stat = Path(f"/proc/{item['pid']}/stat")
            assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        for item in row["processes"]:
            try:
                os.kill(item["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_reused_pid_identity_is_not_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = diagnostic()
    (tmp_path / "process-stale.json").write_text(
        json.dumps({"pid": os.getpid(), "start_ticks": "not-this-process"})
    )

    def forbidden(*args: object) -> None:
        raise AssertionError("must not signal an unrelated process")

    monkeypatch.setattr(module.os, "kill", forbidden)
    monkeypatch.setattr(module.os, "killpg", forbidden)
    module.signal_ranks(tmp_path, signal.SIGUSR1)
    module.signal_ranks(tmp_path, signal.SIGKILL)


@pytest.mark.parametrize("inner_timeout", [False, True])
def test_pytest_plugin_bootstrap_and_inner_timeout(
    tmp_path: Path, inner_timeout: bool
) -> None:
    module = diagnostic()
    child_test = tmp_path / "test_child.py"
    child_test.write_text(
        "import subprocess,sys\ndef test_child():\n"
        + (
            "    subprocess.run([sys.executable,'-c','import time;time.sleep(60)'],timeout=0.05)\n"
            if inner_timeout
            else "    assert 2+2 == 4\n"
        )
    )
    report = module.diagnose(
        tmp_path / "evidence",
        iterations=1,
        iteration_timeout=5,
        total_timeout=10,
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "gloo_diagnostic",
            "-c",
            str(ROOT / "pyproject.toml"),
            str(child_test),
        ],
        environment={"fixture": True},
    )
    row = report["iterations"][0]
    assert row["outcome"] == ("timeout" if inner_timeout else "passed")
    assert len(row["tests"]) == 1
    assert row["tests"][0]["outcome"] == ("failed" if inner_timeout else "passed")
    assert bool(row["inner_timeouts"]) == inner_timeout


@pytest.mark.parametrize(
    "settings",
    [
        {"iterations": 0},
        {"iterations": 101},
        {"iterations": True},
        {"iteration_timeout": 0},
        {"total_timeout": float("inf")},
        {"total_timeout": -1},
    ],
)
def test_invalid_bounds_do_not_launch(tmp_path: Path, settings: dict[str, Any]) -> None:
    module = diagnostic()
    with pytest.raises(ValueError, match="bounds"):
        module.diagnose(tmp_path, environment={"fixture": True}, **settings)
    assert not (tmp_path / "report.json").exists()


def test_torchrun_hook_assigns_and_records_fresh_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = diagnostic()
    calls = []

    def fake_popen(args: list[str], **kwargs: Any) -> object:
        calls.append(args)
        return object()

    monkeypatch.setenv("QVELT_GLOO_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setenv("QVELT_GLOO_DIAGNOSTIC_PORT", "12345")
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    module.pytest_configure()
    for _ in range(2):
        module.subprocess.Popen(
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "worker.py"]
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "launches.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 and rows[0]["port"] == 12345
    assert rows[0]["port"] != rows[1]["port"]
    for args, row in zip(calls, rows, strict=True):
        assert "--standalone" not in args
        assert f"--rdzv-endpoint=127.0.0.1:{row['port']}" in args
        assert args[-1] == "worker.py"
    assert module.os.environ["QVELT_GLOO_DIAGNOSTIC_DIR"] == str(tmp_path)
