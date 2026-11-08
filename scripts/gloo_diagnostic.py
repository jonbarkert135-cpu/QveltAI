"""Opt-in Linux CPU Gloo evidence collector; successful repeats do not prove a fix."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/integration/test_distributed.py",
    "tests/integration/test_pretraining.py::test_two_rank_packed_training_replay_and_uneven_window_oracle",
    "tests/integration/test_phase5_diagnostics.py::test_two_rank_nan_recovery_completes_backward_hooks",
]
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GLOO_SOCKET_IFNAME",
    "TORCH_DISTRIBUTED_DEBUG",
)
_stack_file: TextIO | None = None


def start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]


def free_port(used: set[int]) -> int:
    """Selection is not reservation: another process can bind after socket close."""
    for _ in range(100):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = int(listener.getsockname()[1])
        if port not in used:
            used.add(port)
            return port
    raise RuntimeError("Could not select a fresh diagnostic port")


def install_stack_dump() -> None:
    """Temporary sitecustomize calls this in each Python rank before user code."""
    global _stack_file
    directory = Path(os.environ["QVELT_GLOO_DIAGNOSTIC_DIR"])
    rank, pid = os.environ.get("RANK"), os.getpid()
    if rank is None:
        return
    _stack_file = (directory / f"rank-{rank}-pid-{pid}.stack").open("w")
    faulthandler.enable(file=_stack_file, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=_stack_file, all_threads=True)
    # Inner test timeouts can kill torchrun before the outer supervisor notices.
    faulthandler.dump_traceback_later(10, repeat=True, file=_stack_file)
    (directory / f"process-{pid}.json").write_text(
        json.dumps(
            {
                "pid": pid,
                "start_ticks": start_ticks(pid),
                "rank": rank,
                "started_unix_seconds": time.time(),
                "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
            }
        )
    )


def pytest_configure() -> None:
    """Only loaded with explicit pytest -p; leave assertions and workers untouched."""
    directory_text = os.environ.get("QVELT_GLOO_DIAGNOSTIC_DIR")
    if directory_text is None:
        return
    directory = Path(directory_text)
    original = subprocess.Popen
    used: set[int] = set()

    def launch(args: Any, *positional: Any, **kwargs: Any) -> Any:
        if isinstance(args, list) and args[1:3] == ["-m", "torch.distributed.run"]:
            port = (
                int(os.environ["QVELT_GLOO_DIAGNOSTIC_PORT"])
                if not used
                else free_port(used)
            )
            used.add(port)
            args = [arg for arg in args if arg != "--standalone"]
            args[3:3] = [
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint=127.0.0.1:{port}",
                f"--rdzv-id={uuid.uuid4().hex}",
            ]
            with (directory / "launches.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "port": port,
                            "command": args,
                            "started_unix_seconds": time.time(),
                        }
                    )
                    + "\n"
                )
        return original(args, *positional, **kwargs)

    subprocess.Popen = launch  # type: ignore[assignment, misc]


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    if call.excinfo is not None and isinstance(
        call.excinfo.value, subprocess.TimeoutExpired
    ):
        directory = Path(os.environ["QVELT_GLOO_DIAGNOSTIC_DIR"])
        with (directory / "timeouts.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {"nodeid": item.nodeid, "timeout": call.excinfo.value.timeout}
                )
                + "\n"
            )


def pytest_runtest_logreport(report: Any) -> None:
    if report.when == "call" or report.failed:
        directory = Path(os.environ["QVELT_GLOO_DIAGNOSTIC_DIR"])
        with (directory / "tests.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "nodeid": report.nodeid,
                        "phase": report.when,
                        "outcome": report.outcome,
                        "duration_seconds": report.duration,
                    }
                )
                + "\n"
            )


def environment_info(timeout: float) -> dict[str, Any]:
    code = (
        "import json,torch;print(json.dumps({'torch':torch.__version__,"
        "'torch_git':torch.version.git_version,'gloo_available':torch.distributed.is_gloo_available(),"
        "'gloo_version':None,'gloo_version_boundary':'No public version API in installed Torch',"
        "'probe_torch_threads':torch.get_num_threads(),'probe_torch_interop_threads':torch.get_num_interop_threads()}))"
    )
    probe = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return {
        **json.loads(probe.stdout),
        "python": sys.version,
        "platform": platform.platform(),
        "base_commit": revision.stdout.strip(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_average": list(os.getloadavg()),
        "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
    }


def json_lines(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def signal_ranks(directory: Path, signum: signal.Signals) -> None:
    for path in directory.glob("process-*.json"):
        record = json.loads(path.read_text())
        pid = record["pid"]
        try:
            if start_ticks(pid) != record["start_ticks"]:
                continue
            if signum == signal.SIGKILL and os.getpgid(pid) == pid:
                os.killpg(pid, signum)
            else:
                os.kill(pid, signum)
        except (FileNotFoundError, ProcessLookupError):
            pass


def iteration(
    number: int,
    directory: Path,
    command: list[str],
    env: dict[str, str],
    timeout: float,
    environment: dict[str, Any],
    port: int,
) -> dict[str, Any]:
    directory.mkdir()
    started = time.monotonic()
    expired = False
    with (directory / "child.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=max(0.001, timeout - 0.25))
        except subprocess.TimeoutExpired:
            expired = True
        finally:
            if expired or process.returncode != 0:
                signal_ranks(directory, signal.SIGUSR1)
                time.sleep(0.05)
            # Torchrun ranks have their own sessions, outside the pytest group.
            signal_ranks(directory, signal.SIGKILL)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=0.15)
    inner_timeouts = json_lines(directory / "timeouts.jsonl")
    row = {
        "iteration": number,
        "directory": str(directory),
        "duration_seconds": time.monotonic() - started,
        "outcome": "timeout"
        if expired or inner_timeouts
        else "passed"
        if process.returncode == 0
        else "failed",
        "returncode": process.returncode,
        "selected_port": port,
        "command": command,
        "environment": environment,
        "last_log_lines": "\n".join(
            (directory / "child.log").read_text(errors="replace").splitlines()[-60:]
        ),
        "launches": json_lines(directory / "launches.jsonl"),
        "tests": json_lines(directory / "tests.jsonl"),
        "inner_timeouts": inner_timeouts,
        "processes": [
            json.loads(p.read_text()) for p in sorted(directory.glob("process-*.json"))
        ],
        "stack_dumps": {
            p.name: p.read_text(errors="replace")
            for p in sorted(directory.glob("*.stack"))
        },
    }
    return row


def diagnose(
    output: Path,
    *,
    iterations: int = 10,
    iteration_timeout: float = 180,
    total_timeout: float = 900,
    command: list[str] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        type(iterations) is not int
        or not 1 <= iterations <= 100
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 < x <= 86400
            for x in (iteration_timeout, total_timeout)
        )
    ):
        raise ValueError(
            "Invalid diagnostic bounds: iterations1..100, finite seconds0..86400"
        )
    if sys.platform != "linux":
        raise RuntimeError("Diagnostic process-group/stack supervision requires Linux")
    started = time.monotonic()
    deadline = started + total_timeout
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    if environment is None:
        environment = environment_info(min(15, total_timeout / 4))
    bootstrap = output / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "import os\nif 'RANK' in os.environ:\n"
        "    from gloo_diagnostic import install_stack_dump\n"
        "    install_stack_dump()\n"
    )
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(bootstrap), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requested_iterations": iterations,
        "iteration_timeout_seconds": iteration_timeout,
        "total_timeout_seconds": total_timeout,
        "environment": environment,
        "started_unix_seconds": time.time(),
        "iterations": [],
        "instrumentation": "Explicit fresh c10d loopback endpoints replace --standalone; Python stack watchdog10s/SIGUSR1; unique pytest tmp per iteration. Thread settings inherited; tests force OMP1. Port selection has a bind/close race. No C++ stacks. Not a fix or historical-condition reproduction.",
    }
    used: set[int] = set()
    for number in range(1, iterations + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0.25:
            break
        directory = output / f"iteration-{number:03d}"
        port = free_port(used)
        child_env = {
            **env,
            "QVELT_GLOO_DIAGNOSTIC_DIR": str(directory),
            "QVELT_GLOO_DIAGNOSTIC_PORT": str(port),
        }
        child_command = (
            command
            if command is not None
            else [
                sys.executable,
                "-m",
                "pytest",
                "-vv",
                "-s",
                "-p",
                "no:cacheprovider",
                "-p",
                "gloo_diagnostic",
                f"--basetemp={directory / 'pytest-tmp'}",
                *TESTS,
            ]
        )
        row = iteration(
            number,
            directory,
            child_command,
            child_env,
            min(iteration_timeout, remaining),
            environment,
            port,
        )
        report["iterations"].append(row)
        used.update(launch["port"] for launch in row["launches"])
        report["completed_iterations"] = len(report["iterations"])
        report["failures"] = sum(r["outcome"] != "passed" for r in report["iterations"])
        report["duration_seconds"] = time.monotonic() - started
        report["stop_reason"] = (
            "iteration_cap"
            if number == iterations
            else "total_wall_time_limit"
            if deadline - time.monotonic() <= 0.25
            else "running"
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"iteration={number} outcome={row['outcome']} seconds={row['duration_seconds']:.3f} port={port}",
            flush=True,
        )
    report.update(
        completed_iterations=len(report["iterations"]),
        failures=sum(r["outcome"] != "passed" for r in report["iterations"]),
        duration_seconds=time.monotonic() - started,
        stop_reason="iteration_cap"
        if len(report["iterations"]) == iterations
        else "total_wall_time_limit",
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations", type=int, default=10, help="Default10, hard maximum100"
    )
    parser.add_argument("--iteration-timeout", type=float, default=180)
    parser.add_argument("--total-timeout", type=float, default=900)
    args = parser.parse_args()
    output = Path(tempfile.mkdtemp(prefix="qvelt-gloo-diagnostic-"))
    print(f"Report: {output / 'report.json'}", flush=True)
    report = diagnose(
        output,
        iterations=args.iterations,
        iteration_timeout=args.iteration_timeout,
        total_timeout=args.total_timeout,
    )
    return int(
        report["failures"] > 0 or report["completed_iterations"] < args.iterations
    )


if __name__ == "__main__":
    raise SystemExit(main())
