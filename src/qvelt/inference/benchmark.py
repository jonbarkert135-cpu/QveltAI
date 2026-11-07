"""Small actual loopback HTTP workload;measure the isolated server process,not client RSS."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import math
import os
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from qvelt.eval.config import ArtifactRef
from qvelt.inference.serving_config import ServingConfig


def _post(port: int, key: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    started = time.perf_counter()
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        response = connection.getresponse()
        data = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(data) > 1024 * 1024:
            raise RuntimeError("Loopback benchmark request failed")
        return json.loads(data), time.perf_counter() - started
    finally:
        connection.close()


def benchmark_server(
    reference: ArtifactRef, directory: Path, *, requests: int = 8, concurrency: int = 4
) -> dict[str, Any]:
    if (
        type(requests) is not int
        or not 4 <= requests <= 32
        or type(concurrency) is not int
        or not 1 <= concurrency <= 8
    ):
        raise ValueError("Bounded HTTP benchmark needs4..32 requests/concurrency1..8")
    directory.mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = ServingConfig(
        reference, port=port, batch_wait_ms=10, request_timeout=5.0
    )
    config = directory / "serving.yaml"
    config.write_text(yaml.safe_dump(asdict(settings)))
    key = secrets.token_urlsafe(24)
    child_env = {
        **os.environ,
        settings.api_key_env: key,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "qvelt", "serve", str(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env,
    )
    measurements: dict[str, Any] = {}
    try:
        deadline = time.monotonic() + 30
        while True:
            if process.poll() is not None:
                raise RuntimeError("Benchmark server failed startup")
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
            try:
                connection.request("GET", "/readyz")
                ready = connection.getresponse()
                if ready.status == 200:
                    ready.read()
                    break
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
            if time.monotonic() > deadline:
                raise TimeoutError("Benchmark server startup deadline")
            time.sleep(0.05)
        body = {
            "model": "qvelt",
            "messages": [{"role": "user", "content": "What is two plus two?"}],
            "max_tokens": 16,
            "temperature": 0.8,
            "top_k": 32,
            "top_p": 0.9,
            "seed": 20261107,
        }
        _post(port, key, body)
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_post, port, key, {**body, "seed": 20261107 + i})
                for i in range(requests)
            ]
            rows = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        latencies = sorted(row[1] for row in rows)
        tokens = sum(row[0]["usage"]["completion_tokens"] for row in rows)
        expected, _ = _post(port, key, body)
        stream_start = time.perf_counter()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        chunks = []
        first_token = None
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(
                    {**body, "stream": True, "stream_options": {"include_usage": True}}
                ).encode(),
                {"Content-Type": "application/json", "Authorization": "Bearer " + key},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError("Full SSE probe failed")
            while True:
                line = response.readline(65536)
                if not line:
                    raise RuntimeError("SSE ended without DONE")
                if line.strip() == b"data: [DONE]":
                    break
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event["choices"]:
                    content = event["choices"][0]["delta"].get("content", "")
                    if content and first_token is None:
                        first_token = time.perf_counter() - stream_start
                    chunks.append(content)
            if "".join(chunks) != expected["choices"][0]["message"]["content"]:
                raise RuntimeError("HTTP streaming/nonstreaming text mismatch")
        finally:
            connection.close()
        # Read an SSE role chunk,then disconnect before the queued decode starts.
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps({**body, "stream": True, "max_tokens": 128}).encode(),
                {"Content-Type": "application/json", "Authorization": "Bearer " + key},
            )
            response = connection.getresponse()
            if response.status != 200 or not response.readline(8192).startswith(
                b"data: "
            ):
                raise RuntimeError("Loopback streaming contract failed")
            response.close()
        finally:
            connection.close()
        time.sleep(0.1)
        status = Path(f"/proc/{process.pid}/status").read_text()
        highwater = next(
            line for line in status.splitlines() if line.startswith("VmHWM:")
        )
        hardware = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        measurements = {
            "protocol": "qvelt-loopback-http-v1",
            "device": "cpu",
            "dtype": "float32",
            "cpu_model": hardware,
            "threads": 1,
            "snapshot_sha256": reference.sha256,
            "serving_policy": asdict(settings),
            "requests": requests,
            "concurrency": concurrency,
            "warmup_requests": 1,
            "body": body,
            "completed_tokens": tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens / elapsed,
            "latency_p50_seconds": statistics.median(latencies),
            "latency_p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "server_peak_rss_bytes": int(highwater.split()[1]) * 1024,
            "rss_scope": "isolated server process lifetime VmHWM,includes startup",
            "stream_disconnected_after_role": True,
            "stream_text_matches_nonstream": True,
            "stream_first_content_seconds": first_token,
            "scope": "bounded CPU loopback workload,not high-throughput/GPU/production capacity",
        }
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError("Server did not stop cooperatively") from None
        if key in stdout or key in stderr:
            raise RuntimeError("Serving logs exposed authorization")
        (directory / "server.stdout.log").write_text(stdout)
        (directory / "server.stderr.log").write_text(stderr)
    if process.returncode not in (0, -signal.SIGTERM):
        raise RuntimeError("Benchmark server failed shutdown")
    summaries = [
        json.loads(line) for line in stdout.splitlines() if line.startswith("{")
    ]
    if not summaries or summaries[-1].get("pending_after_shutdown") != 0:
        raise RuntimeError("Server did not release pending requests")
    summary = summaries[-1]
    if summary.get("cancelled_or_expired", 0) < 1:
        raise RuntimeError("Disconnected stream did not cancel generation")
    if concurrency > 1 and max(summary.get("batches", [0])) <= 1:
        raise RuntimeError("HTTP workload did not form a real microbatch")
    measurements["server_shutdown"] = summary
    measurements["server_returncode"] = process.returncode
    (directory / "result.json").write_text(
        json.dumps(measurements, sort_keys=True, allow_nan=False) + "\n"
    )
    return measurements
