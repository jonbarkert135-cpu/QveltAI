"""Exercise the built nonroot images:actual tiny training and authenticated HTTP/SSE."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from qvelt.config.pretraining import dump_pretraining_job
from qvelt.eval.config import ArtifactRef
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.inference.artifacts import export_snapshot
from qvelt.inference.serving_config import ServingConfig
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.manifests import manifest_digest


def docker(*arguments: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["docker", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    ).stdout


def run(
    directory: Path, repository: Path, training_image: str, serving_image: str
) -> dict[str, Any]:
    directory.parent.mkdir(parents=True, exist_ok=True)
    root, repository = create_directory(directory).resolve(), repository.resolve()
    fixture = create_evaluation_fixture(root / "fixture", repository)
    manifest = export_snapshot(
        fixture.checkpoint, fixture.tokenizer, root / "snapshot", repository
    )
    expected_source = next(
        f.sha256 for f in manifest.files if f.role == "dependency_lock"
    )
    # Default image user must be nonroot before overriding UID for owned bind mounts.
    users = {
        image: int(
            docker(
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "/app/.venv/bin/python",
                image,
                "-c",
                "import os;print(os.geteuid())",
            ).strip()
        )
        for image in (training_image, serving_image)
    }
    assert all(value == 10001 for value in users.values())
    doctor = docker(
        "run", "--rm", "--network", "none", training_image, "doctor", "--device", "cpu"
    )
    (root / "doctor.json").write_text(doctor)
    recipe = demo_recipe(
        root / "fixture/model-data/prepared/manifest.json", repository, steps=3
    )
    recipe = replace(
        recipe, data_manifest="/work/fixture/model-data/prepared/manifest.json"
    )
    (root / "training.yaml").write_text(dump_pretraining_job(recipe))
    uid = f"{os.getuid()}:{os.getgid()}"
    train_output = docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        uid,
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        "--mount",
        f"type=bind,src={root},dst=/work",
        "--mount",
        f"type=bind,src={repository},dst=/repo,readonly",
        training_image,
        "pretrain",
        "/work/training.yaml",
        "--directory",
        "/work/training",
        "--repository",
        "/repo",
    )
    (root / "training.stdout.log").write_text(train_output)
    checkpoint = ArtifactStore(
        root / "training/checkpoints/step-00000003"
    ).read_manifest()
    assert checkpoint.kind == "checkpoint"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = ServingConfig(
        ArtifactRef("/work/snapshot/manifest.json", manifest_digest(manifest)),
        host="0.0.0.0",
        port=8000,
    )
    (root / "serving.yaml").write_text(yaml.safe_dump(asdict(settings)))
    key, name = secrets.token_urlsafe(24), "qvelt-phase8-" + uuid.uuid4().hex[:12]
    env = {**os.environ, "QVELT_API_KEY": key}
    try:
        docker(
            "run",
            "--detach",
            "--name",
            name,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--user",
            uid,
            "--env",
            "QVELT_API_KEY",
            "--env",
            "OMP_NUM_THREADS=1",
            "--publish",
            f"127.0.0.1:{port}:8000",
            "--mount",
            f"type=bind,src={root},dst=/work,readonly",
            serving_image,
            "serve",
            "/work/serving.yaml",
            env=env,
        )
        deadline = time.monotonic() + 30
        while True:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            try:
                connection.request("GET", "/readyz")
                response = connection.getresponse()
                if response.status == 200:
                    response.read()
                    break
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
            if time.monotonic() > deadline:
                raise TimeoutError("Container readiness deadline")
            time.sleep(0.1)

        def request(body: dict[str, Any], auth: bool = True) -> tuple[int, bytes]:
            client = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                headers = {"Content-Type": "application/json"}
                if auth:
                    headers["Authorization"] = "Bearer " + key
                client.request(
                    "POST", "/v1/chat/completions", json.dumps(body).encode(), headers
                )
                reply = client.getresponse()
                return reply.status, reply.read(1024 * 1024)
            finally:
                client.close()

        body = {
            "model": "qvelt",
            "messages": [{"role": "user", "content": "2+2?"}],
            "max_tokens": 4,
            "temperature": 0,
        }
        assert request(body, auth=False)[0] == 401
        status, raw = request(body)
        assert status == 200
        completion = json.loads(raw)["choices"][0]["message"]["content"]
        status, raw = request({**body, "stream": True})
        lines = [
            line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")
        ]
        assert status == 200 and lines[-1] == "[DONE]"
        streamed = "".join(
            json.loads(line)["choices"][0]["delta"].get("content", "")
            for line in lines[:-1]
        )
        assert streamed == completion
        docker("stop", "--time", "15", name)
        logs = docker("logs", name)
        assert key not in logs
        (root / "serving.stdout.log").write_text(logs)
        summary = [
            json.loads(line) for line in logs.splitlines() if line.startswith("{")
        ][-1]
        assert summary["pending_after_shutdown"] == 0
        result = {
            "training_image_id": docker(
                "image", "inspect", "--format", "{{.Id}}", training_image
            ).strip(),
            "serving_image_id": docker(
                "image", "inspect", "--format", "{{.Id}}", serving_image
            ).strip(),
            "default_users": users,
            "dependency_lock_sha256": expected_source,
            "training_updates": 3,
            "checkpoint_sha256": manifest_digest(checkpoint),
            "authenticated_json_sse_exact": True,
            "unauthenticated_status": 401,
            "shutdown": summary,
            "scope": "CPU tiny image smoke;no CUDA or production claim",
        }
        (root / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        return result
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, timeout=30
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--training-image", default="qvelt-training:phase8")
    parser.add_argument("--serving-image", default="qvelt-serving:phase8")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.directory, args.repository, args.training_image, args.serving_image
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
