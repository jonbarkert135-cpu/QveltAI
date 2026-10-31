"""Actual torchrun/Gloo workers; CPU success does not establish NCCL/FSDP."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]


def workers(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            *args,
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


def test_two_rank_all_reduce_and_distributed_checkpoint(tmp_path: Path) -> None:
    probe = workers("scripts/all_reduce.py", "--device", "cpu")
    values = [
        json.loads(line) for line in probe.stdout.splitlines() if line.startswith("{")
    ]
    assert len(values) == 1 and all(
        x["sum"] == 3 and x["world_size"] == 2 for x in values
    )
    output = tmp_path / "distributed"
    workers("tests/integration/distributed_worker.py", str(output))
    result = json.loads((output / "distributed-test-result.json").read_text())
    assert (
        result["uneven_mask_gradient_oracle"]
        and result["dcp_weights_optimizer_rng_equal"]
    )


def test_ddp_training_cli_resume_matches_continuous(tmp_path: Path) -> None:
    base = [
        "-m",
        "qvelt",
        "train",
        "configs/train/smoke.yaml",
        "--repository",
        str(ROOT),
        "--set",
        "training.steps=4",
        "--set",
        "training.warmup_steps=1",
        "--set",
        "training.decay_steps=1",
        "--set",
        "training.checkpoint_every=2",
        "--set",
        "training.strategy=ddp",
    ]
    workers(*base, "--directory", str(tmp_path / "continuous"))
    workers(*base, "--directory", str(tmp_path / "replayed"), "--stop-after", "2")
    workers(*base, "--directory", str(tmp_path / "replayed"), "--resume", "latest")
    a = load_file(
        str(tmp_path / "continuous/checkpoints/step-00000004/model.safetensors")
    )
    b = load_file(
        str(tmp_path / "replayed/checkpoints/step-00000004/model.safetensors")
    )
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
