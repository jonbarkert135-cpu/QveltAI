"""Actual process death,recovery,diagnostics and corrupted-state rejection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import torch

from qvelt.config.pretraining import TokenSchedule
from qvelt.train.diagnostics import kill_resume_check, scaling_sanity
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.train.report import archive_run
from qvelt.utils.artifacts import ArtifactIntegrityError
from qvelt.utils.checkpoints import select_valid_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def test_final_window_stops_at_exact_target_horizon(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = replace(
        demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=5),
        schedule=TokenSchedule(0, 7, 3),
    )
    result = train_pretraining(recipe, tmp_path / "exact-horizon", ROOT)
    assert result["processed_tokens"] == 10
    assert result["optimizer_step"] == 1 and result["optimizer_updates"] == 1
    assert result["schedule_complete"]


def test_actual_kill_replays_every_loss(prepared_demo: Path, tmp_path: Path) -> None:
    base = demo_recipe(
        prepared_demo / "prepared/manifest.json", ROOT, steps=8, accumulation=1
    )
    recipe = replace(
        base,
        job=replace(base.job, training=replace(base.job.training, checkpoint_every=2)),
        keep_checkpoints=2,
    )
    result = kill_resume_check(recipe, tmp_path / "actual-kill", ROOT, kill_after=3)
    assert result["killed_returncode"] == -9
    assert len(result["loss_curve"]) == 8 and result["maximum_loss_difference"] == 0
    assert result["optimizer_bit_exact"] and result["rng_bit_exact"]


def test_scaling_fit_matches_independent_centered_formula(
    prepared_demo: Path, tmp_path: Path
) -> None:
    result = scaling_sanity(
        prepared_demo / "prepared/manifest.json", tmp_path / "scaling", ROOT, steps=3
    )
    rows = result["rows"]
    assert len({r["parameters"] for r in rows}) == 4
    assert len({r["tokens"] for r in rows}) == 1
    x = np.log(np.array([r["parameters"] for r in rows]))
    y = np.log(np.array([r["loss"] for r in rows]))
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    assert result["slope"] == pytest.approx(slope, abs=1e-12)


def test_nonfinite_optimizer_state_rolls_back_to_initial(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = demo_recipe(
        prepared_demo / "prepared/manifest.json", ROOT, steps=3, accumulation=1
    )
    recipe = replace(recipe, nonfinite_policy="rollback")
    original = torch.optim.AdamW.step
    counter = {"value": 0}

    def step(self: torch.optim.AdamW, *args: object, **kwargs: object) -> object:
        value = original(self, *args, **kwargs)
        if counter["value"] == 0:
            with torch.no_grad():
                self.param_groups[0]["params"][0].flatten()[0] = float("inf")
        counter["value"] += 1
        return value

    with patch.object(torch.optim.AdamW, "step", step):
        result = train_pretraining(recipe, tmp_path / "rollback", ROOT)
    assert result["rollbacks"] == 1 and result["optimizer_updates"] == 2
    assert list((tmp_path / "rollback").glob("failure-*.json"))


def test_checkpoint_recovery_rejects_symlink(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    (tmp_path / "checkpoints/step-00000001").symlink_to(
        external, target_is_directory=True
    )
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        select_valid_checkpoint(tmp_path)
    assert external.exists()


def test_archive_excludes_unrelated_text_and_existing_output(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = demo_recipe(
        prepared_demo / "prepared/manifest.json", ROOT, steps=1, accumulation=1
    )
    train_pretraining(recipe, tmp_path / "run", ROOT)
    (tmp_path / "run/unrelated.txt").write_text("private operator notes")
    names = archive_run(tmp_path / "run", tmp_path / "logs.tar.gz")
    assert "unrelated.txt" not in names and "summary.json" in names
    assert not any("model.safetensors" in x or "optimizer.pt" in x for x in names)
    with pytest.raises(FileExistsError):
        archive_run(tmp_path / "run", tmp_path / "logs.tar.gz")


def test_two_rank_nan_recovery_completes_backward_hooks(
    prepared_demo: Path, tmp_path: Path
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "tests/integration/phase5_worker.py",
            str(prepared_demo / "prepared/manifest.json"),
            str(tmp_path / "distributed"),
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        json.loads((tmp_path / "distributed/recovery-result.json").read_text())[
            "rollbacks"
        ]
        == 1
    )


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_packed_fsdp2_replay_on_two_actual_gpus(
    prepared_demo: Path, tmp_path: Path, precision: Literal["fp32", "bf16"]
) -> None:
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
    )
    from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

    from qvelt.config.pretraining import dump_pretraining_job
    from qvelt.model.decoder import Decoder
    from qvelt.train.optim import make_optimizer
    from qvelt.train.smoke import _equal
    from qvelt.utils.artifacts import ArtifactStore
    from qvelt.utils.checkpoints import latest_checkpoint

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two actual CUDA GPUs;CPU DCP is not FSDP2 validation")
    if precision == "bf16":
        for index in (0, 1):
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    pytest.skip("Native BF16 required on every selected device")
    base = demo_recipe(
        prepared_demo / "prepared/manifest.json",
        ROOT,
        steps=3,
        accumulation=2,
        world_size=2,
    )
    recipe = replace(
        base,
        validation_batches=1,
        job=replace(
            base.job,
            experiment=replace(
                base.job.experiment,
                runtime=replace(
                    base.job.experiment.runtime, device="cuda", precision=precision
                ),
            ),
            training=replace(base.job.training, strategy="fsdp", fused_optimizer=False),
        ),
    )
    path = tmp_path / "fsdp.yaml"
    path.write_text(dump_pretraining_job(recipe))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "qvelt",
        "pretrain",
        str(path),
        "--repository",
        str(ROOT),
    ]
    for directory, extra in [
        ("full", []),
        ("replayed", ["--stop-after", "1"]),
        ("replayed", ["--resume", "auto"]),
    ]:
        result = subprocess.run(
            [*command, "--directory", str(tmp_path / directory), *extra],
            cwd=ROOT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    states = []
    for name in ("full", "replayed"):
        checkpoint = latest_checkpoint(tmp_path / name)
        ArtifactStore(checkpoint).read_manifest()
        model = Decoder(recipe.job.experiment.model)
        optimizer = make_optimizer(model, recipe.job.training)
        state = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optimizer),
        }
        # DCP reshards the actual multi-rank GPU checkpoint into full CPU tensors.
        with pytest.warns(
            UserWarning, match="uninitialized|single process|single-process"
        ):
            dcp_load(
                {"model": state["model"]},
                checkpoint_id=checkpoint / "model",
                no_dist=True,
            )
        with pytest.warns(
            UserWarning, match="uninitialized|single process|single-process"
        ):
            dcp_load(
                {"optimizer": state["optimizer"]},
                checkpoint_id=checkpoint / "optimizer",
                no_dist=True,
            )
        states.append(state)
    _equal(states[0], states[1])
