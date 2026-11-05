"""Phase5 fault tolerance: NaN skip/rollback, SIGTERM, auto-resume, sweeps."""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import load_file
from torch import Tensor

from qvelt.cli import main
from qvelt.config import ConfigError
from qvelt.config.pretraining import PretrainingJob, dump_pretraining_job
from qvelt.data.loader import PackedDataset
from qvelt.model.decoder import Decoder
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.train.recipes import build_recipe
from qvelt.train.sweep import expand, load_sweep
from qvelt.utils.checkpoints import committed_steps, latest_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def recipe_for(prepared: Path, steps: int = 6, **changes: Any) -> PretrainingJob:
    recipe = demo_recipe(prepared / "prepared/manifest.json", ROOT, steps=steps)
    recipe = replace(recipe, validation_batches=1, validation_every=steps, **changes)
    return replace(
        recipe,
        job=replace(
            recipe.job, training=replace(recipe.job.training, checkpoint_every=2)
        ),
    )


def metrics(directory: Path) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    paths = sorted(
        directory.glob("metrics-*.jsonl"), key=lambda p: p.stat().st_mtime_ns
    )
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            merged[record["step"]] = {**merged.get(record["step"], {}), **record}
    return merged


@contextmanager
def training_forward_hook(action: Any) -> Iterator[None]:
    """Call action(index, logits) on every *training* Decoder forward."""
    original = Decoder.forward
    calls = {"n": 0}

    def wrapped(self: Decoder, *args: Any, **kwargs: Any) -> Tensor:
        logits = original(self, *args, **kwargs)
        if self.training and torch.is_grad_enabled():
            index = calls["n"]
            calls["n"] += 1
            return cast(Tensor, action(index, logits))
        return logits

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Decoder, "forward", wrapped)
        yield


def nan_at(*indices: int) -> Any:
    return lambda i, logits: logits * float("nan") if i in indices else logits


def final_weights(directory: Path) -> dict[str, Tensor]:
    return load_file(str(latest_checkpoint(directory) / "model.safetensors"))


def test_nonfinite_skip_keeps_cursor_and_logs(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = recipe_for(prepared_demo, nonfinite_policy="skip", max_nonfinite_windows=1)
    with training_forward_hook(nan_at(4)):  # step3, first microbatch
        result = train_pretraining(recipe, tmp_path / "skip", ROOT)
    logged = metrics(tmp_path / "skip")
    assert result["optimizer_step"] == 6 and result["nonfinite_windows"] == 1
    assert result["consumed_microbatches"] == 12  # skipped data still consumed
    assert logged[3]["nonfinite_window"] == 1.0 and logged[3]["loss"] is None
    assert logged[3]["gradient_norm"] is None and logged[4]["loss"] is not None
    with training_forward_hook(nan_at(2, 4)):
        with pytest.raises(FloatingPointError, match="budget"):
            train_pretraining(recipe, tmp_path / "budget", ROOT)


def test_nonfinite_rollback_restores_last_checkpoint(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = recipe_for(prepared_demo, nonfinite_policy="rollback")
    with training_forward_hook(nan_at(5)):  # step3 second microbatch
        result = train_pretraining(recipe, tmp_path / "rollback", ROOT)
    logged = metrics(tmp_path / "rollback")
    assert result["rollbacks"] == 1 and logged[3]["rollback_to_step"] == 2.0
    # Restored weights equal the step-2 checkpoint: replaying step4 from them
    # with the same data must match a clean run that skipped window3 itself.
    skipped = recipe_for(prepared_demo, nonfinite_policy="skip")
    with training_forward_hook(nan_at(5)):
        train_pretraining(skipped, tmp_path / "skip", ROOT)
    assert logged[4]["loss"] == metrics(tmp_path / "skip")[4]["loss"]
    with training_forward_hook(nan_at(0)):
        early = train_pretraining(recipe, tmp_path / "early", ROOT)
        assert early["rollbacks"] == 1
        assert metrics(tmp_path / "early")[1]["rollback_to_step"] == 0


def test_sigterm_checkpoints_and_auto_resume_is_exact(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = recipe_for(prepared_demo)
    path = tmp_path / "job.yaml"
    path.write_text(dump_pretraining_job(recipe))
    reference = train_pretraining(recipe, tmp_path / "reference", ROOT)

    def terminate(index: int, logits: Tensor) -> Tensor:
        if index == 4:  # during step3: graceful stop after committing step3
            os.kill(os.getpid(), signal.SIGTERM)
        return logits

    command = [
        "pretrain",
        str(path),
        "--directory",
        str(tmp_path / "run"),
        "--repository",
        str(ROOT),
        "--resume",
        "auto",
    ]
    with training_forward_hook(terminate):
        assert main(command) == 3
    assert [s for s, _ in committed_steps(tmp_path / "run")] == [3, 2]
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert main(command) == 0
    resumed = metrics(tmp_path / "run")
    clean = metrics(tmp_path / "reference")
    assert [resumed[s]["loss"] for s in range(1, 7)] == [
        clean[s]["loss"] for s in range(1, 7)
    ]
    torch.testing.assert_close(
        final_weights(tmp_path / "run"),
        final_weights(tmp_path / "reference"),
        rtol=0,
        atol=0,
    )
    assert reference["optimizer_step"] == 6
    assert main(command) == 0  # completed run: auto-resume is a no-op


def test_auto_resume_quarantines_corrupt_and_incomplete(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = recipe_for(prepared_demo)
    train_pretraining(recipe, tmp_path / "reference", ROOT)
    run = tmp_path / "run"
    train_pretraining(recipe, run, ROOT, stop_after=4)
    corrupt = run / "checkpoints/step-00000004/model.safetensors"
    data = bytearray(corrupt.read_bytes())
    data[-1] ^= 0xFF
    corrupt.write_bytes(bytes(data))
    (run / "checkpoints/step-00000005").mkdir()  # killed mid-save: no manifest
    with pytest.raises(RuntimeError):
        train_pretraining(recipe, run, ROOT, resume="latest")  # strict: no fallback
    result = train_pretraining(recipe, run, ROOT, resume="auto")
    assert result["resumed_from"] == 2 and result["optimizer_step"] == 6
    quarantined = sorted(p.name for p in (run / "checkpoints/quarantine").iterdir())
    assert [name[:13] for name in quarantined] == ["step-00000004", "step-00000005"]
    torch.testing.assert_close(
        final_weights(run), final_weights(tmp_path / "reference"), rtol=0, atol=0
    )


def test_checkpoint_retention_keeps_newest(prepared_demo: Path, tmp_path: Path) -> None:
    recipe = recipe_for(prepared_demo, keep_checkpoints=2)
    train_pretraining(recipe, tmp_path / "run", ROOT)
    assert [s for s, _ in committed_steps(tmp_path / "run")] == [6, 4]


def test_sweep_expansion_validates_points_and_preserves_tokens(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = recipe_for(prepared_demo, steps=8)
    base = tmp_path / "base.yaml"
    base.write_text(dump_pretraining_job(recipe))
    config = tmp_path / "sweep.yaml"
    config.write_text(
        "name: demo\nbase: base.yaml\noutput: runs\nstop_after: 4\n"
        "grid:\n  job.training.learning_rate: [0.001, 0.002]\n"
        "  accumulation_steps: [2, 4]\n"
    )
    runs = expand(load_sweep(config))
    assert len(runs) == 4
    doubled = [r for r in runs if "accumulation_steps=4" in r.overrides]
    assert all("job.training.steps=4" in r.overrides for r in doubled)
    assert all(r.stop_after == 2 for r in doubled)
    assert (
        runs[0].command[:2] == ["qvelt", "pretrain"] and "--resume" in runs[0].command
    )
    config.write_text(
        "name: demo\nbase: base.yaml\noutput: runs\n"
        "grid:\n  job.training.learning_rate: [-1.0]\n"
    )
    with pytest.raises(ConfigError):
        expand(load_sweep(config))


def test_small_preset_builds_only_against_matching_data(
    prepared_demo: Path, tmp_path: Path
) -> None:
    preset = ROOT / "configs/pretrain/small.yaml"
    manifest = prepared_demo / "prepared/manifest.json"
    with pytest.raises(ConfigError, match="disagree"):
        build_recipe(preset, manifest, world_size=8)
    with pytest.raises(ConfigError, match="divisible"):
        build_recipe(preset, manifest, world_size=3)
    demo = build_recipe(
        preset,
        manifest,
        world_size=1,
        overrides=[
            f"job.experiment.model.vocab={PackedDataset(manifest).vocab}",
            "job.experiment.model.context=64",
            "job.training.sequence_length=64",
        ],
    )
    assert demo.accumulation_steps == 8 and demo.job.training.strategy == "single"
    assert demo.schedule.decay_start_tokens == 9_000_000_000
