"""Actual packed BPE optimization/replay, partial windows, permissions and corruption."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file

from qvelt.cli import main
from qvelt.config.execution import ModelExecution
from qvelt.config.pretraining import dump_pretraining_job
from qvelt.data.loader import LoaderConfig, ResumableLoader
from qvelt.model.decoder import Decoder
from qvelt.train.pretraining import prepare_inputs, train_pretraining, validation_probe
from qvelt.train.pretraining_smoke import demo_recipe, run_pretraining_smoke
from qvelt.train.runner import train_synthetic
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore
from qvelt.utils.checkpoints import latest_checkpoint
from qvelt.utils.distributed import DistributedContext
from qvelt.utils.manifest_types import ArtifactFile, CheckpointInfo
from qvelt.utils.manifests import manifest_bytes

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
def test_partial_accumulation_exact_checkpoint_replay(
    prepared_demo: Path, tmp_path: Path, backend: str
) -> None:
    recipe = demo_recipe(
        prepared_demo / "prepared/manifest.json", ROOT, steps=3, accumulation=4
    )
    recipe = replace(
        recipe,
        total_batches=9,
        validation_batches=1,
        job=replace(
            recipe.job,
            training=replace(
                recipe.job.training, execution=ModelExecution(backend, True)
            ),
        ),
    )
    full = train_pretraining(recipe, tmp_path / "full", ROOT)
    train_pretraining(recipe, tmp_path / "replayed", ROOT, stop_after=1)
    replay = train_pretraining(recipe, tmp_path / "replayed", ROOT, resume=True)
    assert full["optimizer_step"] == replay["optimizer_step"] == 3
    assert full["consumed_microbatches"] == replay["consumed_microbatches"] == 9
    assert full["processed_tokens"] == replay["processed_tokens"]
    for name in ("model.safetensors", "optimizer.pt"):
        a, b = (
            latest_checkpoint(tmp_path / "full") / name,
            latest_checkpoint(tmp_path / "replayed") / name,
        )
        left = (
            load_file(str(a))
            if name.endswith("safetensors")
            else torch.load(a, weights_only=True)
        )
        right = (
            load_file(str(b))
            if name.endswith("safetensors")
            else torch.load(b, weights_only=True)
        )
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    manifest = ArtifactStore(latest_checkpoint(tmp_path / "replayed")).read_manifest()
    assert isinstance(manifest.metadata, CheckpointInfo)
    assert manifest.metadata.data_sha256 == recipe.data_sha256
    assert manifest.metadata.tokenizer_sha256 == replay["tokenizer_sha256"]
    assert manifest.metadata.processed_tokens == replay["processed_tokens"]
    assert (tmp_path / "replayed/data.manifest.json").exists()
    assert not (tmp_path / "replayed/data-spec.json").exists()
    with pytest.raises(ArtifactIntegrityError, match="matching application"):
        train_synthetic(recipe.job, tmp_path / "replayed", ROOT, resume=True)


def test_prefetched_worker_change_preserves_optimizer_updates(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = demo_recipe(
        prepared_demo / "prepared/manifest.json", ROOT, steps=2, accumulation=2
    )
    recipe = replace(recipe, validation_batches=1)
    train_pretraining(recipe, tmp_path / "full", ROOT)
    train_pretraining(
        replace(recipe, workers=2), tmp_path / "replayed", ROOT, stop_after=1
    )
    train_pretraining(recipe, tmp_path / "replayed", ROOT, resume=True)
    torch.testing.assert_close(
        load_file(str(latest_checkpoint(tmp_path / "full") / "model.safetensors")),
        load_file(str(latest_checkpoint(tmp_path / "replayed") / "model.safetensors")),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.load(
            latest_checkpoint(tmp_path / "full") / "optimizer.pt", weights_only=True
        ),
        torch.load(
            latest_checkpoint(tmp_path / "replayed") / "optimizer.pt", weights_only=True
        ),
        rtol=0,
        atol=0,
    )


def test_validation_probe_preserves_training_cursor_rng_and_mode(
    prepared_demo: Path,
) -> None:
    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=3)
    inputs = prepare_inputs(recipe)
    model = Decoder(recipe.job.experiment.model)
    before = torch.get_rng_state().clone()
    with ResumableLoader(
        inputs.training, LoaderConfig(max_batches=4), inputs.stages
    ) as training:
        next(training)
        cursor = training.state_dict()
        result = validation_probe(
            model, recipe, inputs, DistributedContext(0, 1, 0, torch.device("cpu"))
        )
        assert training.state_dict() == cursor
    assert (
        result["validation_probe_targets"] > 0 and result["validation_probe_loss"] > 0
    )
    assert torch.equal(before, torch.get_rng_state()) and model.training


@pytest.mark.parametrize("bad", ["permission", "data", "schedule", "application"])
def test_invalid_input_and_checkpoint_fail_closed(
    prepared_demo: Path, tmp_path: Path, bad: str
) -> None:
    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=2)
    output = tmp_path / "run"
    if bad == "permission":
        with pytest.raises(RuntimeError):
            train_pretraining(replace(recipe, allow_demo=False), output, ROOT)
        assert not output.exists()
        return
    if bad == "data":
        with pytest.raises(RuntimeError):
            train_pretraining(replace(recipe, data_sha256="0" * 64), output, ROOT)
        assert not output.exists()
        return
    train_pretraining(recipe, output, ROOT, stop_after=1)
    if bad == "schedule":
        changed = replace(recipe, accumulation_steps=3)
        with pytest.raises(RuntimeError, match="resume"):
            train_pretraining(changed, output, ROOT, resume=True)
        return
    checkpoint = latest_checkpoint(output)
    manifest = ArtifactStore(checkpoint).read_manifest()
    path = checkpoint / "application-rank-00000.json"
    state = json.loads(path.read_bytes())
    state["loader"]["step"] += 1
    body = json.dumps(state).encode()
    path.write_bytes(body)
    records = tuple(
        ArtifactFile(f.path, f.role, len(body), hashlib.sha256(body).hexdigest())
        if f.path == path.name
        else f
        for f in manifest.files
    )
    (checkpoint / "manifest.json").write_bytes(
        manifest_bytes(replace(manifest, files=records))
    )
    with patch(
        "qvelt.utils.checkpoints.set_model_state_dict",
        side_effect=AssertionError("must validate before model loading"),
    ):
        with pytest.raises(RuntimeError, match="application"):
            train_pretraining(recipe, output, ROOT, resume=True)


def test_pretrain_cli_and_smoke_module(prepared_demo: Path, tmp_path: Path) -> None:
    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=1)
    path = tmp_path / "job.yaml"
    path.write_text(dump_pretraining_job(recipe))
    assert (
        main(
            [
                "pretrain",
                str(path),
                "--directory",
                str(tmp_path / "cli"),
                "--repository",
                str(ROOT),
            ]
        )
        == 0
    )
    report = run_pretraining_smoke(tmp_path / "smoke", ROOT, steps=3)
    assert (
        report["weights_bit_exact"]
        and report["optimizer_bit_exact"]
        and report["consumer_state_exact"]
    )
    assert report["continuous"]["scope"] == "demo"


def test_two_rank_packed_training_replay_and_uneven_window_oracle(
    prepared_demo: Path, tmp_path: Path
) -> None:
    import os
    import subprocess
    import sys

    def workers(arguments: list[str]) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                *arguments,
            ],
            cwd=ROOT,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    audit = tmp_path / "oracle.json"
    workers(["tests/integration/pretraining_worker.py", str(audit)])
    assert json.loads(audit.read_bytes())["all_ranks_failed_together"]
    recipe = demo_recipe(
        prepared_demo / "prepared/manifest.json",
        ROOT,
        steps=2,
        accumulation=2,
        world_size=2,
    )
    recipe = replace(
        recipe,
        validation_batches=1,
        job=replace(
            recipe.job,
            training=replace(
                recipe.job.training,
                strategy="ddp",
                execution=ModelExecution("sdpa", True),
            ),
        ),
    )
    path = tmp_path / "ddp.yaml"
    path.write_text(dump_pretraining_job(recipe))
    common = ["-m", "qvelt", "pretrain", str(path), "--repository", str(ROOT)]
    workers([*common, "--directory", str(tmp_path / "full")])
    workers([*common, "--directory", str(tmp_path / "replayed"), "--stop-after", "1"])
    workers([*common, "--directory", str(tmp_path / "replayed"), "--resume", "latest"])
    a, b = (
        latest_checkpoint(tmp_path / "full"),
        latest_checkpoint(tmp_path / "replayed"),
    )
    torch.testing.assert_close(
        load_file(str(a / "model.safetensors")),
        load_file(str(b / "model.safetensors")),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.load(a / "optimizer.pt", weights_only=True),
        torch.load(b / "optimizer.pt", weights_only=True),
        rtol=0,
        atol=0,
    )
    for rank in (0, 1):
        name = f"application-rank-{rank:05d}.json"
        left, right = (
            json.loads((a / name).read_bytes()),
            json.loads((b / name).read_bytes()),
        )
        assert (
            left["loader"] == right["loader"]
            and left["processed_tokens"] == right["processed_tokens"]
        )


def test_nonfinite_packed_loss_never_updates_or_commits(
    prepared_demo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any

    from torch import Tensor

    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=1)

    def invalid(self: Decoder, tokens: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return torch.full(
            (*tokens.shape, self.config.vocab), float("nan"), requires_grad=True
        )

    monkeypatch.setattr(Decoder, "forward", invalid)
    # Isolate the training-loss failure from the separately tested validation guard.
    with (
        patch(
            "qvelt.train.pretraining.validation_probe",
            return_value={
                "validation_probe_loss": 1.0,
                "validation_probe_targets": 1.0,
            },
        ),
        patch.object(torch.optim.AdamW, "step") as step,
    ):
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            train_pretraining(recipe, tmp_path / "bad", ROOT)
        step.assert_not_called()
    assert not list((tmp_path / "bad/checkpoints").glob("*/manifest.json"))


@pytest.mark.gpu
def test_actual_single_cuda_packed_replay(prepared_demo: Path, tmp_path: Path) -> None:
    import os

    if not torch.cuda.is_available():
        pytest.skip("Actual CUDA hardware required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        pytest.skip(
            "Set CUBLAS_WORKSPACE_CONFIG before starting Python for deterministic CUDA"
        )
    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=2)
    recipe = replace(
        recipe,
        validation_batches=1,
        job=replace(
            recipe.job,
            experiment=replace(
                recipe.job.experiment,
                runtime=replace(recipe.job.experiment.runtime, device="cuda"),
            ),
        ),
    )
    train_pretraining(recipe, tmp_path / "gpu-full", ROOT)
    train_pretraining(recipe, tmp_path / "gpu-replay", ROOT, stop_after=1)
    train_pretraining(recipe, tmp_path / "gpu-replay", ROOT, resume=True)
    a, b = (
        latest_checkpoint(tmp_path / "gpu-full"),
        latest_checkpoint(tmp_path / "gpu-replay"),
    )
    torch.testing.assert_close(
        load_file(str(a / "model.safetensors")),
        load_file(str(b / "model.safetensors")),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.load(a / "optimizer.pt", weights_only=True, map_location="cpu"),
        torch.load(b / "optimizer.pt", weights_only=True, map_location="cpu"),
        rtol=1e-5,
        atol=1e-6,
    )
