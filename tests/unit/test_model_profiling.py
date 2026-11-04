"""Exact storage-count oracle and graph-memory/timing sanity without speed thresholds."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qvelt.config.execution import ModelExecution
from qvelt.config.training import load_training_job
from qvelt.model.profiling import profile_cpu, saved_tensor_storage

ROOT = Path(__file__).resolve().parents[2]


def test_unique_storage_count_deduplicates_views_and_excludes_parameters() -> None:
    x = torch.ones(3, 4, requires_grad=True)
    result, size = saved_tensor_storage(lambda: (x * x.view_as(x)).sum())
    assert size == 3 * 4 * 4
    torch.autograd.backward(result)
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))
    _, excluded = saved_tensor_storage(lambda: (x * x).sum(), [x])
    assert excluded == 0


def test_checkpoint_reduces_saved_graph_storage_and_profiles_restore_rng() -> None:
    job = load_training_job(
        ROOT / "configs/train/smoke.yaml",
        ["experiment.model.layers=4", "training.sequence_length=16"],
    )
    before = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    plain = profile_cpu(job, warmup=1, repeats=1)
    checkpointed = profile_cpu(
        replace(
            job,
            training=replace(
                job.training, execution=ModelExecution(activation_checkpointing=True)
            ),
        ),
        warmup=1,
        repeats=1,
    )
    assert (
        checkpointed["unique_saved_nonparameter_storage_bytes"]
        < plain["unique_saved_nonparameter_storage_bytes"]
    )
    assert plain["cache_storage_bytes"] == checkpointed["cache_storage_bytes"] > 0
    assert plain["cached_logits_close"] and checkpointed["cached_logits_close"]
    assert plain["train_targets_per_second"] > 0
    assert threads == torch.get_num_threads() and torch.equal(
        before, torch.get_rng_state()
    )
    with pytest.raises(ValueError):
        profile_cpu(job, repeats=0)


def test_operator_profile_script_runs_on_cpu_and_refuses_missing_cuda(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = importlib.util.spec_from_file_location(
        "profile_model", ROOT / "scripts/profile_model.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    model = str(ROOT / "configs/model/tiny.yaml")
    report = script.main(
        ["--model", model, "--device", "cpu", "--length", "8", "--steps", "1"]
    )
    capsys.readouterr()
    assert report["verified_scope"] == "cpu harness"
    assert report["parameters"] == 19_338_560
    assert report["top_operators"]
    assert report["estimated_mfu"] is None
    assert "no optimizer updates" in report["profile_scope"]
    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit, match="peak-tflops"):
            script.main(["--device", "cpu", "--peak-tflops", value])
    with pytest.raises(SystemExit, match="context"):
        script.main(["--device", "cpu", "--length", "513"])
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="CUDA"):
            script.main(["--model", model])
