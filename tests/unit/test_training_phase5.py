"""Phase5 schedule/z-loss/stability/optimizer/sweep/report units (no processes)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from qvelt.config import ConfigError
from qvelt.config.pretraining import TokenSchedule, parse_pretraining_job
from qvelt.config.training import TrainingConfig, TrainingJob, load_training_job
from qvelt.model.decoder import Decoder
from qvelt.model.loss import causal_loss, z_loss_sum
from qvelt.train.accumulation import batch_loss, batch_objective, token_learning_rate
from qvelt.train.optim import make_optimizer
from qvelt.train.parallel import parallelize
from qvelt.train.report import red_flags, summarize_run
from qvelt.train.stability import (
    ActivationRecorder,
    SpikeMonitor,
    layer_gradient_norms,
)
from qvelt.utils.distributed import DistributedContext

ROOT = Path(__file__).resolve().parents[2]
CPU = DistributedContext(0, 1, 0, torch.device("cpu"))


def _job() -> TrainingJob:
    return load_training_job(ROOT / "configs/train/smoke.yaml")


def test_wsd_and_cosine_shapes_hit_rfc_formulas() -> None:
    wsd = TokenSchedule(100, 700, 300)
    assert token_learning_rate(1.0, wsd, 0, 50) == pytest.approx(0.5)
    assert token_learning_rate(1.0, wsd, 600, 100) == 1.0
    assert token_learning_rate(1.0, wsd, 800, 50) == pytest.approx(0.5)
    assert token_learning_rate(1.0, wsd, 999, 1) == 0.0
    floored = replace(wsd, min_lr_ratio=0.1)
    assert token_learning_rate(1.0, floored, 999, 1) == pytest.approx(0.1)
    assert token_learning_rate(1.0, floored, 800, 50) == pytest.approx(0.55)
    cosine = TokenSchedule(100, 700, 300, shape="cosine", min_lr_ratio=0.1)
    for position in (100, 325, 550, 775, 1000):
        progress = (position - 100) / 900
        rfc = 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
        assert token_learning_rate(1.0, cosine, position - 1, 1) == pytest.approx(rfc)
    zero = replace(cosine, min_lr_ratio=0.0)
    assert token_learning_rate(1.0, zero, 999, 1) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ConfigError):
        TokenSchedule(1, 2, 3, shape="linear")
    with pytest.raises(ConfigError):
        TokenSchedule(1, 2, 3, min_lr_ratio=1.0)


def test_recipe_schema_defaults_and_rejections(prepared_demo: Path) -> None:
    from qvelt.config.pretraining import dump_pretraining_job
    from qvelt.train.pretraining_smoke import demo_recipe

    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=2)
    raw = yaml.safe_load(dump_pretraining_job(recipe))
    for key in ("shape", "min_lr_ratio"):
        raw["schedule"].pop(key)
    legacy = parse_pretraining_job(yaml.safe_dump(raw))
    assert legacy.schedule.shape == "wsd" and legacy.nonfinite_policy == "abort"
    for field, value in (
        ("nonfinite_policy", "ignore"),
        ("z_loss", -1.0),
        ("sample_every", 3),  # prompts missing
        ("keep_checkpoints", -1),
    ):
        with pytest.raises(ConfigError):
            parse_pretraining_job(yaml.safe_dump({**raw, field: value}))
    fsdp = {**raw, "job": {**raw["job"], "training": {**raw["job"]["training"]}}}
    fsdp["job"]["training"]["strategy"] = "fsdp"
    assert parse_pretraining_job(yaml.safe_dump(fsdp)).job.training.strategy == "fsdp"


@pytest.mark.parametrize("name", ["tensor_parallel", "pipeline_parallel"])
def test_tensor_and_pipeline_parallel_are_reserved_not_faked(name: str) -> None:
    with pytest.raises(ConfigError, match="not implemented"):
        TrainingConfig(**{name: 2})  # type: ignore[arg-type]
    job = _job()
    model = Decoder(job.experiment.model)
    forced = replace(job, training=replace(job.training))
    object.__setattr__(forced.training, name, 2)
    with pytest.raises(NotImplementedError, match="reserved"):
        parallelize(model, forced, CPU)


def test_z_loss_matches_manual_logsumexp_and_is_off_by_default() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 40)
    targets = torch.randint(2, 40, (2, 5))
    targets[0, 0] = 0  # PAD never supervised
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 4] = False
    valid = mask & (targets > 1)
    expected = (torch.logsumexp(logits, -1).square() * valid).sum()
    torch.testing.assert_close(z_loss_sum(logits, targets, mask), expected)
    job = _job()
    model = Decoder(job.experiment.model)
    batch = {
        "input_ids": torch.randint(2, 288, (1, 8)),
        "labels": torch.randint(2, 288, (1, 8)),
        "positions": torch.arange(8)[None],
        "document_ids": torch.zeros(1, 8, dtype=torch.int64),
        "valid_mask": torch.ones(1, 8, dtype=torch.bool),
        "loss_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    plain = batch_loss(model, batch, "fp32", torch.device("cpu"))
    loss, count, z = batch_objective(model, batch, "fp32", torch.device("cpu"), 0.0)
    assert z is None and torch.equal(loss, plain[0]) and torch.equal(count, plain[1])
    loss, count, z = batch_objective(model, batch, "fp32", torch.device("cpu"), 1e-4)
    logits = model(batch["input_ids"], batch["positions"])
    reference = causal_loss(logits, batch["labels"], batch["loss_mask"])[0]
    torch.testing.assert_close(loss, reference)
    assert z is not None and float(z.detach()) > 0


def test_spike_monitor_detects_ignores_and_roundtrips() -> None:
    monitor = SpikeMonitor(threshold=6.0, warmup=5)
    for value in [3.0, 3.01, 2.99, 3.0, 3.02, 2.98, 3.0]:
        assert monitor.update(value)[1] is False
    mean = monitor.mean
    score, spike = monitor.update(6.0)
    assert spike and score is not None and score > 6
    assert monitor.mean == mean and monitor.spikes == 1  # spike not absorbed
    restored = SpikeMonitor(threshold=6.0, warmup=5)
    restored.load_state_dict(json.loads(json.dumps(monitor.state_dict())))
    assert restored == monitor
    with pytest.raises(ValueError):
        SpikeMonitor(threshold=3.0, warmup=5).load_state_dict(monitor.state_dict())
    with pytest.raises(ValueError):
        monitor.update(float("nan"))
    disabled = SpikeMonitor(threshold=0.0, warmup=0)
    for value in [1.0, 1.0, 100.0]:
        assert disabled.update(value)[1] is False


def test_layer_gradient_norms_partition_the_global_norm_and_hooks_record() -> None:
    torch.manual_seed(1)
    job = _job()
    model = Decoder(job.experiment.model)
    recorder = ActivationRecorder(model)
    tokens = torch.randint(2, 288, (2, 16))
    with recorder.recording():
        loss = model(tokens).float().logsumexp(-1).mean()
    torch.autograd.backward(loss)
    assert not model.blocks[0]._forward_hooks  # handles removed
    norms = layer_gradient_norms(model, CPU, sharded=False)
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9, foreach=False)
    squared = sum(value**2 for value in norms.values())
    assert squared == pytest.approx(float(total) ** 2, rel=1e-5)
    stats = recorder.reduce(CPU)
    assert set(stats) == {
        f"layer_{i:02d}_act_{kind}" for i in range(2) for kind in ("rms", "absmax")
    }
    assert all(v > 0 and math.isfinite(v) for v in stats.values())
    assert stats["layer_00_act_absmax"] >= stats["layer_00_act_rms"]


def test_fused_adamw_matches_reference_update() -> None:
    job = _job()
    torch.manual_seed(2)
    models = [Decoder(job.experiment.model) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    fused = make_optimizer(models[0], replace(job.training, fused_optimizer=True))
    reference = make_optimizer(
        models[1],
        replace(job.training, fused_optimizer=False),
    )
    assert fused.defaults["fused"] is True and reference.defaults["foreach"] is False
    tokens = torch.randint(2, 288, (2, 16))
    for _ in range(3):
        for model, optimizer in ((models[0], fused), (models[1], reference)):
            optimizer.zero_grad()
            torch.autograd.backward(model(tokens).float().logsumexp(-1).mean())
            optimizer.step()
    for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_report_merges_attempts_and_flags(tmp_path: Path) -> None:
    import os

    first = tmp_path / "metrics-a.jsonl"
    second = tmp_path / "metrics-b.jsonl"
    rows = [{"step": s, "loss": 5.0 - 0.1 * s, "clipped": 0.0} for s in range(1, 6)]
    first.write_text("".join(json.dumps(r) + "\n" for r in rows))
    replay = [
        {"step": 4, "loss": 1.0, "loss_spike": 1.0, "nonfinite_window": 0.0},
        {"step": 6, "loss": 4.0, "validation_probe_loss": 4.5},
    ]
    second.write_text("".join(json.dumps(r) + "\n" for r in replay))
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    summary = summarize_run(tmp_path)
    assert summary["attempts"] == 2 and summary["last_step"] == 6
    assert summary["loss_spike_steps"] == [4]
    assert summary["min_loss"] == 1.0  # replayed step 4 replaced the first value
    assert any("spike" in flag for flag in summary["red_flags"])
    assert red_flags({**summary, "loss_spike_steps": []}, []) == []


def test_z_loss_gradient_matches_independent_numpy_reference() -> None:
    import numpy as np

    logits = torch.tensor([[[0.2, -0.3, 0.7], [0.1, 0.5, -0.2]]], requires_grad=True)
    targets = torch.tensor([[2, 0]])
    actual = z_loss_sum(logits, targets)
    torch.autograd.backward(actual)
    values = logits.detach().numpy().astype(np.float64)
    exponent = np.exp(values)
    partition = exponent.sum(-1)
    reference = np.log(partition[0, 0]) ** 2
    gradient = 2 * np.log(partition)[..., None] * exponent / partition[..., None]
    gradient[0, 1] = 0
    assert float(actual.detach()) == pytest.approx(reference, rel=1e-6)
    assert logits.grad is not None
    np.testing.assert_allclose(logits.grad.numpy(), gradient, rtol=1e-6, atol=1e-6)


def test_exact_token_budget_matches_selected_global_target_gradient() -> None:
    from qvelt.train.accumulation import accumulate_gradients

    job = load_training_job(ROOT / "configs/train/smoke.yaml")
    torch.manual_seed(11)
    model = Decoder(job.experiment.model)
    reference = Decoder(job.experiment.model)
    reference.load_state_dict(model.state_dict())
    batch = {
        "input_ids": torch.tensor([[32, 33, 34, 35]]),
        "labels": torch.tensor([[33, 34, 35, 36]]),
        "positions": torch.arange(4)[None],
        "document_ids": torch.zeros(1, 4, dtype=torch.int64),
        "valid_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    stats = accumulate_gradients(model, lambda: batch, 2, CPU, target_budget=5)
    # Four targets of microbatch1 plus only the first target of microbatch2.
    logits = reference(batch["input_ids"])
    a, _ = causal_loss(logits, batch["labels"])
    mask = torch.tensor([[True, False, False, False]])
    b, _ = causal_loss(reference(batch["input_ids"]), batch["labels"], mask)
    torch.autograd.backward((a + b) / 5)
    assert stats.targets == 5
    for left, right in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)


def test_report_ignores_only_truncated_final_record(tmp_path: Path) -> None:
    from qvelt.train.report import merged_metrics

    path = tmp_path / "metrics-interrupted.jsonl"
    path.write_text('{"step":1,"loss":2.0}\n{"step":2,"loss":')
    records, _ = merged_metrics(tmp_path)
    assert set(records) == {1}
    path.write_text('{"step":1,"loss":}\n')
    with pytest.raises(ValueError):
        merged_metrics(tmp_path)


def test_dcp_initial_optimizer_snapshot_preserves_adam_time_and_first_update() -> None:
    from qvelt.utils.checkpoints import _sharded_optimizer_state

    torch.manual_seed(31)
    models = [torch.nn.Linear(3, 2, bias=False) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.1, foreach=False)
        for m in models
    ]
    initial = models[0].weight.detach().clone()
    state = _sharded_optimizer_state(models[0], optimizers[0], 0)
    assert state and torch.equal(models[0].weight, initial)
    assert all(float(s["step"]) == 0 for s in optimizers[0].state.values())
    for model, optimizer in zip(models, optimizers, strict=True):
        torch.autograd.backward(model(torch.ones(1, 3)).square().sum())
        optimizer.step()
    torch.testing.assert_close(models[0].weight, models[1].weight, rtol=0, atol=0)
