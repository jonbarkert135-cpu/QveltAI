"""Runtime config, SDPA reference gradients and non-reentrant recomputation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from qvelt.config import ModelConfig
from qvelt.config.execution import ModelExecution
from qvelt.config.training import (
    dump_training_job,
    load_training_job,
    parse_training_job,
    training_job_digest,
)
from qvelt.model.decoder import Decoder
from qvelt.model.layers import Attention
from qvelt.model.loss import causal_loss

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ModelConfig(
    layers=2, hidden=32, heads=4, kv_heads=2, ffn=64, vocab=288, context=16
)


@pytest.fixture(autouse=True)
def one_thread() -> None:
    torch.set_num_threads(1)


def test_execution_settings_roundtrip_and_identity() -> None:
    job = load_training_job(
        ROOT / "configs/train/smoke.yaml",
        [
            "training.execution.attention_backend=sdpa",
            "training.execution.activation_checkpointing=true",
        ],
    )
    assert job.training.execution == ModelExecution("sdpa", True)
    assert parse_training_job(dump_training_job(job)) == job
    assert training_job_digest(job) != training_job_digest(
        replace(job, training=replace(job.training, execution=ModelExecution()))
    )
    for backend in ("flash", "unknown", ""):
        with pytest.raises(ValueError):
            ModelExecution(backend)
    with pytest.raises(ValueError):
        parse_training_job(
            dump_training_job(job).replace(
                "activation_checkpointing: true", "activation_checkpointing: 1"
            )
        )
    with pytest.raises(ValueError):
        parse_training_job(
            dump_training_job(job).replace(
                "attention_backend: sdpa", "attention_backend: sdpa\n    unknown: 1"
            )
        )


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("padding", [False, True])
def test_sdpa_attention_values_and_all_gradients(kv_heads: int, padding: bool) -> None:
    config = replace(CONFIG, kv_heads=kv_heads)
    reference, native = Attention(config), Attention(config, "sdpa")
    native.load_state_dict(reference.state_dict())
    x = torch.randn(2, 6, config.hidden, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]]).expand(2, -1)
    documents = torch.tensor([[7, 7, 7, 8, 8, 8], [3, 3, 3, 4, 4, 4]])
    valid = torch.ones(2, 6, dtype=torch.bool)
    if padding:
        valid[0, -2:] = False
        valid[1] = False  # Entirely padded sequence must stay finite with zero output.
    expected = reference(x, positions, documents, valid)
    actual = native(other, positions, documents, valid)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(actual)
    torch.autograd.backward((expected * upstream).sum())
    torch.autograd.backward((actual * upstream).sum())
    torch.testing.assert_close(other.grad, x.grad, rtol=4e-5, atol=4e-6)
    for (_, a), (_, b) in zip(
        reference.named_parameters(), native.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, b.grad, rtol=4e-5, atol=4e-6)
    assert bool(torch.isfinite(actual).all())
    assert torch.equal(actual[~valid], torch.zeros_like(actual[~valid]))


def test_reference_scores_stay_float32_under_cpu_autocast() -> None:
    layer = Attention(CONFIG)
    x = torch.randn(1, 5, CONFIG.hidden)
    ids = torch.zeros(1, 5, dtype=torch.int64)
    positions = torch.arange(5)[None]
    valid = torch.ones_like(ids, dtype=torch.bool)
    with patch(
        "qvelt.model.layers.F.softmax", wraps=torch.nn.functional.softmax
    ) as softmax:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = layer(x, positions, ids, valid)
    assert softmax.call_args.args[0].dtype == torch.float32
    assert output.dtype == torch.bfloat16 and bool(torch.isfinite(output).all())


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
@pytest.mark.parametrize("autocast", [False, True])
def test_activation_checkpoint_gradient_and_rng_parity(
    backend: str, autocast: bool
) -> None:
    reference = Decoder(CONFIG, ModelExecution(backend))
    recomputed = Decoder(CONFIG, ModelExecution(backend, True))
    recomputed.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[32, 33, 34, 35, 36], [38, 38, 39, 40, 41]])
    positions = torch.tensor([[0, 1, 0, 1, 2]]).expand(2, -1)
    documents = torch.tensor([[1, 1, 2, 2, 2]]).expand(2, -1)
    valid = torch.tensor([[True, True, True, True, False]]).expand(2, -1)
    target = torch.tensor([[33, 2, 35, 2, 0], [38, 2, 40, 2, 0]])
    saved = torch.get_rng_state().clone()
    for model in (reference, recomputed):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            logits = model(tokens, positions, documents, valid)
            total, count = causal_loss(logits, target, valid)
        torch.autograd.backward(total / count)
    assert torch.equal(saved, torch.get_rng_state())
    for (_, a), (_, b) in zip(
        reference.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    # Parameter names/storage semantics remain compatible with ordinary checkpoints.
    assert reference.state_dict().keys() == recomputed.state_dict().keys()


def test_checkpoint_recomputes_but_evaluation_and_no_grad_skip_it() -> None:
    model = Decoder(CONFIG, ModelExecution(activation_checkpointing=True))
    tokens = torch.tensor([[32, 33, 34]])
    with patch.object(
        model.blocks[0], "forward", wraps=model.blocks[0].forward
    ) as block:
        logits = model(tokens)
        torch.autograd.backward(logits.square().sum())
        assert block.call_count == 2
    model.eval()
    with patch(
        "qvelt.model.decoder.checkpoint",
        side_effect=AssertionError("must not checkpoint"),
    ):
        model(tokens)
        model.train()
        with torch.no_grad():
            model(tokens)


def test_sdpa_bfloat16_outputs_and_gradients_against_reference() -> None:
    reference = Decoder(CONFIG)
    native = Decoder(CONFIG, ModelExecution("sdpa"))
    native.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[32, 34, 38, 40]])
    outputs: list[Tensor] = []
    for model in (reference, native):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(tokens)
            outputs.append(output)
            loss, count = causal_loss(output, torch.tensor([[34, 38, 40, 2]]))
        torch.autograd.backward(loss / count)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.04, atol=0.004)
    for a, b in zip(reference.parameters(), native.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0.08, atol=0.004)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
def test_cuda_sdpa_checkpoint_and_cached_parity(precision: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    if precision == torch.bfloat16 and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        pytest.skip("Native BF16 hardware required")
    reference = Decoder(CONFIG).cuda()
    native = Decoder(CONFIG, ModelExecution("sdpa", True)).cuda()
    native.load_state_dict(reference.state_dict())
    ids = torch.tensor([[32, 33, 34, 35]], device="cuda")
    outputs = []
    for model in (reference, native):
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=precision == torch.bfloat16
        ):
            output = model(ids)
            outputs.append(output)
            loss, count = causal_loss(
                output, torch.tensor([[33, 34, 35, 2]], device="cuda")
            )
        torch.autograd.backward(loss / count)
    tolerance = 0.05 if precision == torch.bfloat16 else 5e-4
    torch.testing.assert_close(outputs[0], outputs[1], rtol=tolerance, atol=tolerance)
    for a, b in zip(reference.parameters(), native.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=tolerance, atol=tolerance)
    native.eval()
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=precision == torch.bfloat16
    ):
        full = native(ids)
        first, cache = native.forward_cached(ids[:, :2])
        second, _ = native.forward_cached(ids[:, 2:], cache)
    torch.testing.assert_close(
        torch.cat((first, second), dim=1), full, rtol=tolerance, atol=tolerance
    )
