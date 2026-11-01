"""Independent numerical/gradient oracles for RFC-001's portable decoder."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from qvelt.config import ModelConfig, load_config
from qvelt.model.decoder import Block, Decoder
from qvelt.model.layers import Attention, RMSNorm, RotaryEmbedding, SwiGLU
from qvelt.model.loss import causal_loss

ROOT = Path(__file__).resolve().parents[2]
SMALL = ModelConfig(
    layers=2, hidden=32, heads=4, kv_heads=2, ffn=64, vocab=288, context=16
)


@pytest.fixture(autouse=True)
def single_thread() -> None:
    torch.set_num_threads(1)


def test_rmsnorm_value_and_gradient() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 3, 8, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    layer = RMSNorm(8, 1e-6)
    layer.weight.data.copy_(torch.linspace(0.2, 1.5, 8))
    gain = layer.weight.detach().clone().requires_grad_()
    expected = ref / (ref.square().sum(-1, keepdim=True) / 8 + 1e-6).sqrt() * gain
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    torch.autograd.backward(actual.square().sum())
    torch.autograd.backward(expected.square().sum())
    torch.testing.assert_close(x.grad, ref.grad)
    torch.testing.assert_close(layer.weight.grad, gain.grad)


def test_rope_independent_rotation_and_gradients() -> None:
    x = torch.randn(2, 4, 3, 8, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    positions = torch.tensor([[0, 1, 9], [2, 3, 4]])
    entries = []
    for batch in range(2):
        heads = []
        for head in range(4):
            times = []
            for time in range(3):
                coordinates = []
                for pair in range(4):
                    angle = float(positions[batch, time]) * 10000 ** (-2 * pair / 8)
                    a, b = ref[batch, head, time, 2 * pair : 2 * pair + 2]
                    coordinates.extend(
                        [
                            a * math.cos(angle) - b * math.sin(angle),
                            a * math.sin(angle) + b * math.cos(angle),
                        ]
                    )
                times.append(torch.stack(coordinates))
            heads.append(torch.stack(times))
        entries.append(torch.stack(heads))
    expected = torch.stack(entries)
    actual = RotaryEmbedding(8, 10000)(x, positions)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    weights = torch.randn_like(actual)
    torch.autograd.backward((actual * weights).sum())
    torch.autograd.backward((expected * weights).sum())
    torch.testing.assert_close(x.grad, ref.grad, rtol=1e-5, atol=1e-6)


def test_attention_against_loop_oracle() -> None:
    torch.manual_seed(9)
    layer = Attention(SMALL)
    x = torch.randn(1, 4, 32, requires_grad=True)
    positions = torch.tensor([[0, 1, 0, 1]])
    documents = torch.tensor([[10, 10, 20, 20]])
    valid = torch.tensor([[True, True, True, False]])
    # Oracle uses loops, scalar head mapping, manual rotations and dot products.
    ref = x.detach().clone().requires_grad_()
    matrices = {
        name: p.detach().clone().requires_grad_()
        for name, p in layer.named_parameters()
    }
    q = F.linear(ref, matrices["query.weight"]).view(1, 4, 4, 8)
    k = F.linear(ref, matrices["key.weight"]).view(1, 4, 2, 8)
    v = F.linear(ref, matrices["value.weight"]).view(1, 4, 2, 8)

    def rotate(z: Tensor, pos: int) -> Tensor:
        output = []
        for j in range(4):
            angle = pos * 10000 ** (-2 * j / 8)
            output.extend(
                [
                    z[2 * j] * math.cos(angle) - z[2 * j + 1] * math.sin(angle),
                    z[2 * j] * math.sin(angle) + z[2 * j + 1] * math.cos(angle),
                ]
            )
        return torch.stack(output)

    rows = []
    for t in range(4):
        heads = []
        for h in range(4):
            if not valid[0, t]:
                heads.append(torch.zeros(8))
                continue
            keys = [
                s
                for s in range(t + 1)
                if bool(valid[0, s]) and documents[0, s] == documents[0, t]
            ]
            query = rotate(q[0, t, h], int(positions[0, t]))
            scores = torch.stack(
                [
                    (query * rotate(k[0, s, h // 2], int(positions[0, s]))).sum()
                    / math.sqrt(8)
                    for s in keys
                ]
            )
            weights = scores.softmax(0)
            heads.append(
                sum(
                    (weights[i] * v[0, s, h // 2] for i, s in enumerate(keys)),
                    torch.zeros(8),
                )
            )
        rows.append(torch.cat(heads))
    expected = F.linear(torch.stack(rows)[None], matrices["output.weight"])
    actual = layer(x, positions, documents, valid)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(actual)
    torch.autograd.backward((actual * upstream).sum())
    torch.autograd.backward((expected * upstream).sum())
    torch.testing.assert_close(x.grad, ref.grad, rtol=2e-5, atol=2e-6)
    for name, p in layer.named_parameters():
        torch.testing.assert_close(p.grad, matrices[name].grad, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(actual).all() and torch.equal(actual[0, 3], torch.zeros(32))


def test_swiglu_reference_and_gradients() -> None:
    layer = SwiGLU(SMALL)
    x = torch.randn(2, 3, 32, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    weights = [
        p.detach().clone().requires_grad_()
        for p in [layer.gate.weight, layer.up.weight, layer.down.weight]
    ]
    a = reference @ weights[0].T
    expected = (a * a.sigmoid() * (reference @ weights[1].T)) @ weights[2].T
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    torch.autograd.backward(actual.sum())
    torch.autograd.backward(expected.sum())
    torch.testing.assert_close(x.grad, reference.grad)
    for actual_p, reference_p in zip(
        [layer.gate.weight, layer.up.weight, layer.down.weight], weights, strict=True
    ):
        torch.testing.assert_close(actual_p.grad, reference_p.grad)


@pytest.mark.parametrize("name", ["tiny", "small", "base", "large"])
def test_actual_parameter_counts_on_meta(name: str) -> None:
    cfg = load_config(ROOT / f"configs/model/{name}.yaml").model
    with torch.device("meta"):
        model = Decoder(cfg)
    assert sum(p.numel() for p in model.parameters()) == cfg.parameter_count
    assert not any(n.endswith("bias") for n, _ in model.named_parameters())
    assert "lm_head.weight" not in model.state_dict()


def test_real_tiny_cpu_forward_backward() -> None:
    config = load_config(ROOT / "configs/model/tiny.yaml").model
    model = Decoder(config)
    logits = model(torch.tensor([[32, 33, 34]]))
    assert logits.shape == (1, 3, 32768)
    loss, count = causal_loss(logits, torch.tensor([[33, 34, 2]]))
    torch.autograd.backward(loss / count)
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_causality_and_packed_document_isolation() -> None:
    model = Decoder(SMALL)
    a = torch.tensor([[32, 33, 34, 35]])
    b = a.clone()
    b[0, 2:] = torch.tensor([90, 91])
    torch.testing.assert_close(model(a)[:, :2], model(b)[:, :2], rtol=0, atol=0)
    documents = torch.tensor([[1, 1, 2, 2]])
    positions = torch.tensor([[0, 1, 0, 1]])
    c = a.clone()
    c[0, :2] = torch.tensor([99, 100])
    torch.testing.assert_close(
        model(a, positions, documents)[:, 2:],
        model(c, positions, documents)[:, 2:],
        rtol=0,
        atol=0,
    )


def test_loss_reference_empty_mask_and_gradient() -> None:
    logits = torch.randn(1, 4, 288, requires_grad=True)
    target = torch.tensor([[3, 0, 1, -100]])
    value, count = causal_loss(logits, target)
    expected = torch.logsumexp(logits[0, 0].float(), 0) - logits[0, 0, 3]
    torch.testing.assert_close(value, expected)
    assert count.item() == 1
    torch.autograd.backward(value)
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 1:], torch.zeros(3, 288))
    zero, n = causal_loss(logits, target, torch.zeros_like(target, dtype=torch.bool))
    assert zero.item() == 0 and n.item() == 0


@pytest.mark.parametrize(
    "case", ["empty", "context", "dtype", "packed", "positions", "mask"]
)
def test_invalid_forward_inputs(case: str) -> None:
    model = Decoder(SMALL)
    ids = torch.tensor([[32, 33]])
    with pytest.raises((ValueError, RuntimeError)):
        if case == "empty":
            model(ids[:, :0])
        if case == "context":
            model(torch.zeros(1, 17, dtype=torch.int64))
        if case == "dtype":
            model(ids.float())
        if case == "packed":
            model(ids, document_ids=ids)
        if case == "positions":
            model(ids, torch.tensor([[-1, 0]]))
        if case == "mask":
            model(ids, valid_mask=ids)


def test_seeded_initialization_and_residual_variances() -> None:
    config = ModelConfig(
        layers=3, hidden=128, heads=8, kv_heads=2, ffn=256, vocab=288, context=8
    )
    with torch.random.fork_rng():
        torch.manual_seed(41)
        first = Decoder(config)
        torch.manual_seed(41)
        second = Decoder(config)
        assert all(
            torch.equal(value, second.state_dict()[name])
            for name, value in first.state_dict().items()
        )
        torch.manual_seed(42)
        different = Decoder(config)
    assert not torch.equal(first.embedding.weight, different.embedding.weight)
    scaled = {
        id(p)
        for block in first.blocks
        for p in (
            cast(Block, block).attention.output.weight,
            cast(Block, block).ffn.down.weight,
        )
    }
    for parameter in first.parameters():
        if parameter.ndim == 1:
            torch.testing.assert_close(
                parameter, torch.ones_like(parameter), rtol=0, atol=0
            )
            continue
        variance = 0.02**2 / (2 * config.layers) if id(parameter) in scaled else 0.02**2
        # At least4096 elements per matrix; this checks the distribution, not init calls.
        assert parameter.numel() >= 4096
        torch.testing.assert_close(
            parameter.detach().var(unbiased=True),
            torch.tensor(variance),
            rtol=0.08,
            atol=0,
        )
        assert abs(float(parameter.detach().mean())) < 0.1 * math.sqrt(variance)


def test_tied_embedding_gradient_equals_independent_input_and_output_sum() -> None:
    model = Decoder(SMALL)
    tokens = torch.tensor([[32, 32, 33, 34]])
    positions = torch.arange(4)[None]
    documents = torch.zeros_like(tokens)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    input_weight = model.embedding.weight.detach().clone().requires_grad_()
    output_weight = model.embedding.weight.detach().clone().requires_grad_()
    hidden = F.embedding(tokens, input_weight)
    for block in model.blocks:
        hidden = block(hidden, positions, documents, valid)
    separate_logits = F.linear(model.norm(hidden), output_weight)
    upstream = torch.randn_like(separate_logits)
    input_grad, output_grad = torch.autograd.grad(
        (separate_logits * upstream).sum(), (input_weight, output_weight)
    )
    actual_logits = model(tokens, positions, documents, valid)
    (tied_grad,) = torch.autograd.grad(
        (actual_logits * upstream).sum(), (model.embedding.weight,)
    )
    torch.testing.assert_close(
        tied_grad, input_grad + output_grad, rtol=2e-5, atol=2e-6
    )
    assert bool((input_grad[torch.tensor([32, 33, 34])] != 0).any())
    assert torch.equal(input_grad[:32], torch.zeros_like(input_grad[:32]))
    assert bool((output_grad[:32] != 0).any())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_are_not_silently_sanitized(value: float) -> None:
    logits = torch.full((1, 2, 288), value, requires_grad=True)
    targets = torch.tensor([[32, 0]])
    total, count = causal_loss(logits, targets)
    assert count.item() == 1 and not bool(torch.isfinite(total))
    norm = RMSNorm(8, 1e-6)
    assert not bool(torch.isfinite(norm(torch.full((1, 8), value))).all())


def test_nonfinite_padding_logits_still_surface_at_training_boundary() -> None:
    logits = torch.zeros(1, 2, 288)
    logits[:, 1] = float("nan")
    loss, count = causal_loss(logits, torch.tensor([[32, 0]]))
    assert count.item() == 1 and not bool(torch.isfinite(loss))


def test_fp32_seeded_parameters_do_not_follow_ambient_float64() -> None:
    previous = torch.get_default_dtype()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(59)
            reference = Decoder(SMALL)
            torch.set_default_dtype(torch.float64)
            torch.manual_seed(59)
            actual = Decoder(SMALL)
        assert all(
            parameter.dtype == torch.float32 for parameter in actual.parameters()
        )
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous)
