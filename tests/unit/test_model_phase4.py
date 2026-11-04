"""Phase4 acceptance: gradcheck, determinism, random KV parity, overfit, depth scale,
compilation, RoPE interpolation and parameter/FLOP agreement with the RFC table."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch.func import functional_call

from qvelt.config import ConfigError, ModelConfig, load_config, parse_config
from qvelt.config.execution import ModelExecution
from qvelt.model.decoder import Block, Decoder
from qvelt.model.layers import RotaryEmbedding
from qvelt.model.loss import causal_loss
from qvelt.train.optim import flops_per_token

ROOT = Path(__file__).parents[2]
SMALL = ModelConfig(
    layers=2, hidden=16, heads=4, kv_heads=2, ffn=32, vocab=288, context=24
)
BACKENDS = ["reference", "sdpa"]


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(previous)


def _packed(lengths: list[int], batch: int) -> tuple[Tensor, Tensor]:
    """Reset positions and document IDs for consecutive packed documents."""
    positions = torch.cat([torch.arange(n) for n in lengths])
    documents = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])
    return (
        positions.expand(batch, -1).contiguous(),
        documents.expand(batch, -1).contiguous(),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_float64_gradcheck_of_full_packed_padded_decoder(backend: str) -> None:
    torch.manual_seed(3)
    model = Decoder(SMALL, ModelExecution(backend)).double()
    tokens = torch.randint(32, SMALL.vocab, (2, 7))
    positions, documents = _packed([3, 4], 2)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[1, -2:] = False
    names = [n for n, _ in model.named_parameters()]
    values = tuple(p.detach().clone().requires_grad_() for p in model.parameters())

    def logits(*params: Tensor) -> Tensor:
        output = functional_call(
            model,
            dict(zip(names, params, strict=True)),
            (tokens, positions, documents, valid),
        )
        return cast(Tensor, output)

    # Fast mode checks random projections of the full Jacobian for every parameter.
    assert torch.autograd.gradcheck(logits, values, fast_mode=True, atol=1e-6)


def _train_steps(seed: int, steps: int) -> tuple[Tensor, dict[str, Tensor]]:
    torch.manual_seed(seed)
    model = Decoder(SMALL, ModelExecution("sdpa", True)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tokens = torch.randint(32, SMALL.vocab, (2, 13))
    losses = []
    for _ in range(steps):
        total, count = causal_loss(model(tokens[:, :-1]), tokens[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(total / count)
        optimizer.step()
        losses.append(total.detach() / count)
    return torch.stack(losses), {k: v.clone() for k, v in model.state_dict().items()}


def test_seeded_training_is_bitwise_deterministic_on_cpu() -> None:
    first, first_state = _train_steps(7, 3)
    second, second_state = _train_steps(7, 3)
    other, _ = _train_steps(8, 3)
    assert torch.equal(first, second)
    assert all(torch.equal(v, second_state[k]) for k, v in first_state.items())
    assert not torch.equal(first, other)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [1.0, 3.0])
def test_cached_decode_matches_full_forward_for_random_lengths(
    backend: str, scale: float
) -> None:
    generator = torch.Generator().manual_seed(11)

    def draw(high: int) -> int:
        return int(torch.randint(1, high + 1, (), generator=generator))

    model = Decoder(replace(SMALL, rope_scale=scale), ModelExecution(backend)).eval()
    for _ in range(30):
        batch, length = draw(3), draw(SMALL.context)
        cuts = sorted({draw(length) for _ in range(draw(3))} | {length})
        lengths = [b - a for a, b in zip([0, *cuts[:-1]], cuts, strict=True)]
        positions, documents = _packed(lengths, batch)
        tokens = torch.randint(32, SMALL.vocab, (batch, length), generator=generator)
        with torch.no_grad():
            expected = model(tokens, positions, documents)
        prefix = draw(length)
        logits, cache = model.forward_cached(
            tokens[:, :prefix], None, positions[:, :prefix], documents[:, :prefix]
        )
        parts = [logits]
        start = prefix
        while start < length:
            stop = min(length, start + draw(3))
            logits, cache = model.forward_cached(
                tokens[:, start:stop],
                cache,
                positions[:, start:stop],
                documents[:, start:stop],
            )
            parts.append(logits)
            start = stop
        assert cache.length == length
        torch.testing.assert_close(
            torch.cat(parts, dim=1), expected, rtol=3e-5, atol=3e-6
        )


def test_tiny_model_overfits_one_batch() -> None:
    torch.manual_seed(0)
    config = load_config(ROOT / "configs/model/tiny.yaml").model
    model = Decoder(config)
    tokens = torch.randint(288, config.vocab, (2, 17))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for _ in range(100):
        total, count = causal_loss(model(tokens[:, :-1]), tokens[:, 1:])
        loss = total / count
        if float(loss.detach()) < 0.05:
            break
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimizer.step()
    # Initial loss is ~ln(32768)=10.4; memorizing 32 random targets must succeed.
    assert float(loss.detach()) < 0.05


def _depth_profile(layers: int, scaled: bool) -> tuple[float, float]:
    """Added residual variance at the last block and block gradient-norm spread."""
    torch.manual_seed(5)
    config = ModelConfig(
        layers=layers, hidden=128, heads=4, kv_heads=2, ffn=384, vocab=512, context=64
    )
    model = Decoder(config)
    if not scaled:
        for name, parameter in model.named_parameters():
            if name.endswith(("output.weight", "down.weight")):
                nn.init.normal_(parameter, std=0.02)
    tokens = torch.randint(288, config.vocab, (4, 64))
    outputs: list[Tensor] = []
    hooks = [
        block.register_forward_hook(lambda _m, _i, out: outputs.append(out.detach()))
        for block in model.blocks
    ]
    total, count = causal_loss(model(tokens[:, :-1]), tokens[:, 1:])
    torch.autograd.backward(total / count)
    for hook in hooks:
        hook.remove()
    embedded = model.embedding(tokens[:, :-1]).detach()
    added = float(outputs[-1].square().mean() - embedded.square().mean())
    norms = torch.stack(
        [
            torch.linalg.vector_norm(
                torch.cat([cast(Tensor, p.grad).flatten() for p in block.parameters()])
            )
            for block in model.blocks
        ]
    )
    return added, float(norms.max() / norms.min())


def test_residual_scaled_init_keeps_activation_and_gradient_scale_across_depth() -> (
    None
):
    shallow, _ = _depth_profile(4, True)
    deep, spread = _depth_profile(24, True)
    shallow_plain, _ = _depth_profile(4, False)
    deep_plain, _ = _depth_profile(24, False)
    # Branch variance ~1/(2L), so the total added variance is ~depth independent.
    assert deep / shallow < 2.0
    assert deep_plain / shallow_plain > 4.0
    assert spread < 2.5


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("recompute", [False, True])
def test_torch_compile_matches_eager_values_and_gradients(
    backend: str, recompute: bool
) -> None:
    _compile_parity(backend, recompute, "aot_eager")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("cc") is None, reason="inductor needs a C compiler")
def test_inductor_compile_matches_eager() -> None:
    _compile_parity("sdpa", False, "inductor")


def _compile_parity(backend: str, recompute: bool, compiler: str) -> None:
    torch.manual_seed(1)
    model = Decoder(SMALL, ModelExecution(backend, recompute)).train()
    tokens = torch.randint(32, SMALL.vocab, (2, 9))
    positions, documents = _packed([4, 5], 2)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[1, -2:] = False
    expected = model(tokens, positions, documents, valid)
    expected.square().mean().backward()
    gradients = [cast(Tensor, p.grad).clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=compiler)
    actual = compiled(tokens, positions, documents, valid)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=1e-4, atol=1e-6)
    # Blocks trace without graph breaks; Decoder retains host-side validation.
    torch._dynamo.reset()
    block = torch.compile(model.blocks[0], backend="aot_eager", fullgraph=True)
    x = torch.randn(2, 9, SMALL.hidden)
    torch.testing.assert_close(
        block(x, positions, documents, valid),
        cast(Block, model.blocks[0])(x, positions, documents, valid),
    )
    torch._dynamo.reset()


def test_rope_linear_interpolation_hook() -> None:
    x = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    positions = torch.arange(5).expand(2, 5)
    base = RotaryEmbedding(8, 10000.0)
    stretched = RotaryEmbedding(8, 10000.0, scale=4.0)
    # Position 4p at scale 4 is exactly the baseline rotation at position p.
    torch.testing.assert_close(stretched(x, 4 * positions), base(x, positions))
    assert ModelConfig().rope_scale == 1.0
    assert parse_config("model: {rope_scale: 2}").model.rope_scale == 2.0
    with pytest.raises(ConfigError, match="rope_scale"):
        ModelConfig(rope_scale=0.5)
    scaled = Decoder(replace(SMALL, rope_scale=2.0))
    assert cast(Block, scaled.blocks[0]).attention.rope.scale == 2.0


def _rfc_table() -> dict[str, list[str]]:
    text = (ROOT / "docs/research/architecture_rfc.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (tiny|small|base|large) \|(.+)\|$", text, re.MULTILINE)
    return {name: [c.strip() for c in cells.split("|")] for name, cells in rows[:4]}


@pytest.mark.parametrize("name", ["tiny", "small", "base", "large"])
def test_parameter_and_flop_counts_match_rfc_table(name: str) -> None:
    cells = _rfc_table()[name]
    layers, hidden, heads, kv_heads, ffn, vocab, context = map(int, cells[:7])
    total, embedding, non_embedding = (int(c.replace(",", "")) for c in cells[7:10])
    config = load_config(ROOT / f"configs/model/{name}.yaml").model
    assert (config.layers, config.hidden, config.heads, config.kv_heads) == (
        layers,
        hidden,
        heads,
        kv_heads,
    )
    assert (config.ffn, config.vocab, config.context) == (ffn, vocab, context)
    with torch.device("meta"):
        model = Decoder(config)
    counted = sum(p.numel() for p in model.parameters())
    assert counted == config.parameter_count == total
    assert model.embedding.weight.numel() == embedding
    assert counted - embedding == non_embedding
    gflops = flops_per_token(config, config.context) / 1e9
    assert round(gflops, 6) == float(cells[10])
