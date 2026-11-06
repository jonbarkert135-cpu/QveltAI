"""Independent naive decoding/sampler references and bounded cache lifecycle."""

from __future__ import annotations

import gc
import math
import weakref
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor

from qvelt.config import ModelConfig
from qvelt.inference.config import EngineLimits, GenerationOptions, TokenRequest
from qvelt.inference.generation import (
    GenerationEngine,
    cache_bytes,
    display_text,
    sample_token,
)
from qvelt.model.decoder import Decoder
from qvelt.tokenizer.core import TokenizerSettings, train_tokenizer


@pytest.fixture()
def engine() -> GenerationEngine:
    tokenizer = train_tokenizer(
        ["Original CPU generation fixture. Привет 😀 café"],
        TokenizerSettings(vocab_size=288),
    )
    torch.manual_seed(17)
    model = Decoder(
        ModelConfig(
            layers=2,
            hidden=32,
            heads=4,
            kv_heads=1,
            ffn=64,
            vocab=tokenizer.vocab_size,
            context=64,
        )
    ).eval()
    return GenerationEngine(model, tokenizer)


@pytest.mark.parametrize("seed", [1, 19, 123])
@pytest.mark.parametrize("strategy", ["greedy", "sample"])
def test_batched_cache_and_full_prefix_match_independent_unpadded_reference(
    engine: GenerationEngine, seed: int, strategy: str
) -> None:
    opts = GenerationOptions(
        max_new_tokens=6,
        stop_tokens=(),
        strategy=strategy,
        temperature=0 if strategy == "greedy" else 0.8,
        seed=seed,
    )
    requests = tuple(
        TokenRequest(
            str(i), tuple([1] + list(range(32, 32 + n))), replace(opts, seed=seed + i)
        )
        for i, n in enumerate((0, 5, 13))
    )
    saved = torch.get_rng_state().clone()
    cached = engine.generate(requests)
    plain = engine.generate(requests, use_cache=False)
    for request, a, b in zip(requests, cached, plain, strict=True):
        assert a.token_ids == b.token_ids
        assert a.text == b.text and a.reason == b.reason == "max_tokens"
        seq = list(request.tokens)
        rng = torch.Generator().manual_seed(request.options.seed)
        generated = []
        for _ in range(6):
            with torch.no_grad():
                logits = engine.model(torch.tensor([seq], dtype=torch.int64))[0, -1]
            if strategy == "greedy":
                value = int(logits.argmax())
            else:
                values = [float(x) / 0.8 for x in logits]
                peak = max(values)
                mass = [math.exp(x - peak) for x in values]
                probabilities = torch.tensor(
                    [x / sum(mass) for x in mass], dtype=torch.float32
                )
                value = int(torch.multinomial(probabilities, 1, generator=rng))
            seq.append(value)
            generated.append(value)
        assert list(a.token_ids) == generated
    assert torch.equal(saved, torch.get_rng_state())
    reverse = engine.generate(requests[::-1])
    assert {r.identifier: r.token_ids for r in reverse} == {
        r.identifier: r.token_ids for r in cached
    }


def test_sampler_distribution_matches_scalar_exponentials_and_has_no_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def draw(values: Tensor, count: int, *, generator: torch.Generator) -> Tensor:
        observed.append(values)
        assert count == 1
        return torch.tensor([2])

    monkeypatch.setattr(torch, "multinomial", draw)
    logits = torch.tensor([-3.0, 0.0, 2.0, 5.0], requires_grad=True)
    opts = GenerationOptions(strategy="sample", temperature=0.5)
    assert sample_token(logits, opts, torch.Generator()) == 2
    expected = [math.exp(2 * x) for x in (-3, 0, 2, 5)]
    assert observed[0].tolist() == pytest.approx(
        [x / sum(expected) for x in expected], rel=1e-6
    )
    assert not observed[0].requires_grad and logits.grad is None


def test_padded_cache_logits_match_each_naive_unpadded_prefix(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []
    original = engine.model.forward_cached

    def cached(*args: Any, **kwargs: Any) -> Any:
        logits, cache = original(*args, **kwargs)
        captured.append(logits[:, -1].clone())
        return logits, cache

    monkeypatch.setattr(engine.model, "forward_cached", cached)
    opts = GenerationOptions(max_new_tokens=4, stop_tokens=())
    requests = [
        TokenRequest(str(i), tuple([1] + list(range(32, 32 + n))), opts)
        for i, n in enumerate((1, 5, 13))
    ]
    result = engine.generate(requests)
    for step, batch in enumerate(captured):
        for i, (request, output) in enumerate(zip(requests, result, strict=True)):
            with torch.no_grad():
                expected = engine.model(
                    torch.tensor([[*request.tokens, *output.token_ids[:step]]])
                )[0, -1]
            torch.testing.assert_close(batch[i], expected, rtol=3e-5, atol=3e-6)


def test_finished_slot_is_masked_while_other_request_decodes(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = engine.model.forward_cached
    masks = []
    token = engine.tokenizer.encode("A")[0]

    def cached(*args: Any, **kwargs: Any) -> Any:
        logits, cache = original(*args, **kwargs)
        logits = logits.clone()
        logits[:, -1, :] = -1000
        logits[0, -1, 6] = 1000
        logits[1, -1, token] = 1000
        masks.append(cache.valid.clone())
        return logits, cache

    monkeypatch.setattr(engine.model, "forward_cached", cached)
    requests = [
        TokenRequest("early", (1, 32), GenerationOptions(max_new_tokens=3)),
        TokenRequest("later", (1, 33, 34), GenerationOptions(max_new_tokens=3)),
    ]
    results = engine.generate(requests)
    assert results[0].token_ids == (6,) and results[0].reason == "stop_token"
    assert results[1].token_ids == (token, token, token) and results[1].text == "AAA"
    assert masks[-1][0].tolist() == [False, True, True, False, False]
    assert masks[-1][1].all()


def test_engine_forces_fp32_without_leaking_ambient_autocast(
    engine: GenerationEngine,
) -> None:
    request = TokenRequest(
        "test", (1, 32), GenerationOptions(max_new_tokens=3, stop_tokens=())
    )
    expected = engine.generate([request])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = engine.generate([request])
        assert torch.is_autocast_enabled("cpu")
    assert actual == expected


def test_cache_payload_matches_actual_tensor_storage_and_true_batch_shape(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = engine.model.forward_cached
    seen = []

    def forward(*args: Any, **kwargs: Any) -> Any:
        logits, state = original(*args, **kwargs)
        actual = sum(
            t.numel() * t.element_size()
            for layer in state.layers
            for t in (layer.key, layer.value)
        )
        actual += sum(
            t.numel() * t.element_size()
            for t in (state.positions, state.documents, state.valid)
        )
        assert cache_bytes(engine.model.config, 2, state.length) == actual
        seen.append(args[0].shape[0])
        return logits, state

    monkeypatch.setattr(engine.model, "forward_cached", forward)
    requests = (
        TokenRequest("a", (1, 32), GenerationOptions(max_new_tokens=3, stop_tokens=())),
        TokenRequest("b", (1, 32, 33, 34)),
    )
    engine.generate(requests)
    assert seen and set(seen) == {2}


def test_stream_close_releases_cache_and_does_not_leak_grad_mode(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = engine.model.forward_cached
    refs = []

    def forward(*args: Any, **kwargs: Any) -> Any:
        logits, state = original(*args, **kwargs)
        refs.append(weakref.ref(state))
        return logits, state

    monkeypatch.setattr(engine.model, "forward_cached", forward)
    rng = torch.get_rng_state().clone()
    stream = engine.stream(
        [TokenRequest("a", (1,), GenerationOptions(max_new_tokens=4, stop_tokens=()))]
    )
    assert torch.is_grad_enabled()
    first = next(stream)
    assert not first.done and torch.is_grad_enabled()
    stream.close()
    assert torch.is_grad_enabled()
    gc.collect()
    assert refs and all(r() is None for r in refs)
    assert torch.equal(rng, torch.get_rng_state())


def oracle(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch, text: str
) -> list[int]:
    output = engine.tokenizer.encode(text) + [6]

    def forward(ids: Tensor, **kwargs: Any) -> Tensor:
        logits = torch.full((*ids.shape, engine.model.config.vocab), -1000.0)
        logits[:, -1, output[ids.shape[1] - 1]] = 1000
        return logits

    monkeypatch.setattr(engine.model, "forward", forward)
    return output


def test_split_stop_string_partial_prefix_and_unicode_snapshots(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle(engine, monkeypatch, "abcENDtail")
    opts = GenerationOptions(max_new_tokens=20, stop_strings=("END",))
    events = list(engine.stream([TokenRequest("s", (1,), opts)], use_cache=False))
    assert events[-1].reason == "stop_string" and events[-1].text == "abc"
    assert all("END" not in e.text for e in events)
    assert display_text("abcEN", ("END",), final=False) == ("abc", False)
    assert display_text("abcEN", ("END",), final=True) == ("abcEN", False)
    output = oracle(engine, monkeypatch, "é😀")
    events = list(
        engine.stream(
            [TokenRequest("u", (1,), GenerationOptions(max_new_tokens=20))],
            use_cache=False,
        )
    )
    assert events[-1].reason == "stop_token" and events[-1].text == "é😀"
    assert events[-1].generated_tokens == len(output)
    assert any("\ufffd" in e.text for e in events[:-1])


def test_cancellation_deadline_nonfinite_and_output_limit(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = TokenRequest("cancel", (1,), GenerationOptions(max_new_tokens=3))
    events = list(engine.stream([request], cancelled=lambda _: True))
    assert (
        len(events) == 1
        and events[0].reason == "cancelled"
        and events[0].token_id is None
    )
    ticks = iter([0.0, 0.0, 2.0])
    events = list(
        engine.stream(
            [
                replace(
                    request, options=GenerationOptions(max_new_tokens=3, max_seconds=1)
                )
            ],
            clock=lambda: next(ticks),
        )
    )
    assert events[-1].reason == "timeout" and events[-1].generated_tokens == 0
    monkeypatch.setattr(
        engine.model,
        "forward",
        lambda ids, **kwargs: torch.full(
            (*ids.shape, engine.model.config.vocab), float("nan")
        ),
    )
    assert list(engine.stream([request], use_cache=False))[-1].reason == "nonfinite"
    oracle(engine, monkeypatch, "abcdef")
    small = replace(
        request, options=GenerationOptions(max_new_tokens=10, max_output_chars=2)
    )
    result = engine.generate([small], use_cache=False)[0]
    assert result.reason == "output_limit" and result.text == "ab"


@pytest.mark.parametrize("cached", [True, False])
def test_model_mutation_is_rejected_mid_stream(
    engine: GenerationEngine, cached: bool
) -> None:
    stream = engine.stream(
        [TokenRequest("s", (1,), GenerationOptions(max_new_tokens=3, stop_tokens=()))],
        use_cache=cached,
    )
    next(stream)
    with torch.no_grad():
        engine.model.embedding.weight.add_(0.01)
    with pytest.raises(ValueError, match="changed"):
        next(stream)


def test_capacity_rejections_precede_forward_and_uncached_uses_no_kv(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = engine.model.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine.model, "forward", forward)
    for request in (
        TokenRequest("long", tuple([1] * 60), GenerationOptions(max_new_tokens=8)),
        TokenRequest("vocab", (99999,)),
        TokenRequest(
            "kv", (1,), GenerationOptions(max_new_tokens=2, max_cache_bytes=1)
        ),
    ):
        with pytest.raises(ValueError):
            list(engine.stream([request]))
    assert not calls
    request = TokenRequest(
        "kv",
        (1,),
        GenerationOptions(max_new_tokens=2, max_cache_bytes=1, stop_tokens=()),
    )
    assert engine.generate([request], use_cache=False)[0].cache_bytes_at_stop == 0
    limited = GenerationEngine(
        engine.model, engine.tokenizer, EngineLimits(max_total_tokens=1)
    )
    with pytest.raises(ValueError, match="capacity"):
        list(limited.stream([request]))


@pytest.mark.parametrize("bad", [None, [], [None], list(range(9))])
def test_all_public_batch_entries_reject_invalid_inputs(
    engine: GenerationEngine, bad: Any
) -> None:
    with pytest.raises(ValueError):
        engine.generate(bad)
    with pytest.raises(ValueError):
        list(engine.stream(bad))
    with pytest.raises(ValueError):
        engine.encode_requests(bad)
