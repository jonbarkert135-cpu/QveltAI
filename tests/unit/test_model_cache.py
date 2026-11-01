"""Cached/full causal and packed parity, compact KV storage and stale-cache rejection."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from qvelt.config import ModelConfig
from qvelt.config.execution import ModelExecution
from qvelt.model.cache import LayerKV, parameter_stamp
from qvelt.model.decoder import Decoder

CONFIG = ModelConfig(
    layers=2, hidden=32, heads=4, kv_heads=2, ffn=64, vocab=288, context=8
)


@pytest.fixture(autouse=True)
def one_thread() -> None:
    torch.set_num_threads(1)


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("chunks", [(1, 1, 2, 4), (3, 2, 3), (8,)])
def test_full_and_cached_logits_and_compact_kv(
    backend: str, kv_heads: int, chunks: tuple[int, ...]
) -> None:
    model = Decoder(replace(CONFIG, kv_heads=kv_heads), ModelExecution(backend)).eval()
    tokens = torch.tensor(
        [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47]]
    )
    with torch.no_grad():
        expected = model(tokens)
    cache = None
    parts = []
    consumed = 0
    saved_rng = torch.get_rng_state().clone()
    for size in chunks:
        old = cache
        old_keys = None if old is None else old.layers[0].key.clone()
        logits, cache = model.forward_cached(
            tokens[:, consumed : consumed + size], cache
        )
        consumed += size
        parts.append(logits)
        assert cache.length == consumed and len(cache.layers) == CONFIG.layers
        assert cache.layers[0].key.shape == (2, kv_heads, consumed, 8)
        assert not logits.requires_grad and not cache.layers[0].key.requires_grad
        if old is not None:
            assert old.length == consumed - size
            assert torch.equal(old.layers[0].key, old_keys)
    assert torch.equal(torch.get_rng_state(), saved_rng)
    torch.testing.assert_close(torch.cat(parts, dim=1), expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
def test_packed_documents_padding_and_reset_positions(backend: str) -> None:
    model = Decoder(CONFIG, ModelExecution(backend)).eval()
    tokens = torch.tensor([[32, 33, 34, 35, 36, 37], [38, 39, 40, 41, 42, 43]])
    positions = torch.tensor([[0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5]])
    documents = torch.tensor([[7, 7, 8, 8, 9, 9], [3, 3, 3, 3, 3, 3]])
    valid = torch.tensor(
        [[True, True, True, False, True, True], [False, False, True, True, True, False]]
    )
    with torch.no_grad():
        expected = model(tokens, positions, documents, valid)
    first, cache = model.forward_cached(
        tokens[:, :3],
        positions=positions[:, :3],
        document_ids=documents[:, :3],
        valid_mask=valid[:, :3],
    )
    second, cache = model.forward_cached(
        tokens[:, 3:], cache, positions[:, 3:], documents[:, 3:], valid[:, 3:]
    )
    torch.testing.assert_close(
        torch.cat((first, second), dim=1), expected, rtol=3e-5, atol=3e-6
    )
    assert torch.equal(cache.positions, positions) and torch.equal(
        cache.documents, documents
    )
    assert torch.equal(cache.valid, valid)


def test_cache_context_batch_and_training_mode_fail_without_advancing() -> None:
    model = Decoder(CONFIG)
    tokens = torch.tensor([[32, 33]])
    with pytest.raises(ValueError, match="eval"):
        model.forward_cached(tokens)
    model.eval()
    _, cache = model.forward_cached(tokens)
    before = cache.layers[0].key.clone()
    with pytest.raises(ValueError, match="context"):
        model.forward_cached(torch.full((1, 7), 32), cache)
    with pytest.raises(ValueError, match="metadata"):
        model.forward_cached(tokens.expand(2, -1), cache)
    with pytest.raises(ValueError, match="another model"):
        Decoder(CONFIG).eval().forward_cached(tokens, cache)
    assert cache.length == 2 and torch.equal(cache.layers[0].key, before)


@pytest.mark.parametrize("mutation", ["inplace", "load", "replace"])
def test_parameter_updates_invalidate_cache(mutation: str) -> None:
    model = Decoder(CONFIG).eval()
    tokens = torch.tensor([[32, 33]])
    _, cache = model.forward_cached(tokens)
    if mutation == "inplace":
        with torch.no_grad():
            model.embedding.weight.add_(0.01)
    elif mutation == "load":
        model.load_state_dict(model.state_dict())
    else:
        model.embedding.weight = torch.nn.Parameter(
            model.embedding.weight.detach().clone()
        )
    with pytest.raises(ValueError, match="stale weights"):
        model.forward_cached(tokens, cache)


@pytest.mark.parametrize(
    "field",
    [
        "positions",
        "documents",
        "valid",
        "layers",
        "key",
        "gradient",
        "execution",
        "config",
    ],
)
def test_malformed_cache_is_rejected(field: str) -> None:
    model = Decoder(CONFIG).eval()
    tokens = torch.tensor([[32, 33]])
    _, cache = model.forward_cached(tokens)
    if field == "positions":
        cache = replace(cache, positions=cache.positions.float())
    elif field == "documents":
        cache = replace(cache, documents=cache.documents[:, 0])
    elif field == "valid":
        cache = replace(cache, valid=cache.valid.long())
    elif field == "layers":
        cache = replace(cache, layers=())
    elif field in ("key", "gradient"):
        first = cache.layers[0]
        key = (
            first.key[:, :, :1]
            if field == "key"
            else first.key.clone().requires_grad_()
        )
        cache = replace(cache, layers=(LayerKV(key, first.value), *cache.layers[1:]))
    elif field == "execution":
        cache = replace(cache, execution=ModelExecution("sdpa"))
    else:
        cache = replace(cache, config=replace(CONFIG, context=9))
    with pytest.raises(ValueError):
        model.forward_cached(tokens, cache)


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
def test_cpu_bfloat16_cached_logits_and_policy_change(backend: str) -> None:
    model = Decoder(CONFIG, ModelExecution(backend)).eval()
    tokens = torch.tensor([[32, 33, 34, 35]])
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(tokens)
        first, cache = model.forward_cached(tokens[:, :2])
        second, suffix = model.forward_cached(tokens[:, 2:], cache)
        torch.testing.assert_close(
            torch.cat((first, second), dim=1), expected, rtol=0.04, atol=0.004
        )
    assert suffix.length == 4 and cache.layers[0].key.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="dtype"):
        model.forward_cached(tokens[:, 2:], cache)
    assert cache.length == 2


def test_metadata_is_not_aliased_to_caller_and_parameter_stamp_is_checked() -> None:
    model = Decoder(CONFIG).eval()
    positions = torch.tensor([[0, 1]])
    docs = torch.tensor([[7, 7]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    _, cache = model.forward_cached(
        torch.tensor([[32, 33]]),
        positions=positions,
        document_ids=docs,
        valid_mask=mask,
    )
    positions.zero_()
    docs.zero_()
    mask.zero_()
    assert (
        cache.positions[0, 1] == 1
        and cache.documents[0, 0] == 7
        and bool(cache.valid.all())
    )
    before = parameter_stamp(model)
    with torch.no_grad():
        model.norm.weight.add_(1)
    assert parameter_stamp(model) != before
    with torch.inference_mode():
        inference_parameters = Decoder(CONFIG).eval()
    with pytest.raises(ValueError, match="version counters"):
        inference_parameters.forward_cached(torch.tensor([[32, 33]]))


def test_cache_owner_is_weak_and_cannot_outlive_model_identity() -> None:
    import gc
    from weakref import ref

    model = Decoder(CONFIG).eval()
    tokens = torch.tensor([[32, 33]])
    _, cache = model.forward_cached(tokens)
    owner = ref(model)
    del model
    gc.collect()
    assert owner() is None and cache.owner() is None
    with pytest.raises(ValueError, match="another model"):
        Decoder(CONFIG).eval().forward_cached(tokens, cache)
