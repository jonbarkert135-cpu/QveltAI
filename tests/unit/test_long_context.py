"""Exact position-interpolation identity,cache parity and original-weight immutability."""

import pytest
import torch

from qvelt.config import ModelConfig
from qvelt.inference.long_context import extend_context
from qvelt.model.decoder import Decoder


def test_interpolated_doubled_positions_match_original_and_cache() -> None:
    torch.manual_seed(37)
    original = Decoder(
        ModelConfig(
            layers=2, hidden=32, heads=4, kv_heads=1, ffn=64, vocab=288, context=32
        )
    ).eval()
    saved = {name: p.clone() for name, p in original.state_dict().items()}
    state = torch.get_rng_state().clone()
    extended = extend_context(original, 2)
    assert (
        torch.equal(state, torch.get_rng_state())
        and extended.config.context == 64
        and extended.config.rope_scale == 2
    )
    ids = torch.tensor([[1, 32, 33, 34, 35]])
    with torch.no_grad():
        expected = original(ids)
        actual = extended(ids, positions=2 * torch.arange(5)[None, :])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        long_ids = torch.randint(32, 288, (1, 50))
        full = extended(long_ids)
        _, cache = extended.forward_cached(long_ids[:, :31])
        cached, _ = extended.forward_cached(long_ids[:, 31:], cache)
        torch.testing.assert_close(cached, full[:, 31:], atol=2e-6, rtol=2e-5)
    assert original.config.context == 32 and all(
        torch.equal(original.state_dict()[name], tensor)
        for name, tensor in saved.items()
    )
    with pytest.raises(ValueError):
        extend_context(original, 3)


def test_cached_last_logit_selection_matches_full_vocab_projection() -> None:
    model = Decoder(
        ModelConfig(
            layers=1, hidden=32, heads=4, kv_heads=1, ffn=64, vocab=288, context=64
        )
    ).eval()
    ids = torch.tensor([[1, 32, 33, 34]])
    full, cache = model.forward_cached(ids)
    last, last_cache = model.forward_cached(ids, last_logits_only=True)
    assert last.shape == (1, 1, 288) and last_cache.length == cache.length
    torch.testing.assert_close(last, full[:, -1:], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError):
        model.forward_cached(ids, last_logits_only=1)  # type: ignore[arg-type]  # Invalid boundary input.
