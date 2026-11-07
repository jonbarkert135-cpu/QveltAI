"""Actual HQQ CPU packing,independent dequantized references and data-only reload."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from qvelt.eval.config import ArtifactRef
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.inference.artifacts import export_snapshot, load_snapshot
from qvelt.inference.config import GenerationOptions, TokenRequest
from qvelt.inference.generation import GenerationEngine
from qvelt.inference.quantization import (
    compact_state,
    export_quantized,
    load_quantized,
    quantize_model,
    restore_compact,
)
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("bits", [4, 8])
def test_actual_hqq_numeric_reference_and_standalone_reload(
    tmp_path: Path, bits: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("hqq")
    import hqq.core.quantize

    def reject_encoded_metadata(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Encoded HQQ dtype metadata must never be decoded")

    monkeypatch.setattr(
        hqq.core.quantize, "decode_safetensor_type", reject_encoded_metadata
    )
    job = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    m = export_snapshot(job.checkpoint, job.tokenizer, tmp_path / "native", ROOT)
    ref = ArtifactRef(str(tmp_path / "native/manifest.json"), manifest_digest(m))
    native = load_snapshot(ref)
    rng = torch.get_rng_state().clone()
    quant = quantize_model(native.model, bits)
    assert torch.equal(rng, torch.get_rng_state())
    ids = torch.tensor([[1, 32, 33, 34, 35]])
    with torch.no_grad():
        logits = quant(ids)
        assert torch.isfinite(logits).all()
        linear = cast(Any, quant.blocks[0]).attention.query
        inputs = torch.arange(64, dtype=torch.float32).reshape(1, 64) / 64
        torch.testing.assert_close(
            linear(inputs), inputs @ linear.dequantize().T, rtol=0, atol=0
        )
        # Every quantized projection agrees with an independent dense reconstruction.
        dense = load_snapshot(ref).model
        for name, module in quant.named_modules():
            if hasattr(module, "dequantize"):
                dict(dense.named_parameters())[name + ".weight"].copy_(
                    cast(Any, module).dequantize()
                )
        torch.testing.assert_close(quant(ids), dense(ids), atol=2e-6, rtol=2e-5)
    assert torch.equal(native.model.embedding.weight, quant.embedding.weight)
    before = compact_state(quant)
    assert sum(t.numel() * t.element_size() for t in before.values()) < sum(
        p.numel() * p.element_size() for p in native.model.parameters()
    )
    manifest = export_quantized(ref, tmp_path / "quant", ROOT, bits=bits)
    loaded = load_quantized(
        ArtifactRef(str(tmp_path / "quant/manifest.json"), manifest_digest(manifest))
    )
    for name, tensor in compact_state(loaded.model).items():
        assert torch.equal(tensor, before[name])
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(ids), logits, rtol=0, atol=0)
    request = TokenRequest(
        "test", (1, 32, 33), GenerationOptions(max_new_tokens=4, stop_tokens=())
    )
    assert (
        GenerationEngine(loaded.model, loaded.tokenizer)
        .generate([request])[0]
        .token_ids
        == GenerationEngine(quant, native.tokenizer).generate([request])[0].token_ids
    )
    bad = {**before, "encoded_dtype_payload": torch.tensor([1])}
    with pytest.raises(ValueError):
        restore_compact(quant, bad)
