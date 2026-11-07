"""Actual offline AutoModel/AutoTokenizer and independent native logits/cache parity."""

from pathlib import Path

import pytest
import torch

from qvelt.eval.config import ArtifactRef
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.inference.artifacts import export_snapshot, load_snapshot
from qvelt.inference.huggingface import export_huggingface, split_rotary_rows
from qvelt.posttrain.chat import Message, encode_chat
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]


def test_row_permutation_independent_index_oracle() -> None:
    weight = torch.arange(48).reshape(12, 4)
    expected = weight[[0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11]]
    assert torch.equal(split_rotary_rows(weight, 3), expected)
    with pytest.raises(ValueError):
        split_rotary_rows(weight, 4)


def test_real_offline_builtin_hf_export_and_cache(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    job = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    manifest = export_snapshot(job.checkpoint, job.tokenizer, tmp_path / "native", ROOT)
    ref = ArtifactRef(str(tmp_path / "native/manifest.json"), manifest_digest(manifest))
    snapshot = load_snapshot(ref)
    export_huggingface(ref, tmp_path / "hf", ROOT)
    hf = AutoModelForCausalLM.from_pretrained(
        tmp_path / "hf",
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tmp_path / "hf", local_files_only=True, trust_remote_code=False
    )
    assert (
        hf.get_input_embeddings().weight.data_ptr()
        == hf.get_output_embeddings().weight.data_ptr()
    )
    for text in ("Привет 😀", "A = 12/3; café", ""):
        assert tokenizer.encode(
            text, add_special_tokens=False
        ) == snapshot.tokenizer.encode(text)
    assert tokenizer.encode("Hello") == [1, *snapshot.tokenizer.encode("Hello")]
    messages = (
        Message("system", "You solve math."),
        Message("user", "2+2?"),
        Message("assistant", "4"),
        Message("user", "3+3?"),
        Message("assistant", "6"),
    )
    native, _ = encode_chat(messages, snapshot.tokenizer, 256)
    hf_ids = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    assert hf_ids == native
    ids = torch.tensor([native])
    with torch.no_grad():
        expected = snapshot.model(ids)
        actual = hf(ids, use_cache=True)
        torch.testing.assert_close(actual.logits, expected, atol=4e-6, rtol=4e-5)
        suffix = torch.tensor([[32, 33, 34]])
        cached = hf(
            suffix, past_key_values=actual.past_key_values, use_cache=True
        ).logits
        full = snapshot.model(torch.cat((ids, suffix), dim=1))[:, -3:]
        torch.testing.assert_close(cached, full, atol=4e-6, rtol=4e-5)
    with pytest.raises(Exception, match="Reserved control"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|assistant|>"}],
            tokenize=True,
            add_generation_prompt=True,
        )


def test_streamed_checkpoint_shards_and_extended_positions(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from qvelt.inference.hf_checkpoint import (
        export_checkpoint_huggingface,
        weight_layout,
    )
    from qvelt.inference.long_context import extend_context

    job = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    native_manifest = export_snapshot(
        job.checkpoint, job.tokenizer, tmp_path / "native", ROOT
    )
    snapshot = load_snapshot(
        ArtifactRef(
            str(tmp_path / "native/manifest.json"), manifest_digest(native_manifest)
        )
    )
    before = torch.get_rng_state().clone()
    manifest = export_checkpoint_huggingface(
        job.checkpoint,
        job.tokenizer,
        tmp_path / "hf",
        ROOT,
        context_extension=2,
        shard_bytes=32768,
    )
    assert torch.equal(before, torch.get_rng_state())
    assert sum(file.role == "hf_weights" for file in manifest.files) > 1
    assert set(weight_layout(snapshot.model.config)) == set(snapshot.model.state_dict())
    hf = AutoModelForCausalLM.from_pretrained(
        tmp_path / "hf",
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    native = extend_context(snapshot.model, 2)
    ids = (
        torch.arange(snapshot.model.config.context + 3)
        .remainder(native.config.vocab)
        .unsqueeze(0)
    )
    with torch.no_grad():
        torch.testing.assert_close(hf(ids).logits, native(ids), atol=4e-6, rtol=4e-5)
    with pytest.raises(ValueError, match="shard budget"):
        export_checkpoint_huggingface(
            job.checkpoint, job.tokenizer, tmp_path / "bad", ROOT, shard_bytes=0
        )
