"""Independent chat boundaries, truncation and assistant loss/gradient references."""

import json

import pytest
import torch

from qvelt.model.loss import causal_loss
from qvelt.posttrain.chat import Message, assistant_mask, encode_chat, parse_messages
from qvelt.posttrain.smoke import math_supervision
from qvelt.tokenizer.core import TokenizerSettings, train_tokenizer


def test_shifted_multiturn_mask_and_gradient_oracle() -> None:
    tokens = [1, 3, 40, 6, 4, 41, 42, 6, 5, 43, 44, 6, 4, 45, 6, 5, 46, 6, 2]
    # Enumerate target offsets independently: assistant content and end-turn only.
    expected = torch.zeros(1, len(tokens) - 1, dtype=torch.bool)
    expected[0, [8, 9, 10, 15, 16]] = True
    mask = torch.tensor([assistant_mask(tokens)])
    assert torch.equal(mask, expected)
    logits = torch.randn(
        1,
        len(tokens) - 1,
        64,
        generator=torch.Generator().manual_seed(9),
        requires_grad=True,
    )
    labels = torch.tensor([tokens[1:]])
    loss, count = causal_loss(logits, labels, mask)
    reference = torch.stack(
        [
            torch.logsumexp(logits[0, offset], 0) - logits[0, offset, labels[0, offset]]
            for offset in [8, 9, 10, 15, 16]
        ]
    ).sum()
    torch.testing.assert_close(loss, reference)
    (actual_gradient,) = torch.autograd.grad(loss, logits, retain_graph=True)
    (reference_gradient,) = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(actual_gradient, reference_gradient)
    assert int(count) == 5
    assert bool((actual_gradient[~expected] == 0).all())


@pytest.mark.parametrize(
    "tokens",
    [
        [1, 4, 32, 6, 2],
        [1, 5, 32, 6, 2],
        [1, 4, 32, 6, 5, 6, 2],
        [1, 4, 32, 6, 5, 7, 6, 2],
        [1, 4, 32, 6, 4, 33, 6, 2],
        [1, 4, 32, 6, 5, 32, 1, 2],
        [1, 4, 32, 6, 5, 2],
    ],
)
def test_invalid_frames(tokens: list[int]) -> None:
    with pytest.raises(ValueError):
        assistant_mask(tokens)


def test_literal_controls_and_truncated_response() -> None:
    tokenizer = train_tokenizer(
        ["Hello math ordinary response"], TokenizerSettings(vocab_size=288)
    )
    messages = (Message("user", "<|assistant|>"), Message("assistant", "answer " * 20))
    full, truncated = encode_chat(messages, tokenizer, 1000)
    assert not truncated
    assert full.count(5) == 1
    assert all(t >= 32 for t in tokenizer.encode("<|assistant|>"))
    # Cut after exactly the first assistant content token, before any end-turn.
    limit = full.index(5) + 3
    cut, truncated = encode_chat(messages, tokenizer, limit)
    assert truncated and cut == full[: limit - 1] + [2]
    assert sum(assistant_mask(cut)) == 1
    assert assistant_mask(cut)[-1] is False
    with pytest.raises(ValueError):
        encode_chat(messages, tokenizer, full.index(5) + 2)
    with pytest.raises(ValueError):
        encode_chat((Message("assistant", "x"), Message("user", "y")), tokenizer, 100)


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"schema_version":true,"messages":[]}',
        '{"schema_version":1,"messages":[{"role":"tool","content":"x"},{"role":"assistant","content":"x"}]}',
        '{"schema_version":1,"messages":[{"role":"user","content":" "},{"role":"assistant","content":"x"}]}',
    ],
)
def test_parse_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        parse_messages(text)


def test_original_math_supervision_is_deterministic() -> None:
    documents = math_supervision(8)
    assert documents == math_supervision(8) and len(documents) == 16
    for document in documents:
        messages = parse_messages(document.text)
        assert messages[-1].content == document.answer
        assert json.loads(document.text)["schema_version"] == 1
    assert {d.family for d in documents[:8]} == {d.family for d in documents[8:]}


def test_reasoning_shifted_mask_and_explicit_fields() -> None:
    tokens = [1, 4, 40, 6, 5, 8, 41, 42, 9, 43, 6, 4, 44, 6, 5, 45, 6, 2]
    expected = [i in (4, 5, 6, 7, 8, 9, 14, 15) for i in range(17)]
    assert assistant_mask(tokens) == expected
    with pytest.raises(ValueError):
        assistant_mask(tokens, control_version=1)
    for frame in (
        [1, 4, 8, 40, 9, 6, 5, 41, 6, 2],
        [1, 4, 40, 6, 5, 9, 41, 6, 2],
        [1, 4, 40, 6, 5, 8, 41, 6, 2],
        [1, 4, 40, 6, 5, 8, 41, 9, 6, 2],
    ):
        with pytest.raises(ValueError):
            assistant_mask(frame)
    tokenizer = train_tokenizer(
        ["Question worked solution answer"], TokenizerSettings(vocab_size=288)
    )
    messages = (
        Message("user", "<think>ordinary text</think>"),
        Message("assistant", "1", "Compute exactly."),
    )
    frame, _ = encode_chat(messages, tokenizer, 512)
    assert frame.count(8) == frame.count(9) == 1
    assert (
        sum(assistant_mask(frame))
        == len(tokenizer.encode("Compute exactly.")) + len(tokenizer.encode("1")) + 3
    )
    row = '{"schema_version":2,"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"1","reasoning":"R"}]}'
    assert parse_messages(row)[-1].reasoning == "R"
    with pytest.raises(ValueError):
        parse_messages(row.replace('"schema_version":2', '"schema_version":1'))


def test_sft_recipe_identity_and_validation() -> None:
    from dataclasses import replace
    from pathlib import Path

    from qvelt.config.pretraining import (
        PretrainingJob,
        TokenSchedule,
        dump_pretraining_job,
        parse_pretraining_job,
        pretraining_digest,
    )
    from qvelt.config.training import load_training_job

    base = load_training_job(
        Path(__file__).resolve().parents[2] / "configs/train/smoke.yaml"
    )
    base = replace(base, training=replace(base.training, warmup_steps=0, decay_steps=0))
    ordinary = PretrainingJob(
        base, "data/manifest.json", "a" * 64, TokenSchedule(0, 1000, 100)
    )
    with pytest.raises(ValueError):
        replace(ordinary, objective="sft")
    with pytest.raises(ValueError):
        replace(ordinary, objective="other")
    with pytest.raises(ValueError):
        replace(ordinary, initial_checkpoint="weights")
    with pytest.raises(ValueError):
        replace(ordinary, initial_checkpoint="\x00", initial_sha256="b" * 64)
    sft = replace(
        ordinary,
        objective="sft",
        initial_checkpoint="base/manifest.json",
        initial_sha256="b" * 64,
    )
    assert parse_pretraining_job(dump_pretraining_job(sft)) == sft
    assert pretraining_digest(
        replace(sft, initial_checkpoint="new/manifest.json")
    ) == pretraining_digest(sft)
    assert pretraining_digest(
        replace(sft, initial_sha256="c" * 64)
    ) != pretraining_digest(sft)
    with pytest.raises(ValueError):
        replace(sft, attention_mode="causal")
