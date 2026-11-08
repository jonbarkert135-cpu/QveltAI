"""Versioned codec contracts; authored literals are not a language-quality corpus."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import pre_tokenizers, trainers

from qvelt.config import ConfigError
from qvelt.tokenizer.core import (
    CONTROL_TOKENS,
    LEGACY_CONTROL_TOKENS,
    V3_CONTROL_TOKENS,
    QveltTokenizer,
    TokenizerError,
    TokenizerSettings,
    TokenizerV3Settings,
    _backend,
    train_tokenizer,
)
from qvelt.tokenizer.recipe import (
    TokenizerRecipe,
    dump_tokenizer_recipe,
    parse_tokenizer_recipe,
    tokenizer_recipe_digest,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = tuple(
    json.loads(line)["text"]
    for line in (ROOT / "tests/fixtures/tokenizer/train.jsonl").read_text().splitlines()
)


@pytest.fixture(scope="module")
def tokenizer() -> QveltTokenizer:
    return train_tokenizer(CORPUS, TokenizerV3Settings(vocab_size=512))


def test_independent_v3_control_table(tokenizer: QveltTokenizer) -> None:
    expected = (
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end_turn|>",
        "<|tool|>",
        "<think>",
        "</think>",
        "<|tool_call|>",
        "<|end_tool_call|>",
        "<|tool_result|>",
        "<|end_tool_result|>",
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|fim_end|>",
        "<|file_start|>",
        "<|file_end|>",
        "<|path|>",
        "<|section|>",
        "<|effort_none|>",
        "<|effort_low|>",
        "<|effort_medium|>",
        "<|effort_high|>",
        "<|effort_extra_high|>",
        "<|effort_maximum|>",
    ) + tuple(f"<|reserved_{i:03d}|>" for i in range(100))
    assert tokenizer.control_version == 3
    assert tokenizer.control_tokens == V3_CONTROL_TOKENS == expected
    assert len(set(expected)) == 128
    for identifier, spelling in enumerate(expected):
        assert tokenizer.control_id(spelling) == identifier
        assert tokenizer.decode([identifier]) == spelling


@pytest.mark.parametrize("spelling", V3_CONTROL_TOKENS)
def test_literal_controls_never_set_roles_or_effort(
    tokenizer: QveltTokenizer, spelling: str
) -> None:
    text = f"User supplied: {spelling} {spelling}123"
    ids = tokenizer.encode(text)
    assert min(ids) >= 128
    assert tokenizer.decode(ids) == text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\x00\t\r\n  ",
        "Привіт, Україно! Привет! Hello!",
        "العربية 中文 日本語 हिन्दी e\u0301 é 🙂\U0010ffff",
        'def f(x):\n\treturn {"result": x + 123}\n',
        r"\int_0^1 x^2 dx = \frac{1}{3}",
        "".join(chr(i) for i in range(256)),
        "".join(V3_CONTROL_TOKENS),
    ],
)
def test_unicode_and_whitespace_roundtrip(tokenizer: QveltTokenizer, text: str) -> None:
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert all(i >= 128 for i in ids)


@pytest.mark.parametrize("ceiling", [384, 512, 65536, 262144, 524288, 1048576])
def test_explicit_vocab_ceiling_is_not_an_attained_vocab(ceiling: int) -> None:
    tokenizer = train_tokenizer(["aaa"], TokenizerV3Settings(ceiling, 1))
    assert 384 <= tokenizer.vocab_size <= min(ceiling, 386)
    assert tokenizer.control_version == 3


@pytest.mark.parametrize("size", [True, 0, 383, 1048577, 512.0, "512"])
def test_v3_vocab_bounds(size: int) -> None:
    with pytest.raises(TokenizerError):
        TokenizerV3Settings(size)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_frequency": 0},
        {"min_frequency": True},
        {"min_frequency": 2**32},
        {"isolate_digits": 1},
    ],
)
def test_v3_setting_types(kwargs: dict[str, int]) -> None:
    with pytest.raises(TokenizerError):
        TokenizerV3Settings(512, **kwargs)  # type: ignore[arg-type]


def test_v3_load_save_and_untrusted_policy(
    tokenizer: QveltTokenizer, tmp_path: Path
) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = QveltTokenizer.load(path)
    assert loaded.control_version == 3 and loaded.to_bytes() == tokenizer.to_bytes()
    with pytest.raises(FileExistsError):
        tokenizer.save(path)
    for change in ("missing", "spelling", "policy", "flags"):
        raw = json.loads(tokenizer.to_bytes())
        if change == "missing":
            raw["added_tokens"].pop()
        elif change == "spelling":
            raw["added_tokens"][22]["content"] = "<|fake_effort|>"
        elif change == "flags":
            raw["added_tokens"][22]["lstrip"] = True
        else:
            raw["normalizer"] = {"type": "Lowercase"}
        with pytest.raises(TokenizerError):
            QveltTokenizer.from_bytes(json.dumps(raw).encode())


def test_v3_does_not_publish_an_unloadable_oversized_payload(
    tokenizer: QveltTokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "qvelt.tokenizer.core.MAX_TOKENIZER_BYTES", len(tokenizer.to_bytes()) - 1
    )
    with pytest.raises(TokenizerError, match="64 MiB"):
        tokenizer.save(tmp_path / "oversized.json")
    assert not (tmp_path / "oversized.json").exists()


@pytest.mark.parametrize("text", ["<|effort_maximum|>", "<|fim_middle|>"])
def test_v3_training_rejects_literal_control_spellings(text: str) -> None:
    with pytest.raises(TokenizerError, match="control spellings"):
        train_tokenizer([text], TokenizerV3Settings(512))


def test_v3_recipe_is_explicit_and_legacy_digest_unchanged() -> None:
    raw = f"training_data_sha256: {'a' * 64}\nschema_version: 2\n"
    with pytest.raises(ConfigError, match="explicit"):
        parse_tokenizer_recipe(raw)
    recipe = parse_tokenizer_recipe(raw + "tokenizer: {vocab_size: 524288}\n")
    assert recipe.tokenizer == TokenizerV3Settings(524288)
    assert parse_tokenizer_recipe(dump_tokenizer_recipe(recipe)) == recipe
    old = TokenizerRecipe("a" * 64)
    # Captured on unmodified main09446b4 with tokenizers0.22.2 before this change.
    assert (
        tokenizer_recipe_digest(old)
        == "9226a30a804dc8b3c7f60d6efa69f3bca25936c8df870a83b9fd8d0c99151436"
    )
    assert (
        hashlib.sha256(dump_tokenizer_recipe(old).encode()).hexdigest()
        == "53a949a29b25419b81413fcd5069a4ecaba1e737c4e09c360728cf09c56e35fe"
    )
    for version, settings in (
        (1, TokenizerV3Settings(512)),
        (2, TokenizerSettings(512)),
    ):
        with pytest.raises(ConfigError, match="match"):
            TokenizerRecipe("a" * 64, schema_version=version, tokenizer=settings)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("digits", [False, True])
def test_legacy_payload_hashes_from_unchanged_main(version: int, digits: bool) -> None:
    hashes = {
        (1, False): "d218290ff23043874cfc9ca1b51defe140a233a61a2454829b3f01b4baeafb4d",
        (1, True): "ef65d66aac3eae0d66f7870d16e2cb4dcff0f7727d4263a10474f6a13c7c2185",
        (2, False): "b89997e958cfced270350e4f118f964b52682d68114453c1dc1fbb54549cc590",
        (2, True): "08e0d3f2eda176f7ebf326b1bd9d21c3b2a7f08c26c6ad68673c63d96a3c2876",
    }
    if version == 1:
        backend = _backend(digits)
        backend.train_from_iterator(
            ["aaa"],
            trainer=trainers.BpeTrainer(
                vocab_size=288,
                min_frequency=1,
                special_tokens=list(LEGACY_CONTROL_TOKENS),
                initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
                show_progress=False,
            ),
        )
        tokenizer = QveltTokenizer.from_bytes(backend.to_str().encode())
    else:
        tokenizer = train_tokenizer(["aaa"], TokenizerSettings(288, 1, digits))
    assert tokenizer.control_version == version
    assert len(tokenizer.control_tokens) == len(CONTROL_TOKENS) == 32
    assert hashlib.sha256(tokenizer.to_bytes()).hexdigest() == hashes[version, digits]


def test_v3_artifact_hash_across_processes_and_thread_counts() -> None:
    script = """
import hashlib,json
from pathlib import Path
from qvelt.tokenizer.core import TokenizerV3Settings,train_tokenizer
texts=[json.loads(s)['text'] for s in Path('tests/fixtures/tokenizer/train.jsonl').read_text().splitlines()]
t=train_tokenizer(texts,TokenizerV3Settings(512))
print(hashlib.sha256(t.to_bytes()).hexdigest())
"""
    hashes = []
    for seed, threads in (("1", "1"), ("47", "2"), ("99", "4")):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed, "RAYON_NUM_THREADS": threads},
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        hashes.append(result.stdout.strip())
    assert len(set(hashes)) == 1 and len(hashes[0]) == 64
