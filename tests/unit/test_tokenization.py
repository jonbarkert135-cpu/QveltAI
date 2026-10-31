"""Authored fixtures test contracts, not language or mathematical quality."""

from pathlib import Path

import pytest

from qvelt.tokenizer.core import (
    CONTROL_TOKENS,
    QveltTokenizer,
    TokenizerError,
    TokenizerSettings,
    train_tokenizer,
)

CORPUS = (
    "A small authored corpus. Spaces and\nnewlines remain.\n",
    "Привет! Проверяем русский текст и числа 1234567890.",
    r"\frac{a+b}{2} = 17.5; x^2 + y^2 = z^2.",
) * 4


@pytest.fixture(scope="module")
def tokenizer() -> QveltTokenizer:
    return train_tokenizer(CORPUS, TokenizerSettings(vocab_size=384))


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\t\r\n  ",
        "Hello, world!",
        "Привет, ёж!",
        "αβγ 中文 日本語 🙂🚀",
        "e\u0301 é",
        "\x00\x01\x7f",
        r"\int_0^1 x^2 dx = \frac{1}{3}",
        "1234567890 -17.25 1e-9",
        "".join(chr(i) for i in range(256)),
        "<|user|>hello<|eos|>",
        "\U0010ffff",
    ],
)
def test_round_trip(tokenizer: QveltTokenizer, text: str) -> None:
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert all(i >= 32 for i in ids)


def test_control_ids(tokenizer: QveltTokenizer) -> None:
    assert len(CONTROL_TOKENS) == 32
    for i, token in enumerate(CONTROL_TOKENS):
        assert tokenizer.decode([i]) == token
        assert i not in tokenizer.encode(token)


def test_digits_isolated(tokenizer: QveltTokenizer) -> None:
    assert tokenizer.encode("123456") == sum(
        (tokenizer.encode(i) for i in "123456"), []
    )


def test_save_load_and_no_overwrite(tokenizer: QveltTokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = QveltTokenizer.load(path)
    assert restored.to_bytes() == tokenizer.to_bytes()
    for text in CORPUS:
        assert restored.encode(text) == tokenizer.encode(text)
    with pytest.raises(FileExistsError):
        tokenizer.save(path)


def test_reproducible() -> None:
    settings = TokenizerSettings(vocab_size=384)
    a = train_tokenizer(iter(CORPUS), settings)
    b = train_tokenizer(iter(CORPUS), settings)
    assert a.to_bytes() == b.to_bytes()
    assert a.vocab_size <= settings.vocab_size


@pytest.mark.parametrize("digits", [False, True])
def test_small_vocab(tmp_path: Path, digits: bool) -> None:
    tokenizer = train_tokenizer(["aaa"], TokenizerSettings(288, 1, digits))
    assert tokenizer.vocab_size == 288
    assert tokenizer.decode(tokenizer.encode("Unseen 🧮 Я 123")) == "Unseen 🧮 Я 123"
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert QveltTokenizer.load(path).vocab_size == 288


@pytest.mark.parametrize("ids", [[-1], [True], [999999]])
def test_invalid_ids(tokenizer: QveltTokenizer, ids: list[int]) -> None:
    with pytest.raises(TokenizerError):
        tokenizer.decode(ids)


@pytest.mark.parametrize("corpus", [[], [""], ["\ud800"], ["<|eos|>"]])
def test_invalid_training(corpus: list[str]) -> None:
    with pytest.raises(TokenizerError, match="nonempty|surrogates|control"):
        train_tokenizer(corpus)


def test_invalid_unicode(tokenizer: QveltTokenizer) -> None:
    with pytest.raises(TokenizerError, match="surrogates"):
        tokenizer.encode("\udfff")
    with pytest.raises(TokenizerError, match="limit"):
        tokenizer.encode("a" * (1024 * 1024 + 1))


@pytest.mark.parametrize("size", [True, 0, 287, 262145])
def test_invalid_settings(size: int) -> None:
    with pytest.raises(TokenizerError):
        TokenizerSettings(vocab_size=size)


def test_reject_modified_policy(tokenizer: QveltTokenizer, tmp_path: Path) -> None:
    import json

    data = json.loads(tokenizer.to_bytes())
    data["normalizer"] = {"type": "Lowercase"}
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TokenizerError, match="policy"):
        QveltTokenizer.load(path)
    path.write_text("bad json")
    with pytest.raises(TokenizerError, match="Invalid"):
        QveltTokenizer.load(path)
