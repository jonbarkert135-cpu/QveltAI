"""Seeded Unicode round trips and disk-backed packing against original document pairs."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qvelt.data.demo import authored_source
from qvelt.data.loader import PackedDataset
from qvelt.data.shards import ShardWriter
from qvelt.tokenizer.core import (
    QveltTokenizer,
    TokenizerError,
    TokenizerSettings,
    train_tokenizer,
)
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifest_types import DataInfo, Manifest


@pytest.fixture(scope="module")
def byte_tokenizer() -> QveltTokenizer:
    return train_tokenizer(
        ["Original property fixture. Русский текст. café 123 😀"],
        TokenizerSettings(vocab_size=320),
    )


def test_decode_queries_vocabulary_once(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = train_tokenizer(
        ["Example original text."], TokenizerSettings(vocab_size=320)
    )
    original = tokenizer._backend
    calls = []

    def vocabulary() -> int:
        calls.append(True)
        return int(original.get_vocab_size())

    monkeypatch.setattr(
        tokenizer,
        "_backend",
        SimpleNamespace(get_vocab_size=vocabulary, decode=original.decode),
    )
    tokens = original.encode("Original").ids
    assert tokenizer.decode(tokens * 128) == "Original" * 128
    assert len(calls) == 1
    with pytest.raises(TokenizerError):
        tokenizer.decode([*tokens, original.get_vocab_size()])


@settings(max_examples=200, derandomize=True, database=None, deadline=None)
@given(st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=512))
def test_unicode_roundtrip_has_no_implicit_controls(
    byte_tokenizer: QveltTokenizer, text: str
) -> None:
    tokens = byte_tokenizer.encode(text)
    assert byte_tokenizer.decode(tokens) == text
    assert all(32 <= token < byte_tokenizer.vocab_size for token in tokens)
    assert QveltTokenizer.from_bytes(byte_tokenizer.to_bytes()).encode(text) == tokens


@st.composite
def documents(draw: Any) -> tuple[int, int, bool, list[list[int]]]:
    context = draw(st.integers(2, 24))
    whole = draw(st.booleans())
    bodies = draw(
        st.lists(
            st.lists(
                st.integers(32, 65536),
                min_size=0,
                max_size=context - 1 if whole else 3 * context,
            ),
            min_size=1,
            max_size=6,
        )
    )
    return context, draw(st.integers(1, 64)), whole, [[1, *body, 2] for body in bodies]


@settings(max_examples=70, derandomize=True, database=None, deadline=None)
@given(documents())
def test_packing_preserves_pairs_and_document_boundaries(
    case: tuple[int, int, bool, list[list[int]]],
) -> None:
    context, shard_tokens, whole, frames = case
    with tempfile.TemporaryDirectory(prefix="qvelt-packing-property-") as folder:
        root = Path(folder)
        store = ArtifactStore(root)
        writer = ShardWriter(
            store,
            "shards/source/train",
            65537,
            context,
            shard_tokens,
            whole_documents=whole,
        )
        try:
            for index, frame in enumerate(frames):
                writer.add(frame, index, {"family": str(index)})
            descriptor = writer.finish()
        finally:
            writer.close()
        layout = {
            "schema_version": 1,
            "context": context,
            "vocab": 65537,
            "control_version": 2,
            "tokenizer_sha256": "a" * 64,
            "sources": {"source": {"train": descriptor}},
            "mixing": [],
        }
        record = store.add_bytes(
            "layout.json", json.dumps(layout).encode(), role="packing_layout"
        )
        store.publish(
            Manifest(
                "data",
                "property-packing",
                "2026-11-07T00:00:00Z",
                DataInfo(
                    "token_ids",
                    (authored_source("property-packing"),),
                    len(frames),
                    sum(map(len, frames)),
                    "a" * 64,
                ),
                (*writer.files, record),
            )
        )
        data = PackedDataset(root / "manifest.json")
        actual: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(frames))}
        block_for_document: dict[int, int] = {}
        for index in range(len(data)):
            batch = data[index]
            previous = -1
            position = 0
            for j in range(context):
                if not batch["valid_mask"][j]:
                    assert not batch["loss_mask"][j] and batch["document_ids"][j] == -1
                    continue
                doc = int(batch["document_ids"][j])
                if whole:
                    assert block_for_document.setdefault(doc, index) == index
                position = position + 1 if doc == previous else 0
                assert int(batch["positions"][j]) == position
                actual[doc].append(
                    (int(batch["input_ids"][j]), int(batch["labels"][j]))
                )
                previous = doc
        assert actual == {
            i: list(zip(frame[:-1], frame[1:], strict=True))
            for i, frame in enumerate(frames)
        }


def test_shard_writer_refuses_coordinate_replacement_and_oversized_whole_document() -> (
    None
):
    with tempfile.TemporaryDirectory(prefix="qvelt-shard-guards-") as directory:
        writer = ShardWriter(
            ArtifactStore(directory), "shards/source", 288, 2, 8, whole_documents=True
        )
        try:
            with pytest.raises(ValueError, match="storage coordinates"):
                writer.add([1, 32, 2], 0, {"shard": 99})
            with pytest.raises(ValueError, match="Whole document"):
                writer.add([1, 32, 33, 2], 0, {})
            with pytest.raises(ValueError, match="empty"):
                writer.finish()
        finally:
            writer.close()


def test_tokenizer_policy_bounds(
    byte_tokenizer: QveltTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokenizers import Tokenizer, models

    for bad in (
        {"min_frequency": 0},
        {"min_frequency": True},
        {"isolate_digits": "yes"},
    ):
        with pytest.raises(TokenizerError):
            TokenizerSettings(**bad)
    with pytest.raises(TokenizerError, match="Control token unavailable"):
        byte_tokenizer.control_id("<|unknown|>")
    with pytest.raises(TokenizerError, match="Unrecognized"):
        QveltTokenizer(Tokenizer(models.BPE(vocab={"unknown": 0}, merges=[])))
    with pytest.raises(TokenizerError, match="Only byte-BPE"):
        QveltTokenizer.from_bytes(
            Tokenizer(models.WordLevel({"unknown": 0})).to_str().encode()
        )
    monkeypatch.setattr("qvelt.tokenizer.core.MAX_TOKENIZER_BYTES", 16)
    with pytest.raises(TokenizerError, match="64 MiB"):
        QveltTokenizer.from_bytes(b"x" * 17)
