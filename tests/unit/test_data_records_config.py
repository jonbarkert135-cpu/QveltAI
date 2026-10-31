"""Raw metadata alignment, strict resolved recipes and control-version compatibility."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from tokenizers import AddedToken, pre_tokenizers, trainers

from qvelt.config import ConfigError
from qvelt.data.config import (
    MixStage,
    SourceSpec,
    dump_data_recipe,
    load_data_recipe,
    parse_data_recipe,
    recipe_digest,
)
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.records import RawDocument, iter_records, write_raw_bundle
from qvelt.tokenizer.core import (
    CONTROL_TOKENS,
    LEGACY_CONTROL_TOKENS,
    QveltTokenizer,
    TokenizerSettings,
    _backend,
    train_tokenizer,
)
from qvelt.utils.artifacts import ArtifactIntegrityError
from qvelt.utils.manifest_types import ArtifactFile
from qvelt.utils.manifests import manifest_bytes, manifest_digest


def test_raw_manifest_is_deterministic_and_metadata_roundtrips(tmp_path: Path) -> None:
    docs = [
        RawDocument("ё\r\n\t e\u0301 😀", "doc", "family", "ru", "hard", "2/3"),
        RawDocument("a second raw document", "two"),
    ]
    first = write_raw_bundle(
        iter(docs), tmp_path / "first", authored_source("original"), PERMISSION
    )
    second = write_raw_bundle(
        iter(docs), tmp_path / "second", authored_source("original"), PERMISSION
    )
    assert manifest_digest(first) == manifest_digest(second)
    assert (
        list(iter_records(tmp_path / "first/manifest.json", manifest_digest(first)))
        == docs
    )
    (tmp_path / "first/data.jsonl").write_text('{"text":"tampered"}\n')
    with pytest.raises(ArtifactIntegrityError):
        list(iter_records(tmp_path / "first/manifest.json", manifest_digest(first)))


@pytest.mark.parametrize(
    "bad",
    [
        b"",
        b'{"identifier":"x"}\n',
        b'{"identifier":"x","identifier":"y","family":"","language":"","difficulty":"","answer":""}\n',
    ],
)
def test_metadata_count_and_duplicate_fields_are_rejected(
    tmp_path: Path, bad: bytes
) -> None:
    manifest = write_raw_bundle(
        [RawDocument("document to check alignment")],
        tmp_path / "raw",
        authored_source("alignment"),
        PERMISSION,
    )
    path = tmp_path / "raw/documents.jsonl"
    path.write_bytes(bad)
    files = tuple(
        ArtifactFile(f.path, f.role, len(bad), hashlib.sha256(bad).hexdigest())
        if f.path == "documents.jsonl"
        else f
        for f in manifest.files
    )
    altered = replace(manifest, files=files)
    (tmp_path / "raw/manifest.json").write_bytes(manifest_bytes(altered))
    with pytest.raises((ValueError, ArtifactIntegrityError)):
        list(iter_records(tmp_path / "raw/manifest.json", manifest_digest(altered)))


def test_resolved_recipe_roundtrip_and_known_defaults(prepared_demo: Path) -> None:
    recipe = load_data_recipe(prepared_demo / "recipe.yaml")
    assert parse_data_recipe(dump_data_recipe(recipe)) == recipe
    assert recipe_digest(parse_data_recipe(dump_data_recipe(recipe))) == recipe_digest(
        recipe
    )
    assert replace(recipe, tokenizer=TokenizerSettings()).tokenizer.vocab_size == 32768
    assert [s.start_step for s in recipe.mixing] == [0, 5]
    for key, value in [
        ("ngram_size", True),
        ("context", "64"),
        ("validation_fraction", float("nan")),
    ]:
        raw: dict[str, Any] = yaml.safe_load(dump_data_recipe(recipe))
        raw[key] = value
        with pytest.raises(ValueError):
            parse_data_recipe(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        parse_data_recipe("schema_version: 1\n")
    with pytest.raises(ValueError):
        replace(recipe, purpose="production")
    with pytest.raises(ValueError):
        MixStage(0, (("source", -1),))
    with pytest.raises(ValueError):
        SourceSpec("source", "manifest.json", "a" * 64, False)
    with pytest.raises(ValueError):
        parse_data_recipe(dump_data_recipe(recipe) + "unknown: true\n")


def test_control_v2_plain_text_and_legacy_loading() -> None:
    tokenizer = train_tokenizer(
        ["Numbers 1234, русский текст and unicode 😀"] * 3,
        TokenizerSettings(vocab_size=320),
    )
    assert len(CONTROL_TOKENS) == len(LEGACY_CONTROL_TOKENS) == 32
    assert tokenizer.control_version == 2
    for token in (
        "<think>",
        "</think>",
        "<|tool_call|>",
        "<|end_tool_call|>",
        "<|tool_result|>",
    ):
        assert tokenizer.control_id(token) < 32
        assert tokenizer.decode(tokenizer.encode(token)) == token
        assert all(i >= 32 for i in tokenizer.encode(token))
    backend = _backend(True)
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        min_frequency=1,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[
            AddedToken(token, special=True, normalized=False)
            for token in LEGACY_CONTROL_TOKENS
        ],
    )
    backend.train_from_iterator(
        ["Legacy numbers 123 and кириллица"] * 3, trainer=trainer
    )
    legacy = QveltTokenizer.from_bytes(backend.to_str().encode())
    assert legacy.control_version == 1 and legacy.control_id("<|reserved_00|>") == 8
    assert legacy.control_id("<|bos|>") == 1
    with pytest.raises(ValueError):
        legacy.control_id("<think>")
    assert (
        legacy.decode(legacy.encode("<think>ё 123</think>")) == "<think>ё 123</think>"
    )


@pytest.mark.parametrize(
    "text",
    [
        "\ufeff e\u0301\u00a0\r\n\t\u200d",
        "Русская ёжиха съела ещё хлеба.\n",
        'def f(x):\n\treturn "a  b"\r\n',
        r"\frac{12}{37}+\sum_{i=1}^{20} i = 210",
        "العربية Ελληνικά 中文 👩\u200d💻",
    ],
)
def test_new_tokenizer_tricky_roundtrip(text: str) -> None:
    tokenizer = train_tokenizer(
        ["a small authored training example"] * 2, TokenizerSettings(vocab_size=300)
    )
    assert tokenizer.decode(tokenizer.encode(text)) == text
