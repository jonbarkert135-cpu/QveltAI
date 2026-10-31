"""Strict YAML recipes reuse the experiment parser's trust-boundary checks."""

from pathlib import Path

import pytest

from qvelt.config import ConfigError
from qvelt.manifest_types import ManifestError
from qvelt.tokenization import TokenizerError, TokenizerSettings
from qvelt.tokenizer_recipe import (
    TokenizerRecipe,
    dump_tokenizer_recipe,
    load_tokenizer_recipe,
    parse_tokenizer_recipe,
    tokenizer_recipe_digest,
)

DIGEST = "a" * 64


def test_defaults_roundtrip_and_semantic_digest(tmp_path: Path) -> None:
    minimal = f"training_data_sha256: {DIGEST}\n"
    recipe = parse_tokenizer_recipe(minimal)
    assert recipe == TokenizerRecipe(DIGEST)
    assert parse_tokenizer_recipe(dump_tokenizer_recipe(recipe)) == recipe
    assert tokenizer_recipe_digest(
        parse_tokenizer_recipe("# comment\n" + minimal)
    ) == tokenizer_recipe_digest(recipe)
    path = tmp_path / "recipe.yaml"
    path.write_text(dump_tokenizer_recipe(recipe))
    assert load_tokenizer_recipe(path) == recipe
    assert tokenizer_recipe_digest(
        TokenizerRecipe(DIGEST, tokenizer=TokenizerSettings(512))
    ) != tokenizer_recipe_digest(recipe)


@pytest.mark.parametrize(
    "extra",
    [
        "schema_version: 2",
        "schema_version: true",
        'schema_version: "1"',
        "name: true",
        "name: ../unsafe",
        "unknown: value",
        "tokenizer: []",
        "tokenizer: {vocab_size: true}",
        "tokenizer: {vocab_size: 287}",
        "tokenizer: {min_frequency: 0}",
        "tokenizer: {min_frequency: 4294967296}",
        'tokenizer: {isolate_digits: "false"}',
        "tokenizer: {isolate_digits: yes}",
        "tokenizer: {typo: 1}",
        "tokenizer: {vocab_size: 512, vocab_size: 1024}",
        "name: first\nname: second",
        "name: &anchor abc",
        "name: !anything abc",
        "tokenizer: {<<: {vocab_size: 512}}",
    ],
)
def test_invalid_recipe(extra: str) -> None:
    with pytest.raises((ConfigError, TokenizerError, ManifestError)):
        parse_tokenizer_recipe(f"training_data_sha256: {DIGEST}\n" + extra)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "{}",
        "training_data_sha256: nope",
        "training_data_sha256: 42",
        "---\n{}\n---\n{}",
        "x: [",
    ],
)
def test_invalid_root(text: str) -> None:
    with pytest.raises((ConfigError, ManifestError)):
        parse_tokenizer_recipe(text)


def test_file_bounds_and_encoding(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_tokenizer_recipe(path)
    path.write_bytes(b" " * (128 * 1024 + 1))
    with pytest.raises(ConfigError, match="128 KiB"):
        load_tokenizer_recipe(path)
