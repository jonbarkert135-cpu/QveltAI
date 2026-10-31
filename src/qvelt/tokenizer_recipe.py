"""Strict, path-independent YAML recipes for byte-BPE artifact production."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import yaml

from qvelt.config import MAX_CONFIG_BYTES, ConfigError, parse_yaml_mapping
from qvelt.manifest_types import digest, identifier
from qvelt.tokenization import TokenizerSettings


@dataclass(frozen=True)
class TokenizerRecipe:
    """A resolved training policy pinned to one raw-data manifest identity."""

    training_data_sha256: str
    schema_version: int = 1
    name: str = "byte-bpe"
    tokenizer: TokenizerSettings = field(default_factory=TokenizerSettings)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigError("tokenizer recipe: only integer schema 1 is supported")
        identifier(self.name, "recipe.name")
        digest(self.training_data_sha256, "recipe.training_data_sha256")
        if not isinstance(self.tokenizer, TokenizerSettings):
            raise ConfigError("recipe.tokenizer must be validated TokenizerSettings")


def parse_tokenizer_recipe(text: str) -> TokenizerRecipe:
    """Apply defaults after strict YAML construction; reject all unknown fields."""
    root = parse_yaml_mapping(text)
    if set(root) - {"schema_version", "name", "training_data_sha256", "tokenizer"}:
        raise ConfigError("tokenizer recipe: unknown fields")
    settings = root.get("tokenizer", {})
    if not isinstance(settings, dict) or set(settings) - {
        "vocab_size",
        "min_frequency",
        "isolate_digits",
    }:
        raise ConfigError("recipe.tokenizer: expected known settings mapping")
    defaults = TokenizerSettings()
    return TokenizerRecipe(
        training_data_sha256=cast(str, root.get("training_data_sha256")),
        schema_version=cast(int, root.get("schema_version", 1)),
        name=cast(str, root.get("name", "byte-bpe")),
        tokenizer=TokenizerSettings(
            vocab_size=cast(int, settings.get("vocab_size", defaults.vocab_size)),
            min_frequency=cast(
                int, settings.get("min_frequency", defaults.min_frequency)
            ),
            isolate_digits=cast(
                bool, settings.get("isolate_digits", defaults.isolate_digits)
            ),
        ),
    )


def load_tokenizer_recipe(path: str | Path) -> TokenizerRecipe:
    """Read bounded UTF-8 YAML without initializing a model or touching a corpus."""
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("Tokenizer recipe exceeds 128 KiB")
    try:
        return parse_tokenizer_recipe(data.decode("utf-8-sig"))
    except UnicodeError as error:
        raise ConfigError("Tokenizer recipe must be UTF-8") from error


def dump_tokenizer_recipe(recipe: TokenizerRecipe) -> str:
    """Emit a deterministic resolved recipe, including defaults."""
    return yaml.safe_dump(asdict(recipe), sort_keys=True, allow_unicode=True)


def tokenizer_recipe_digest(recipe: TokenizerRecipe) -> str:
    """Identify meaningful settings independently of YAML formatting and paths."""
    data = json.dumps(asdict(recipe), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
