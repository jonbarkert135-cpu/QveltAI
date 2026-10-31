"""Configuration validation, bounded YAML parsing and analytical parity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path
from typing import cast

import pytest
import yaml

from qvelt.config import (
    MAX_CONFIG_BYTES,
    MAX_DEPTH,
    MAX_EVENTS,
    ConfigError,
    Device,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    config_digest,
    dump_config,
    load_config,
    parse_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_valid_immutable_and_round_trip() -> None:
    config = parse_config("{}")
    assert config == ExperimentConfig()
    assert parse_config(dump_config(config)) == config
    for item in fields(config.model):
        with pytest.raises(FrozenInstanceError):
            setattr(config.model, item.name, getattr(config.model, item.name))


def test_hand_counted_parameters() -> None:
    # Embedding 1200; Q/O 32; K/V 16; MLP 72; block gains 8; final gain 4.
    config = ModelConfig(layers=1, hidden=4, heads=2, kv_heads=1, ffn=6, vocab=300)
    assert config.parameter_count == 1332


@pytest.mark.parametrize("name", ["tiny", "small", "base", "large"])
def test_presets_match_research_dimensions_and_unique_counts(name: str) -> None:
    config = load_config(ROOT / "configs" / f"{name}.yaml")
    report = json.loads((ROOT / "docs/research/estimates.json").read_text())[name]
    assert config.model.parameter_count == report["parameters"]
    for field in ("layers", "hidden", "heads", "kv_heads", "ffn", "vocab", "context"):
        assert getattr(config.model, field) == report[field]


def test_laptop_preset_changes_context_not_parameter_storage() -> None:
    config = load_config(ROOT / "configs/laptop-debug.yaml")
    assert config.model.context == 128
    assert config.model.parameter_count == ModelConfig().parameter_count
    assert config.runtime.device == "cuda"  # Requested, not validated hardware.


@pytest.mark.parametrize(
    "text",
    [
        "",
        "---\n",
        "[]",
        "42",
        "model: []",
        "runtime: null",
        "model: {other: 1}",
        "optimizer: {}",
        "schema_version: 2",
        "schema_version: true",
        "schema_version: '1'",
        "model: {layers: true}",
        "model: {layers: 0}",
        "model: {layers: -1}",
        "model: {layers: '8'}",
        "model: {layers: 010}",
        "model: {layers: 0x10}",
        "model: {heads: 0}",
        "model: {heads: 3}",
        "model: {kv_heads: 2}",
        "model: {hidden: 15}",
        "model: {vocab: 287}",
        "model: {norm_eps: false}",
        "model: {norm_eps: 0}",
        "model: {norm_eps: -.inf}",
        "model: {norm_eps: .nan}",
        "model: {norm_eps: 1e999}",
        "model: {rope_theta: 1}",
        "runtime: {seed: -1}",
        "runtime: {seed: 4294967296}",
        "runtime: {seed: false}",
        "runtime: {deterministic: yes}",
        "runtime: {deterministic: 'true'}",
        "runtime: {device: cuda:0}",
        "runtime: {precision: fp16}",
        "runtime: {precision: bf16}",
        "runtime: {log_level: info}",
        "name: 123",
        "name: ../path",
        "name: ''",
        "name: [bad]",
    ],
)
def test_reject_invalid_schema_and_scalar_types(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_config(text)


@pytest.mark.parametrize(
    "text",
    [
        "name: first\nname: second",
        "model: {layers: 1, layers: 2}",
        "model: &model {layers: 1}",
        "model: *undefined",
        "name: !!str explicit",
        "!!python/object/apply:os.system ['false']",
        "model: {<<: {layers: 1}}",
        "1: bad",
        "? [bad, key]\n: value",
        "{}\n---\n{}",
        "model: [",
    ],
)
def test_reject_ambiguous_or_unsafe_yaml(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_config(text)


def test_bounded_parse_and_bounded_file_read(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="128 KiB"):
        parse_config(" " * (MAX_CONFIG_BYTES + 1))
    path = tmp_path / "huge.yaml"
    path.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ConfigError, match="128 KiB"):
        load_config(path)
    with pytest.raises(ConfigError, match="nesting"):
        parse_config("[" * (MAX_DEPTH + 1) + "]" * (MAX_DEPTH + 1))
    with pytest.raises(ConfigError, match="too many"):
        parse_config("[" + ",".join("1" for _ in range(MAX_EVENTS)) + "]")
    with pytest.raises(ConfigError):
        parse_config("model: {layers: " + "9" * 5000 + "}")


def test_utf8_and_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("\ufeffname: utf8-bom\n", encoding="utf-8")
    assert load_config(path).name == "utf8-bom"
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(path)
    with pytest.raises(ConfigError, match="UTF-8"):
        parse_config("\ud800")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_scalar_policy_does_not_mutate_pyyaml_global_loader() -> None:
    assert parse_config("name: on").name == "on"
    assert parse_config("name: 2026-10-31").name == "2026-10-31"
    assert yaml.safe_load("flag: on") == {"flag": True}
    config = parse_config("model: {norm_eps: 1e-6, rope_theta: 1e4}")
    assert config.model.norm_eps == 1e-6
    assert config.model.rope_theta == 10000.0


def test_direct_constructors_enforce_invariants() -> None:
    with pytest.raises(ConfigError):
        ModelConfig(layers=True)
    with pytest.raises(ConfigError):
        ModelConfig(norm_eps=float("nan"))
    with pytest.raises(ConfigError):
        ModelConfig(rope_theta=10**1000)
    with pytest.raises(ConfigError):
        RuntimeConfig(device=cast(Device, "unknown"))
    with pytest.raises(ConfigError):
        RuntimeConfig(deterministic=cast(bool, 1))
    with pytest.raises(ConfigError):
        ExperimentConfig(model=cast(ModelConfig, {}))


def test_canonical_digest_uses_resolved_defaults_not_input_order() -> None:
    a = parse_config("name: sample\nmodel: {norm_eps: 0.000001}\n")
    b = parse_config("model: {norm_eps: 1e-6}\nname: sample # comment\n")
    assert config_digest(a) == config_digest(b)
    canonical = json.dumps(
        asdict(a),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )
    assert config_digest(a) == hashlib.sha256(canonical.encode()).hexdigest()
    assert config_digest(a) != config_digest(replace(a, name="different"))
    assert config_digest(
        ExperimentConfig(model=ModelConfig(rope_theta=10000))
    ) == config_digest(parse_config("model: {rope_theta: 10000}"))


def test_invalid_values_are_not_echoed_in_errors() -> None:
    secret = "do-not-echo-this-value"
    for text in (f"model: {{layers: '{secret}'}}", f"name: [{secret}"):
        with pytest.raises(ConfigError) as caught:
            parse_config(text)
        assert secret not in str(caught.value)


def test_dimension_bounds_and_full_uint32_seed_range() -> None:
    with pytest.raises(ConfigError):
        parse_config("model: {layers: 2147483648}")
    assert parse_config("runtime: {seed: 4294967295}").runtime.seed == 2**32 - 1
