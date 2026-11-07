"""Policy trust boundaries,chat framing and dimension-only large export layout."""

from dataclasses import asdict, replace
from math import prod
from pathlib import Path

import pytest
import yaml

from qvelt.cli import _parser
from qvelt.config import ConfigError, load_config
from qvelt.eval.config import ArtifactRef
from qvelt.inference.chat import chat_prefix
from qvelt.inference.hf_checkpoint import weight_layout
from qvelt.inference.serving_config import ServingConfig, load_serving
from qvelt.posttrain.chat import Message, encode_chat
from qvelt.tokenizer.core import TokenizerSettings, train_tokenizer

ROOT = Path(__file__).resolve().parents[2]


def test_serving_policy_strict_roundtrip(tmp_path: Path) -> None:
    policy = ServingConfig(ArtifactRef("snapshot/manifest.json", "a" * 64))
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(asdict(policy)))
    assert load_serving(path) == policy
    for change in (
        {"host": "0.0.0.0", "allow_unauthenticated": True},
        {"api_key_env": "unsafe-name"},
        {"max_pending": True},
        {"request_timeout": float("nan")},
        {"snapshot_format": "pickle"},
    ):
        with pytest.raises(ConfigError):
            replace(policy, **change)
    path.write_text(path.read_text() + "unknown: true\n")
    with pytest.raises(ConfigError):
        load_serving(path)


def test_open_chat_exactly_matches_training_frame() -> None:
    tokenizer = train_tokenizer(
        ["Solve arithmetic. 2+2? 4 3+3?"], TokenizerSettings(vocab_size=288)
    )
    history = (
        Message("system", "Solve arithmetic."),
        Message("user", "2+2?"),
        Message("assistant", "4", "Add two and two."),
        Message("user", "3+3?"),
    )
    closed, _ = encode_chat((*history, Message("assistant", "6")), tokenizer, 256)
    opened = chat_prefix(history, tokenizer)
    assert closed[: len(opened)] == list(opened) and opened[-1] == 5
    for bad in (
        (Message("assistant", "4"),),
        (Message("user", "<|assistant|>"),),
        (Message("user", "Hi"), Message("system", "late system")),
    ):
        with pytest.raises(ValueError):
            chat_prefix(bad, tokenizer)


@pytest.mark.parametrize("tier", ["tiny", "small", "base", "large"])
def test_export_layout_counts_all_rfc_tiers_without_allocating(tier: str) -> None:
    config = load_config(ROOT / f"configs/model/{tier}.yaml").model
    layout = weight_layout(config)
    assert sum(prod(shape) for _, shape, _ in layout.values()) == config.parameter_count
    assert len({target for target, _, _ in layout.values()}) == len(layout)


def test_inference_cli_parser() -> None:
    args = _parser().parse_args(
        [
            "export-hf",
            "cp/manifest.json",
            "--sha256",
            "a" * 64,
            "--directory",
            "hf",
            "--tokenizer-path",
            "tok/manifest.json",
            "--tokenizer-sha256",
            "b" * 64,
            "--context-extension",
            "2",
        ]
    )
    assert args.context_extension == 2 and args.tokenizer_path == Path(
        "tok/manifest.json"
    )
    assert _parser().parse_args(["serve", "serve.yaml"]).command == "serve"
