"""Packed adapter SFT uses genuine shared optimizer/checkpoint replay."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from qvelt.config.pretraining import load_pretraining_job
from qvelt.posttrain.smoke import run_sft_smoke
from qvelt.train.pretraining import train_pretraining
from qvelt.train.smoke import _equal
from qvelt.utils.checkpoints import latest_checkpoint


def test_adapter_sft_frozen_base_and_exact_resume(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    run_sft_smoke(tmp_path / "fixture", repository, steps=6)
    recipe = replace(
        load_pretraining_job(tmp_path / "fixture/sft.yaml"), lora_rank=2, lora_alpha=4
    )
    full = train_pretraining(recipe, tmp_path / "full", repository)
    train_pretraining(recipe, tmp_path / "replay", repository, stop_after=2)
    resumed = train_pretraining(recipe, tmp_path / "replay", repository, resume=True)
    left, right = (latest_checkpoint(tmp_path / name) for name in ("full", "replay"))
    weights = load_file(str(left / "model.safetensors"))
    _equal(weights, load_file(str(right / "model.safetensors")))
    assert recipe.initial_checkpoint is not None
    base = load_file(str(Path(recipe.initial_checkpoint).parent / "model.safetensors"))
    for name, value in base.items():
        assert torch.equal(weights[name], value)
    assert any(
        bool(value.abs().sum())
        for name, value in weights.items()
        if name.endswith(".b")
    )
    for filename in ("optimizer.pt", "rng-rank-00000.pt"):
        _equal(
            torch.load(left / filename, weights_only=True),
            torch.load(right / filename, weights_only=True),
        )
    assert json.loads((left / "application-rank-00000.json").read_text()) == json.loads(
        (right / "application-rank-00000.json").read_text()
    )
    assert full["final_training"] == resumed["final_training"]
    with pytest.raises(ValueError):
        replace(recipe, objective="pretraining")
