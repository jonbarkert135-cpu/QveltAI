"""Independent single-factor recipe controls and exact seeded initialization."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType

import pytest
import torch

from qvelt.config.pretraining import PretrainingJob, TokenSchedule
from qvelt.config.training import load_training_job
from qvelt.eval.config import ArtifactRef

ROOT = Path(__file__).resolve().parents[2]


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rope_ablation", ROOT / "scripts/rope_ablation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe() -> PretrainingJob:
    job = load_training_job(ROOT / "configs/train/smoke.yaml")
    job = replace(job, training=replace(job.training, warmup_steps=0, decay_steps=0))
    return PretrainingJob(job, "data.json", "a" * 64, TokenSchedule(0, 1000, 1))


def test_only_rope_scale_and_derived_parent_can_change() -> None:
    module = script()
    base = recipe()
    variant = module.variant_recipe(base)
    expected = asdict(base)
    expected["job"]["experiment"]["model"]["rope_scale"] = 2.0
    assert asdict(variant) == expected
    with pytest.raises(ValueError, match="scale1"):
        module.variant_recipe(variant)
    sft = replace(
        base, objective="sft", initial_checkpoint="old.json", initial_sha256="b" * 64
    )
    with pytest.raises(ValueError, match="parent"):
        module.variant_recipe(sft)
    parent = ArtifactRef("new.json", "c" * 64)
    changed = module.variant_recipe(sft, parent)
    assert (
        changed.initial_checkpoint == parent.path
        and changed.initial_sha256 == parent.sha256
    )
    assert (
        changed.job.training == sft.job.training
        and changed.data_sha256 == sft.data_sha256
    )


def test_rope_changes_no_initial_weights_or_caller_rng() -> None:
    module = script()
    base = recipe()
    variant = module.variant_recipe(base)
    before = torch.get_rng_state().clone()
    assert len(module.initial_weights(base, variant)) == 64
    assert torch.equal(before, torch.get_rng_state())
    other_seed = replace(
        variant,
        job=replace(
            variant.job,
            experiment=replace(
                variant.job.experiment,
                runtime=replace(variant.job.experiment.runtime, seed=1),
            ),
        ),
    )
    with pytest.raises(ValueError, match="initial parameters"):
        module.initial_weights(base, other_seed)
