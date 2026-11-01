"""Strict packed recipes, exact valid-token accumulation and explicit schedule boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from qvelt.config import ModelConfig
from qvelt.config.pretraining import (
    TokenSchedule,
    dump_pretraining_job,
    load_pretraining_job,
    parse_pretraining_job,
    pretraining_digest,
)
from qvelt.model.decoder import Decoder
from qvelt.train.accumulation import accumulate_gradients, token_learning_rate
from qvelt.train.pretraining import prepare_inputs
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.utils.distributed import DistributedContext
from qvelt.utils.manifest_types import ManifestError

ROOT = Path(__file__).resolve().parents[2]


def test_recipe_defaults_partial_window_and_runtime_only_changes(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = demo_recipe(
        prepared_demo / "prepared/manifest.json", ROOT, steps=3, accumulation=4
    )
    recipe = replace(recipe, total_batches=9)
    assert recipe.batches == 9
    assert parse_pretraining_job(dump_pretraining_job(recipe)) == recipe
    path = tmp_path / "job.yaml"
    path.write_text(dump_pretraining_job(recipe))
    changed = load_pretraining_job(path, ["workers=2", "prefetch=1"])
    assert changed.workers == 2 and pretraining_digest(changed) == pretraining_digest(
        recipe
    )
    assert pretraining_digest(
        replace(recipe, data_manifest="moved/data/manifest.json")
    ) == pretraining_digest(recipe)
    assert pretraining_digest(
        replace(recipe, schedule=TokenSchedule(0, 500, 200))
    ) != pretraining_digest(recipe)
    assert TokenSchedule(10**10, 2 * 10**10, 10**10).decay_tokens == 10**10


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("accumulation_steps", 0),
        ("workers", -1),
        ("workers", True),
        ("allow_demo", "true"),
        ("total_batches", 1),
        ("validation_every", 0),
        ("attention_mode", "unknown"),
    ],
)
def test_recipe_refuses_invalid_values(
    prepared_demo: Path, field: str, value: Any
) -> None:
    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=3)
    with pytest.raises(ValueError):
        replace(recipe, **{field: value})
    with pytest.raises(ValueError):
        parse_pretraining_job("schema_version: 1\n")
    with pytest.raises(ValueError):
        parse_pretraining_job(dump_pretraining_job(recipe) + "unknown: true\n")


def test_legacy_schedule_cannot_silently_compete(prepared_demo: Path) -> None:
    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=3)
    job = replace(recipe.job, training=replace(recipe.job.training, warmup_steps=1))
    with pytest.raises(ValueError, match="legacy"):
        replace(recipe, job=job)
    raw = yaml.safe_load(dump_pretraining_job(recipe))
    raw["schedule"]["unknown"] = 9
    with pytest.raises(ValueError):
        parse_pretraining_job(yaml.safe_dump(raw))


@pytest.mark.parametrize(
    "processed,update,expected",
    [
        (0, 5, 0.05),
        (5, 5, 0.1),
        (10, 5, 0.1),
        (15, 5, 0.1),
        (20, 5, 0.05),
        (25, 5, 0.0),
        (40, 1, 0.0),
    ],
)
def test_token_schedule_matches_declared_piecewise_function(
    processed: int, update: int, expected: float
) -> None:
    assert token_learning_rate(
        0.1, TokenSchedule(10, 20, 10), processed, update
    ) == pytest.approx(expected)
    with pytest.raises(ValueError):
        token_learning_rate(float("nan"), TokenSchedule(0, 10, 10), 0, 1)


def test_demo_permission_and_actual_data_shapes_are_enforced(
    prepared_demo: Path,
) -> None:
    recipe = demo_recipe(prepared_demo / "prepared/manifest.json", ROOT, steps=3)
    assert prepare_inputs(recipe).scope == "demo"
    with pytest.raises(ValueError, match="allow_demo"):
        prepare_inputs(replace(recipe, allow_demo=False))
    wrong = replace(
        recipe.job,
        experiment=replace(
            recipe.job.experiment,
            model=replace(
                recipe.job.experiment.model, vocab=recipe.job.experiment.model.vocab + 1
            ),
        ),
    )
    with pytest.raises(ValueError, match="vocabulary"):
        prepare_inputs(replace(recipe, job=wrong))
    with pytest.raises(ManifestError):
        prepare_inputs(replace(recipe, data_sha256="0" * 64))


def batch(tokens: list[list[int]], masks: list[list[bool]]) -> dict[str, Tensor]:
    ids = torch.tensor(tokens)
    return {
        "input_ids": ids,
        "labels": ids + 1,
        "positions": torch.arange(ids.shape[1]).expand_as(ids),
        "document_ids": torch.zeros_like(ids),
        "valid_mask": torch.ones_like(ids, dtype=torch.bool),
        "loss_mask": torch.tensor(masks),
    }


@pytest.mark.parametrize("windows", [1, 2, 3])
def test_accumulated_gradients_equal_one_global_masked_batch(windows: int) -> None:
    torch.set_num_threads(1)
    config = ModelConfig(
        layers=1, hidden=16, heads=2, kv_heads=1, ffn=32, vocab=288, context=4
    )
    actual, reference = Decoder(config), Decoder(config)
    reference.load_state_dict(actual.state_dict())
    batches = [
        batch([[32, 33, 34, 35]], [[True, True, True, True]]),
        batch(
            [[40, 41, 42, 43], [44, 45, 46, 47]],
            [[True, False, False, False], [True, True, False, False]],
        ),
        batch([[50, 51, 52, 53]], [[True, False, False, False]]),
    ][:windows]
    iterator = iter(batches)
    stats = accumulate_gradients(
        actual,
        lambda: next(iterator),
        windows,
        DistributedContext(0, 1, 0, torch.device("cpu")),
    )
    combined = {
        key: torch.cat([value[key] for value in batches], dim=0) for key in batches[0]
    }
    logits = reference(
        combined["input_ids"],
        combined["positions"],
        combined["document_ids"],
        combined["valid_mask"],
    )
    selected = combined["loss_mask"]
    expected = F.cross_entropy(logits[selected], combined["labels"][selected])
    torch.autograd.backward(expected)
    assert stats.targets == int(selected.sum()) and stats.microbatches == windows
    assert stats.loss == pytest.approx(float(expected.detach()), rel=1e-5)
    for a, b in zip(actual.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=5e-5, atol=5e-6)


def test_empty_window_and_rank_local_loader_failure_stop() -> None:
    model = Decoder(
        ModelConfig(
            layers=1, hidden=16, heads=2, kv_heads=1, ffn=32, vocab=288, context=4
        )
    )
    empty = batch([[32, 33, 34, 35]], [[False] * 4])
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    with pytest.raises(FloatingPointError, match="Empty global"):
        accumulate_gradients(model, lambda: empty, 1, context)

    def broken() -> dict[str, Tensor]:
        raise OSError("Deliberate read failure")

    with pytest.raises(RuntimeError, match="next training batch"):
        accumulate_gradients(model, broken, 1, context)
