"""Actual post-save evaluation,crash repair,plots and unchanged optimization/RNG."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file

from qvelt.eval.config import EvalJob, dump_eval
from qvelt.eval.regression import evaluate_checkpoint, regression_policy
from qvelt.eval.runner import run_evaluation
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.train.smoke import _equal
from qvelt.utils.artifacts import hash_file
from qvelt.utils.checkpoints import latest_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def test_each_checkpoint_evaluated_and_crash_resume_preserves_training(
    tmp_path: Path,
) -> None:
    fixture = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    policy = replace(
        fixture, protocol="qvelt-eval-v2", max_cases=4, training_sources=()
    )
    path = tmp_path / "regression.yaml"
    path.write_text(dump_eval(policy))
    recipe = demo_recipe(
        tmp_path / "fixture/model-data/prepared/manifest.json", ROOT, steps=3
    )
    recipe = replace(
        recipe,
        job=replace(
            recipe.job,
            training=replace(
                recipe.job.training,
                checkpoint_every=1,
                regression_recipe=str(path),
                regression_sha256=hash_file(path).sha256,
            ),
        ),
    )
    train_pretraining(recipe, tmp_path / "full", ROOT)
    seen = False

    def fail_once(job: EvalJob, directory: Path, repository: Path) -> dict[str, Any]:
        nonlocal seen
        if directory.name == "step-00000002" and not seen:
            seen = True
            raise RuntimeError(
                "Deliberate evaluation interruption after checkpoint publication"
            )
        return run_evaluation(job, directory, repository)

    with patch("qvelt.eval.regression.run_evaluation", side_effect=fail_once):
        with pytest.raises(RuntimeError, match="Deliberate"):
            train_pretraining(recipe, tmp_path / "replay", ROOT)
        assert (tmp_path / "replay/checkpoints/step-00000002/manifest.json").exists()
        resumed = train_pretraining(recipe, tmp_path / "replay", ROOT, resume=True)
    assert resumed["resumed_from"] == 2
    ordinary = replace(
        recipe,
        job=replace(
            recipe.job,
            training=replace(
                recipe.job.training, regression_recipe=None, regression_sha256=None
            ),
        ),
    )
    train_pretraining(ordinary, tmp_path / "without-eval", ROOT)
    a, b, c = (
        latest_checkpoint(tmp_path / name)
        for name in ("full", "replay", "without-eval")
    )
    for other in (b, c):
        _equal(
            load_file(str(a / "model.safetensors")),
            load_file(str(other / "model.safetensors")),
        )
        for filename in ("optimizer.pt", "rng-rank-00000.pt"):
            _equal(
                torch.load(a / filename, weights_only=True),
                torch.load(other / filename, weights_only=True),
            )
    for name in ("full", "replay"):
        base = tmp_path / name / "regression"
        assert len(list(base.glob("step-*/manifest.json"))) == 3
        curve = json.loads((base / "curve-step-00000003.json").read_text())
        assert [r["step"] for r in curve["points"]] == [1, 2, 3]
        assert (
            (base / "curve-step-00000003.png")
            .read_bytes()
            .startswith(b"\x89PNG\r\n\x1a\n")
        )
    rng = torch.get_rng_state()
    evaluate_checkpoint(tmp_path / "full", 3, recipe.job, ROOT)
    assert torch.equal(rng, torch.get_rng_state())
    path.write_text(path.read_text() + "# changed bytes\n")
    with pytest.raises(ValueError, match="bytes changed"):
        regression_policy(recipe.job)
