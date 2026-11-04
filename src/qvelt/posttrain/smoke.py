"""Original verified math supervision, base-to-SFT optimization and exact replay."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qvelt.config.pretraining import TokenSchedule, dump_pretraining_job
from qvelt.data.demo import PERMISSION, authored_source, create_demo
from qvelt.data.math import generate_problems
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.posttrain.data import prepare_sft
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.train.smoke import _equal
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.checkpoints import latest_checkpoint
from qvelt.utils.manifests import manifest_digest


def math_supervision(count: int = 32) -> list[RawDocument]:
    """Gold values are independently checked by the original stdlib/SymPy generator."""
    documents = []
    for language in ("en", "ru"):
        for problem in generate_problems(count, 20261104, language):
            messages = [
                {"role": "user", "content": problem.question},
                {"role": "assistant", "content": problem.answer},
            ]
            documents.append(
                RawDocument(
                    json.dumps(
                        {"schema_version": 1, "messages": messages}, ensure_ascii=False
                    ),
                    problem.identifier,
                    problem.family,
                    language,
                    problem.difficulty,
                    problem.answer,
                )
            )
    return documents


def run_sft_smoke(
    directory: Path, repository: Path, *, steps: int = 80
) -> dict[str, Any]:
    if type(steps) is not int or steps < 3:
        raise ValueError("SFT replay requires at least three steps")
    root = create_directory(directory)
    create_demo(root / "base-data", repository, documents=32)
    base_recipe = demo_recipe(
        root / "base-data/prepared/manifest.json", repository, steps=3
    )
    base_recipe = replace(
        base_recipe,
        job=replace(
            base_recipe.job,
            experiment=replace(
                base_recipe.job.experiment,
                model=replace(base_recipe.job.experiment.model, context=128),
            ),
        ),
    )
    train_pretraining(base_recipe, root / "base", repository)
    checkpoint = latest_checkpoint(root / "base")
    initial = ArtifactStore(checkpoint).read_manifest()
    tokenizer_path = root / "base-data/prepared/tokenizer/manifest.json"
    tokenizer_manifest = ArtifactStore(tokenizer_path.parent).read_manifest()
    raw = write_raw_bundle(
        math_supervision(),
        root / "supervision",
        authored_source("sft-exact-math"),
        PERMISSION,
    )
    data = prepare_sft(
        root / "supervision/manifest.json",
        manifest_digest(raw),
        tokenizer_path,
        manifest_digest(tokenizer_manifest),
        root / "data",
        context=128,
        approved_for_training=True,
    )
    recipe = demo_recipe(
        root / "data/manifest.json", repository, steps=steps, accumulation=1
    )
    recipe = replace(
        recipe,
        objective="sft",
        initial_checkpoint=str(checkpoint / "manifest.json"),
        initial_sha256=manifest_digest(initial),
        schedule=TokenSchedule(0, 10**9, 1),
        job=replace(
            recipe.job, training=replace(recipe.job.training, learning_rate=0.003)
        ),
    )
    (root / "sft.yaml").write_text(dump_pretraining_job(recipe))
    full = train_pretraining(recipe, root / "continuous", repository)
    stop = steps // 3
    train_pretraining(recipe, root / "replayed", repository, stop_after=stop)
    replay = train_pretraining(recipe, root / "replayed", repository, resume=True)
    a, b = (latest_checkpoint(root / name) for name in ("continuous", "replayed"))
    _equal(
        load_file(str(a / "model.safetensors")), load_file(str(b / "model.safetensors"))
    )
    for filename in ("optimizer.pt", "rng-rank-00000.pt"):
        _equal(
            torch.load(a / filename, weights_only=True),
            torch.load(b / filename, weights_only=True),
        )
    state_a, state_b = (
        json.loads((p / "application-rank-00000.json").read_text()) for p in (a, b)
    )
    assert state_a == state_b
    logs = list((root / "continuous").glob("metrics-*.jsonl"))
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    losses = [r["loss"] for r in records if "loss" in r]
    assert losses[-1] < losses[0], "SFT did not learn its original training fixture"
    assert (
        full["final_training"]["validation_probe_loss"]
        < full["initial_training"]["validation_probe_loss"]
    ), "Fixed SFT train probe did not improve"
    result = {
        "scope": "original exact-answer demo SFT, not reasoning accuracy",
        "steps": steps,
        "stop_after": stop,
        "seed": recipe.job.experiment.runtime.seed,
        "initial_checkpoint_sha256": manifest_digest(initial),
        "data_sha256": manifest_digest(data),
        "recipe": asdict(recipe),
        "continuous": full,
        "replayed": replay,
        "first_training_loss": losses[0],
        "last_training_loss": losses[-1],
        "weights_bit_exact": True,
        "optimizer_bit_exact": True,
        "rng_bit_exact": True,
        "consumer_state_exact": True,
    }
    (root / "smoke-result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return result
