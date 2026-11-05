"""Original verifier-labeled math preferences and exact DPO update replay."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qvelt.config.pretraining import load_pretraining_job
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.math import generate_problems
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.posttrain.dpo import DPOJob, dump_dpo, train_dpo
from qvelt.posttrain.preferences import load_preferences, prepare_preferences
from qvelt.posttrain.smoke import run_sft_smoke
from qvelt.train.smoke import _equal
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.checkpoints import latest_checkpoint, source_digest
from qvelt.utils.manifests import manifest_digest


def math_preferences() -> list[RawDocument]:
    documents = []
    for problem in generate_problems(32, 20261105, "en"):
        wrong = str(Fraction(problem.answer) + 1)
        assert problem.reward(problem.answer) == 1 and problem.reward(wrong) == 0
        row = {
            "schema_version": 1,
            "prompt": problem.question,
            "chosen": problem.answer,
            "rejected": wrong,
        }
        documents.append(
            RawDocument(
                json.dumps(row, sort_keys=True),
                problem.identifier,
                problem.family,
                "en",
                problem.difficulty,
                problem.answer,
            )
        )
    return documents


def run_preference_smoke(
    directory: Path, repository: Path, *, steps: int = 40
) -> dict[str, Any]:
    if type(steps) is not int or steps < 3:
        raise ValueError("Preference smoke requires at least three steps")
    root = create_directory(directory)
    run_sft_smoke(root / "sft", repository, steps=30)
    sft = load_pretraining_job(root / "sft/sft.yaml")
    checkpoint = latest_checkpoint(root / "sft/continuous")
    initial = ArtifactStore(checkpoint).read_manifest()
    token_path = root / "sft/data/tokenizer/manifest.json"
    tokenizer = ArtifactStore(token_path.parent).read_manifest()
    raw = write_raw_bundle(
        math_preferences(),
        root / "source",
        authored_source("verifier-math-preferences"),
        PERMISSION,
    )
    data = prepare_preferences(
        root / "source/manifest.json",
        manifest_digest(raw),
        token_path,
        manifest_digest(tokenizer),
        root / "pairs",
        context=sft.job.training.sequence_length,
        approved_for_training=True,
    )
    job = replace(
        sft.job,
        training=replace(
            sft.job.training,
            steps=steps,
            learning_rate=0.001,
            weight_decay=0.0,
            checkpoint_every=max(1, steps // 2),
        ),
    )
    recipe = DPOJob(
        job,
        str(root / "pairs/manifest.json"),
        manifest_digest(data),
        str(checkpoint / "manifest.json"),
        manifest_digest(initial),
    )
    (root / "dpo.yaml").write_text(dump_dpo(recipe))
    full = train_dpo(recipe, root / "continuous", repository)
    stopped = train_dpo(recipe, root / "replayed", repository, stop_after=steps // 3)
    replay = train_dpo(recipe, root / "replayed", repository, resume=True)
    a, b = (latest_checkpoint(root / name) for name in ("continuous", "replayed"))
    _equal(
        load_file(str(a / "model.safetensors")), load_file(str(b / "model.safetensors"))
    )
    for name in ("optimizer.pt", "rng-rank-00000.pt"):
        _equal(
            torch.load(a / name, weights_only=True),
            torch.load(b / name, weights_only=True),
        )
    assert json.loads((a / "application-rank-00000.json").read_text()) == json.loads(
        (b / "application-rank-00000.json").read_text()
    )
    assert (
        full["losses_this_attempt"]
        == stopped["losses_this_attempt"] + replay["losses_this_attempt"]
    )
    assert full["reference_frozen"] and replay["reference_frozen"]
    assert full["final_train"]["dpo_loss"] < full["initial_train"]["dpo_loss"]
    _, _, pairs = load_preferences(
        root / "pairs/manifest.json",
        manifest_digest(data),
        job.experiment.model.vocab,
        job.training.sequence_length,
    )
    result = {
        "scope": full["scope"],
        "steps": steps,
        "parameters": job.experiment.model.parameter_count,
        "source_sha256": source_digest(),
        "data_sha256": recipe.data_sha256,
        "reference_sha256": recipe.reference_sha256,
        "beta": recipe.beta,
        "learning_rate": job.training.learning_rate,
        "seed": job.experiment.runtime.seed,
        "split_pairs": {
            name: sum(p.split == name for p in pairs)
            for name in ("train", "validation", "test")
        },
        "initial_train": full["initial_train"],
        "final_train": full["final_train"],
        "initial_validation": full["initial_validation"],
        "final_validation": full["final_validation"],
        "processed_pairs": full["processed_pairs"],
        "processed_targets": full["processed_targets"],
        "weights_bit_exact": True,
        "optimizer_bit_exact": True,
        "rng_bit_exact": True,
        "consumer_state_exact": True,
        "all_losses_exact": True,
        "reference_frozen": full["reference_frozen"],
        "loss_curve": full["losses_this_attempt"],
    }
    (root / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return result
