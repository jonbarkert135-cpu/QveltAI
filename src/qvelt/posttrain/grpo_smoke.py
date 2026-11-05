"""Fresh original SFT to outcome-GRPO with exact rollout/optimizer replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qvelt.posttrain.grpo_training import GRPOJob, dump_grpo, train_grpo
from qvelt.posttrain.smoke import run_sft_smoke
from qvelt.train.smoke import _equal
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.checkpoints import latest_checkpoint, source_digest
from qvelt.utils.manifests import manifest_digest


def run_grpo_smoke(
    directory: Path, repository: Path, *, steps: int = 8
) -> dict[str, Any]:
    if type(steps) is not int or not 3 <= steps <= 100:
        raise ValueError("GRPO smoke needs3..100 rollout windows")
    root = create_directory(directory)
    run_sft_smoke(root / "sft", repository, steps=80)
    cp = latest_checkpoint(root / "sft/continuous")
    source = ArtifactStore(cp).read_manifest()
    recipe = GRPOJob(
        str(root / "sft/sft.yaml"),
        hash_file(root / "sft/sft.yaml").sha256,
        str(cp / "manifest.json"),
        manifest_digest(source),
        steps=steps,
        checkpoint_every=max(1, steps // 2),
    )
    (root / "grpo.yaml").write_text(dump_grpo(recipe))
    full = train_grpo(recipe, root / "continuous", repository)
    stopped = train_grpo(recipe, root / "replayed", repository, stop_after=steps // 2)
    replay = train_grpo(recipe, root / "replayed", repository, resume=True)
    a, b = (latest_checkpoint(root / name) for name in ("continuous", "replayed"))
    left, right = (
        load_file(str(a / "model.safetensors")),
        load_file(str(b / "model.safetensors")),
    )
    _equal(left, right)
    initial = load_file(str(cp / "model.safetensors"))
    assert any(not torch.equal(value, initial[name]) for name, value in left.items()), (
        "No GRPO parameter update occurred"
    )
    for filename in ("optimizer.pt", "rng-rank-00000.pt"):
        _equal(
            torch.load(a / filename, weights_only=True),
            torch.load(b / filename, weights_only=True),
        )
    assert json.loads((a / "application-rank-00000.json").read_text()) == json.loads(
        (b / "application-rank-00000.json").read_text()
    )
    assert (
        full["losses_this_attempt"]
        == stopped["losses_this_attempt"] + replay["losses_this_attempt"]
    )
    assert (
        full["rollout_sha256_this_attempt"]
        == stopped["rollout_sha256_this_attempt"]
        + replay["rollout_sha256_this_attempt"]
    )
    assert full["counters"]["policy_updates"] > 0, (
        "Only zero-signal/KL windows occurred"
    )
    assert full["reference_frozen"] and replay["reference_frozen"]
    result = {
        "scope": full["scope"],
        "steps": steps,
        "source_sha256": source_digest(),
        "reference_sha256": recipe.reference_sha256,
        "sft_recipe_sha256": recipe.sft_recipe_sha256,
        "seed": recipe.seed,
        "groups": recipe.groups,
        "prompts_per_window": recipe.prompts,
        "max_tokens": recipe.max_tokens,
        "learning_rate": recipe.learning_rate,
        "clip": recipe.clip,
        "kl": recipe.kl,
        "counters": full["counters"],
        "initial_validation": full["initial_validation"],
        "final_validation": full["final_validation"],
        "all_losses_exact": True,
        "all_rollouts_exact": True,
        "weights_bit_exact": True,
        "optimizer_bit_exact": True,
        "rng_bit_exact": True,
        "consumer_state_exact": True,
        "reference_frozen": True,
        "loss_curve": full["losses_this_attempt"],
        "rollout_sha256": full["rollout_sha256_this_attempt"],
    }
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    return result
