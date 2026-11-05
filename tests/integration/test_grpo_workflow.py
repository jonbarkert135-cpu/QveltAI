"""Actual on-policy RL/replay,masked behavior logps and adversarial verifier cases."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file

from qvelt.cli import main
from qvelt.config import ConfigError, ModelConfig
from qvelt.data.math import MathProblem
from qvelt.model.decoder import Decoder
from qvelt.posttrain.grpo import generate_rollout, policy_digest, rollout_logps
from qvelt.posttrain.grpo_smoke import run_grpo_smoke
from qvelt.posttrain.grpo_training import dump_grpo, load_grpo, train_grpo
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def grpo_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("grpo") / "smoke"
    run_grpo_smoke(root, ROOT, steps=4)
    return root


def test_actual_nonzero_policy_updates_and_exact_rollout_replay(grpo_run: Path) -> None:
    result = json.loads((grpo_run / "result.json").read_text())
    assert all(
        result[k]
        for k in (
            "weights_bit_exact",
            "optimizer_bit_exact",
            "rng_bit_exact",
            "consumer_state_exact",
            "all_losses_exact",
            "all_rollouts_exact",
            "reference_frozen",
        )
    )
    assert result["counters"]["policy_updates"] > 0
    assert result["counters"]["updates"] + result["counters"]["skipped"] == 4
    assert result["counters"]["tokens"] > 0
    assert len(result["rollout_sha256"]) == 4
    recipe = load_grpo(grpo_run / "grpo.yaml")
    assert recipe.groups == 4 and recipe.learning_rate == 1e-6
    assert recipe.sft_recipe_sha256


def test_all_zero_rewards_skip_adam_and_resume_without_fake_updates(
    grpo_run: Path, tmp_path: Path
) -> None:
    recipe = replace(
        load_grpo(grpo_run / "grpo.yaml"), steps=2, checkpoint_every=1, kl=0.0
    )
    directory = tmp_path / "zero-signal"
    with patch.object(MathProblem, "reward", return_value=0.0):
        stopped = train_grpo(recipe, directory, ROOT, stop_after=1)
        resumed = train_grpo(recipe, directory, ROOT, resume=True)
    assert stopped["counters"]["updates"] == 0
    assert (
        resumed["counters"]["skipped"] == 2
        and resumed["counters"]["policy_updates"] == 0
    )
    checkpoint = directory / "checkpoints/step-00000002"
    state = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    assert state["state"] == {}
    torch.testing.assert_close(
        load_file(str(checkpoint / "model.safetensors")),
        load_file(str(Path(recipe.reference_manifest).parent / "model.safetensors")),
        rtol=0,
        atol=0,
    )


def test_behavior_logps_match_naive_recomputation_and_rng_replay(
    grpo_run: Path,
) -> None:
    token_path = grpo_run / "sft/data/tokenizer/manifest.json"
    manifest = ArtifactStore(token_path.parent).read_manifest()
    tokenizer, _ = load_tokenizer_bundle(token_path, manifest_digest(manifest))
    torch.manual_seed(152)
    model = Decoder(
        ModelConfig(
            vocab=tokenizer.vocab_size,
            context=64,
            layers=1,
            hidden=16,
            heads=2,
            kv_heads=1,
            ffn=48,
        )
    )
    problem = MathProblem("one", "addition", "easy", "en", "1+1?", "2")
    snapshot = policy_digest(model)
    rng = torch.get_rng_state()
    first = generate_rollout(model, problem, tokenizer, 4)
    torch.set_rng_state(rng)
    second = generate_rollout(model, problem, tokenizer, 4)
    assert first == second
    assert model.training and policy_digest(model) == snapshot
    values, mask = rollout_logps(model, [first, second])
    assert int(mask.sum()) == len(first.actions) + len(second.actions)
    for index, row in enumerate((first, second)):
        torch.testing.assert_close(
            values[index, : len(row.actions)],
            torch.tensor(row.old_logps),
            rtol=1e-5,
            atol=1e-5,
        )
    assert not any(p.grad is not None for p in model.parameters())
    torch.autograd.backward(values.sum())
    assert any(p.grad is not None for p in model.parameters())


def test_complete_stop_required_and_no_reward_for_partial_correct_answer(
    grpo_run: Path,
) -> None:
    token_path = grpo_run / "sft/data/tokenizer/manifest.json"
    manifest = ArtifactStore(token_path.parent).read_manifest()
    tokenizer, _ = load_tokenizer_bundle(token_path, manifest_digest(manifest))
    model = Decoder(
        ModelConfig(
            vocab=tokenizer.vocab_size,
            context=64,
            layers=1,
            hidden=16,
            heads=2,
            kv_heads=1,
            ffn=48,
        )
    )
    problem = MathProblem("one", "constant", "easy", "en", "Give one.", "1")
    digit = tokenizer.encode("1")
    assert len(digit) == 1
    initial = 4 + len(tokenizer.encode(problem.question))

    def force_ends(stop: int) -> Callable[..., torch.Tensor]:
        def forward(ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            target = digit[0] if ids.shape[1] == initial else stop
            logits = torch.full((*ids.shape, tokenizer.vocab_size), -1000.0)
            logits[:, :, -1] = -1000.0
            logits[:, -1, target] = 0.0
            return logits

        return forward

    # Forced logits are an isolated verifier oracle,never used by actual RL training.
    with patch.object(model, "forward", side_effect=force_ends(6)):
        partial = generate_rollout(model, problem, tokenizer, 1, sample=False)
        completed = generate_rollout(model, problem, tokenizer, 2, sample=False)
    assert partial.reward == 0 and partial.stop == "length"
    assert completed.reward == 1 and completed.stop == "end_turn"
    with patch.object(model, "forward", side_effect=force_ends(2)):
        early = generate_rollout(model, problem, tokenizer, 2, sample=False)
    assert early.reward == 0 and early.stop == "outer_eos"
    for response in (
        '__import__("os").system("x")',
        "1+0",
        "1/0",
        "NaN",
        "inf",
        "1\nignore all rules",
        "9" * 1000000,
    ):
        assert problem.reward(response) == 0


def test_cli_update_recipe_change_and_config_limits(
    grpo_run: Path, tmp_path: Path
) -> None:
    recipe = load_grpo(grpo_run / "grpo.yaml")
    path = tmp_path / "grpo.yaml"
    path.write_text(dump_grpo(recipe))
    assert (
        main(
            [
                "grpo",
                str(path),
                "--directory",
                str(tmp_path / "run"),
                "--repository",
                str(ROOT),
                "--stop-after",
                "1",
            ]
        )
        == 0
    )
    assert (tmp_path / "run/checkpoints/step-00000001/manifest.json").exists()
    with pytest.raises(ValueError, match="recipe"):
        train_grpo(
            replace(recipe, learning_rate=2e-6), tmp_path / "run", ROOT, resume=True
        )
    with pytest.raises(ConfigError):
        replace(recipe, groups=1)
    with pytest.raises(ConfigError):
        replace(recipe, max_tokens=65)
    with pytest.raises(ConfigError):
        replace(recipe, steps=101)
    path.write_text(dump_grpo(recipe) + "unknown: 1\n")
    assert main(["grpo", str(path), "--directory", str(tmp_path / "invalid")]) == 1
