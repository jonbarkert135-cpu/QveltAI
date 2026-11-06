"""Real fixed-budget SFT->DPO->native GRPO, including independent offline replay."""

from pathlib import Path

from qvelt.posttrain.phase6_smoke import run_phase6_smoke


def test_arithmetic_mechanism_reward_increases(tmp_path: Path) -> None:
    result = run_phase6_smoke(tmp_path / "control", Path(__file__).resolve().parents[2])
    assert (
        result["sft_steps"] == 20
        and result["dpo_steps"] == 4
        and result["rl_steps"] == 60
    )
    assert result["training_reward_last10"] > result["training_reward_first10"]
    assert result["validation_after"]["reward"] > result["validation_before"]["reward"]
    assert result["online_offline_weights_adam_rng_losses_exact"]
    assert result["rl"]["counters"]["policy_updates"] > 0
    assert result["final800_scored"] == 0
