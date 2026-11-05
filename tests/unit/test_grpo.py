"""Independent clipped-GRPO values/gradients and bounded group statistics."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from qvelt.data.math import MathProblem
from qvelt.posttrain.grpo import group_advantages, grpo_loss, verified_reward


def test_bounded_exact_verifier_deadline_and_nonfinite_contract() -> None:
    problem = MathProblem("two", "addition", "easy", "en", "1+1?", "2")
    assert verified_reward(problem, "4/2", "end_turn") == 1
    assert verified_reward(problem, "2", "length") == 0
    assert verified_reward(problem, "2.0", "end_turn") == 0
    assert verified_reward(problem, "x" * 1000000, "end_turn") == 0
    with patch("qvelt.posttrain.grpo.monotonic", side_effect=[0.0, 1.1]):
        with pytest.raises(TimeoutError):
            verified_reward(problem, "2", "end_turn")
    with patch.object(MathProblem, "reward", return_value=float("nan")):
        with pytest.raises(ValueError, match="finite"):
            verified_reward(problem, "2", "end_turn")


def test_group_normalization_matches_population_reference_and_constant_groups() -> None:
    raw = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    rewards = torch.tensor(raw, requires_grad=True)
    got = group_advantages(rewards)
    centered = raw - raw.mean(axis=1, keepdims=True)
    expected = centered / np.maximum(raw.std(axis=1, keepdims=True), 1e-6)
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-14)
    assert not got.requires_grad
    assert bool((got[2:] == 0).all())
    np.testing.assert_allclose(got[:2].mean(1).numpy(), [0, 0], atol=1e-15)


def test_clipped_objective_matches_numpy_and_analytic_policy_gradient() -> None:
    ratios = np.array([[[1.5, 0.5, 1.1], [1.5, 0.5, 0.9]]])
    old = np.full((1, 2, 3), -2.0)
    policy = old + np.log(ratios)
    reference = policy + np.array([[[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]]])
    advantage = np.array([[1.0, -1.0]])
    mask = np.array([[[True, True, False], [True, True, True]]])
    a = advantage[:, :, None]
    clipped = np.clip(ratios, 0.8, 1.2)
    surrogate = np.minimum(ratios * a, clipped * a)
    delta = reference - policy
    kl = np.expm1(delta) - delta
    weight = mask / mask.sum(-1, keepdims=True) / 2
    expected = ((-surrogate + 0.07 * kl) * weight).sum()
    current = torch.tensor(policy, requires_grad=True)
    previous = torch.tensor(old, requires_grad=True)
    ref = torch.tensor(reference, requires_grad=True)
    advantages = torch.tensor(advantage, requires_grad=True)
    loss = grpo_loss(
        current, previous, ref, advantages, torch.tensor(mask), clip=0.2, kl=0.07
    )
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-14)
    torch.autograd.backward(loss)
    active = ratios * a <= clipped * a
    gradient = (-ratios * a * active + 0.07 * (1 - np.exp(delta))) * weight
    assert current.grad is not None
    np.testing.assert_allclose(current.grad.numpy(), gradient, atol=1e-14)
    assert previous.grad is None and ref.grad is None and advantages.grad is None
    assert current.grad[0, 0, -1] == 0


def test_zero_advantage_groups_have_no_policy_gradient_without_kl() -> None:
    value = torch.zeros(2, 4, 3, requires_grad=True)
    loss = grpo_loss(
        value,
        value.detach(),
        value.detach(),
        torch.zeros(2, 4),
        torch.ones_like(value, dtype=torch.bool),
        kl=0,
    )
    torch.autograd.backward(loss)
    assert (
        float(loss.detach()) == 0
        and value.grad is not None
        and bool((value.grad == 0).all())
    )


def test_invalid_rewards_empty_masks_and_kl_overflow_fail() -> None:
    for rewards in (
        torch.ones(1, 1),
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[-1.0, 1.0]]),
    ):
        with pytest.raises(ValueError):
            group_advantages(rewards)
    policy = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError):
        grpo_loss(
            policy,
            policy,
            policy,
            torch.ones(1, 2),
            torch.zeros_like(policy, dtype=torch.bool),
        )
    with pytest.raises(FloatingPointError):
        grpo_loss(
            policy,
            policy,
            torch.full_like(policy, 1000),
            torch.ones(1, 2),
            torch.ones_like(policy, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        grpo_loss(
            policy,
            policy,
            policy,
            torch.ones(1, 2),
            torch.ones_like(policy, dtype=torch.bool),
            clip=0,
        )
