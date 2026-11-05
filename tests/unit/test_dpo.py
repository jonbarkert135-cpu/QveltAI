"""Independent sequence-probability and analytic DPO-gradient references."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qvelt.config import ConfigError
from qvelt.posttrain.dpo import dpo_loss, response_logps
from qvelt.posttrain.preferences import PreferencePair


def test_dpo_matches_numpy_value_and_closed_form_gradient() -> None:
    p = np.array([[-2.0, -4.0], [-5.0, -1.0], [-3.0, -3.0]], dtype=np.float64)
    r = np.array([[-3.0, -3.0], [-2.0, -4.0], [-3.0, -3.0]], dtype=np.float64)
    beta = 0.3
    margin = (p[:, 0] - p[:, 1]) - (r[:, 0] - r[:, 1])
    expected = np.logaddexp(0, -beta * margin)
    policy = torch.tensor(p, requires_grad=True)
    reference = torch.tensor(r, requires_grad=True)
    losses = dpo_loss(policy, reference, beta)
    np.testing.assert_allclose(losses.detach().numpy(), expected, rtol=1e-14)
    torch.autograd.backward(losses.sum())
    derivative = -beta / (1 + np.exp(beta * margin))
    assert policy.grad is not None
    np.testing.assert_allclose(
        policy.grad.numpy(), np.stack([derivative, -derivative], axis=1), rtol=1e-14
    )
    assert reference.grad is None
    assert float(losses[-1].detach()) == pytest.approx(np.log(2))


def test_response_sum_and_gradient_match_independent_numpy() -> None:
    raw = np.random.default_rng(17).normal(size=(2, 4, 5))
    labels = np.array([[0, 1, 3, 2], [3, 2, 1, 4]])
    mask = np.array([[False, True, False, False], [True, True, False, True]])
    prob = np.exp(raw - raw.max(axis=-1, keepdims=True))
    prob /= prob.sum(axis=-1, keepdims=True)
    expected = np.array(
        [
            sum(np.log(prob[b, t, labels[b, t]]) for t in range(4) if mask[b, t])
            for b in range(2)
        ]
    )
    logits = torch.tensor(raw, requires_grad=True)
    result = response_logps(logits, torch.tensor(labels), torch.tensor(mask))
    np.testing.assert_allclose(result.detach().numpy(), expected, rtol=1e-14)
    torch.autograd.backward(result.sum())
    grad = -prob
    for b in range(2):
        for t in range(4):
            grad[b, t, labels[b, t]] += 1
    grad *= mask[:, :, None]
    assert logits.grad is not None
    np.testing.assert_allclose(logits.grad.numpy(), grad, atol=1e-15)
    # Equal uniform logits scale with response length,not mean token likelihood.
    uniform = response_logps(
        torch.zeros(2, 4, 5), torch.tensor(labels), torch.tensor(mask)
    )
    torch.testing.assert_close(
        uniform, torch.tensor([-np.log(5), -3 * np.log(5)], dtype=torch.float32)
    )


@pytest.mark.parametrize("beta", [0.0, -1.0, float("nan"), float("inf"), True])
def test_dpo_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ConfigError):
        dpo_loss(torch.zeros(2, 2), torch.zeros(2, 2), beta)


def test_nonfinite_empty_and_shape_inputs_are_rejected() -> None:
    with pytest.raises(FloatingPointError):
        dpo_loss(torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, 2), 0.1)
    with pytest.raises(ValueError):
        dpo_loss(torch.zeros(2, 3), torch.zeros(2, 3), 0.1)
    with pytest.raises(ValueError):
        response_logps(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.int64),
            torch.zeros(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        response_logps(
            torch.zeros(1, 2, 3),
            torch.full((1, 2), 3, dtype=torch.int64),
            torch.ones(1, 2, dtype=torch.bool),
        )
    value = dpo_loss(
        torch.tensor([[-10000.0, 0.0], [0.0, -10000.0]]), torch.zeros(2, 2), 0.1
    )
    assert bool(torch.isfinite(value).all())
    assert value.tolist() == [1000.0, 0.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chosen": " "},
        {"chosen": "2", "rejected": "2"},
        {"family": ""},
        {"identifier": "bad\nidentifier"},
    ],
)
def test_pair_boundaries(kwargs: dict[str, str]) -> None:
    values = {
        "prompt": "1+1?",
        "chosen": "2",
        "rejected": "3",
        "family": "addition",
        "identifier": "one",
    } | kwargs
    with pytest.raises(ValueError):
        PreferencePair(**values)


def test_bf16_logps_accumulate_in_fp32() -> None:
    logits = torch.randn(1, 6, 7, dtype=torch.bfloat16)
    labels = torch.arange(6).reshape(1, 6)
    mask = torch.ones_like(labels, dtype=torch.bool)
    got = response_logps(logits, labels, mask)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, response_logps(logits.float(), labels, mask))
