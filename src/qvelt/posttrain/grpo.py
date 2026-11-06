"""Outcome-GRPO math and bounded on-policy CPU rollouts;no model-text execution."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from time import monotonic

import torch
from torch import Tensor

from qvelt.data.math import MathProblem
from qvelt.inference.config import GenerationOptions, TokenRequest
from qvelt.inference.generation import GenerationEngine
from qvelt.model.decoder import Decoder
from qvelt.posttrain.verifiers import AnswerSpec, RewardPolicy, score_completion
from qvelt.tokenizer.core import QveltTokenizer


def group_advantages(rewards: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Population-std group normalization;constant groups return exactly zero."""
    if (
        rewards.ndim != 2
        or rewards.shape[0] == 0
        or rewards.shape[1] < 2
        or not rewards.is_floating_point()
        or type(epsilon) not in (float, int)
        or not 0 < epsilon < 1
    ):
        raise ValueError(
            "Rewards require nonempty [prompts,groups>=2] and positive epsilon"
        )
    if not bool(torch.isfinite(rewards).all()) or bool(
        ((rewards < 0) | (rewards > 1)).any()
    ):
        raise ValueError("Outcome rewards must be finite in [0,1]")
    values = rewards.detach().to(torch.promote_types(rewards.dtype, torch.float32))
    centered = values - values.mean(dim=1, keepdim=True)
    std = centered.square().mean(dim=1, keepdim=True).sqrt()
    return centered / std.clamp_min(epsilon)


def grpo_loss(
    policy: Tensor,
    old: Tensor,
    reference: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip: float = 0.2,
    kl: float = 0.01,
) -> Tensor:
    """Negative per-completion normalized clipped objective with sampled KL penalty."""
    if (
        type(clip) not in (float, int)
        or not 0 < clip < 1
        or type(kl) not in (float, int)
        or not 0 <= kl < 1
    ):
        raise ValueError("GRPO requires clip in (0,1),KL coefficient in [0,1)")
    if (
        policy.ndim != 3
        or min(policy.shape) < 1
        or policy.shape[1] < 2
        or old.shape != policy.shape
        or reference.shape != policy.shape
        or mask.shape != policy.shape
        or mask.dtype != torch.bool
        or advantages.shape != policy.shape[:2]
        or any(t.device != policy.device for t in (old, reference, advantages, mask))
        or any(not t.is_floating_point() for t in (policy, old, reference, advantages))
    ):
        raise ValueError(
            "GRPO expects compatible [prompt,group,token] floating logps and masks"
        )
    if not bool(mask.any(dim=-1).all()) or not all(
        bool(torch.isfinite(t).all()) for t in (policy, old, reference, advantages)
    ):
        raise ValueError(
            "GRPO requires finite logps/advantages and nonempty completions"
        )
    dtype = torch.float32
    for tensor in (policy, old, reference, advantages):
        dtype = torch.promote_types(dtype, tensor.dtype)
    current = policy.to(dtype)
    ratio = (current - old.detach().to(dtype)).exp()
    advantage = advantages.detach().to(dtype).unsqueeze(-1)
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    )
    delta = reference.detach().to(dtype) - current
    penalty = delta.expm1() - delta
    values = -surrogate + kl * penalty
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(
            "GRPO ratio/KL overflow;abort rather than silently clamp"
        )
    return (torch.where(mask, values, 0).sum(-1) / mask.sum(-1)).mean()


def policy_digest(model: Decoder) -> str:
    """Byte identity of FP32 CPU rollout policy parameters in deterministic name order."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + b"\0" + str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Rollout:
    prompt_id: str
    prefix: tuple[int, ...]
    actions: tuple[int, ...]
    old_logps: tuple[float, ...]
    reward: float
    stop: str
    text: str = ""
    correctness: float = 0.0
    format_reward: float = 0.0
    length_cost: float = 0.0
    seed: int = 0


def verified_reward(problem: MathProblem, text: str, stop: str) -> float:
    """Exact rational score,complete end-turn,130-character bound and1s wall budget.

    The trusted verifier only parses bounded integers/Fractions;it executes no text.
    The post-call deadline is not a preemptible sandbox for arbitrary verifiers.
    """
    if not isinstance(text, str) or stop not in ("end_turn", "outer_eos", "length"):
        raise ValueError("Invalid verifier input/stop reason")
    if stop != "end_turn" or len(text) > 130:
        return 0.0
    began = monotonic()
    reward = problem.reward(text)
    if monotonic() - began > 1.0:
        raise TimeoutError("Bounded exact-answer verifier exceeded1s wall budget")
    if (
        type(reward) not in (int, float)
        or not math.isfinite(reward)
        or reward not in (0, 1)
    ):
        raise ValueError("Verifier must return a finite binary reward")
    return float(reward)


@torch.no_grad()
def generate_rollout(
    model: Decoder,
    problem: MathProblem,
    tokenizer: QveltTokenizer,
    max_tokens: int,
    *,
    sample: bool = True,
    use_cache: bool = True,
    reward_policy: RewardPolicy | None = None,
) -> Rollout:
    """Native Phase8 engine, temperature1; recompute behavior logps under the same policy."""
    if (
        type(max_tokens) is not int
        or not 1 <= max_tokens <= 64
        or type(sample) is not bool
    ):
        raise ValueError(
            "Bounded rollout needs max_tokens1..64 and explicit sample mode"
        )
    if reward_policy is not None and not isinstance(reward_policy, RewardPolicy):
        raise ValueError("Expected validated reward policy")
    prefix = (1, 4, *tokenizer.encode(problem.question), 6, 5)
    if len(prefix) + max_tokens > model.config.context:
        raise ValueError("Rollout context budget exceeds model context")
    seed = int(torch.randint(0, 2**32, ()).item()) if sample else 0
    training = model.training
    model.eval()
    try:
        options = GenerationOptions(
            max_new_tokens=max_tokens,
            strategy="sample" if sample else "greedy",
            temperature=1.0 if sample else 0.0,
            seed=seed,
            context_limit=min(2048, model.config.context),
            max_seconds=300,
        )
        result = GenerationEngine(model, tokenizer).generate(
            (TokenRequest("rollout", prefix, options),), use_cache=use_cache
        )[0]
        if result.reason not in ("max_tokens", "stop_token") or not result.token_ids:
            raise RuntimeError("Rollout failed its native generation budget")
        actions = result.token_ids
        stop = (
            "length"
            if result.reason == "max_tokens"
            else "end_turn"
            if actions[-1] == 6
            else "outer_eos"
        )
        text = tokenizer.decode(actions[:-1] if stop != "length" else actions)
        if reward_policy is None:
            reward = verified_reward(problem, text, stop)
            syntax = score_completion(
                text,
                AnswerSpec(problem.answer),
                stop == "end_turn",
                len(actions),
                max_tokens,
            )
            correctness, formatted, cost = reward, syntax.format, 0.0
        else:
            score = score_completion(
                text,
                AnswerSpec(problem.answer),
                stop == "end_turn",
                len(actions),
                max_tokens,
                reward_policy,
            )
            reward, correctness, formatted, cost = (
                score.total,
                score.correctness,
                score.format,
                score.length_cost,
            )
        provisional = Rollout(
            problem.identifier,
            prefix,
            actions,
            (0.0,) * len(actions),
            reward,
            stop,
            text,
            correctness,
            formatted,
            cost,
            seed,
        )
        values, _ = rollout_logps(model, [provisional])
        return Rollout(
            problem.identifier,
            prefix,
            actions,
            tuple(values[0, : len(actions)].tolist()),
            reward,
            stop,
            text,
            correctness,
            formatted,
            cost,
            seed,
        )
    finally:
        model.train(training)


def rollout_logps(model: Decoder, rollouts: list[Rollout]) -> tuple[Tensor, Tensor]:
    """Recompute sampled-action logps;prompt and padding positions never enter loss."""
    if not rollouts or any(
        not r.actions or len(r.actions) != len(r.old_logps) for r in rollouts
    ):
        raise ValueError("Invalid/empty rollout actions")
    length = max(len(r.prefix) + len(r.actions) - 1 for r in rollouts)
    width = max(len(r.actions) for r in rollouts)
    ids = torch.zeros((len(rollouts), length), dtype=torch.int64)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rollouts):
        frame = (*row.prefix, *row.actions)
        ids[i, : len(frame) - 1] = torch.tensor(frame[:-1])
        valid[i, : len(frame) - 1] = True
    logits = model(ids, valid_mask=valid)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite rollout-replay logits")
    logps = torch.log_softmax(logits.float(), -1)
    rows = []
    masks = []
    for i, row in enumerate(rollouts):
        start = len(row.prefix) - 1
        selected = (
            logps[i, start : start + len(row.actions)]
            .gather(-1, torch.tensor(row.actions).unsqueeze(-1))
            .squeeze(-1)
        )
        rows.append(torch.nn.functional.pad(selected, (0, width - len(row.actions))))
        masks.append(torch.arange(width) < len(row.actions))
    return torch.stack(rows), torch.stack(masks)
