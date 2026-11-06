"""Checksum-bound GRPO rollout replay, with current-policy and reward revalidation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import torch

from qvelt.data.math import MathProblem
from qvelt.eval.config import ArtifactRef
from qvelt.model.decoder import Decoder
from qvelt.posttrain.grpo import Rollout, policy_digest, rollout_logps, verified_reward
from qvelt.posttrain.verifiers import AnswerSpec, RewardPolicy, score_completion
from qvelt.tokenizer.core import QveltTokenizer
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import _reject_constant, _unique_mapping


def read_rollout_file(reference: ArtifactRef) -> list[dict[str, Any]]:
    path = Path(reference.path)
    with ArtifactStore(path.parent).open_file(path.name) as stream:
        data = stream.read(8 * 1024**2 + 1)
    if len(data) > 8 * 1024**2 or hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError("Offline rollout byte limit/checksum mismatch")
    lines = data.splitlines()
    if not 1 <= len(lines) <= 100:
        raise ValueError("Offline rollout file requires1..100 windows")
    result = []
    for line in lines:
        value = json.loads(
            line, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
        )
        if not isinstance(value, dict) or set(value) != {
            "step",
            "policy_sha256",
            "reference_sha256",
            "problems",
            "rows",
        }:
            raise ValueError("Invalid offline window schema")
        result.append(value)
    return result


@torch.no_grad()
def replay_window(
    window: dict[str, Any],
    step: int,
    selected: list[MathProblem],
    groups: int,
    model: Decoder,
    tokenizer: QveltTokenizer,
    reference_sha: str,
    max_tokens: int,
    reward_policy: RewardPolicy | None,
) -> list[Rollout]:
    if (
        type(window["step"]) is not int
        or window["step"] != step
        or window["policy_sha256"] != policy_digest(model)
        or window["reference_sha256"] != reference_sha
        or window["problems"] != [asdict(p) for p in selected]
        or not isinstance(window["rows"], list)
        or len(window["rows"]) != len(selected) * groups
    ):
        raise ValueError(
            "Offline rollout policy/reference/window mismatch; stale off-policy data is unsupported"
        )
    rows = []
    training = model.training
    model.eval()
    try:
        for index, raw in enumerate(window["rows"]):
            problem = selected[index // groups]
            if not isinstance(raw, dict) or set(raw) != {
                f.name for f in fields(Rollout)
            }:
                raise ValueError("Invalid offline rollout schema")
            for key in ("prefix", "actions"):
                if (
                    not isinstance(raw[key], list)
                    or not 1
                    <= len(raw[key])
                    <= (2048 if key == "prefix" else max_tokens)
                    or any(
                        type(t) is not int or not 0 <= t < model.config.vocab
                        for t in raw[key]
                    )
                ):
                    raise ValueError("Invalid offline rollout token IDs")
            if (
                not isinstance(raw["old_logps"], list)
                or len(raw["old_logps"]) != len(raw["actions"])
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v > 0
                    for v in raw["old_logps"]
                )
            ):
                raise ValueError("Invalid behavior log probabilities")
            for key in ("reward", "correctness", "format_reward", "length_cost"):
                if (
                    type(raw[key]) not in (int, float)
                    or not math.isfinite(raw[key])
                    or not 0 <= raw[key] <= 1
                ):
                    raise ValueError("Invalid offline reward fields")
            prefix = (1, 4, *tokenizer.encode(problem.question), 6, 5)
            actions = tuple(raw["actions"])
            expected_seed = int(torch.randint(0, 2**32, ()).item())
            if (
                raw["prompt_id"] != problem.identifier
                or tuple(raw["prefix"]) != prefix
                or len(prefix) + len(actions) > model.config.context
                or type(raw["seed"]) is not int
                or raw["seed"] != expected_seed
                or any(t in (2, 6) for t in actions[:-1])
            ):
                raise ValueError("Offline prompt/seed/action identity mismatch")
            stop = (
                "end_turn"
                if actions[-1] == 6
                else "outer_eos"
                if actions[-1] == 2
                else "length"
            )
            if raw["stop"] != stop or stop == "length" and len(actions) != max_tokens:
                raise ValueError("Invalid offline completion stop")
            text = tokenizer.decode(actions[:-1] if stop != "length" else actions)
            if text != raw["text"]:
                raise ValueError(
                    "Offline human-readable audit disagrees with sampled actions"
                )
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
            if (
                raw["reward"],
                raw["correctness"],
                raw["format_reward"],
                raw["length_cost"],
            ) != (reward, correctness, formatted, cost):
                raise ValueError("Offline verifier/reward mismatch")
            row = Rollout(
                problem.identifier,
                prefix,
                actions,
                tuple(raw["old_logps"]),
                reward,
                stop,
                text,
                correctness,
                formatted,
                cost,
                expected_seed,
            )
            current, _ = rollout_logps(model, [row])
            values = current[0, : len(actions)]
            if not torch.allclose(
                values, torch.tensor(row.old_logps), rtol=1e-5, atol=1e-5
            ):
                raise ValueError(
                    "Offline behavior log probabilities disagree with current policy"
                )
            # Use canonical recomputed values; accepted rounding cannot alter updates.
            rows.append(replace(row, old_logps=tuple(values.tolist())))
    finally:
        model.train(training)
    return rows
