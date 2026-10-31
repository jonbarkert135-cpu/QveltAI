"""RFC AdamW grouping and token-count-based warmup/stable/linear-decay policy."""

from __future__ import annotations

import torch
from torch import nn

from qvelt.config import ModelConfig
from qvelt.config.training import TrainingConfig


def make_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    decay = []
    no_decay = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            raise ValueError("Parameter appears in more than one optimizer group")
        seen.add(id(parameter))
        if parameter.ndim == 1 or name.endswith("embedding.weight"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )


def learning_rate(
    config: TrainingConfig, processed: int, update_targets: int, total_targets: int
) -> float:
    """Synthetic fixed-size updates land exactly on all RFC token boundaries."""
    if (
        min(processed, update_targets, total_targets) < 0
        or update_targets == 0
        or processed + update_targets > total_targets
    ):
        raise ValueError("Invalid schedule token window")
    warmup = config.warmup_steps * update_targets
    tail = config.decay_steps * update_targets
    if processed < warmup:
        return config.learning_rate * (processed + update_targets) / warmup
    if tail and processed >= total_targets - tail:
        return config.learning_rate * (total_targets - processed) / tail
    return config.learning_rate


def flops_per_token(config: ModelConfig, length: int) -> int:
    """Analytical training-operation estimate; excludes optimizer, recompute and IO."""
    d, k = config.hidden, config.kv_heads * (config.hidden // config.heads)
    return (
        6
        * (
            config.vocab * d
            + config.layers * (2 * d * d + 2 * d * k + 3 * d * config.ffn)
        )
        + 12 * config.layers * length * d
    )
