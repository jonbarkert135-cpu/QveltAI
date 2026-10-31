"""Stateless authored token sequences for engineering checks, not pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticBatch:
    """Sample identity determines an entire T+1 sequence before target shifting."""

    inputs: Tensor
    targets: Tensor
    sample_ids: Tensor


def synthetic_batch(
    step: int,
    batch_size: int,
    length: int,
    vocab: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device | None = None,
) -> SyntheticBatch:
    """Allocate disjoint same-topology sample IDs; reserve control IDs0–31."""
    if (
        any(
            type(value) is not int
            for value in (step, batch_size, length, vocab, rank, world_size)
        )
        or min(step, rank) < 0
        or min(batch_size, length, world_size) < 1
        or vocab < 288
        or rank >= world_size
    ):
        raise ValueError("Invalid synthetic batch dimensions or rank")
    ids = (step * world_size + rank) * batch_size + torch.arange(
        batch_size, device=device
    )
    full = 32 + (ids[:, None] + torch.arange(length + 1, device=device)) % (vocab - 32)
    return SyntheticBatch(full[:, :-1].contiguous(), full[:, 1:].contiguous(), ids)
