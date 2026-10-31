"""FP32 masked next-target loss with explicit sum/count normalization."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def causal_loss(
    logits: Tensor, targets: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Targets are already shifted within documents; exclude PAD, BOS and -100."""
    if (
        logits.ndim != 3
        or targets.shape != logits.shape[:2]
        or targets.dtype != torch.int64
    ):
        raise ValueError("Expected logits [B,T,V] and aligned int64 targets [B,T]")
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    if mask.shape != targets.shape or mask.dtype != torch.bool:
        raise ValueError("Loss mask must be boolean and match targets")
    valid = mask & (targets != 0) & (targets != 1) & (targets != -100)
    selected = targets.masked_fill(~valid, 0)
    values = F.cross_entropy(
        logits.float().flatten(0, 1), selected.flatten(), reduction="none"
    ).view_as(targets)
    return (values * valid).sum(), valid.sum()
