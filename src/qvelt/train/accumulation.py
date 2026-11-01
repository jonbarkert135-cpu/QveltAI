"""Global valid-target loss-sum accumulation and an explicit token-axis WSD schedule."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from qvelt.config import _integer
from qvelt.config.pretraining import TokenSchedule
from qvelt.model.loss import causal_loss
from qvelt.utils.distributed import DistributedContext

Batch = Mapping[str, Tensor]


@dataclass(frozen=True)
class WindowStats:
    loss: float
    targets: int
    microbatches: int


def token_learning_rate(
    peak: float, schedule: TokenSchedule, processed: int, update: int
) -> float:
    """Evaluate warmup/stable/linear-decay at the completed valid-token window end."""
    if type(peak) not in (int, float) or not math.isfinite(peak) or peak <= 0:
        raise ValueError("Peak LR must be finite and positive")
    _integer(processed, "processed tokens", minimum=0, maximum=2**63 - 1)
    _integer(update, "window targets", maximum=2**63 - 1)
    position = processed + update
    if schedule.warmup_tokens and position < schedule.warmup_tokens:
        return peak * position / schedule.warmup_tokens
    if position <= schedule.decay_start_tokens:
        return float(peak)
    return peak * max(
        0.0, 1 - (position - schedule.decay_start_tokens) / schedule.decay_tokens
    )


def device_batch(batch: Batch, device: torch.device) -> dict[str, Tensor]:
    return {
        name: batch[name].to(device)
        for name in (
            "input_ids",
            "labels",
            "positions",
            "document_ids",
            "valid_mask",
            "loss_mask",
        )
    }


def batch_loss(
    model: nn.Module, batch: Batch, precision: str, device: torch.device
) -> tuple[Tensor, Tensor]:
    if precision not in ("fp32", "bf16"):
        raise ValueError("Unsupported training precision")
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(
            batch["input_ids"],
            batch["positions"],
            batch["document_ids"],
            batch["valid_mask"],
        )
        return causal_loss(logits, batch["labels"], batch["loss_mask"])


def accumulate_gradients(
    model: nn.Module,
    next_batch: Callable[[], Batch],
    microbatches: int,
    context: DistributedContext,
    precision: str = "fp32",
) -> WindowStats:
    """Sum first, normalize once; DDP's world average is compensated after reduction."""
    _integer(microbatches, "window.microbatches")
    model.zero_grad(set_to_none=True)
    count = torch.zeros((), dtype=torch.int64, device=context.device)
    total = torch.zeros((), dtype=torch.float64, device=context.device)
    for index in range(microbatches):
        synchronization = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index + 1 < microbatches
            else nullcontext()
        )
        with synchronization:
            raw = context.collective(
                next_batch, "A rank could not read its next training batch"
            )
            batch = context.collective(
                partial(device_batch, raw, context.device),
                "A rank could not transfer its training batch",
            )
            loss, targets = context.collective(
                partial(batch_loss, model, batch, precision, context.device),
                "A rank could not compute its training loss",
            )
            finite = context.sum(torch.isfinite(loss).to(torch.int64))
            if int(finite.item()) != context.world_size:
                raise FloatingPointError("Nonfinite accumulated training loss")
            torch.autograd.backward(loss)
            count += targets.detach()
            total += loss.detach().double()
    global_count = int(context.sum(count).item())
    if global_count <= 0:
        raise FloatingPointError("Empty global accumulation window")
    scale = context.world_size / global_count
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    mean = float(context.sum(total).item()) / global_count
    return WindowStats(mean, global_count, microbatches)
