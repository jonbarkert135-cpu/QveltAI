"""Global valid-target loss-sum accumulation and an explicit token-axis WSD schedule."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from qvelt.config import _integer
from qvelt.config.pretraining import TokenSchedule
from qvelt.model.loss import causal_loss, valid_targets, z_loss_sum
from qvelt.utils.distributed import DistributedContext

Batch = Mapping[str, Tensor]


@dataclass(frozen=True)
class WindowStats:
    loss: float
    targets: int
    microbatches: int
    z_loss: float | None = None
    finite: bool = True


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
    floor = schedule.min_lr_ratio
    if schedule.shape == "cosine":
        # RFC-002 cosine control: decays from warmup end to D=decay_start+decay;
        # min_lr_ratio=0.1 reproduces peak*[0.1+0.45*(1+cos(pi*p))].
        horizon = schedule.decay_start_tokens + schedule.decay_tokens
        span = horizon - schedule.warmup_tokens
        progress = min(1.0, max(0.0, (position - schedule.warmup_tokens) / span))
        return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
    if position <= schedule.decay_start_tokens:
        return float(peak)
    remaining = max(
        0.0, 1 - (position - schedule.decay_start_tokens) / schedule.decay_tokens
    )
    return peak * (floor + (1 - floor) * remaining) if floor else peak * remaining


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


def batch_objective(
    model: nn.Module,
    batch: Batch,
    precision: str,
    device: torch.device,
    z_weight: float,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Cross-entropy sum, valid count and (if enabled) the summed squared logZ."""
    if not z_weight:
        loss, count = batch_loss(model, batch, precision, device)
        return loss, count, None
    if precision not in ("fp32", "bf16"):
        raise ValueError("Unsupported training precision")
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(
            batch["input_ids"],
            batch["positions"],
            batch["document_ids"],
            batch["valid_mask"],
        )
        loss, count = causal_loss(logits, batch["labels"], batch["loss_mask"])
        return loss, count, z_loss_sum(logits, batch["labels"], batch["loss_mask"])


def accumulate_gradients(
    model: nn.Module,
    next_batch: Callable[[], Batch],
    microbatches: int,
    context: DistributedContext,
    precision: str = "fp32",
    *,
    z_weight: float = 0.0,
    stop_on_nonfinite: bool = True,
    target_budget: int | None = None,
) -> WindowStats:
    """Sum first, normalize once; DDP/FSDP world averaging is compensated after.

    With ``stop_on_nonfinite=False`` a nonfinite microbatch (agreed by all ranks)
    marks the window bad: later microbatches are still consumed and counted so the
    data cursor stays aligned, but no further forward/backward is run.
    """
    _integer(microbatches, "window.microbatches")
    if target_budget is not None:
        _integer(target_budget, "window.target_budget")
    remaining = target_budget
    model.zero_grad(set_to_none=True)
    count = torch.zeros((), dtype=torch.int64, device=context.device)
    total = torch.zeros((), dtype=torch.float64, device=context.device)
    z_total = torch.zeros((), dtype=torch.float64, device=context.device)
    finite_window = True
    for index in range(microbatches):
        last = index + 1 == microbatches
        synchronization = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not last
            else nullcontext()
        )
        # FSDP2 reduce-scatters every microbatch: sharded grads accumulate in
        # param.grad (cleared by zero_grad), so an aborted window cannot leak
        # unsharded no-sync gradients into the next one.
        with synchronization:
            raw = context.collective(
                next_batch, "A rank could not read its next training batch"
            )
            batch = context.collective(
                partial(device_batch, raw, context.device),
                "A rank could not transfer its training batch",
            )
            if remaining is not None:
                active = valid_targets(batch["labels"], batch["loss_mask"])
                local_count = active.sum().to(dtype=torch.int64).reshape(1)
                counts = [
                    torch.zeros_like(local_count) for _ in range(context.world_size)
                ]
                if context.world_size > 1:
                    dist.all_gather(counts, local_count)
                else:
                    counts = [local_count]
                sizes = [int(value.item()) for value in counts]
                allowed = max(
                    0, min(sizes[context.rank], remaining - sum(sizes[: context.rank]))
                )
                keep = active.flatten().nonzero().flatten()[:allowed]
                mask = torch.zeros_like(active).flatten()
                mask[keep] = True
                batch["loss_mask"] = mask.reshape_as(active)
                remaining = max(0, remaining - sum(sizes))
            loss, targets, z_sum = context.collective(
                partial(
                    batch_objective, model, batch, precision, context.device, z_weight
                ),
                "A rank could not compute its training loss",
            )
            healthy = torch.isfinite(loss) & (
                torch.isfinite(z_sum) if z_sum is not None else True
            )
            finite = context.sum(healthy.to(torch.int64))
            if int(finite.item()) != context.world_size:
                if stop_on_nonfinite:
                    raise FloatingPointError("Nonfinite accumulated training loss")
                finite_window = False
                # Complete reducer/FSDP backward hooks on every rank before
                # discarding the bad window; never step these gradients.
            objective = loss if z_sum is None else loss + z_weight * z_sum
            torch.autograd.backward(objective)
            count += targets.detach()
            total += loss.detach().double()
            if z_sum is not None:
                z_total += z_sum.detach().double()
    global_count = int(context.sum(count).item())
    if global_count <= 0:
        raise FloatingPointError("Empty global accumulation window")
    if not finite_window:
        model.zero_grad(set_to_none=True)
        return WindowStats(math.nan, global_count, microbatches, None, False)
    scale = context.world_size / global_count
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    mean = float(context.sum(total).item()) / global_count
    z_mean = float(context.sum(z_total).item()) / global_count if z_weight else None
    return WindowStats(mean, global_count, microbatches, z_mean, True)
