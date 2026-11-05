"""Wrap the decoder for single/DDP/FSDP2; TP/PP are reserved, never emulated."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from qvelt.config.training import TrainingJob
from qvelt.model.decoder import Decoder
from qvelt.utils.distributed import DistributedContext

STRATEGIES = ("single", "ddp", "fsdp")


def fsdp_policy(precision: str) -> MixedPrecisionPolicy:
    """FP32 sharded master weights/optimizer; BF16 gathered compute, FP32 reduce."""
    if precision == "bf16":
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
    return MixedPrecisionPolicy()


def parallelize(
    model: Decoder, job: TrainingJob, context: DistributedContext
) -> nn.Module:
    """Apply compile (per block, before wrapping) and the data-parallel strategy.

    ``tensor_parallel``/``pipeline_parallel`` are validated to equal 1 by the
    config; this function rejects any other value again rather than faking it.
    """
    training = job.training
    if training.tensor_parallel != 1 or training.pipeline_parallel != 1:
        raise NotImplementedError(
            "Tensor/pipeline parallelism is reserved but not implemented (ADR-0042)"
        )
    if training.strategy not in STRATEGIES:
        raise ValueError("Unknown data-parallel strategy")
    if (training.strategy == "single") != (context.world_size == 1):
        raise ValueError("single requires one process; ddp/fsdp require several")
    if training.strategy == "fsdp" and context.device.type != "cuda":
        raise ValueError("FSDP2 requires CUDA; CPU DCP is not FSDP2 validation")
    if training.strategy == "fsdp" and training.compile:
        raise ValueError("Compiled FSDP2 awaits hardware validation; use compile=false")
    if training.compile:
        for block in model.blocks:
            # Module.compile keeps state-dict keys (no _orig_mod prefix), so
            # weights retain names; exact resume still binds the compile recipe.
            cast(Any, block).compile()
    if training.strategy == "ddp":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    if training.strategy == "fsdp":
        if context.device.type != "cuda":
            raise ValueError("FSDP2 requires CUDA; CPU DCP is not FSDP2 validation")
        policy = fsdp_policy(job.experiment.runtime.precision)
        for block in model.blocks:
            fully_shard(block, mp_policy=policy)
        # The root group owns the tied embedding/head and final norm.
        fully_shard(model, mp_policy=policy)
    return model


def unwrap(model: nn.Module) -> Decoder:
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(inner, Decoder):
        raise TypeError("Expected a (wrapped) Decoder")
    return inner


def is_sharded(model: nn.Module) -> bool:
    return isinstance(model, FSDPModule)


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def full_scalar(value: torch.Tensor) -> float:
    """Materialize a (possibly DTensor) reduced scalar such as a grad norm."""
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return float(value)


def persistent_state_finite(
    model: nn.Module, optimizer: torch.optim.Optimizer, context: DistributedContext
) -> bool:
    """All ranks agree before checkpointing finite local parameters/Adam state."""
    healthy = torch.ones((), dtype=torch.int64, device=context.device)
    values = [
        *model.parameters(),
        *(
            value
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ),
    ]
    for value in values:
        healthy &= (
            torch.isfinite(local_tensor(value).detach())
            .all()
            .to(context.device, dtype=torch.int64)
        )
    return int(context.sum(healthy).item()) == context.world_size
