"""Explicit torchrun process groups, rank-zero operations and all-reduce checks."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def sum(self, value: torch.Tensor) -> torch.Tensor:
        result = value.detach().clone()
        if self.world_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def rank_zero(self, action: Callable[[], T]) -> T | None:
        """Propagate rank-zero IO errors before another collective is entered."""
        result = None
        error = None
        if self.rank == 0:
            try:
                result = action()
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
        messages: list[str | None] = [error]
        if self.world_size > 1:
            dist.broadcast_object_list(messages, src=0, device=self.device)
        if messages[0] is not None:
            raise RuntimeError("Rank-zero operation failed: " + messages[0])
        return result


@contextmanager
def distributed_context(device: str) -> Iterator[DistributedContext]:
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError("Invalid torchrun rank environment")
    if dist.is_initialized():
        raise RuntimeError("Qvelt must own its process group")
    if device == "cuda":
        if not torch.cuda.is_available() or local >= torch.cuda.device_count():
            raise RuntimeError("Requested local CUDA device is unavailable")
        torch.cuda.set_device(local)
    target = torch.device(device, local) if device == "cuda" else torch.device("cpu")
    started = False
    try:
        if world > 1:
            dist.init_process_group(
                backend="nccl" if device == "cuda" else "gloo",
                timeout=timedelta(seconds=120),
            )
            started = True
        yield DistributedContext(rank, world, local, target)
    finally:
        if started:
            dist.destroy_process_group()


def all_reduce_probe(device: str) -> dict[str, int | str]:
    """Every rank contributes rank+1; verify the exact integer sum on all ranks."""
    with distributed_context(device) as context:
        value = torch.tensor(context.rank + 1, dtype=torch.int64, device=context.device)
        reduced = int(context.sum(value).item())
        expected = context.world_size * (context.world_size + 1) // 2
        if reduced != expected:
            raise RuntimeError("All-reduce returned an incorrect sum")
        return {
            "rank": context.rank,
            "world_size": context.world_size,
            "device": str(context.device),
            "sum": reduced,
            "expected": expected,
        }
