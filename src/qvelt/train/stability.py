"""Loss-spike detection and optional per-layer gradient/activation statistics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from qvelt.model.decoder import Decoder
from qvelt.train.parallel import local_tensor
from qvelt.utils.distributed import DistributedContext


@dataclass
class SpikeMonitor:
    """EMA mean/variance z-score of the window loss; spikes do not update the EMA.

    ``threshold=0`` disables detection. Detection starts after ``warmup`` healthy
    windows. ``alpha`` = 2/(N+1) gives an effective memory of ~N=99 windows.
    """

    threshold: float
    warmup: int
    alpha: float = 0.02
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0
    spikes: int = 0

    def update(self, loss: float) -> tuple[float | None, bool]:
        if not math.isfinite(loss):
            raise ValueError("Spike monitor accepts finite losses only")
        score = None
        if self.count >= max(1, self.warmup):
            # Floor avoids infinite z on an artificially flat curve.
            spread = max(math.sqrt(self.variance), 1e-4 * max(1.0, abs(self.mean)))
            score = (loss - self.mean) / spread
        spike = bool(self.threshold and score is not None and score > self.threshold)
        if spike:
            self.spikes += 1
            return score, True
        if self.count == 0:
            self.mean = loss
        else:
            difference = loss - self.mean
            increment = self.alpha * difference
            self.mean += increment
            self.variance = (1 - self.alpha) * (self.variance + difference * increment)
        self.count += 1
        return score, False

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = set(asdict(self))
        if set(state) != expected:
            raise ValueError("Unknown spike-monitor state fields")
        if (
            state["threshold"] != self.threshold
            or state["warmup"] != self.warmup
            or state["alpha"] != self.alpha
            or type(state["count"]) is not int
            or type(state["spikes"]) is not int
            or min(state["count"], state["spikes"]) < 0
            or not all(
                isinstance(state[k], int | float) and math.isfinite(state[k])
                for k in ("mean", "variance")
            )
        ):
            raise ValueError("Spike-monitor state disagrees with the recipe")
        if state["variance"] < 0:
            raise ValueError("Spike variance must be nonnegative")
        self.count, self.spikes = state["count"], state["spikes"]
        self.mean, self.variance = float(state["mean"]), float(state["variance"])


def _squared(parameters: Iterable[Tensor]) -> Tensor:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            grad = local_tensor(parameter.grad)
            total = total.to(grad.device) + grad.detach().double().square().sum()
    return total


def layer_gradient_norms(
    decoder: Decoder, context: DistributedContext, *, sharded: bool
) -> dict[str, float]:
    """Pre-clip L2 norm of each block plus the tied embedding (root) parameters.

    DDP gradients are already identical on every rank; FSDP2 holds disjoint
    shards, so the local squared sums are all-reduced before the square root.
    """
    groups = [decoder.blocks[i].parameters() for i in range(len(decoder.blocks))]
    root = [decoder.embedding.weight, *decoder.norm.parameters()]
    values = torch.stack(
        [_squared(g).to(context.device) for g in groups]
        + [_squared(root).to(context.device)]
    )
    if sharded:
        values = context.sum(values)
    norms = values.sqrt().tolist()
    result = {f"layer_{i:02d}_grad_norm": norms[i] for i in range(len(groups))}
    result["embedding_norm_grad_norm"] = norms[-1]
    return result


class ActivationRecorder:
    """Forward hooks storing each block output's RMS and abs-max (last call)."""

    def __init__(self, decoder: Decoder) -> None:
        self.decoder = decoder
        self.values: dict[int, tuple[float, float]] = {}

    def _hook(self, index: int) -> Any:
        def record(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
            with torch.no_grad():
                value = output.detach().float()
                self.values[index] = (
                    float(value.square().mean().sqrt()),
                    float(value.abs().max()),
                )

        return record

    @contextmanager
    def recording(self) -> Iterator[ActivationRecorder]:
        handles = [
            block.register_forward_hook(self._hook(index))
            for index, block in enumerate(self.decoder.blocks)
        ]
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def reduce(self, context: DistributedContext) -> dict[str, float]:
        """Rank-mean RMS and global abs-max; empty if no forward ran."""
        layers = len(self.decoder.blocks)
        if len(self.values) != layers:
            return {}
        rms = torch.tensor(
            [self.values[i][0] for i in range(layers)],
            dtype=torch.float64,
            device=context.device,
        )
        peak = torch.tensor(
            [self.values[i][1] for i in range(layers)],
            dtype=torch.float64,
            device=context.device,
        )
        if context.world_size > 1:
            dist.all_reduce(rms, op=dist.ReduceOp.SUM)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            rms /= context.world_size
        result: dict[str, float] = {}
        for i in range(layers):
            result[f"layer_{i:02d}_act_rms"] = float(rms[i])
            result[f"layer_{i:02d}_act_absmax"] = float(peak[i])
        return result
