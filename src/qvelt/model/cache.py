"""Explicit trusted, immutable request cache; tensor contents must not be mutated."""

from __future__ import annotations

from dataclasses import dataclass
from weakref import ReferenceType

import torch
from torch import Tensor, nn

from qvelt.config import ModelConfig
from qvelt.config.execution import ModelExecution

ParameterStamp = tuple[tuple[str, int, int, int, str, str], ...]


def parameter_stamp(model: nn.Module) -> ParameterStamp:
    """Track normal in-place updates/replacements without copying model weights.

    Tensor._version is a pinned torch2.9.1 interface checked by regression tests.
    Unsafe .data writes bypass counters and are outside this trusted cache API.
    """
    try:
        return tuple(
            (name, id(p), p.data_ptr(), p._version, str(p.device), str(p.dtype))
            for name, p in model.named_parameters()
        )
    except RuntimeError as error:
        raise ValueError(
            "Cached models require normal parameter version counters; construct outside inference_mode"
        ) from error


@dataclass(frozen=True)
class LayerKV:
    key: Tensor
    value: Tensor

    def validate(
        self, batch: int, heads: int, length: int, width: int, device: torch.device
    ) -> None:
        expected = (batch, heads, length, width)
        if (
            self.key.shape != expected
            or self.value.shape != expected
            or self.key.device != device
            or self.value.device != device
            or self.key.dtype != self.value.dtype
            or self.key.requires_grad
            or self.value.requires_grad
        ):
            raise ValueError("Invalid cached KV dimensions/device/dtype/grad state")


@dataclass(frozen=True)
class KVCache:
    owner: ReferenceType[nn.Module]
    parameters: ParameterStamp
    config: ModelConfig
    execution: ModelExecution
    layers: tuple[LayerKV, ...]
    positions: Tensor
    documents: Tensor
    valid: Tensor

    @property
    def length(self) -> int:
        return self.documents.shape[1]

    def validate(
        self,
        model: nn.Module,
        config: ModelConfig,
        execution: ModelExecution,
        batch: int,
        device: torch.device,
    ) -> None:
        if (
            self.owner() is not model
            or self.config != config
            or self.execution != execution
            or self.parameters != parameter_stamp(model)
        ):
            raise ValueError(
                "Cache belongs to another model, execution policy or stale weights"
            )
        if (
            self.documents.ndim != 2
            or self.documents.shape[0] != batch
            or not 1 <= self.length <= config.context
            or len(self.layers) != config.layers
        ):
            raise ValueError("Invalid cache metadata dimensions")
        for value, dtype in (
            (self.positions, torch.int64),
            (self.documents, torch.int64),
            (self.valid, torch.bool),
        ):
            if (
                value.shape != (batch, self.length)
                or value.dtype != dtype
                or value.device != device
                or value.requires_grad
            ):
                raise ValueError("Invalid cache metadata dtype/device/shape")
        for layer in self.layers:
            layer.validate(
                batch,
                config.kv_heads,
                self.length,
                config.hidden // config.heads,
                device,
            )
