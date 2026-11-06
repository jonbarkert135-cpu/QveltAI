"""Original unquantized low-rank updates; full base tensors remain in checkpoints."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from qvelt.model.decoder import Decoder


class LoRALinear(nn.Module):
    """W_eff = W + (alpha/r) B A; W frozen, B initially zero, no dropout/bias."""

    weight: Tensor
    a: Tensor
    b: Tensor
    scale: float

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if (
            base.bias is not None
            or type(rank) is not int
            or not 1 <= rank <= min(64, base.in_features, base.out_features)
            or type(alpha) not in (int, float)
            or not math.isfinite(alpha)
            or not 0 < alpha <= 128
        ):
            raise ValueError("LoRA requires bias-free linear, bounded rank/alpha")
        self.weight = nn.Parameter(base.weight.detach(), requires_grad=False)
        self.scale = float(alpha) / rank
        self.a = nn.Parameter(self.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(self.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.a, std=1 / math.sqrt(base.in_features))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight) + self.scale * F.linear(
            F.linear(x, self.a), self.b
        )


def attach_lora(model: Decoder, rank: int, alpha: float) -> int:
    """Freeze decoder and adapt all block attention/FFN linears; no tied-head change."""
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if any(isinstance(module, LoRALinear) for module in model.modules()) or not modules:
        raise ValueError("LoRA requires an unadapted decoder")
    # Validate/construct before mutating the caller's module tree.
    replacements = [(name, LoRALinear(module, rank, alpha)) for name, module in modules]
    model.requires_grad_(False)
    for name, replacement in replacements:
        replacement.a.requires_grad_(True)
        replacement.b.requires_grad_(True)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def merge_lora(model: Decoder) -> None:
    """Replace adapters with ordinary frozen linears; caller must discard old KV caches."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            linear = nn.Linear(
                module.weight.shape[1],
                module.weight.shape[0],
                bias=False,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            linear.weight.copy_(module.weight + module.scale * (module.b @ module.a))
            linear.requires_grad_(False)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, linear)
