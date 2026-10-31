"""Original portable RMSNorm, interleaved RoPE, grouped attention and SwiGLU."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from qvelt.config import ModelConfig


class RMSNorm(nn.Module):
    """g*x/sqrt(mean(x²)+eps), with FP32 statistics and a learned gain."""

    def __init__(self, hidden: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        values = x.float()
        return (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
            * self.weight.float()
        ).to(x.dtype)


class RotaryEmbedding(nn.Module):
    inverse: Tensor

    """Rotate adjacent even/odd Q/K coordinates at explicit document positions."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.register_buffer(
            "inverse",
            theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim),
            persistent=False,
        )

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        angles = positions.float()[:, None, :, None] * self.inverse[None, None, None, :]
        a, b = x.float()[..., 0::2], x.float()[..., 1::2]
        cos, sin = angles.cos(), angles.sin()
        return (
            torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
            .flatten(-2)
            .to(x.dtype)
        )


class Attention(nn.Module):
    """Full GQA with causal/document/padding masks; padded queries return zero."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads, self.kv_heads = config.heads, config.kv_heads
        self.head_dim = config.hidden // config.heads
        kv = self.kv_heads * self.head_dim
        self.query = nn.Linear(config.hidden, config.hidden, bias=False)
        self.key = nn.Linear(config.hidden, kv, bias=False)
        self.value = nn.Linear(config.hidden, kv, bias=False)
        self.output = nn.Linear(config.hidden, config.hidden, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self, x: Tensor, positions: Tensor, documents: Tensor, valid: Tensor
    ) -> Tensor:
        b, t, _ = x.shape
        q = self.query(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, positions), self.rope(k, positions)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & (
            documents[:, None, :, None] == documents[:, None, None, :]
        )
        allowed = allowed & valid[:, None, :, None] & valid[:, None, None, :]
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, -torch.inf)
        # A padded query must not feed an all-infinity row into softmax.
        scores = torch.where(valid[:, None, :, None], scores, 0.0)
        weights = F.softmax(scores, dim=-1).to(v.dtype)
        attended = (weights @ v).transpose(1, 2).reshape(b, t, -1)
        return cast(Tensor, self.output(attended) * valid[..., None])


class SwiGLU(nn.Module):
    """Wdown(SiLU(Wgate*x) * Wup*x), all projections bias-free."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden, config.ffn, bias=False)
        self.up = nn.Linear(config.hidden, config.ffn, bias=False)
        self.down = nn.Linear(config.ffn, config.hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.down(F.silu(self.gate(x)) * self.up(x)))
