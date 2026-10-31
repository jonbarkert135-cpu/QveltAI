"""RFC-001 decoder; no pretrained weights, cache or custom kernels."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from qvelt.config import ModelConfig
from qvelt.model.layers import Attention, RMSNorm, SwiGLU


class Block(nn.Module):
    """Independent pre-normalized attention and feed-forward residual branches."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden, config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self, x: Tensor, positions: Tensor, documents: Tensor, valid: Tensor
    ) -> Tensor:
        x = x + self.attention(self.attention_norm(x), positions, documents, valid)
        return cast(Tensor, x + self.ffn(self.ffn_norm(x)))


class Decoder(nn.Module):
    """Tied E input/output decoder returning [batch,time,vocabulary] logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab, config.hidden)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.hidden, config.norm_eps)
        for name, parameter in self.named_parameters():
            if parameter.ndim == 1:
                nn.init.ones_(parameter)
            else:
                std = (
                    0.02 / math.sqrt(2 * config.layers)
                    if name.endswith(("output.weight", "down.weight"))
                    else 0.02
                )
                nn.init.normal_(parameter, std=std)

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor | None = None,
        document_ids: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.int64:
            raise ValueError("token_ids must be a batch/time int64 tensor")
        b, t = token_ids.shape
        if b == 0 or not 1 <= t <= self.config.context:
            raise ValueError("Empty batch or sequence exceeds the configured context")
        if document_ids is not None and positions is None:
            raise ValueError("Packed documents require explicit reset positions")
        if positions is None:
            positions = torch.arange(t, device=token_ids.device).expand(b, t)
        if document_ids is None:
            document_ids = torch.zeros_like(token_ids)
        if valid_mask is None:
            valid_mask = torch.ones_like(token_ids, dtype=torch.bool)
        for value in (positions, document_ids):
            if (
                value.shape != token_ids.shape
                or value.dtype != torch.int64
                or value.device != token_ids.device
            ):
                raise ValueError("Position/document tensors must match token IDs")
        if (
            valid_mask.shape != token_ids.shape
            or valid_mask.dtype != torch.bool
            or valid_mask.device != token_ids.device
        ):
            raise ValueError("valid_mask must match token IDs with boolean dtype")
        if bool((positions < 0).any()) or bool(
            (positions >= self.config.context).any()
        ):
            raise ValueError("Positions must lie inside the configured context")
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, positions, document_ids, valid_mask)
        return F.linear(self.norm(x), self.embedding.weight)
