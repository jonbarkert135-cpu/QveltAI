"""Original RFC decoder with explicit execution and request-local cached decoding."""

from __future__ import annotations

import math
from typing import cast
from weakref import ref

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from qvelt.config import ModelConfig
from qvelt.config.execution import ModelExecution
from qvelt.model.cache import KVCache, LayerKV, parameter_stamp
from qvelt.model.layers import Attention, RMSNorm, SwiGLU

_DEFAULT_EXECUTION = ModelExecution()


class Block(nn.Module):
    """Independent pre-normalized attention and feed-forward residual branches."""

    def __init__(
        self, config: ModelConfig, execution: ModelExecution = _DEFAULT_EXECUTION
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden, config.norm_eps)
        self.attention = Attention(config, execution.attention_backend)
        self.ffn_norm = RMSNorm(config.hidden, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self, x: Tensor, positions: Tensor, documents: Tensor, valid: Tensor
    ) -> Tensor:
        x = x + self.attention(self.attention_norm(x), positions, documents, valid)
        return cast(Tensor, x + self.ffn(self.ffn_norm(x)))

    def forward_cached(
        self,
        x: Tensor,
        positions: Tensor,
        documents: Tensor,
        valid: Tensor,
        key_documents: Tensor,
        key_valid: Tensor,
        past: LayerKV | None,
    ) -> tuple[Tensor, LayerKV]:
        attended, state = self.attention.forward_cached(
            self.attention_norm(x),
            positions,
            documents,
            valid,
            key_documents,
            key_valid,
            past,
        )
        x = x + attended
        return x + self.ffn(self.ffn_norm(x)), state


class Decoder(nn.Module):
    """Tied E input/output decoder returning [batch,time,vocabulary] logits."""

    def __init__(
        self, config: ModelConfig, execution: ModelExecution = _DEFAULT_EXECUTION
    ) -> None:
        super().__init__()
        if not isinstance(execution, ModelExecution):
            raise ValueError("Expected validated ModelExecution")
        self.config, self.execution = config, execution
        self.embedding = nn.Embedding(config.vocab, config.hidden, dtype=torch.float32)
        self.blocks = nn.ModuleList(
            Block(config, execution) for _ in range(config.layers)
        )
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

    def _inputs(
        self,
        token_ids: Tensor,
        positions: Tensor | None,
        document_ids: Tensor | None,
        valid_mask: Tensor | None,
        start: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if token_ids.ndim != 2 or token_ids.dtype != torch.int64:
            raise ValueError("token_ids must be a batch/time int64 tensor")
        b, t = token_ids.shape
        if b == 0 or t == 0 or start + t > self.config.context:
            raise ValueError("Empty batch or sequence exceeds the configured context")
        if token_ids.device != self.embedding.weight.device:
            raise ValueError("Token and model devices must match")
        if document_ids is not None and positions is None:
            raise ValueError("Packed documents require explicit reset positions")
        if positions is None:
            positions = torch.arange(start, start + t, device=token_ids.device).expand(
                b, t
            )
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
        return positions, document_ids, valid_mask

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor | None = None,
        document_ids: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        positions, documents, valid = self._inputs(
            token_ids, positions, document_ids, valid_mask
        )
        x = self.embedding(token_ids)
        for block in self.blocks:
            if (
                self.execution.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                x = cast(
                    Tensor,
                    checkpoint(
                        block,
                        x,
                        positions,
                        documents,
                        valid,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    ),
                )
            else:
                x = block(x, positions, documents, valid)
        return F.linear(self.norm(x), self.embedding.weight)

    @torch.no_grad()
    def forward_cached(
        self,
        token_ids: Tensor,
        cache: KVCache | None = None,
        positions: Tensor | None = None,
        document_ids: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, KVCache]:
        """Append a bounded suffix; failed calls never advance the original cache.

        The caller must use eval mode and must not mutate cache tensors or .data.
        Packed metadata must be supplied explicitly with the same semantics as forward.
        """
        if self.training:
            raise ValueError("Cached decoding requires model.eval()")
        if token_ids.ndim != 2:
            raise ValueError("Cached token IDs must have batch/time dimensions")
        if cache is not None:
            if not isinstance(cache, KVCache):
                raise ValueError("Expected an explicit KVCache")
            cache.validate(
                self, self.config, self.execution, token_ids.shape[0], token_ids.device
            )
        stamp = parameter_stamp(self) if cache is None else cache.parameters
        start = 0 if cache is None else cache.length
        positions, documents, valid = self._inputs(
            token_ids, positions, document_ids, valid_mask, start
        )
        all_positions = (
            positions.clone()
            if cache is None
            else torch.cat((cache.positions, positions), dim=1)
        )
        all_documents = (
            documents.clone()
            if cache is None
            else torch.cat((cache.documents, documents), dim=1)
        )
        all_valid = (
            valid.clone() if cache is None else torch.cat((cache.valid, valid), dim=1)
        )
        x = self.embedding(token_ids)
        states = []
        for index, block in enumerate(self.blocks):
            past = None if cache is None else cache.layers[index]
            x, state = cast(Block, block).forward_cached(
                x, positions, documents, valid, all_documents, all_valid, past
            )
            states.append(state)
        logits = F.linear(self.norm(x), self.embedding.weight)
        result = KVCache(
            ref(self),
            stamp,
            self.config,
            self.execution,
            tuple(states),
            all_positions,
            all_documents,
            all_valid,
        )
        return logits, result
