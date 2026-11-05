"""Per-source heldout perplexity and greedy generation samples during pretraining."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from qvelt.data.loader import PackedDataset
from qvelt.tokenizer.core import QveltTokenizer
from qvelt.train.accumulation import batch_loss, device_batch
from qvelt.utils.distributed import DistributedContext

_FIELDS = (
    "input_ids",
    "labels",
    "positions",
    "document_ids",
    "valid_mask",
    "loss_mask",
)


def domain_indices(
    dataset: PackedDataset,
    source: int,
    batch: int,
    batch_size: int,
    context: DistributedContext,
) -> list[int]:
    """Fixed, rank-disjoint block indices of one source; wraps on tiny sources."""
    blocks = dataset.offsets[source + 1] - dataset.offsets[source]
    first = (batch * context.world_size + context.rank) * batch_size
    return [dataset.offsets[source] + (first + k) % blocks for k in range(batch_size)]


def _batch(
    dataset: PackedDataset, indices: Sequence[int], attention_mode: str
) -> dict[str, Tensor]:
    samples = [dataset[index] for index in indices]
    batch = {name: torch.stack([s[name] for s in samples]) for name in _FIELDS}
    if attention_mode == "causal":
        # Mirror ResumableLoader's causal mode: one stream, positions 0..T-1.
        batch["document_ids"] = torch.where(batch["valid_mask"], 0, -1)
        batch["positions"] = torch.where(
            batch["valid_mask"], torch.arange(dataset.context), 0
        )
    return batch


@torch.no_grad()
def domain_validation(
    model: nn.Module,
    dataset: PackedDataset,
    *,
    batches: int,
    batch_size: int,
    attention_mode: str,
    precision: str,
    context: DistributedContext,
) -> dict[str, float | None]:
    """Mean next-token loss and perplexity for every validation source.

    Every rank evaluates the same number of batches per source (required for
    FSDP collectives); the deterministic blocks are independent of training RNG.
    """
    mode = model.training
    model.eval()
    try:
        result: dict[str, float | None] = {}
        for source, name in enumerate(dataset.names):
            total = torch.zeros((), dtype=torch.float64, device=context.device)
            count = torch.zeros((), dtype=torch.int64, device=context.device)
            for index in range(batches):
                indices = domain_indices(dataset, source, index, batch_size, context)
                raw = context.collective(
                    partial(_batch, dataset, indices, attention_mode),
                    "Domain validation data failure",
                )
                batch = context.collective(
                    partial(device_batch, raw, context.device),
                    "Domain validation transfer failure",
                )
                loss, targets = context.collective(
                    partial(batch_loss, model, batch, precision, context.device),
                    "Domain validation forward failure",
                )
                total += loss.double()
                count += targets
            loss_sum = float(context.sum(total).item())
            targets_sum = int(context.sum(count).item())
            if targets_sum <= 0 or not math.isfinite(loss_sum):
                raise FloatingPointError(f"Invalid domain validation for {name}")
            mean = loss_sum / targets_sum
            result[f"validation_{name}_loss"] = mean
            result[f"validation_{name}_ppl"] = math.exp(mean) if mean < 700 else None
            result[f"validation_{name}_targets"] = float(targets_sum)
        return result
    finally:
        model.train(mode)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    tokenizer: QveltTokenizer,
    prompts: Sequence[str],
    *,
    tokens: int,
    context_length: int,
    precision: str,
    context: DistributedContext,
) -> list[dict[str, Any]]:
    """Greedy, uncached decoding run on every rank (FSDP needs all ranks).

    Rank0's argmax is broadcast each step so ranks never diverge. Decoding uses
    no RNG, so sampling cannot perturb the training random streams.
    """
    if (
        precision not in ("fp32", "bf16")
        or type(tokens) is not int
        or tokens < 1
        or type(context_length) is not int
        or context_length < 2
    ):
        raise ValueError(
            "Generation requires supported precision,positive tokens,context>=2"
        )
    mode = model.training
    model.eval()
    bos, eos = tokenizer.control_id("<|bos|>"), tokenizer.control_id("<|eos|>")
    device = context.device
    samples = []
    try:
        for prompt in prompts:
            encoded = tokenizer.encode(prompt)
            body_limit = max(0, context_length - min(tokens, context_length - 1) - 1)
            prefix = [bos, *(encoded[-body_limit:] if body_limit else [])]
            ids = list(prefix)
            while len(ids) - len(prefix) < tokens and len(ids) < context_length:
                inputs = torch.tensor([ids], dtype=torch.int64, device=device)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
                ):
                    logits = model(inputs)
                choice = logits[0, -1].float().argmax().reshape(1)
                if context.world_size > 1:
                    dist.broadcast(choice, src=0)
                token = int(choice.item())
                ids.append(token)
                if token == eos:
                    break
            completion = ids[len(prefix) :]
            samples.append(
                {
                    "prompt": prompt,
                    "prompt_truncated": len(prefix) < len(encoded) + 1,
                    "completion": tokenizer.decode(completion),
                    "completion_ids": completion,
                    "stopped_at_eos": bool(completion and completion[-1] == eos),
                }
            )
        return samples
    finally:
        model.train(mode)
