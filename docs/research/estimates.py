"""Analytical design estimates; no model execution or measured performance."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    """Bias-free, tied-embedding GQA decoder dimensions and token budget."""

    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    ffn: int
    vocab: int
    context: int
    tokens: int

    def __post_init__(self) -> None:
        for value in (
            self.layers,
            self.hidden,
            self.heads,
            self.kv_heads,
            self.ffn,
            self.vocab,
            self.context,
            self.tokens,
        ):
            if value <= 0:
                raise ValueError("Dimensions and token budget must be positive")
        if self.hidden % self.heads or self.heads % self.kv_heads:
            raise ValueError("hidden/heads and heads/kv_heads must divide exactly")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim


SPECS = (
    ModelSpec("tiny", 8, 320, 5, 1, 896, 32768, 512, 10_000_000),
    ModelSpec("small", 16, 768, 12, 4, 2048, 32768, 2048, 10_000_000_000),
    ModelSpec("base", 26, 1280, 20, 5, 3456, 32768, 4096, 50_000_000_000),
    ModelSpec("large", 32, 2048, 32, 8, 5632, 32768, 4096, 150_000_000_000),
)


def tensor_shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    """Enumerate unique trainable tensors; the output head aliases embedding."""
    shapes: dict[str, tuple[int, ...]] = {"embedding": (spec.vocab, spec.hidden)}
    for layer in range(spec.layers):
        prefix = f"layers.{layer}."
        shapes.update(
            {
                prefix + "q": (spec.hidden, spec.hidden),
                prefix + "k": (spec.kv_width, spec.hidden),
                prefix + "v": (spec.kv_width, spec.hidden),
                prefix + "o": (spec.hidden, spec.hidden),
                prefix + "gate": (spec.ffn, spec.hidden),
                prefix + "up": (spec.ffn, spec.hidden),
                prefix + "down": (spec.hidden, spec.ffn),
                prefix + "attention_norm": (spec.hidden,),
                prefix + "ffn_norm": (spec.hidden,),
            }
        )
    shapes["final_norm"] = (spec.hidden,)
    return shapes


def parameter_count(spec: ModelSpec) -> int:
    """V*d + L*(2*d² + 2*d*k + 3*d*f + 2*d) + d."""
    d, k, f = spec.hidden, spec.kv_width, spec.ffn
    return (
        spec.vocab * d + spec.layers * (2 * d * d + 2 * d * k + 3 * d * f + 2 * d) + d
    )


def flops_per_token(spec: ModelSpec) -> int:
    """Six FLOPs/matrix parameter plus full-square attention, fwd+bwd.

    Includes tied dense vocabulary projection once, not embedding lookup.
    Assumes multiply-add = 2 FLOPs; excludes norm/softmax/optimizer/recomputation.
    Full-square attention contributes 12*L*S*d; triangular kernels may do less.
    """
    d, k, f = spec.hidden, spec.kv_width, spec.ffn
    matrices = spec.vocab * d + spec.layers * (2 * d * d + 2 * d * k + 3 * d * f)
    return 6 * matrices + 12 * spec.layers * spec.context * d


def memory_bytes(spec: ModelSpec, microbatch: int = 1, shards: int = 1) -> float:
    """Planning model for checkpointed BF16 compute and FP32 AdamW state.

    16P persistent bytes plus 2P compute-cache allowance, sharded across ranks.
    Adds checkpoint inputs, one block's heuristic workspace, FP32 full logits,
    two BF16 parameter all-gathers when sharded, and 2 GiB fixed margin.
    This is not a measured peak or guaranteed upper bound. See architecture RFC.
    """
    if microbatch < 1 or shards < 1:
        raise ValueError("Microbatch and shard count must be positive")
    p = parameter_count(spec)
    b, s, d, f, k = microbatch, spec.context, spec.hidden, spec.ffn, spec.kv_width
    activations = (
        2 * b * s * spec.layers * d
        + 2 * b * s * (8 * d + 3 * f + 2 * k)
        + 4 * b * s * spec.vocab
    )
    block = (p - spec.vocab * d - d) // spec.layers
    gathers = 4 * max(block, spec.vocab * d) if shards > 1 else 0
    return 18 * p / shards + activations + gathers + 2 * GIB


def compute_hours(flops: int, effective_tflops: float) -> float:
    """Convert operation count and assumed sustained throughput to ideal hours."""
    if flops < 0 or effective_tflops <= 0 or not math.isfinite(effective_tflops):
        raise ValueError("Nonnegative work and finite positive throughput required")
    return flops / (effective_tflops * 10**12 * 3600)


def report() -> dict[str, dict[str, int | float]]:
    """Produce deterministic resource estimates, not benchmark results."""
    rows: dict[str, dict[str, int | float]] = {}
    for spec in SPECS:
        p = parameter_count(spec)
        enumerated = sum(math.prod(shape) for shape in tensor_shapes(spec).values())
        if p != enumerated:
            raise ArithmeticError("Closed-form and tensor counts disagree")
        flops = flops_per_token(spec)
        rows[spec.name] = {
            "layers": spec.layers,
            "hidden": spec.hidden,
            "heads": spec.heads,
            "kv_heads": spec.kv_heads,
            "ffn": spec.ffn,
            "vocab": spec.vocab,
            "context": spec.context,
            "parameters": p,
            "embedding_parameters": spec.vocab * spec.hidden,
            "non_embedding_parameters": p - spec.vocab * spec.hidden,
            "flops_per_token": flops,
            "tokens": spec.tokens,
            "chinchilla_reference_tokens": 20 * p,
            "tokens_per_parameter": spec.tokens / p,
            "state_gib": 16 * p / GIB,
            "checkpoint_gib": 12 * p / GIB,
            "training_gib_b1": memory_bytes(spec) / GIB,
            "fsdp8_gib_b1": memory_bytes(spec, shards=8) / GIB,
            "unfused_attention_extra_gib": 8 * spec.heads * spec.context**2 / GIB,
            "kv_cache_gib_b1": 4 * spec.context * spec.layers * spec.kv_width / GIB,
            "total_flops": flops * spec.tokens,
            "ideal_hours_cpu_0_05tf": compute_hours(flops * spec.tokens, 0.05),
            "ideal_hours_laptop_2tf": compute_hours(flops * spec.tokens, 2),
            "ideal_hours_laptop_8tf": compute_hours(flops * spec.tokens, 8),
            "ideal_hours_one_gpu_30tf": compute_hours(flops * spec.tokens, 30),
            "ideal_hours_eight_gpu_800tf": compute_hours(flops * spec.tokens, 800),
        }
    return rows


def main() -> None:
    """Write the analytical report to stdout or an explicitly selected path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = json.dumps(report(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
