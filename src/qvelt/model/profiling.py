"""Small CPU execution measurements; saved graph storage is not allocator peak memory."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor

from qvelt.config.training import TrainingJob
from qvelt.data.synthetic import synthetic_batch
from qvelt.model.decoder import Decoder
from qvelt.model.loss import causal_loss


def saved_tensor_storage(
    function: Callable[[], Tensor], excluded: Iterable[Tensor] = ()
) -> tuple[Tensor, int]:
    """Count unique underlying storages saved for backward, excluding model weights."""
    ignored = {t.untyped_storage().data_ptr() for t in excluded}
    storages: dict[int, int] = {}

    def pack(tensor: Tensor) -> Tensor:
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in ignored:
            storages[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = function()
    return result, sum(storages.values())


def profile_cpu(
    job: TrainingJob, *, warmup: int = 2, repeats: int = 8
) -> dict[str, Any]:
    """Measure the resolved small FP32 CPU job without optimizer/weight updates."""
    if (
        job.experiment.runtime.device != "cpu"
        or job.experiment.runtime.precision != "fp32"
        or any(type(v) is not int or v < 1 for v in (warmup, repeats))
        or job.training.sequence_length < 2
    ):
        raise ValueError(
            "CPU profiler requires FP32 CPU, positive repetitions and sequence length>=2"
        )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(job.training.num_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(job.experiment.runtime.seed)
            return _profile(job, warmup, repeats)
    finally:
        torch.set_num_threads(previous_threads)


def _profile(job: TrainingJob, warmup: int, repeats: int) -> dict[str, Any]:
    config, training = job.experiment.model, job.training
    model = Decoder(config, training.execution)
    batch = synthetic_batch(
        0, training.batch_size, training.sequence_length, config.vocab
    )

    def objective() -> Tensor:
        total, count = causal_loss(model(batch.inputs), batch.targets)
        return total / count

    loss, retained_bytes = saved_tensor_storage(objective, model.parameters())
    torch.autograd.backward(loss)

    def train_iteration() -> None:
        model.zero_grad(set_to_none=True)
        torch.autograd.backward(objective())

    for _ in range(warmup):
        train_iteration()
    started = time.perf_counter()
    for _ in range(repeats):
        train_iteration()
    train_seconds = time.perf_counter() - started
    model.eval()
    prefix = training.sequence_length // 2
    suffix = min(8, training.sequence_length - prefix)

    def cached() -> tuple[Tensor, int]:
        output, cache = model.forward_cached(batch.inputs[:, :prefix])
        for step in range(prefix, prefix + suffix):
            output, cache = model.forward_cached(
                batch.inputs[:, step : step + 1], cache
            )
        tensors = [
            cache.positions,
            cache.documents,
            cache.valid,
            *(t for layer in cache.layers for t in (layer.key, layer.value)),
        ]
        storage = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in tensors
        }
        return output, sum(storage.values())

    @torch.no_grad()
    def uncached() -> Tensor:
        output: Tensor = model(batch.inputs[:, :prefix])
        for step in range(prefix, prefix + suffix):
            output = model(batch.inputs[:, : step + 1])
        return output[:, -1:]

    actual, cache_bytes = cached()
    expected = uncached()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)
    timings = {}
    for name, function in (("cached", cached), ("uncached", uncached)):
        for _ in range(warmup):
            function()
        started = time.perf_counter()
        for _ in range(repeats):
            function()
        timings[name] = time.perf_counter() - started
    return {
        "scope": "CPU FP32 numeric-fixture graph/timing profile, no optimizer updates or GPU measurement",
        "job": asdict(job),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "warmup": warmup,
        "repeats": repeats,
        "train_forward_backward_seconds": train_seconds,
        "train_targets_per_second": repeats * batch.targets.numel() / train_seconds,
        "unique_saved_nonparameter_storage_bytes": retained_bytes,
        "memory_scope": "Unique graph-saved underlying storages; excludes weights and is not total/peak allocator memory",
        "cache_storage_bytes": cache_bytes,
        "prefill_tokens_per_sequence": prefix,
        "decode_tokens_per_sequence": suffix,
        "inference_seconds": timings,
        "cached_logits_close": True,
    }
