"""Host batch-delivery measurements, explicitly separate from GPU training throughput."""

from __future__ import annotations

import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from qvelt.data.config import load_data_recipe
from qvelt.data.loader import LoaderConfig, PackedDataset, ResumableLoader


def benchmark_loader(
    path: Path, *, workers: int, batches: int = 64, warmup: int = 8, batch_size: int = 8
) -> dict[str, Any]:
    if type(batches) is not int or type(warmup) is not int or batches < 1 or warmup < 1:
        raise ValueError("Benchmark needs positive warmup and measured batches")
    dataset = PackedDataset(path)
    stages = load_data_recipe(path.parent / "recipe.yaml").mixing
    config = LoaderConfig(
        batch_size=batch_size, workers=workers, max_batches=warmup + batches
    )
    started = time.perf_counter()
    with ResumableLoader(dataset, config, stages) as loader:
        next(loader)
        startup = time.perf_counter() - started
        for _ in range(warmup - 1):
            next(loader)
        started = time.perf_counter()
        targets = capacity = 0
        for _ in range(batches):
            batch = next(loader)
            targets += int(batch["loss_mask"].sum())
            capacity += batch["input_ids"].numel()
        elapsed = time.perf_counter() - started
        state = loader.state_dict()
    return {
        "scope": "CPU host batch delivery, not GPU saturation or model training",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "loader": asdict(config),
        "warmup_batches": warmup,
        "measured_batches": batches,
        "dataset_sha256": dataset.identity,
        "unique_available_blocks": len(dataset),
        "startup_seconds": startup,
        "steady_seconds": elapsed,
        "valid_targets": targets,
        "capacity_tokens": capacity,
        "valid_targets_per_second": targets / elapsed,
        "capacity_tokens_per_second": capacity / elapsed,
        "final_source_cursors": state["counts"],
        "epoch_reuse": "Intentional source repetition after exhaustion; filesystem cache may be warm",
    }
