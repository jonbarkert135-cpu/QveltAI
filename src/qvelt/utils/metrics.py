"""Structured local training metrics with optional offline backend adapters."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

MetricValues = Mapping[str, float | None]


class MetricSink(Protocol):
    def log(self, step: int, values: MetricValues) -> None: ...
    def close(self) -> None: ...


def _validate(step: int, values: MetricValues) -> None:
    if (
        "step" in values
        or type(step) is not int
        or step < 0
        or any(v is not None and not math.isfinite(v) for v in values.values())
    ):
        raise ValueError("Metrics require a nonnegative step and finite scalar values")


class JsonMetricSink:
    def __init__(self, path: Path) -> None:
        self.stream = path.open("x", encoding="utf-8")

    def log(self, step: int, values: MetricValues) -> None:
        _validate(step, values)
        self.stream.write(
            json.dumps({"step": step, **values}, sort_keys=True, allow_nan=False) + "\n"
        )
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


class TensorBoardSink:
    def __init__(self, directory: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = cast(Any, SummaryWriter)(str(directory))

    def log(self, step: int, values: MetricValues) -> None:
        _validate(step, values)
        for key, value in values.items():
            if value is not None:
                self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class WandbSink:
    """Only offline logging; no telemetry or Hub upload is authorized by this class."""

    def __init__(self, directory: Path) -> None:
        import wandb

        self.run = wandb.init(project="qvelt", mode="offline", dir=str(directory))

    def log(self, step: int, values: MetricValues) -> None:
        _validate(step, values)
        self.run.log({k: v for k, v in values.items() if v is not None}, step=step)

    def close(self) -> None:
        self.run.finish()


def optional_sink(backend: str, directory: Path) -> MetricSink | None:
    if backend == "json":
        return None
    try:
        if backend == "tensorboard":
            return TensorBoardSink(directory)
        if backend == "wandb":
            return WandbSink(directory)
    except ImportError as error:
        raise RuntimeError(
            f"Install the {backend} extra to use this metrics backend"
        ) from error
    raise ValueError("Unknown metrics backend")
