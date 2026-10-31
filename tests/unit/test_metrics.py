"""Structured metric values and actual optional local backend output."""

import json
from pathlib import Path

import pytest

from qvelt.utils.metrics import (
    JsonMetricSink,
    TensorBoardSink,
    WandbSink,
    optional_sink,
)


def test_json_metrics_and_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = JsonMetricSink(path)
    sink.log(1, {"loss": 2.0, "estimated_mfu": None})
    with pytest.raises(ValueError):
        sink.log(2, {"loss": float("nan")})
    with pytest.raises(ValueError):
        sink.log(-1, {"loss": 0.0})
    sink.close()
    assert json.loads(path.read_text()) == {
        "step": 1,
        "loss": 2.0,
        "estimated_mfu": None,
    }
    assert optional_sink("json", tmp_path) is None
    with pytest.raises(ValueError):
        optional_sink("unknown", tmp_path)


def test_tensorboard_actual_event_file(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    sink = TensorBoardSink(tmp_path)
    sink.log(3, {"loss": 1.25, "estimated_mfu": None})
    sink.close()
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert accumulator.Scalars("loss")[0].step == 3
    assert accumulator.Scalars("loss")[0].value == 1.25


def test_wandb_actual_offline_run(tmp_path: Path) -> None:
    pytest.importorskip("wandb")
    sink = WandbSink(tmp_path)
    assert sink.run.settings.mode == "offline"
    sink.log(1, {"loss": 0.5})
    sink.close()
    assert list(tmp_path.rglob("*.wandb"))
