"""Strict bounded evaluation protocols and location-independent artifact identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from qvelt.config import ConfigError
from qvelt.eval.config import ArtifactRef, EvalJob, dump_eval, eval_digest, load_eval


def base_job() -> EvalJob:
    ref = ArtifactRef("/local/manifest.json", "a" * 64)
    return EvalJob(ref, ref, ref)


@pytest.mark.parametrize(
    "change",
    [
        {"samples": True},
        {"sampling": "beam"},
        {"temperature": float("nan")},
        {"numeric_atol": -1},
        {"pass_k": (2,)},
        {"pass_k": ([],)},
        {"lm_window": 513},
        {"approved_for_evaluation": 1},
        {"sampling": "sample", "temperature": 0},
        {"max_new_tokens": 129},
        {"protocol": "v2"},
    ],
)
def test_reject_invalid_protocol(change: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        replace(base_job(), **change)


def test_yaml_round_trip_unknown_fields_and_path_relocation(tmp_path: Path) -> None:
    job = base_job()
    file = tmp_path / "eval.yaml"
    file.write_text(dump_eval(job))
    assert load_eval(file) == job
    assert eval_digest(job) == eval_digest(
        replace(job, checkpoint=replace(job.checkpoint, path="/other/manifest.json"))
    )
    assert eval_digest(job) != eval_digest(replace(job, seed=0))
    file.write_text(dump_eval(job) + "unknown: true\n")
    with pytest.raises(ConfigError):
        load_eval(file)
    file.write_text("checkpoint: {path: x}\ntokenizer: {}\ndataset: {}\n")
    with pytest.raises(ConfigError):
        load_eval(file)
