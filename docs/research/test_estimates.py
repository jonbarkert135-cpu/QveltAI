"""Independent arithmetic and failure checks for the research calculator."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from estimates import (
    GIB,
    SPECS,
    ModelSpec,
    compute_hours,
    flops_per_token,
    memory_bytes,
    parameter_count,
    report,
    tensor_shapes,
)


def test_hand_counted_one_layer() -> None:
    spec = ModelSpec("hand", 1, 4, 2, 1, 6, 10, 3, 100)
    # Embedding 40; Q/O 32; K/V 16; FFN 72; block norms 8; final norm 4.
    assert parameter_count(spec) == 172
    assert sum(math.prod(x) for x in tensor_shapes(spec).values()) == 172
    assert flops_per_token(spec) == 6 * (40 + 32 + 16 + 72) + 12 * 1 * 3 * 4


@pytest.mark.parametrize("spec", SPECS, ids=lambda x: x.name)
def test_independent_tensor_enumeration(spec: ModelSpec) -> None:
    shapes = tensor_shapes(spec)
    assert len(shapes) == 9 * spec.layers + 2
    assert "output_head" not in shapes  # Tied storage, not a second parameter.
    assert sum(math.prod(shape) for shape in shapes.values()) == parameter_count(spec)
    assert spec.head_dim == 64


@pytest.mark.parametrize("field", range(8))
def test_reject_nonpositive(field: int) -> None:
    dimensions = [8, 320, 5, 1, 896, 32768, 512, 10_000_000]
    dimensions[field] = 0
    with pytest.raises(ValueError):
        ModelSpec("bad", *dimensions)


@pytest.mark.parametrize("hidden,heads,kv", [(10, 3, 1), (16, 4, 3), (6, 2, 1)])
def test_reject_incompatible_heads(hidden: int, heads: int, kv: int) -> None:
    with pytest.raises(ValueError):
        ModelSpec("bad", 1, hidden, heads, kv, 12, 10, 4, 100)


def test_attention_growth_and_tying() -> None:
    spec = SPECS[1]
    assert (
        flops_per_token(replace(spec, context=2 * spec.context)) - flops_per_token(spec)
        == 12 * spec.layers * spec.context * spec.hidden
    )
    assert parameter_count(replace(spec, context=2 * spec.context)) == parameter_count(
        spec
    )
    assert (
        parameter_count(replace(spec, vocab=spec.vocab + 1)) - parameter_count(spec)
        == spec.hidden
    )


def test_memory_accounting_and_shards() -> None:
    spec = ModelSpec("hand", 1, 4, 2, 1, 6, 10, 3, 100)
    activation_bytes = 24 + 324 + 120
    assert memory_bytes(spec) == 18 * 172 + activation_bytes + 2 * GIB
    # Two BF16 gathers of the larger of embedding=40 or block=128 parameters.
    assert (
        memory_bytes(spec, shards=2)
        == 18 * 172 / 2 + activation_bytes + 4 * 128 + 2 * GIB
    )
    assert memory_bytes(SPECS[-1], shards=8) < memory_bytes(SPECS[-1])


@pytest.mark.parametrize("microbatch,shards", [(0, 1), (1, 0), (-1, 2)])
def test_invalid_memory_inputs(microbatch: int, shards: int) -> None:
    with pytest.raises(ValueError):
        memory_bytes(SPECS[0], microbatch, shards)


def test_cost_units() -> None:
    assert compute_hours(3600 * 10**12, 1) == 1
    assert compute_hours(0, 1) == 0
    for work, rate in [(-1, 1), (1, 0), (1, -1), (1, math.inf), (1, math.nan)]:
        with pytest.raises(ValueError):
            compute_hours(work, rate)


def test_checked_in_report_is_current() -> None:
    saved = json.loads(Path(__file__).with_name("estimates.json").read_text())
    assert saved == report()


@pytest.mark.parametrize(
    "spec,expected",
    list(
        zip(SPECS, [19_338_560, 125_854_464, 493_553_920, 1_510_082_560], strict=True)
    ),
)
def test_rfc_parameter_totals(spec: ModelSpec, expected: int) -> None:
    assert parameter_count(spec) == expected


def test_stage_mixture_totals() -> None:
    mixtures = json.loads(Path(__file__).with_name("data_mix.json").read_text())
    assert len(mixtures) == 5
    for weights in mixtures.values():
        assert sum(weights.values()) == 100
        assert all(0 <= value <= 100 for value in weights.values())


def test_cli_output_and_file(tmp_path: Path) -> None:
    script = Path(__file__).with_name("estimates.py")
    stdout = subprocess.check_output([sys.executable, str(script)], text=True)
    assert json.loads(stdout) == report()
    output = tmp_path / "report.json"
    subprocess.run([sys.executable, str(script), "--output", str(output)], check=True)
    assert output.read_text() == stdout


def test_source_manifest_scope() -> None:
    manifest = json.loads(Path(__file__).with_name("sources.json").read_text())
    assert len(manifest["dataset_cards"]) == 13
    assert all(not card["ingestion_approved"] for card in manifest["dataset_cards"])
    rejected = [
        p for p in manifest["papers"] if p["status"] == "excluded_title_mismatch"
    ]
    assert [paper["id"] for paper in rejected] == ["S41"]
    for paper in manifest["papers"]:
        if paper["status"] == "retrieved":
            assert len(paper["sha256"]) == 64
            assert paper["url"].endswith("v1")
