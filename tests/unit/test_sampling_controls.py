"""Independent scalar probability/filter oracle and malformed sampling controls."""

import math
from dataclasses import replace
from typing import Any

import pytest
import torch

from qvelt.inference.config import GenerationOptions
from qvelt.inference.generation import sampling_logits


def test_filters_match_scalar_reference() -> None:
    logits = [-3.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    options = GenerationOptions(
        strategy="sample",
        temperature=0.8,
        top_k=4,
        top_p=0.85,
        min_p=0.15,
        repetition_penalty=1.5,
        frequency_penalty=0.2,
        presence_penalty=0.3,
    )
    history, generated = [1, 3, 3, 5], [3, 3, 5]
    values = [
        ((v * 1.5 if v < 0 else v / 1.5) if i in history else v)
        - 0.2 * generated.count(i)
        - 0.3 * (i in generated)
        for i, v in enumerate(logits)
    ]
    values = [v / 0.8 for v in values]
    threshold = sorted(values, reverse=True)[3]
    kept = [i for i, v in enumerate(values) if v >= threshold]
    masses = {i: math.exp(values[i] - max(values)) for i in kept}
    total = sum(masses.values())
    mass = 0.0
    nucleus = []
    for i in sorted(kept, key=lambda i: (-values[i], i)):
        if mass >= 0.85:
            break
        nucleus.append(i)
        mass += masses[i] / total
    expected = [
        i
        for i in nucleus
        if values[i] >= max(values[j] for j in nucleus) + math.log(0.15)
    ]
    actual = sampling_logits(
        torch.tensor(logits), options, history=history, generated=generated
    )
    assert torch.isfinite(actual).nonzero().flatten().tolist() == sorted(expected)
    assert actual[expected].tolist() == pytest.approx(
        [values[i] for i in expected], abs=1e-6
    )


def test_topk_ties_nucleus_boundary_and_minp() -> None:
    options = GenerationOptions(strategy="sample", temperature=1, top_k=1)
    assert torch.isfinite(
        sampling_logits(torch.tensor([2.0, 2.0, 1.0]), options)
    ).tolist() == [True, True, False]
    for p in (1e-6, 0.01, 0.5, 1.0):
        scores = sampling_logits(
            torch.tensor([2.0, 1.0, 0.0]), replace(options, top_k=0, top_p=p, min_p=1.0)
        )
        assert torch.isfinite(scores).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        sampling_logits(torch.ones(2), options, history=[True])


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_k", True),
        ("top_k", -1),
        ("top_p", 0),
        ("min_p", 1.1),
        ("repetition_penalty", 0),
        ("frequency_penalty", float("nan")),
        ("presence_penalty", 3),
    ],
)
def test_invalid_sampling_controls(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        replace(GenerationOptions(), **{field: value})
