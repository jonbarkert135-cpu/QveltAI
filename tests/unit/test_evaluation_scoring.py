"""Independent exact subset/count references and adversarial answer formats."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from statistics import NormalDist

import pytest
import sympy as sp

from qvelt.eval.scoring import (
    extract_answer,
    parse_number,
    pass_at_k,
    perplexity,
    score_answer,
    wilson_interval,
)


def test_pass_at_k_matches_exhaustive_success_subsets() -> None:
    for n in range(1, 8):
        for c in range(n + 1):
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(range(n), k))
                expected = Fraction(
                    sum(any(i < c for i in subset) for subset in subsets), len(subsets)
                )
                assert pass_at_k(n, c, k) == float(expected)
    for args in ((0, 0, 1), (3, 4, 1), (2, 1, 3), (True, 1, 1)):
        with pytest.raises(ValueError):
            pass_at_k(*args)


@pytest.mark.parametrize(
    "correct,total",
    [(0, 1), (1, 1), (0, 20), (3, 20), (10, 20), (19, 20), (20, 20), (43, 200)],
)
def test_wilson_matches_high_precision_score_equation(correct: int, total: int) -> None:
    p = sp.Symbol("p")
    z = sp.Rational(str(NormalDist().inv_cdf(0.975)))
    roots = sp.solve(
        (sp.Rational(correct, total) - p) ** 2 - z**2 * p * (1 - p) / total, p
    )
    expected = sorted(float(root.evalf(50)) for root in roots)
    assert wilson_interval(correct, total) == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize(
    "bad",
    [
        "1/0",
        "NaN",
        "Infinity",
        "1 + 1",
        "2 kilograms",
        "__import__('os').system('false')",
        "2\n3",
        "1e999",
        "0." + "1" * 61,
        "9" * 131,
    ],
)
def test_unsafe_or_ambiguous_numeric_formats_are_invalid(bad: str) -> None:
    assert parse_number(bad, "numeric") is None
    assert not score_answer(bad, "2", "numeric", "whole-v1", complete=True)["correct"]


def test_exact_numeric_tolerance_and_completion_contract() -> None:
    assert score_answer(" 2/4 ", "1/2", "math", "whole-v1", complete=True)["correct"]
    assert not score_answer("0.5", "1/2", "math", "whole-v1", complete=True)[
        "format_valid"
    ]
    assert score_answer("2.50000001e-1", "0.25", "numeric", "whole-v1", complete=True)[
        "correct"
    ]
    assert not score_answer("0.251", "0.25", "numeric", "whole-v1", complete=True)[
        "correct"
    ]
    assert not score_answer("1/2", "1/2", "math", "whole-v1", complete=False)["correct"]
    assert extract_answer("Reasoning.\nFinal answer: 2/4", "final-line-v1") == "2/4"
    assert extract_answer("Final answer: 2\nFinal answer: 2", "final-line-v1") is None
    assert extract_answer("Final answer: 2\nmore text", "final-line-v1") is None
    with pytest.raises(ValueError):
        extract_answer("2", "unknown")


def test_perplexity_is_target_weighted_and_withholds_failed_corpus() -> None:
    # One two-way target and three four-way targets:geometric,not arithmetic mean.
    assert perplexity(math.log(2) + 3 * math.log(4), 4) == pytest.approx(128**0.25)
    assert perplexity(1, 0) is None
    assert perplexity(1, 2, 1) is None
    assert perplexity(1000, 1) is None
    with pytest.raises(ValueError):
        perplexity(float("nan"), 2)
