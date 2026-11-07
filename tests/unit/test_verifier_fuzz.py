"""Deterministic bounded parser fuzzing;generated text never reaches an evaluator."""

from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from qvelt.eval.scoring import extract_answer as eval_extract
from qvelt.eval.scoring import parse_number
from qvelt.posttrain.verifiers import (
    AnswerSpec,
    extract_answer,
    polynomial,
    verify_answer,
)


@settings(max_examples=1200, derandomize=True, database=None, deadline=None)
@given(st.text(max_size=5000))
def test_arbitrary_unicode_answers_are_bounded_and_fail_closed(text: str) -> None:
    for spec in (
        AnswerSpec("1"),
        AnswerSpec("1/3", "numeric", 1e-6),
        AnswerSpec("x+y", "polynomial"),
    ):
        assert type(verify_answer(text, spec)) is bool
    for framed in (True, False):
        answer = extract_answer(text, framed=framed)
        assert answer is None or 1 <= len(answer) <= 130
    for protocol in (
        "whole-v1",
        "final-line-v1",
        "gsm8k-strict-v1",
        "reasoning-final-v2",
    ):
        answer = eval_extract(text, protocol)
        assert answer is None or isinstance(answer, str)
    for mode in ("math", "numeric"):
        number = parse_number(text, mode)
        assert number is None or isinstance(number, Fraction)


@settings(max_examples=250, derandomize=True, database=None, deadline=None)
@given(st.integers(-1000, 1000), st.integers(1, 1000), st.integers(-1000, 1000))
def test_rational_and_polynomial_equivalence_against_exact_oracles(
    a: int, b: int, c: int
) -> None:
    value = Fraction(a, b)
    assert verify_answer(f"{a}/{b}", AnswerSpec(str(value)))
    assert verify_answer(f"{a}*x+{c}*y", AnswerSpec(f"{c}*y+{a}*x", "polynomial"))
    assert not verify_answer(f"{a}*x+{c}*y+1", AnswerSpec(f"{c}*y+{a}*x", "polynomial"))
    assert polynomial(f"({a}*x+{c}*y)**0") == {(0, 0): Fraction(1)}
