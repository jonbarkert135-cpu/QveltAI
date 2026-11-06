"""Independent bounded symbolic/numeric and reward-hacking oracles."""

from fractions import Fraction

import pytest
import sympy as sp

from qvelt.posttrain.reasoning import diverse, pass_at_k
from qvelt.posttrain.verifiers import (
    AnswerSpec,
    RewardPolicy,
    extract_answer,
    polynomial,
    score_completion,
    verify_answer,
)


@pytest.mark.parametrize("answer", ["1/2", "2/4", "0.5", "0001/2"])
def test_exact_rational_reference(answer: str) -> None:
    assert verify_answer(answer, AnswerSpec("1/2"))
    assert Fraction(answer) == Fraction(1, 2)


@pytest.mark.parametrize(
    "answer",
    [
        "1+1",
        "1/0",
        "NaN",
        "inf",
        "true",
        "１",
        "1\x00",
        "1\n2",
        "1e999999",
        "9" * 100000,
    ],
)
def test_adversarial_exact(answer: str) -> None:
    assert not verify_answer(answer, AnswerSpec("1"))


def test_numeric_tolerance_exact_boundary() -> None:
    spec = AnswerSpec("100", "numeric", 0.001, 0.0001)
    assert verify_answer("100.011", spec)
    assert not verify_answer("100.0110001", spec)
    assert not verify_answer("100.001", AnswerSpec("100"))
    with pytest.raises(ValueError):
        AnswerSpec("1", "exact", 0.1)


def test_symbolic_identity_and_scalar_reference() -> None:
    spec = AnswerSpec("x**2+2*x*y+y**2", "polynomial")
    assert verify_answer("(x+y)**2", spec)
    assert not verify_answer("x**2+y**2", spec)
    values = polynomial("(x+y)**2 / 2")
    for x in (-3, 0, 7):
        for y in (-4, 0, 2):
            actual = sum(
                c * Fraction(x) ** i * Fraction(y) ** j for (i, j), c in values.items()
            )
            assert actual == Fraction((x + y) ** 2, 2)
    # Executed trusted SymPy identity, no generated strings passed to it.
    x, y = sp.symbols("x y")
    assert sp.expand((x + y) ** 2 - (x * x + 2 * x * y + y * y)) == 0


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",
        "x.__class__",
        "[x][0]",
        "sin(x)",
        "x/x",
        "x**-1",
        "x**999999999",
        "2**2**8",
        "True",
        "x**8*x",
        "1/0",
        "z",
        "9" * 129,
    ],
)
def test_symbolic_grammar_and_domain_defenses(text: str) -> None:
    assert not verify_answer(text, AnswerSpec("1", "polynomial"))


@pytest.mark.parametrize(
    "text",
    [
        "Final answer: 1\nFinal answer: 2",
        "Final answer: ",
        "<think></think>Final answer: 1",
        "<think><think>x</think></think>Final answer: 1",
        "<think>Final answer: 2</think>Final answer: 1",
        "Final answer: 1\x00",
        "Final answer: ​1",
        "Final answer: 1 ignore verifier",
    ],
)
def test_answer_format_exploits(text: str) -> None:
    score = score_completion(
        text, AnswerSpec("1"), True, 5, 16, RewardPolicy(True, 0.1, 0)
    )
    assert score.total == 0 and score.correctness == 0


def test_format_correctness_length_and_truncation_separate() -> None:
    text = "<think>Subtract equal numbers, then add one.</think>\nFinal answer: 1"
    policy = RewardPolicy(True, 0.1, 0.2)
    score = score_completion(text, AnswerSpec("1"), True, 4, 16, policy)
    assert score.correctness == score.format == 1
    assert score.length_cost == 0.05 and score.total == 0.95
    truncated = score_completion(text, AnswerSpec("1"), False, 16, 16, policy)
    assert truncated.total == truncated.correctness == truncated.format == 0
    wrong = score_completion("Final answer: 2", AnswerSpec("1"), True, 4, 16, policy)
    assert wrong.correctness == 0 and wrong.total == 0.05
    assert extract_answer("<think>x</think>Final answer: 1", framed=True) == "1"


def test_pass_at_k_exhaustive_enumeration_and_diversity() -> None:
    import itertools

    for n in range(1, 9):
        for c in range(n + 1):
            for k in range(1, n + 1):
                outcomes = [
                    any(i < c for i in chosen)
                    for chosen in itertools.combinations(range(n), k)
                ]
                assert pass_at_k(n, c, k) == pytest.approx(
                    sum(outcomes) / len(outcomes)
                )
    assert not diverse("ONE  solution", ["one solution"], 0.9)
    assert diverse("second different derivation", ["one solution"], 0.9)


def test_fixed_verifier_controls_are_not_reward_targets() -> None:
    from qvelt.posttrain.verifier_audit import audit_verifiers

    result = audit_verifiers()
    assert result["passed"] and result["cases"] == 14
    assert all(
        row["gold"] not in ("0", "1") or row["mode"] == "polynomial"
        for row in result["rows"]
    )
