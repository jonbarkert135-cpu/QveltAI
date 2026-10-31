"""Independent exact answers, bounded rewards and conservative text policies."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from qvelt.data.cleaning import Cleaner, CleaningConfig, words
from qvelt.data.math import MathProblem, generate_problems


@pytest.mark.parametrize("language", ["en", "ru"])
def test_math_families_and_exact_oracles(language: str) -> None:
    problems = generate_problems(48, seed=37, language=language)
    assert problems == generate_problems(48, seed=37, language=language)
    assert len({p.family for p in problems}) == 8
    assert {p.difficulty for p in problems} == {"easy", "medium", "hard"}
    assert len({p.identifier for p in problems}) == 48
    for p in problems:
        assert p.reward(p.answer) == 1
        assert p.reward(str(Fraction(p.answer) + 1)) == 0
        numbers = re.findall(r"-?[0-9]+(?:/[0-9]+)?", p.question)
        if p.family == "arithmetic-sum":
            a, b, c = map(int, numbers)
            assert Fraction(p.answer) == a + b * c
        elif p.family == "arithmetic-fraction":
            left, right = map(Fraction, numbers)
            assert Fraction(p.answer) == left + right
        elif p.family.startswith("linear"):
            slope, intercept, rhs = map(Fraction, numbers)
            assert slope * Fraction(p.answer) + intercept == rhs
        elif p.family.startswith("number"):
            a, b = map(int, numbers)
            x, y = a, b
            while y:
                x, y = y, x % y
            assert Fraction(p.answer) == (x if p.family == "number-gcd" else a * b // x)
        else:
            n, k = map(int, numbers)
            product = 1
            for i in range(k):
                product *= n - i
            if p.family == "combinations":
                for i in range(1, k + 1):
                    product //= i
            assert Fraction(p.answer) == product


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",
        "1/0",
        "2**1000000000",
        "9" * 1000,
        "١",
        "NaN",
        "1/01",
        "1e0",
        "<think>1</think>",
    ],
)
def test_untrusted_rewards_are_bounded(text: str) -> None:
    assert MathProblem("id", "family", "easy", "en", "q", "1").reward(text) == 0


def test_rewards_accept_equivalent_rationals() -> None:
    problem = MathProblem("id", "family", "easy", "en", "q", "1/2")
    assert problem.reward(" 2/4\n") == 1
    with pytest.raises(ValueError):
        generate_problems(True)
    with pytest.raises(ValueError):
        generate_problems(2, seed=-1)


def test_pii_and_preserved_math_unicode() -> None:
    cleaner = Cleaner(CleaningConfig(), 11)
    text = (
        "Email alice@example.invalid; телефон: +7 (999) 555-01-23. sk-"
        + "a" * 30
        + "\n123456789012345\tёж e\u0301 😀"
    )
    result = cleaner.clean(text)
    assert result.rejection is None
    assert "alice@" not in result.text and "555-01-23" not in result.text
    assert "a" * 30 not in result.text
    assert (
        result.redactions["email"]
        == result.redactions["phone"]
        == result.redactions["api_key"]
        == 1
    )
    assert "123456789012345\tёж e\u0301 😀" in result.text
    assert cleaner.clean(text) == result
    key = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"
    assert "abc" not in cleaner.clean(key + " enough readable material").text
    assert (
        "12345678901234567890"
        not in cleaner.clean("secret=12345678901234567890 has value").text
    )


def test_cleaning_limits_toxicity_and_seeded_lid() -> None:
    cleaner = Cleaner(CleaningConfig(), 91)
    assert (
        cleaner.clean("You must kill yourself right now.").rejection
        == "toxic_phrase_quarantine"
    )
    assert cleaner.clean("same long line\n" * 10).rejection == "repeated_lines"
    assert (
        cleaner.clean("<think>hidden chain of thought</think>").rejection
        == "control_literal"
    )
    assert cleaner.clean(" \t\n" * 10).rejection == "character_limit"
    text = "Русская математика изучает числа и доказательства. Ученики решают задачи и проверяют результаты вычислений."
    assert cleaner.clean(text).language == "ru"
    assert words("x + 123 = y", casefold=True) == ["x", "+", "123", "=", "y"]
    with pytest.raises(ValueError):
        CleaningConfig(max_repeated_line_fraction=float("nan"))
