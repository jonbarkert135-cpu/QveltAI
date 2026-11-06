"""Preregistered easy arithmetic mechanism fixture, never a general math benchmark."""

from __future__ import annotations

import hashlib

from qvelt.data.math import MathProblem


def arithmetic_control(language: str = "en") -> list[MathProblem]:
    if language not in ("en", "ru"):
        raise ValueError("Control supports EN/RU")
    result = []
    for index in range(32):
        a, answer = index + 2, index % 2
        question = (
            f"Compute {a}-{a}+{answer}. Give an integer only."
            if language == "en"
            else f"Вычислите {a}-{a}+{answer}. Только целое число."
        )
        result.append(
            MathProblem(
                hashlib.sha256(question.encode()).hexdigest(),
                f"arithmetic-control-stratum-{index % 8}",
                "easy",
                language,
                question,
                str(answer),
            )
        )
    return result
