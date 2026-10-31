"""Original exact-answer bilingual math fixtures and bounded reward parsing."""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from fractions import Fraction

import sympy as sp


@dataclass(frozen=True)
class MathProblem:
    identifier: str
    family: str
    difficulty: str
    language: str
    question: str
    answer: str

    def reward(self, response: str) -> float:
        """Only bounded ASCII integer/p/q answers; no sympify/eval of model text."""
        text = response.strip()
        if (
            len(text) > 130
            or re.fullmatch(r"-?[0-9]{1,60}(?:/[1-9][0-9]{0,59})?", text) is None
        ):
            return 0.0
        try:
            return float(Fraction(text) == Fraction(self.answer))
        except (ValueError, ZeroDivisionError):
            return 0.0


def generate_problems(
    count: int, seed: int = 20261031, language: str = "en"
) -> list[MathProblem]:
    """Difficulty scales integer ranges; exact solvers verify every produced gold."""
    if (
        type(count) is not int
        or not 1 <= count <= 100000
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or language not in ("en", "ru")
    ):
        raise ValueError("Invalid problem count or language")
    rng = random.Random(seed)
    output = []
    for index in range(count):
        level = ("easy", "medium", "hard")[index % 3]
        bound = {"easy": 20, "medium": 1000, "hard": 1000000}[level]
        kind = index % 8
        a, b, c = (rng.randint(1, bound) for _ in range(3))
        if kind in (0, 1):
            family = "arithmetic-sum" if kind == 0 else "arithmetic-fraction"
            if kind == 0:
                answer = Fraction(a + b * c)
                expression = f"{a} + {b} * {c}"
                checked = sp.Integer(a) + sp.Integer(b) * sp.Integer(c)
            else:
                denominator = rng.randint(1, max(2, bound // 2))
                answer = Fraction(a, b) + Fraction(c, denominator)
                expression = f"{a}/{b} + {c}/{denominator}"
                checked = sp.Rational(a, b) + sp.Rational(c, denominator)
            question = (
                f"Compute exactly: {expression}. Give an integer or p/q."
                if language == "en"
                else f"Вычислите точно: {expression}. Ответ: целое число или p/q."
            )
        elif kind in (2, 3):
            family = "linear-integer" if kind == 2 else "linear-rational"
            answer = Fraction(
                rng.randint(-bound, bound), 1 if kind == 2 else rng.randint(1, 20)
            )
            rhs = a * answer + b
            question = (
                f"Solve for x: {a}*x + {b} = {rhs}. Give an exact fraction."
                if language == "en"
                else f"Решите уравнение: {a}*x + {b} = {rhs}. Ответьте точной дробью."
            )
            x = sp.Symbol("x")
            checked = sp.solve(
                sp.Eq(a * x + b, sp.Rational(rhs.numerator, rhs.denominator)), x
            )[0]
        elif kind in (4, 5):
            family = "number-gcd" if kind == 4 else "number-lcm"
            answer = Fraction(math.gcd(a, b) if kind == 4 else math.lcm(a, b))
            checked = sp.gcd(a, b) if kind == 4 else sp.lcm(a, b)
            operation = "gcd" if kind == 4 else "lcm"
            question = (
                f"Find the integer {operation} of {a} and {b}."
                if language == "en"
                else f"Найдите целое значение {operation} чисел {a} и {b}."
            )
        else:
            family = "combinations" if kind == 6 else "permutations"
            n = rng.randint(3, {"easy": 10, "medium": 25, "hard": 50}[level])
            k = rng.randint(1, min(n, 10))
            answer = Fraction(math.comb(n, k) if kind == 6 else math.perm(n, k))
            checked = (
                sp.binomial(n, k)
                if kind == 6
                else sp.factorial(n) / sp.factorial(n - k)
            )
            operation = "C" if kind == 6 else "P"
            question = (
                f"Compute {operation}({n},{k}) for unordered combinations C or ordered selections P, without replacement."
                if language == "en"
                else f"Вычислите {operation}({n},{k}): C означает сочетания, P — размещения без повторений."
            )
        if sp.Rational(answer.numerator, answer.denominator) != checked:
            raise ArithmeticError("Independent exact math verification failed")
        identifier = hashlib.sha256(
            f"{seed}:{language}:{index}:{question}".encode()
        ).hexdigest()
        output.append(
            MathProblem(identifier, family, level, language, question, str(answer))
        )
    return output
