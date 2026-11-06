"""Bounded answer-only checks; generated strings never reach Python/SymPy evaluators."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction

import sympy as sp


def rational(text: str) -> Fraction:
    """ASCII integer/rational/decimal, at most 60 digits; no expressions/exponents."""
    if (
        not isinstance(text, str)
        or len(text) > 130
        or re.fullmatch(r"-?[0-9]{1,60}(?:/[1-9][0-9]{0,59}|\.[0-9]{1,30})?", text)
        is None
    ):
        raise ValueError("Expected bounded rational/decimal")
    return Fraction(text)


Polynomial = dict[tuple[int, int], Fraction]


def polynomial(text: str) -> Polynomial:
    """Rational polynomials in x,y, total degree<=8 and <=48 AST nodes.

    Division only by a nonzero rational constant. Thus cancellation never hides
    excluded variable denominator zeros. No calls, attributes or general powers.
    """
    if not isinstance(text, str) or not 1 <= len(text) <= 128:
        raise ValueError("Polynomial text exceeds limit")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Invalid polynomial grammar") from error
    if sum(1 for _ in ast.walk(tree)) > 48:
        raise ValueError("Polynomial AST exceeds limit")

    def bounded(values: Polynomial) -> Polynomial:
        result = {
            powers: coefficient for powers, coefficient in values.items() if coefficient
        }
        if any(
            sum(powers) > 8 or max(abs(c.numerator), c.denominator).bit_length() > 256
            for powers, c in result.items()
        ):
            raise ValueError("Polynomial degree/coefficient exceeds limit")
        return result

    def multiply(a: Polynomial, b: Polynomial) -> Polynomial:
        result: Polynomial = {}
        for (x, y), c in a.items():
            for (u, v), d in b.items():
                key = (x + u, y + v)
                result[key] = result.get(key, Fraction()) + c * d
        return bounded(result)

    def visit(node: ast.AST) -> Polynomial:
        if (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and abs(node.value) <= 1_000_000
        ):
            return bounded({(0, 0): Fraction(node.value)})
        if isinstance(node, ast.Name) and node.id in ("x", "y"):
            return {(1, 0) if node.id == "x" else (0, 1): Fraction(1)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return {k: sign * c for k, c in visit(node.operand).items()}
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            if isinstance(node.op, ast.Pow):
                if (
                    not isinstance(node.right, ast.Constant)
                    or type(node.right.value) is not int
                    or not 0 <= node.right.value <= 8
                ):
                    raise ValueError("Only literal powers0..8 allowed")
                result = {(0, 0): Fraction(1)}
                for _ in range(node.right.value):
                    result = multiply(result, left)
                return result
            right = visit(node.right)
            if isinstance(node.op, ast.Add | ast.Sub):
                sign = -1 if isinstance(node.op, ast.Sub) else 1
                for key, value in right.items():
                    left[key] = left.get(key, Fraction()) + sign * value
                return bounded(left)
            if isinstance(node.op, ast.Mult):
                return multiply(left, right)
            if isinstance(node.op, ast.Div) and set(right) == {(0, 0)}:
                return bounded({k: c / right[(0, 0)] for k, c in left.items()})
        raise ValueError("Unsupported polynomial grammar")

    return visit(tree.body)


@dataclass(frozen=True)
class AnswerSpec:
    gold: str
    mode: str = "exact"
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("exact", "numeric", "polynomial"):
            raise ValueError("Unknown verifier protocol")
        for value in (self.absolute_tolerance, self.relative_tolerance):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 0.1
            ):
                raise ValueError("Tolerance must be finite in [0,0.1]")
        if self.mode != "numeric" and (
            self.absolute_tolerance or self.relative_tolerance
        ):
            raise ValueError("Only numeric mode permits tolerance")
        if self.mode == "polynomial":
            polynomial(self.gold)
        else:
            rational(self.gold)


def verify_answer(answer: str, spec: AnswerSpec) -> bool:
    """Exact Fraction, rational tolerance or SymPy QQ polynomial identity."""
    if not isinstance(spec, AnswerSpec):
        raise ValueError("Expected validated answer specification")
    try:
        if spec.mode == "polynomial":
            actual, gold = polynomial(answer), polynomial(spec.gold)
            x, y = sp.symbols("x y")

            # Only trusted bounded coefficients are passed to SymPy constructors.
            def poly(values: Polynomial) -> sp.Poly:
                return sp.Poly.from_dict(
                    {
                        k: sp.Rational(c.numerator, c.denominator)
                        for k, c in values.items()
                    },
                    x,
                    y,
                    domain=sp.QQ,
                )

            return bool((poly(actual) - poly(gold)).is_zero)
        value, expected = rational(answer), rational(spec.gold)
        if spec.mode == "exact":
            return value == expected
        tolerance = Fraction(str(spec.absolute_tolerance)) + Fraction(
            str(spec.relative_tolerance)
        ) * abs(expected)
        return abs(value - expected) <= tolerance
    except (ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return False


def extract_answer(text: str, *, framed: bool = False) -> str | None:
    """Single strict final answer, optional one initial think block; no nested controls."""
    if (
        not isinstance(text, str)
        or len(text) > 4096
        or any(ord(c) < 32 and c not in "\n\t" for c in text)
    ):
        return None
    text = text.strip()
    if framed:
        if text.startswith("<think>"):
            parts = text[7:].split("</think>")
            if (
                len(parts) != 2
                or not parts[0].strip()
                or "<" in parts[0]
                or "Final answer:" in parts[0]
            ):
                return None
            text = parts[1].strip()
        if not text.startswith("Final answer: "):
            return None
        text = text[14:]
    if not text or len(text) > 130 or any(c in text for c in "\n\r\t<>"):
        return None
    return text


@dataclass(frozen=True)
class RewardPolicy:
    framed: bool = False
    format_weight: float = 0.0
    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        if type(self.framed) is not bool:
            raise ValueError("framed must be bool")
        for value in (self.format_weight, self.length_penalty):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 0.2
            ):
                raise ValueError("Reward shaping coefficients must be in [0,0.2]")


@dataclass(frozen=True)
class RewardScore:
    total: float
    correctness: float
    format: float
    length_cost: float


_DEFAULT_REWARD = RewardPolicy()


def score_completion(
    text: str,
    spec: AnswerSpec,
    complete: bool,
    tokens: int,
    max_tokens: int,
    policy: RewardPolicy = _DEFAULT_REWARD,
) -> RewardScore:
    """No format/correctness reward for truncated or unparseable responses."""
    if (
        type(complete) is not bool
        or type(tokens) is not int
        or type(max_tokens) is not int
        or not 1 <= tokens <= max_tokens <= 256
    ):
        raise ValueError("Invalid completion budget")
    answer = extract_answer(text, framed=policy.framed)
    valid = complete and answer is not None
    correct = float(valid and verify_answer(answer or "", spec))
    formatted = float(valid and _valid_answer_grammar(answer or "", spec.mode))
    cost = policy.length_penalty * tokens / max_tokens
    total = max(
        0.0,
        (1 - policy.format_weight) * correct + policy.format_weight * formatted - cost,
    )
    return RewardScore(total, correct, formatted, cost)


def _valid_answer_grammar(text: str, mode: str) -> bool:
    try:
        polynomial(text) if mode == "polynomial" else rational(text)
        return True
    except ValueError:
        return False
