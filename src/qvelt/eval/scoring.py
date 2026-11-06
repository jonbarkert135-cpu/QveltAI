"""Bounded answer extraction,exact/numeric comparison and explicit count estimators."""

from __future__ import annotations

import math
import re
from decimal import Decimal
from fractions import Fraction
from statistics import NormalDist


def parse_number(text: str, kind: str = "math") -> Fraction | None:
    if not isinstance(text, str) or len(text) > 130:
        return None
    value = text.strip()
    rational = r"-?[0-9]{1,60}(?:/[1-9][0-9]{0,59})?"
    if re.fullmatch(rational, value):
        return Fraction(value)
    if kind != "numeric" or not re.fullmatch(
        r"-?[0-9]{1,60}(?:\.[0-9]{1,60})?(?:[eE][+-]?[0-9]{1,3})?", value
    ):
        return None
    if sum(c.isdigit() for c in value.split("e")[0].split("E")[0]) > 60:
        return None
    number = Decimal(value)
    exponent = number.as_tuple().exponent
    if not number.is_finite() or not isinstance(exponent, int) or abs(exponent) > 100:
        return None
    return Fraction(number)


def extract_answer(text: str, protocol: str) -> str | None:
    if not isinstance(text, str) or len(text) > 8192:
        return None
    if protocol == "whole-v1":
        return text.strip()
    if protocol == "reasoning-final-v2":
        from qvelt.posttrain.verifiers import extract_answer as framed_answer

        return framed_answer(text, framed=True)
    if protocol == "gsm8k-strict-v1":
        matches = re.findall(r"#### (\-?[0-9\.\,]+)", text)
        return matches[0].replace(",", "").removesuffix(".") if matches else None
    if protocol != "final-line-v1":
        raise ValueError("Unknown answer extraction protocol")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    marker = "Final answer:"
    if not lines or text.count(marker) != 1 or not lines[-1].startswith(marker):
        return None
    return lines[-1][len(marker) :].strip()


def score_answer(
    text: str,
    target: str,
    kind: str,
    protocol: str,
    *,
    complete: bool,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, object]:
    if (
        kind not in ("math", "numeric")
        or type(complete) is not bool
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in (atol, rtol)
        )
    ):
        raise ValueError("Invalid numeric scoring contract")
    gold = parse_number(target, kind)
    if gold is None:
        raise ValueError("Malformed evaluation gold answer")
    extracted = extract_answer(text, protocol)
    if protocol == "gsm8k-strict-v1":
        # lm-eval0.4.13 strict-match uses first regex capture and literal equality;
        # a terminal EOS is not a prerequisite of the publisher/harness metric.
        expected = target.replace(",", "").removesuffix(".")
        return {
            "correct": extracted == expected,
            "format_valid": extracted is not None,
            "extracted": extracted,
            "complete": complete,
        }
    predicted = None if extracted is None else parse_number(extracted, kind)
    correct = False
    if complete and predicted is not None:
        tolerance = (
            Fraction(0)
            if kind == "math"
            else Fraction(str(atol)) + Fraction(str(rtol)) * abs(gold)
        )
        correct = abs(predicted - gold) <= tolerance
    return {
        "correct": correct,
        "format_valid": predicted is not None,
        "extracted": extracted,
        "complete": complete,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    """1-C(n-c,k)/C(n,k);failed attempts are included in n,never filtered."""
    if any(type(v) is not int for v in (n, c, k)) or not 1 <= k <= n or not 0 <= c <= n:
        raise ValueError("pass@k requires n>=k>=1 and0<=c<=n")
    return float(
        1 - Fraction(math.comb(n - c, k) if n - c >= k else 0, math.comb(n, k))
    )


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """Descriptive95% Wilson score interval;population use requires iid problems."""
    if (
        type(correct) is not int
        or type(total) is not int
        or total < 1
        or not 0 <= correct <= total
    ):
        raise ValueError("Wilson interval needs0<=correct<=total and positive total")
    z = NormalDist().inv_cdf(0.975)
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def perplexity(nll: float, targets: int, failed_documents: int = 0) -> float | None:
    if (
        not math.isfinite(nll)
        or nll < 0
        or type(targets) is not int
        or targets < 0
        or type(failed_documents) is not int
        or failed_documents < 0
    ):
        raise ValueError("Invalid aggregate LM loss/count")
    if failed_documents or targets == 0:
        return None
    try:
        return math.exp(nll / targets)
    except OverflowError:
        return None
