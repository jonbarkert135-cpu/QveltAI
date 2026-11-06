"""Observed answer votes and checked failure witnesses, never inferred cognition."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from typing import Any

from qvelt.eval.scoring import parse_number


def majority_vote(answers: list[str | None], k: int) -> dict[str, Any]:
    """Plurality of the first k attempts; invalids do not vote, ties use first seen.

    Canonicalize answers before calling. Invalid attempts still count in k and
    never trigger replacement sampling. This is a measured vote, not pass@k.
    """
    if (
        type(k) is not int
        or not 1 <= k <= len(answers) <= 32
        or any(a is not None and not isinstance(a, str) for a in answers)
    ):
        raise ValueError("Votes require1<=k<=n<=32 and canonical strings/invalid None")
    counts = Counter(a for a in answers[:k] if a is not None)
    winner = max(counts, key=lambda a: counts[a]) if counts else None
    return {
        "answer": winner,
        "k": k,
        "valid_votes": sum(counts.values()),
        "winner_votes": counts[winner] if winner is not None else 0,
        "tied": sum(v == max(counts.values()) for v in counts.values()) > 1
        if counts
        else False,
    }


def canonical_answer(
    attempt: dict[str, Any], kind: str, protocol: str = "whole-v1"
) -> str | None:
    if (
        attempt.get("complete") is not True and protocol != "gsm8k-strict-v1"
    ) or attempt.get("format_valid") is not True:
        return None
    if protocol == "gsm8k-strict-v1":
        if attempt.get("stop") not in ("complete", "token_limit"):
            return None
        return str(attempt["extracted"])
    value = parse_number(attempt.get("extracted", ""), kind)
    return str(value) if value is not None else None


def arithmetic_witnesses(text: str) -> list[dict[str, str]]:
    """Check at most16 explicit integer op integer = rational claims, no text eval."""
    if not isinstance(text, str) or len(text) > 8192:
        raise ValueError("Bounded text required for arithmetic audit")
    pattern = r"(?<![\w.])-?[0-9]{1,12}\s*[+*/-]\s*-?[0-9]{1,12}\s*=\s*-?[0-9]{1,24}(?:/[1-9][0-9]{0,23})?(?![\w./])"
    rows = []
    for match in list(re.finditer(pattern, text))[:16]:
        parts = re.fullmatch(
            r"(-?[0-9]+)\s*([+*/-])\s*(-?[0-9]+)\s*=\s*(-?[0-9]+(?:/[0-9]+)?)",
            match.group(),
        )
        assert parts is not None
        a, op, b, claimed = parts.groups()
        left, right = Fraction(a), Fraction(b)
        if op == "/" and right == 0:
            rows.append({"claim": match.group(), "finding": "division_by_zero"})
            continue
        actual = (
            left + right
            if op == "+"
            else left - right
            if op == "-"
            else left * right
            if op == "*"
            else left / right
        )
        if actual != Fraction(claimed):
            rows.append(
                {
                    "claim": match.group(),
                    "finding": "arithmetic_mismatch",
                    "expected": str(actual),
                }
            )
    return rows


def failure_mode(attempt: dict[str, Any], *, expected_method: str | None = None) -> str:
    """Specific labels require witnesses; wrong answer alone stays unclassified."""
    if attempt.get("correct") is True:
        return "correct"
    stop = attempt.get("stop")
    if stop in (
        "token_limit",
        "max_tokens",
        "character_limit",
        "output_limit",
        "context_overflow",
    ):
        return "truncation"
    if stop in ("nonfinite_logits", "nonfinite"):
        return "nonfinite"
    if attempt.get("code_failure") is not None:
        return str(attempt["code_failure"])
    if not attempt.get("format_valid") or not attempt.get("complete"):
        return "format_error"
    text = str(attempt.get("text", ""))
    methods = re.findall(r"(?m)^Method: ([a-z][a-z_-]{0,31})$", text)
    if (
        expected_method is not None
        and len(methods) == 1
        and methods[0] != expected_method
    ):
        return "wrong_method_declared"
    if arithmetic_witnesses(text):
        return "arithmetic_slip_witnessed"
    return "incorrect_unclassified"


def reasoning_text(text: str) -> str:
    """Only explicit think delimiters define reasoning; answer length is separate."""
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return ""
    start, end = text.find("<think>") + 7, text.find("</think>")
    return text[start:end] if end >= start else ""
