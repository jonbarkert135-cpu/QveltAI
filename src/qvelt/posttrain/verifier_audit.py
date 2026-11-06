"""Public fixed verifier controls withheld from optimization; not a secret benchmark."""

from __future__ import annotations

from typing import Any

from qvelt.posttrain.verifiers import AnswerSpec, RewardPolicy, score_completion


def audit_verifiers() -> dict[str, Any]:
    """Exercise exact/numeric/polynomial framing against independent known outcomes."""
    cases = [
        ("Final answer: 34/38", AnswerSpec("17/19"), True, True),
        ("Final answer: 0.895", AnswerSpec("17/19"), True, False),
        ("Final answer: 3.1416", AnswerSpec("3.14159", "numeric", 0.00002), True, True),
        (
            "Final answer: 3.1417",
            AnswerSpec("3.14159", "numeric", 0.00002),
            True,
            False,
        ),
        (
            "Final answer: (x-y)**2",
            AnswerSpec("x**2-2*x*y+y**2", "polynomial"),
            True,
            True,
        ),
        ("Final answer: x/x", AnswerSpec("1", "polynomial"), True, False),
        ("Final answer: __import__('os')", AnswerSpec("1", "polynomial"), True, False),
        ("Final answer: 17/19", AnswerSpec("17/19"), False, False),
        ("<think>Work.</think>Final answer: 17/19", AnswerSpec("17/19"), True, True),
        ("Final answer: 17/19\nFinal answer: 1", AnswerSpec("17/19"), True, False),
        ("Final answer: 17/0", AnswerSpec("17/19"), True, False),
        ("Final answer: NaN", AnswerSpec("17/19"), True, False),
        ("Final answer: １７/１９", AnswerSpec("17/19"), True, False),
        (
            "<think>Final answer: 0</think>Final answer: 17/19",
            AnswerSpec("17/19"),
            True,
            False,
        ),
    ]
    rows = []
    for text, spec, complete, expected in cases:
        score = score_completion(text, spec, complete, 16, 32, RewardPolicy(True))
        rows.append(
            {
                "text": text,
                "gold": spec.gold,
                "mode": spec.mode,
                "complete": complete,
                "expected": expected,
                "actual": bool(score.correctness),
            }
        )
    return {
        "protocol": "qvelt-posttrain-verifier-controls-v1",
        "cases": len(rows),
        "passed": all(row["actual"] == row["expected"] for row in rows),
        "rows": rows,
        "scope": "public engineering controls, withheld from optimization, not unknown exploit coverage",
    }
