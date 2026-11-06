"""Independent votes,witness taxonomy and safe integer-function reference cases."""

import itertools
import json

import pytest

from qvelt.eval.code_eval import score_code
from qvelt.eval.diagnostics import arithmetic_witnesses, failure_mode, majority_vote
from qvelt.eval.scoring import extract_answer, score_answer


def test_vote_exhaustive_prefixes_and_first_ties() -> None:
    for n in range(1, 7):
        for answers in itertools.product((None, "0", "1"), repeat=n):
            for k in range(1, n + 1):
                valid = [a for a in answers[:k] if a is not None]
                expected = (
                    None
                    if not valid
                    else sorted(
                        set(valid),
                        key=lambda value: (-valid.count(value), valid.index(value)),
                    )[0]
                )
                actual = majority_vote(list(answers), k)
                assert actual["answer"] == expected
                assert actual["valid_votes"] == len(valid)
    assert majority_vote(["0", "1"], 2)["tied"]
    with pytest.raises(ValueError):
        majority_vote(["1"], 2)


def test_checked_witnesses_and_unknown_method() -> None:
    assert arithmetic_witnesses("2+3=6; 9/3=3") == [
        {"claim": "2+3=6", "finding": "arithmetic_mismatch", "expected": "5"}
    ]
    assert arithmetic_witnesses("1/0=3")[0]["finding"] == "division_by_zero"
    row = {
        "correct": False,
        "format_valid": True,
        "complete": True,
        "stop": "complete",
        "text": "2+3=6\nFinal answer: 6",
    }
    assert failure_mode(row) == "arithmetic_slip_witnessed"
    row["text"] = "Method: multiply\nFinal answer: 6"
    assert failure_mode(row, expected_method="add") == "wrong_method_declared"
    assert failure_mode(row) == "incorrect_unclassified"
    assert failure_mode({**row, "stop": "token_limit"}) == "truncation"


SPEC = json.dumps(
    {
        "entry_point": "difference",
        "arguments": ["a", "b"],
        "tests": [
            {"args": [a, b], "expected": abs(a - b)}
            for a, b in [(-3, 4), (2, 2), (8, 1), (0, -5)]
        ],
    }
)


def test_code_matches_trusted_function_without_executing_text() -> None:
    source = "def difference(a,b):\n    return a-b if a>=b else b-a\n"
    result = score_code(source, SPEC, complete=True)
    assert result["correct"] and result["tests_passed"] == 4
    assert score_code("    return a-b if a>=b else b-a", SPEC, complete=True)["correct"]
    assert not score_code("    return a+b", SPEC, complete=True)["correct"]
    assert score_code(source, SPEC, complete=False)["code_failure"] == "truncation"
    assert (
        score_code("    return a//0", SPEC, complete=True)["code_failure"]
        == "runtime_error"
    )


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef difference(a,b):\n return a-b",
        "def difference(a,b):\n return __import__('os').system('false')",
        "def difference(a,b):\n while True: pass",
        "@evil()\ndef difference(a,b):\n return a-b",
        "def difference(a,b=evil()):\n return a-b",
        "def difference(a: evil(),b):\n return a-b",
        "def difference(a,b):\n return a if True else evil()",
        "def difference(a,b):\n return (x for x in ())",
        "def difference(a,b):\n return a.__class__",
        "def difference(a,b):\n return [a][0]",
    ],
)
def test_code_rejects_unsafe_python_even_in_dead_branches(source: str) -> None:
    assert not score_code(source, SPEC, complete=True)["correct"]


def test_gsm_strict_literal_protocol_matches_declared_filter() -> None:
    assert extract_answer("Work. #### 1,234", "gsm8k-strict-v1") == "1234"
    assert score_answer("#### 2", "2", "math", "gsm8k-strict-v1", complete=False)[
        "correct"
    ]
    assert not score_answer("#### -0", "0", "math", "gsm8k-strict-v1", complete=True)[
        "correct"
    ]
    assert not score_answer("#### 2.0", "2", "math", "gsm8k-strict-v1", complete=True)[
        "correct"
    ]
    # Strict harness filter takes first capture; this differs from final-line-v1.
    assert extract_answer("#### 2 then #### 3", "gsm8k-strict-v1") == "2"
