"""Original arithmetic exact references and hostile human-source adapter inputs."""

import json
import re
from pathlib import Path

import pytest

from qvelt.eval.config import ArtifactRef
from qvelt.posttrain.control import arithmetic_control
from qvelt.posttrain.gsm8k import import_gsm8k_training, parse_training_rows


def test_two_answer_control_exact_values_and_ids() -> None:
    rows = arithmetic_control()
    assert len(rows) == len({r.identifier for r in rows}) == 32
    for row in rows:
        numbers = re.findall(r"\d+", row.question)
        assert int(numbers[0]) - int(numbers[1]) + int(numbers[2]) == int(row.answer)
    assert {r.answer for r in rows} == {"0", "1"}
    assert len(arithmetic_control("ru")) == 32


def test_original_train_numeric_label_extraction() -> None:
    data = json.dumps(
        {
            "question": "Original fixture: sum one thousand and one.",
            "answer": "A worked solution.\n#### 1,001",
        }
    ).encode()
    row = parse_training_rows(data, 1)[0]
    assert (
        row.answer == "1001"
        and row.difficulty == "unrated"
        and row.family == "gsm8k-original-train"
    )


@pytest.mark.parametrize(
    "answer",
    ["No delimiter", "#### NaN", "#### 1/0", "#### 1 #### 2", "#### __import__('os')"],
)
def test_invalid_source_answers(answer: str) -> None:
    with pytest.raises(ValueError):
        parse_training_rows(
            json.dumps({"question": "Question", "answer": answer}).encode(), 1
        )


def test_unreviewed_snapshot_is_rejected_before_output(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"unreviewed")
    reference = ArtifactRef(str(path), "a" * 64)
    with pytest.raises(ValueError, match="pinned"):
        import_gsm8k_training(reference, reference, tmp_path / "output")
    assert not (tmp_path / "output").exists()
