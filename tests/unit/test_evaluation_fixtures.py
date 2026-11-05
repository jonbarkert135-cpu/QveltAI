"""Prospective immutable family/partition identities and exact generated targets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from qvelt.data.records import iter_records
from qvelt.eval.fixtures import freeze_holdout, protected_cases


def test_frozen_counts_bands_and_deterministic_ids() -> None:
    validation, final = protected_cases("validation"), protected_cases("final")
    assert len(validation) == 200 and len(final) == 800
    assert len({d.identifier for d in [*validation, *final]}) == 1000
    assert len({json.loads(d.text)["prompt"] for d in [*validation, *final]}) == 1000
    assert not {d.family for d in validation} & {d.family for d in final}
    assert set(Counter(d.family for d in validation).values()) == {40}
    assert set(Counter(d.family for d in final).values()) == {160}
    assert validation == protected_cases("validation")
    assert all(json.loads(d.text)["target"] == d.answer for d in [*validation, *final])
    with pytest.raises(ValueError):
        protected_cases("test")


def test_freeze_is_write_once_and_eval_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    registry = freeze_holdout(tmp_path / "frozen", root)
    for part in ("validation", "final"):
        entry = registry["partitions"][part]
        assert (
            len(
                list(
                    iter_records(
                        tmp_path / "frozen" / entry["manifest"],
                        entry["manifest_sha256"],
                    )
                )
            )
            == entry["count"]
        )
        policy = json.loads((tmp_path / "frozen" / part / "usage.json").read_text())
        assert "training" not in policy["allowed_uses"]
    with pytest.raises(FileExistsError):
        freeze_holdout(tmp_path / "frozen", root)
