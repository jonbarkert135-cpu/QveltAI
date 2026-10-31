"""Actual SQLite near-duplicate candidates and mandatory complete benchmark gates."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from qvelt.data.decontamination import (
    MANDATORY_BENCHMARKS,
    BenchmarkRegistry,
    Decontaminator,
)
from qvelt.data.dedup import DedupConfig, DiskDeduplicator, shingles
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.utils.manifests import manifest_digest


def registry(
    tmp_path: Path, names: list[str], *, purpose: str = "demo", fixture: bool = True
) -> tuple[Path, str]:
    raw = write_raw_bundle(
        [
            RawDocument(
                "Protected exact equation α + 7421 = 9837 for all of these explicitly authored test registries."
            )
        ],
        tmp_path / "questions",
        authored_source("test-only"),
        PERMISSION,
    )
    entries = [
        {
            "name": name,
            "manifest": "questions/manifest.json",
            "sha256": manifest_digest(raw),
            "approved": True,
            "fixture": fixture,
        }
        for name in names
    ]
    value = {
        "schema_version": 1,
        "purpose": purpose,
        "required": names,
        "benchmarks": entries,
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(value))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_disk_dedup_exact_near_and_math_distinctions() -> None:
    with sqlite3.connect(":memory:") as db:
        dedup = DiskDeduplicator(db, DedupConfig(), 37)
        text = " ".join(f"unique{i}" for i in range(200))
        assert dedup.duplicate(text) is None
        dedup.add(text, "{}", "train")
        assert dedup.duplicate(text) == "exact_duplicate"
        assert (
            dedup.duplicate(text.replace("unique100", "alternative100"))
            == "near_duplicate"
        )
        assert dedup.duplicate("unrelated mathematical text x + 1 = 2") is None
        assert shingles("x + 1", 5) != shingles("x - 1", 5)
        assert dedup.signature(text) == dedup.signature(text)
        assert dedup.signature("")[0] == set()


def test_lsh_collision_is_verified_and_candidate_cap_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite3.connect(":memory:") as db:
        dedup = DiskDeduplicator(
            db, DedupConfig(num_perm=8, bands=1, max_candidates=2), 2
        )
        monkeypatch.setattr(
            dedup, "signature", lambda text: (shingles(text, 5), [b"same-bucket"])
        )
        dedup.add("all words here differ completely", "{}", "train")
        assert dedup.duplicate("new and unrelated mathematical statement") is None
        dedup.add("a second distinct mathematical sample", "{}", "train")
        dedup.add("a third different mathematical sample", "{}", "train")
        assert (
            dedup.duplicate("query with too many band collisions")
            == "near_candidate_limit"
        )
    with pytest.raises(ValueError):
        DedupConfig(num_perm=63, bands=8)


def test_every_registered_benchmark_is_checked(tmp_path: Path) -> None:
    names = sorted(MANDATORY_BENCHMARKS) + ["extra-planned"]
    path, checksum = registry(tmp_path, names)
    checked = BenchmarkRegistry.load(path, checksum, "demo")
    with sqlite3.connect(":memory:") as db:
        decontam = Decontaminator(db, checked, ngram_size=7)
        text = "prefix PROTECTED EXACT EQUATION α + 7421 = 9837 for all of these explicitly authored test registries. suffix"
        assert decontam.matches(text) == set(names)
        assert (
            decontam.matches("independent unrelated equation a + 127 = 2048") == set()
        )
        assert set(decontam.questions) == set(names)


@pytest.mark.parametrize(
    "mutation", ["missing", "fixture", "unapproved", "hash", "schema", "duplicate"]
)
def test_production_gate_refuses_incomplete_or_unapproved_snapshots(
    tmp_path: Path, mutation: str
) -> None:
    names = sorted(MANDATORY_BENCHMARKS)
    path, _ = registry(tmp_path, names, purpose="production", fixture=False)
    raw: dict[str, Any] = json.loads(path.read_bytes())
    if mutation == "missing":
        raw["benchmarks"].pop()
    elif mutation == "fixture":
        raw["benchmarks"][0]["fixture"] = True
    elif mutation == "unapproved":
        raw["benchmarks"][0]["approved"] = False
    elif mutation == "schema":
        raw["schema_version"] = True
    elif mutation == "duplicate":
        raw["benchmarks"].append(raw["benchmarks"][0])
    path.write_text(json.dumps(raw))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if mutation == "hash":
        checksum = "0" * 64
    with pytest.raises(ValueError):
        BenchmarkRegistry.load(path, checksum, "production")


def test_short_questions_and_unicode_normalization(tmp_path: Path) -> None:
    raw = write_raw_bundle(
        [RawDocument("café + 37?")],
        tmp_path / "short",
        authored_source("short"),
        PERMISSION,
    )
    body = {
        "schema_version": 1,
        "purpose": "demo",
        "required": ["short"],
        "benchmarks": [
            {
                "name": "short",
                "manifest": "short/manifest.json",
                "sha256": manifest_digest(raw),
                "approved": True,
                "fixture": True,
            }
        ],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(body))
    reg = BenchmarkRegistry.load(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), "demo"
    )
    with sqlite3.connect(":memory:") as db:
        clean = Decontaminator(db, reg)
        assert clean.matches("prefix CAFE\u0301 + 37? suffix") == {"short"}
        assert clean.matches("café + 38?") == set()


def test_similar_math_templates_do_not_erase_distinct_numbers_or_signs() -> None:
    template = " ".join(
        ["This is a long repeated mathematical explanation without numbers"] * 30
    )
    with sqlite3.connect(":memory:") as db:
        dedup = DiskDeduplicator(db, DedupConfig(), 37)
        dedup.add(template + " Equation x + 12 = 37.", "{}", "train")
        assert dedup.duplicate(template + " Equation x + 13 = 37.") is None
        assert dedup.duplicate(template + " Equation x - 12 = 37.") is None
