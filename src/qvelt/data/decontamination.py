"""Mandatory planned-benchmark registry and streaming normalized n-gram exclusion."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from qvelt.data.cleaning import words
from qvelt.utils.manifest_types import digest, identifier
from qvelt.utils.manifests import _reject_constant, _unique_mapping

# Include the frozen project-authored final set planned by training RFC section6.
MANDATORY_BENCHMARKS = frozenset(
    {"gsm8k", "math", "aime", "mmlu", "qvelt-math-heldout"}
)


def normalized_tokens(text: str) -> list[str]:
    return words(unicodedata.normalize("NFC", text), casefold=True)


@dataclass(frozen=True)
class BenchmarkRegistry:
    purpose: str
    path: Path
    sha256: str
    entries: tuple[dict[str, object], ...]
    required: frozenset[str]

    @classmethod
    def load(cls, path: Path, expected_sha256: str, purpose: str) -> BenchmarkRegistry:
        with path.open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        if (
            len(data) > 1024 * 1024
            or hashlib.sha256(data).hexdigest() != expected_sha256
        ):
            raise ValueError("Benchmark registry size/hash mismatch")
        raw = json.loads(
            data, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
        )
        if (
            not isinstance(raw, dict)
            or set(raw) != {"schema_version", "purpose", "required", "benchmarks"}
            or type(raw["schema_version"]) is not int
            or raw["schema_version"] != 1
        ):
            raise ValueError("Invalid benchmark registry schema")
        if raw["purpose"] != purpose or purpose not in ("demo", "production"):
            raise ValueError("Benchmark registry purpose mismatch")
        if not isinstance(raw["required"], list) or not all(
            isinstance(v, str) for v in raw["required"]
        ):
            raise ValueError("Invalid required benchmark list")
        if len(raw["required"]) > 128 or len(set(raw["required"])) != len(
            raw["required"]
        ):
            raise ValueError("Required benchmarks must be unique and bounded")
        for name in raw["required"]:
            identifier(name, "required benchmark")
        entries = raw["benchmarks"]
        if not isinstance(entries, list) or not entries or len(entries) > 128:
            raise ValueError("At least one bounded benchmark entry is required")
        names = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "name",
                "manifest",
                "sha256",
                "approved",
                "fixture",
            }:
                raise ValueError("Invalid benchmark entry")
            if not isinstance(entry["name"], str) or entry["name"] in names:
                raise ValueError("Duplicate/invalid benchmark name")
            identifier(entry["name"], "benchmark.name")
            digest(entry["sha256"], "benchmark.sha256")
            if (
                not isinstance(entry["manifest"], str)
                or not entry["manifest"]
                or len(entry["manifest"]) > 2048
                or "\x00" in entry["manifest"]
            ):
                raise ValueError("Invalid benchmark manifest path")
            names.add(entry["name"])
            if (
                type(entry["approved"]) is not bool
                or type(entry["fixture"]) is not bool
                or not entry["approved"]
            ):
                raise ValueError("Benchmark snapshot rights are not approved")
            if purpose == "production" and entry["fixture"]:
                raise ValueError(
                    "Authored stand-ins cannot pass the production benchmark gate"
                )
        required = frozenset(raw["required"]) | (
            MANDATORY_BENCHMARKS if purpose == "production" else frozenset()
        )
        if not required <= names:
            raise ValueError("Missing mandatory evaluation benchmark snapshots")
        return cls(purpose, path, expected_sha256, tuple(entries), required)


class Decontaminator:
    def __init__(
        self,
        connection: sqlite3.Connection,
        registry: BenchmarkRegistry,
        ngram_size: int = 13,
    ) -> None:
        if type(ngram_size) is not int or not 1 <= ngram_size <= 64:
            raise ValueError("Invalid n-gram size")
        self.connection, self.registry, self.ngram_size = (
            connection,
            registry,
            ngram_size,
        )
        self.lengths: set[int] = set()
        connection.execute(
            "CREATE TABLE IF NOT EXISTS eval_grams (gram TEXT, benchmark TEXT, PRIMARY KEY(gram,benchmark))"
        )
        self.questions = {}
        from qvelt.tokenizer.workflow import iter_documents

        for entry in registry.entries:
            path = registry.path.parent / str(entry["manifest"])
            count = 0
            # The raw manifest checks every question/permission payload before use.
            for question in iter_documents(path, str(entry["sha256"])):
                tokens = normalized_tokens(question)
                if not tokens:
                    raise ValueError(
                        "Empty benchmark questions are not a decontamination snapshot"
                    )
                length = min(len(tokens), ngram_size)
                self.lengths.add(length)
                grams = {
                    " ".join(tokens[i : i + length])
                    for i in range(len(tokens) - length + 1)
                }
                connection.executemany(
                    "INSERT OR IGNORE INTO eval_grams VALUES (?,?)",
                    [(g, entry["name"]) for g in sorted(grams)],
                )
                count += 1
            if not count:
                raise ValueError("Benchmark snapshot has no questions")
            self.questions[str(entry["name"])] = count
        connection.commit()

    def matches(self, text: str) -> set[str]:
        tokens = normalized_tokens(text)
        hits: set[str] = set()
        pending = []
        for length in sorted(self.lengths):
            for i in range(len(tokens) - length + 1):
                pending.append(" ".join(tokens[i : i + length]))
                if len(pending) == 400:
                    hits.update(self._query(pending))
                    pending = []
        if pending:
            hits.update(self._query(pending))
        return hits

    def _query(self, grams: list[str]) -> set[str]:
        query = (
            "SELECT DISTINCT benchmark FROM eval_grams WHERE gram IN ("
            + ",".join("?" for _ in grams)
            + ")"
        )
        return {str(row[0]) for row in self.connection.execute(query, grams)}
