"""Disk-backed exact and MinHash-LSH candidate deduplication with Jaccard checks."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass

import numpy as np
from datasketch import MinHash

from qvelt.data.cleaning import words


@dataclass(frozen=True)
class DedupConfig:
    threshold: float = 0.85
    num_perm: int = 64
    bands: int = 8
    shingle_size: int = 5
    max_candidates: int = 1000
    preserve_math_atoms: bool = True

    def __post_init__(self) -> None:
        if type(self.threshold) not in (int, float) or not 0 < self.threshold <= 1:
            raise ValueError("Dedup threshold must lie in (0,1]")
        if type(self.preserve_math_atoms) is not bool:
            raise ValueError("Math atom preservation must be bool")
        if (
            any(
                type(v) is not int or v < 1
                for v in [
                    self.num_perm,
                    self.bands,
                    self.shingle_size,
                    self.max_candidates,
                ]
            )
            or self.num_perm > 1024
            or self.bands > 128
            or self.max_candidates > 10000
            or self.shingle_size > 64
            or self.num_perm % self.bands
        ):
            raise ValueError("Invalid bounded MinHash/LSH parameters")


def shingles(text: str, size: int) -> set[str]:
    tokens = words(text)
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


class DiskDeduplicator:
    def __init__(
        self, connection: sqlite3.Connection, config: DedupConfig, seed: int
    ) -> None:
        self.connection, self.config, self.seed = connection, config, seed
        connection.execute(
            "CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, digest TEXT UNIQUE NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL, split TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS buckets (band INTEGER, bucket BLOB, document INTEGER)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS bucket_lookup ON buckets(band,bucket)"
        )

    def signature(self, text: str) -> tuple[set[str], list[bytes]]:
        grams = shingles(text, self.config.shingle_size)
        value = MinHash(num_perm=self.config.num_perm, seed=self.seed)
        if grams:
            value.update_batch([gram.encode() for gram in sorted(grams)])
        rows = self.config.num_perm // self.config.bands
        bands = [
            hashlib.sha256(
                np.asarray(
                    value.hashvalues[i * rows : (i + 1) * rows], dtype="<u8"
                ).tobytes()
            ).digest()
            for i in range(self.config.bands)
        ]
        return grams, bands

    def duplicate(self, text: str) -> str | None:
        digest = hashlib.sha256(text.encode()).hexdigest()
        if self.connection.execute(
            "SELECT 1 FROM documents WHERE digest=?", (digest,)
        ).fetchone():
            return "exact_duplicate"
        grams, bands = self.signature(text)
        clauses = " OR ".join("(band=? AND bucket=?)" for _ in bands)
        parameters = [value for pair in enumerate(bands) for value in pair]
        candidates = self.connection.execute(
            "SELECT DISTINCT document FROM buckets WHERE "
            + clauses
            + " ORDER BY document LIMIT ?",
            [*parameters, self.config.max_candidates + 1],
        ).fetchall()
        if len(candidates) > self.config.max_candidates:
            return "near_candidate_limit"
        for (identifier,) in candidates:
            stored = self.connection.execute(
                "SELECT text FROM documents WHERE id=?", (identifier,)
            ).fetchone()[0]
            # Near-identical templates can describe different equations/answers.
            if self.config.preserve_math_atoms and re.findall(
                r"\d+|[+*/=<>^×÷≤≥−-]", text
            ) != re.findall(r"\d+|[+*/=<>^×÷≤≥−-]", stored):
                continue
            other = shingles(stored, self.config.shingle_size)
            union = grams | other
            similarity = len(grams & other) / len(union) if union else 1.0
            if similarity >= self.config.threshold:
                return "near_duplicate"
        return None

    def add(self, text: str, metadata: str, split: str) -> int:
        digest = hashlib.sha256(text.encode()).hexdigest()
        cursor = self.connection.execute(
            "INSERT INTO documents(digest,text,metadata,split) VALUES (?,?,?,?)",
            (digest, text, metadata, split),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return the document identity")
        identifier = cursor.lastrowid
        _, bands = self.signature(text)
        self.connection.executemany(
            "INSERT INTO buckets VALUES (?,?,?)",
            [(i, bucket, identifier) for i, bucket in enumerate(bands)],
        )
        return identifier
