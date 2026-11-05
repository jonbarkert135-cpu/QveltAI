"""Version1 prospective project holdout;raw final cases never enter training exports."""

from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

import sympy as sp

from qvelt.data.demo import PERMISSION
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.utils.artifacts import atomic_write, create_directory, hash_file
from qvelt.utils.manifest_types import SourceIdentity
from qvelt.utils.manifests import manifest_digest

VERSION = "qvelt-heldout-v1"
SEED = 20261105
FAMILIES = (
    "remainder",
    "progression",
    "two-equations",
    "rational-product",
    "quadratic",
)


def _candidate(
    rng: random.Random, kind: str, low: int, high: int
) -> tuple[str, Fraction]:
    a, b, c, d = (rng.randint(low, high) for _ in range(4))
    if kind == "remainder":
        remainder = rng.randrange(b)
        value = b * a + remainder
        prompt = f"Find the remainder when {value} is divided by {b}. Give an integer."
        answer = Fraction(remainder)
        checked = sp.Mod(value, b)
    elif kind == "progression":
        prompt = f"An arithmetic progression begins at {a},has common difference {b},and contains {c} terms. Find their sum."
        answer = Fraction(sum(a + i * b for i in range(c)))
        checked = sp.Rational(c) * (2 * a + (c - 1) * b) / 2
    elif kind == "two-equations":
        x, y = Fraction(a) + Fraction(1, c), Fraction(b) + Fraction(1, d)
        total, difference = x + y, x - y
        prompt = f"Solve the simultaneous equations x+y={total} and x-y={difference}. Give x as an exact fraction."
        answer = x
        checked = (
            sp.Matrix([[1, 1], [1, -1]]).inv()
            * sp.Matrix(
                [
                    sp.Rational(total.numerator, total.denominator),
                    sp.Rational(difference.numerator, difference.denominator),
                ]
            )
        )[0]
    elif kind == "rational-product":
        prompt = f"Multiply {a}/{b} by {c}/{d} and give the exact reduced value."
        answer = Fraction(a, b) * Fraction(c, d)
        checked = sp.Rational(a, b) * sp.Rational(c, d)
    elif kind == "quadratic":
        point = d if rng.randrange(2) else -d
        prompt = f"For f(t)={a}*t^2+{b}*t+{c},find f({point}). Give an integer."
        answer = Fraction(a * point * point + b * point + c)
        symbol = sp.Symbol("t")
        checked = sp.Poly(a * symbol**2 + b * symbol + c, symbol).eval(point)
    else:
        raise ValueError("Unknown frozen template family")
    if sp.Rational(answer.numerator, answer.denominator) != checked:
        raise ArithmeticError("Independent frozen-answer verification failed")
    return prompt, answer


def protected_cases(partition: str) -> list[RawDocument]:
    """Fixed200/800 cases;parameter bands and question/answer byte identities freeze v1."""
    if partition not in ("validation", "final"):
        raise ValueError("Frozen holdout partition must be validation/final")
    rng = random.Random(SEED + (partition == "final"))
    low, high, count = (2, 25, 40) if partition == "validation" else (26, 80, 160)
    seen: set[str] = set()
    documents = []
    for kind in FAMILIES:
        accepted = 0
        for _ in range(100000):
            if accepted == count:
                break
            prompt, answer = _candidate(rng, kind, low, high)
            if prompt in seen:
                continue
            seen.add(prompt)
            identifier = hashlib.sha256(
                (VERSION + "\0" + partition + "\0" + prompt).encode()
            ).hexdigest()
            family = f"heldout-{kind}-{partition}"
            row = {
                "schema_version": 1,
                "identifier": identifier,
                "kind": "math",
                "domain": "math",
                "family": family,
                "partition": partition,
                "prompt": prompt,
                "target": str(answer),
                "choices": [],
            }
            documents.append(
                RawDocument(
                    json.dumps(row, sort_keys=True),
                    identifier,
                    family,
                    "en",
                    partition,
                    str(answer),
                )
            )
            accepted += 1
        if accepted != count:
            raise RuntimeError(
                "Frozen template exhausted its deterministic uniqueness budget"
            )
    return documents


def freeze_holdout(directory: Path, repository: Path) -> dict[str, Any]:
    root = create_directory(directory)
    registry: dict[str, Any] = {
        "schema_version": 1,
        "version": VERSION,
        "seed": SEED,
        "freeze_date": "2026-11-05",
        "prospective_only": True,
        "generator_sha256": hash_file(Path(__file__)).sha256,
        "license_sha256": hash_file(repository / "LICENSE").sha256,
        "partitions": {},
    }
    for partition in ("validation", "final"):
        documents = protected_cases(partition)
        source = SourceIdentity(
            f"qvelt-heldout-{partition}",
            VERSION,
            "Apache-2.0",
            "https://github.com/jonbarkert135-cpu/QveltAI/blob/main/LICENSE",
        )
        manifest = write_raw_bundle(
            documents,
            root / partition,
            source,
            PERMISSION,
            allowed_uses=("evaluation", "decontamination"),
        )
        record = next(f for f in manifest.files if f.role == "data")
        registry["partitions"][partition] = {
            "manifest": f"{partition}/manifest.json",
            "manifest_sha256": manifest_digest(manifest),
            "payload_sha256": record.sha256,
            "count": len(documents),
            "cases": [
                {"identifier": d.identifier, "family": d.family} for d in documents
            ],
        }
    payload = (json.dumps(registry, sort_keys=True) + "\n").encode()

    def write(stream: BinaryIO) -> None:
        stream.write(payload)

    atomic_write(root / "freeze.json", write)
    return registry
