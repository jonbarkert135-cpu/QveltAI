"""Original licensed offline corpus with deliberate curation and contamination canaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from qvelt.data.config import DataRecipe, MixStage, SourceSpec, dump_data_recipe
from qvelt.data.math import generate_problems
from qvelt.data.pipeline import prepare_data
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.tokenizer.core import TokenizerSettings
from qvelt.utils.artifacts import create_directory
from qvelt.utils.manifest_types import Manifest, SourceIdentity
from qvelt.utils.manifests import manifest_digest

PERMISSION = b"Original Qvelt authored fixture and mathematical templates. Apache-2.0; retain the repository LICENSE and NOTICE. No external source or teacher.\n"
HOLDOUT = "An original protected benchmark canary asks how many silver triangles remain after exactly seventeen rotations around a square table."


def authored_source(name: str) -> SourceIdentity:
    return SourceIdentity(
        name,
        "authored-phase3-v1",
        "Apache-2.0",
        "https://github.com/jonbarkert135-cpu/QveltAI/blob/main/LICENSE",
    )


def demo_documents(count: int, seed: int, language: str) -> Iterator[RawDocument]:
    for problem in generate_problems(count, seed, language):
        yield RawDocument(
            problem.question + "\nAnswer: " + problem.answer,
            problem.identifier,
            problem.family,
            problem.language,
            problem.difficulty,
            problem.answer,
        )
    # Independent explanatory/code/Unicode text and unique families diversify splits.
    examples = [
        "Python exact arithmetic example:\nfrom fractions import Fraction\nx = Fraction(3, 7)\nassert x + x == Fraction(6, 7)\n",
        "Русский учебный текст: математика изучает числа, геометрию, доказательства и точные вычисления. Сохраним букву ё и пробелы.\t\n",
        r"LaTeX identity: \(\sum_{k=1}^{n} k = n(n+1)/2\). Prove by induction on the positive integer n.",
        "Unicode exactness: café e\u0301 Ελληνικά العربية 中文 😀\u200d🚀\u00a0tab\tend\r\n",
        "Contact example alice@example.invalid, phone: +1 (202) 555-0147. The mathematical integer 123456789012345 must remain intact.",
        "A deliberately harmful phrase kill yourself is quarantined in this authored policy test.",
        HOLDOUT,
    ]
    near = " ".join(f"word{index}" for index in range(120))
    examples += [near, near.replace("word90", "replacement90"), examples[0]]
    for index, text in enumerate(examples):
        yield RawDocument(
            text, f"example-{language}-{index}", f"example-{index}", language
        )


def create_demo(directory: Path, repository: Path, documents: int = 192) -> Manifest:
    root = create_directory(directory)
    sources = []
    for language in ("en", "ru"):
        manifest = write_raw_bundle(
            demo_documents(documents, 20261031, language),
            root / language,
            authored_source("math-" + language),
            PERMISSION,
        )
        sources.append(
            SourceSpec(
                "math-" + language,
                language + "/manifest.json",
                manifest_digest(manifest),
                True,
                True,
            )
        )
    heldout = write_raw_bundle(
        [RawDocument(HOLDOUT)],
        root / "holdout",
        authored_source("authored-holdout"),
        PERMISSION,
    )
    registry = {
        "schema_version": 1,
        "purpose": "demo",
        "required": ["authored-holdout"],
        "benchmarks": [
            {
                "name": "authored-holdout",
                "manifest": "holdout/manifest.json",
                "sha256": manifest_digest(heldout),
                "approved": True,
                "fixture": True,
            }
        ],
    }
    payload = (json.dumps(registry, sort_keys=True) + "\n").encode()
    (root / "benchmarks.json").write_bytes(payload)
    recipe = DataRecipe(
        tuple(sources),
        "benchmarks.json",
        hashlib.sha256(payload).hexdigest(),
        (
            MixStage(0, (("math-en", 0.5), ("math-ru", 0.5))),
            MixStage(5, (("math-en", 0.2), ("math-ru", 0.8)), "annealing"),
        ),
        context=64,
        shard_tokens=2048,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer=TokenizerSettings(vocab_size=1024),
    )
    path = root / "recipe.yaml"
    path.write_text(dump_data_recipe(recipe), encoding="utf-8")
    return prepare_data(path, root / "prepared", repository)
