"""Measure digit policies, EN/RU fertility and actual host loader delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from qvelt.data.benchmark import benchmark_loader
from qvelt.data.demo import create_demo
from qvelt.data.math import generate_problems
from qvelt.tokenizer.core import TokenizerSettings, train_tokenizer
from qvelt.utils.artifacts import create_directory
from qvelt.utils.provenance import capture_code


def digit_experiment() -> dict[str, Any]:
    train = [
        p.question + "\nAnswer: " + p.answer
        for language in ("en", "ru")
        for p in generate_problems(384, seed=113, language=language)
    ]
    heldout = {
        language: [
            p.question + "\nAnswer: " + p.answer
            for p in generate_problems(192, seed=997, language=language)
        ]
        for language in ("en", "ru")
    }
    results = []
    for digits in (True, False):
        settings = TokenizerSettings(vocab_size=1024, isolate_digits=digits)
        started = time.perf_counter()
        tokenizer = train_tokenizer(train, settings)
        seconds = time.perf_counter() - started
        languages = {}
        for language, texts in heldout.items():
            tokens = sum(len(tokenizer.encode(text)) for text in texts)
            words = sum(len(text.split()) for text in texts)
            assert all(
                tokenizer.decode(tokenizer.encode(text)) == text for text in texts
            )
            languages[language] = {
                "documents": len(texts),
                "tokens": tokens,
                "whitespace_words": words,
                "tokens_per_word": tokens / words,
                "roundtrip_failures": 0,
            }
        results.append(
            {
                "settings": asdict(settings),
                "actual_vocab": tokenizer.vocab_size,
                "tokenizer_json_sha256": hashlib.sha256(
                    tokenizer.to_bytes()
                ).hexdigest(),
                "training_seconds": seconds,
                "languages": languages,
                "example_digit_token_ids": tokenizer.encode("1234567890"),
            }
        )
    payload = {"train": train, "heldout": heldout}
    return {
        "scope": "Authored exact math only; different generation seeds, shared templates; no public evaluation content",
        "corpus_sha256": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "training_documents": len(train),
        "heldout_documents": sum(map(len, heldout.values())),
        "policies": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--documents", type=int, default=192)
    args = parser.parse_args()
    root = create_directory(args.directory)
    torch.set_num_threads(1)
    _, code = capture_code(args.repository)
    digits = digit_experiment()
    started = time.perf_counter()
    create_demo(root / "demo", args.repository, args.documents)
    prepare_seconds = time.perf_counter() - started
    path = root / "demo/prepared/manifest.json"
    measures = [benchmark_loader(path, workers=w, batches=64, warmup=8) for w in (0, 2)]
    files = [p for p in (root / "demo").rglob("*") if p.is_file()]
    receipt = {
        "code": asdict(code),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "digit_experiment": digits,
        "demo_prepare_seconds": prepare_seconds,
        "demo_total_file_bytes": sum(p.stat().st_size for p in files),
        "demo_prepared_file_bytes": sum(
            p.stat().st_size for p in files if p.is_relative_to(path.parent)
        ),
        "curation": json.loads((path.parent / "report.json").read_bytes()),
        "loader_benchmarks": measures,
    }
    (root / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"receipt": str(root / "receipt.json"), "production_ready": False},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
