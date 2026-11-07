"""Original bounded needle-in-haystack-style development controls with token budgets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.records import write_raw_bundle
from qvelt.eval.corpus import EvalCase
from qvelt.eval.smoke import case_document
from qvelt.tokenizer.core import QveltTokenizer
from qvelt.utils.manifest_types import Manifest


def needle_cases(
    tokenizer: QveltTokenizer,
    *,
    lengths: tuple[int, ...] = (64, 128, 256, 512, 1024),
    seed: int = 20261107,
    samples: int = 2,
) -> tuple[list[EvalCase], dict[str, Any]]:
    if (
        type(lengths) is not tuple
        or not lengths
        or len(lengths) > 8
        or len(set(lengths)) != len(lengths)
        or any(type(n) is not int or not 48 <= n <= 2048 for n in lengths)
    ):
        raise ValueError("Needle budgets require1..8 distinct lengths48..2048")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(samples) is not int
        or not 1 <= samples <= 10
        or len(lengths) * 3 * samples > 128
    ):
        raise ValueError("Needle seed/sample capacity invalid")
    rng = random.Random(seed)
    cases = []
    records = []
    for length in lengths:
        for depth in (0.1, 0.5, 0.9):
            for sample, answer in enumerate(rng.sample(range(10), samples)):
                needle = f"key={answer}. "
                suffix = "\nkey? "

                def prompt(
                    count: int,
                    fraction: float = depth,
                    fact: str = needle,
                    tail: str = suffix,
                ) -> str:
                    left = int(count * fraction)
                    return "z " * left + fact + "z " * (count - left) + tail

                low, high = 0, length
                while low < high:
                    mid = (low + high + 1) // 2
                    if 1 + len(tokenizer.encode(prompt(mid))) <= length - 8:
                        low = mid
                    else:
                        high = mid - 1
                text = prompt(low)
                tokens = 1 + len(tokenizer.encode(text))
                if tokens > length - 8:
                    raise ValueError("Needle framing exceeds requested context budget")
                identifier = f"needle-{length}-{int(depth * 100)}-{sample}"
                cases.append(
                    EvalCase(
                        identifier,
                        "math",
                        f"needle-L{length}-D{int(depth * 100)}",
                        "needle-retrieval",
                        "validation",
                        text,
                        str(answer),
                        language="en",
                    )
                )
                records.append(
                    {
                        "id": identifier,
                        "context_budget": length,
                        "target_depth_fraction": depth,
                        "prompt_tokens_with_bos": tokens,
                        "needle_character_offset": text.index(needle),
                    }
                )
    return cases, {
        "protocol": "qvelt-original-needle-v1",
        "seed": seed,
        "lengths": list(lengths),
        "depths": [0.1, 0.5, 0.9],
        "samples_per_cell": samples,
        "reserved_generation_tokens": 8,
        "scope": "original diagnostic retrieval;not trained for this task or a standardized benchmark",
        "cases": records,
    }


def create_needles(
    tokenizer: QveltTokenizer,
    directory: Path,
    *,
    lengths: tuple[int, ...] = (64, 128, 256, 512, 1024),
    seed: int = 20261107,
    samples: int = 2,
) -> Manifest:
    cases, construction = needle_cases(
        tokenizer, lengths=lengths, seed=seed, samples=samples
    )
    return write_raw_bundle(
        [case_document(c) for c in cases],
        directory,
        authored_source("original-phase8-needle-v1"),
        PERMISSION,
        allowed_uses=("evaluation", "decontamination"),
        evidence=(
            (
                "construction.json",
                (json.dumps(construction, sort_keys=True) + "\n").encode(),
                "needle_construction",
            ),
        ),
    )
