"""Original mixed EN/RU/math/code fixture settings and resolved evaluation factory."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from qvelt.config import MAX_CONFIG_BYTES, _integer, _mapping, parse_yaml_mapping
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.math import generate_problems
from qvelt.data.records import write_raw_bundle
from qvelt.eval.config import ArtifactRef, EvalJob
from qvelt.eval.corpus import EvalCase
from qvelt.eval.smoke import case_document
from qvelt.utils.manifest_types import Manifest


@dataclass(frozen=True)
class SuiteSettings:
    math_per_language: int = 4
    samples: int = 4
    max_new_tokens: int = 8
    context_limit: int = 256
    seed: int = 20261106
    schema_version: int = 1
    taskset: str = "qvelt-original-mixed-v2"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.taskset != "qvelt-original-mixed-v2"
        ):
            raise ValueError("Unknown mixed-suite settings protocol")
        for key, low, high in (
            ("math_per_language", 1, 16),
            ("samples", 1, 8),
            ("max_new_tokens", 1, 128),
            ("context_limit", 32, 2048),
            ("seed", 0, 2**32 - 1),
        ):
            _integer(getattr(self, key), "suite." + key, minimum=low, maximum=high)


def load_suite_settings(path: Path) -> SuiteSettings:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError("Suite settings exceed byte limit")
    body = _mapping(
        parse_yaml_mapping(data.decode()),
        "suite",
        {f.name for f in fields(SuiteSettings)},
    )
    return SuiteSettings(**cast(dict[str, Any], body))


def original_cases(settings: SuiteSettings) -> list[EvalCase]:
    result = []
    for language in ("en", "ru"):
        for problem in generate_problems(
            settings.math_per_language, settings.seed, language
        ):
            result.append(
                EvalCase(
                    "phase7-" + problem.identifier,
                    "math",
                    "original-math",
                    problem.family,
                    "validation",
                    problem.question,
                    problem.answer,
                    language=language,
                )
            )
    result.extend(
        [
            EvalCase(
                "original-choice-en",
                "choice",
                "original-general",
                "capital-control",
                "validation",
                "What is the capital of France?\nAnswer:",
                " Paris",
                (" Rome", " Paris", " Berlin", " Madrid"),
                language="en",
            ),
            EvalCase(
                "original-choice-ru",
                "choice",
                "original-general",
                "capital-control",
                "validation",
                "Какой город является столицей Франции?\nОтвет:",
                " Париж",
                (" Рим", " Париж", " Берлин", " Мадрид"),
                language="ru",
            ),
            EvalCase(
                "original-lm-en",
                "lm",
                "original-language",
                "language-control",
                "validation",
                "A triangle has three sides.",
                "",
                language="en",
            ),
            EvalCase(
                "original-lm-ru",
                "lm",
                "original-language",
                "language-control",
                "validation",
                "Треугольник имеет три стороны.",
                "",
                language="ru",
            ),
        ]
    )
    for name, args, tests, description in [
        (
            "add",
            ["a", "b"],
            [([2, 3], 5), ([-7, 4], -3), ([0, 0], 0)],
            "Return the sum of integer a and b.",
        ),
        (
            "difference",
            ["a", "b"],
            [([-3, 4], 7), ([2, 2], 0), ([8, 1], 7)],
            "Return the absolute difference of integers a and b.",
        ),
    ]:
        target = json.dumps(
            {
                "entry_point": name,
                "arguments": args,
                "tests": [
                    {"args": values, "expected": expected} for values, expected in tests
                ],
            },
            sort_keys=True,
        )
        prompt = f'def {name}({",".join(args)}):\n    """{description}"""\n'
        result.append(
            EvalCase(
                "original-code-" + name,
                "code",
                "original-code",
                "integer-function-control",
                "validation",
                prompt,
                target,
                language="en",
            )
        )
    return result


def create_original_suite(settings: SuiteSettings, directory: Path) -> Manifest:
    return write_raw_bundle(
        [case_document(c) for c in original_cases(settings)],
        directory,
        authored_source("original-phase7-en-ru-code-v2"),
        PERMISSION,
        allowed_uses=("evaluation", "decontamination"),
    )


def suite_job(
    settings: SuiteSettings,
    checkpoint: ArtifactRef,
    tokenizer: ArtifactRef,
    dataset: ArtifactRef,
    training_sources: tuple[ArtifactRef, ...],
) -> EvalJob:
    return EvalJob(
        checkpoint,
        tokenizer,
        dataset,
        training_sources,
        approved_for_evaluation=True,
        prompt_format="plain-v1",
        protocol="qvelt-eval-v2",
        choice_normalization="character",
        samples=settings.samples,
        sampling="greedy" if settings.samples == 1 else "sample",
        temperature=0 if settings.samples == 1 else 1.0,
        pass_k=tuple(sorted({1, settings.samples})),
        max_new_tokens=settings.max_new_tokens,
        context_limit=settings.context_limit,
        lm_window=min(64, settings.context_limit),
        seed=settings.seed,
        max_cases=64,
    )
