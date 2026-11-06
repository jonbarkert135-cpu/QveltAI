"""Strict bounded native generation recipes;unknown/ambiguous inputs fail closed."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from qvelt.config import ConfigError
from qvelt.eval.config import ArtifactRef
from qvelt.inference.config import (
    GenerationJob,
    GenerationOptions,
    TextRequest,
    dump_generation,
    load_generation,
)


@pytest.mark.parametrize(
    "values",
    [
        {"max_new_tokens": True},
        {"max_new_tokens": 257},
        {"temperature": float("nan")},
        {"strategy": "beam"},
        {"strategy": "sample", "temperature": 0},
        {"max_seconds": 0},
        {"max_cache_bytes": 0},
        {"stop_tokens": ([1],)},
        {"stop_tokens": (2, 2)},
        {"stop_strings": ("",)},
        {"stop_strings": ([],)},
        {"max_output_chars": 16385},
    ],
)
def test_options_reject_invalid_bounds(values: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        GenerationOptions(**values)


def test_generation_recipe_roundtrip_unknown_fields_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    request = TextRequest("one", "Привет 😀", GenerationOptions(stop_strings=("\n\n",)))
    job = GenerationJob(ArtifactRef("/snapshot/manifest.json", "a" * 64), (request,))
    path = tmp_path / "generate.yaml"
    path.write_text(dump_generation(job))
    assert load_generation(path) == job
    path.write_text(dump_generation(job) + "unreviewed: true\n")
    with pytest.raises(ConfigError):
        load_generation(path)
    with pytest.raises(ConfigError):
        replace(job, requests=(request, request))
    with pytest.raises(ConfigError):
        replace(job, use_cache=cast(Any, "yes"))
