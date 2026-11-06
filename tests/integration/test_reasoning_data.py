"""Real checkpoint sampling, trace/terms trust boundaries and selected bundle reuse."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from qvelt.cli import main
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.records import RawDocument, iter_records, write_raw_bundle
from qvelt.eval.config import ArtifactRef
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.inference.generation import GenerationResult
from qvelt.posttrain.reasoning import (
    ReasoningJob,
    build_reasoning_data,
    dump_reasoning,
    ingest_external_traces,
    licensed_records,
    load_reasoning,
)
from qvelt.utils.manifest_types import DataInfo
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]


def test_real_own_checkpoint_candidates_and_replay(tmp_path: Path) -> None:
    fixture = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    recipe = ReasoningJob(
        fixture.checkpoint, fixture.tokenizer, count=2, samples=2, max_tokens=4
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(dump_reasoning(recipe))
    assert load_reasoning(path) == recipe
    rng_reports = []
    for name in ("one", "two"):
        manifest = build_reasoning_data(recipe, tmp_path / name)
        assert isinstance(manifest.metadata, DataInfo)
        assert manifest.metadata.documents <= 4
        rng_reports.append(
            (
                json.loads((tmp_path / name / "report.json").read_text()),
                (tmp_path / name / "audit.jsonl").read_text(),
            )
        )
        assert len(rng_reports[-1][1].splitlines()) == 4
        assert (
            len(
                list(
                    iter_records(
                        tmp_path / name / "manifest.json", manifest_digest(manifest)
                    )
                )
            )
            == manifest.metadata.documents
        )
    assert rng_reports[0] == rng_reports[1]
    assert rng_reports[0][0]["external_synthetic"] is False
    assert (
        main(["reasoning-data", str(path), "--directory", str(tmp_path / "cli")]) == 0
    )
    path.write_text(dump_reasoning(recipe) + "samples: 3\n")
    with pytest.raises(ValueError):
        load_reasoning(path)


def test_rejection_selection_and_diversity_isolated_oracle(tmp_path: Path) -> None:
    fixture = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    source = write_raw_bundle(
        [RawDocument("Compute 1+1.", "p", "sum", "en", "easy", "2")],
        tmp_path / "problems",
        authored_source("test-rejection-problems"),
        PERMISSION,
    )
    recipe = ReasoningJob(
        fixture.checkpoint,
        fixture.tokenizer,
        ArtifactRef(str(tmp_path / "problems/manifest.json"), manifest_digest(source)),
        True,
        samples=4,
        max_tokens=16,
        framed=True,
    )
    outputs = [
        ("<think>Add one and one.</think>Final answer: 2", (40, 6), "stop_token"),
        ("<think>Add one and one.</think>Final answer: 4/2", (40, 6), "stop_token"),
        ("Final answer: 2", (40, 41), "max_tokens"),
        ("Final answer: 3", (40, 6), "stop_token"),
    ]

    def generate(*args: Any, **kwargs: Any) -> list[GenerationResult]:
        text, tokens, reason = outputs.pop(0)
        return [GenerationResult("candidate", tokens, text, reason, 3, 0)]

    # Force candidates only in a selection unit oracle; the test above samples real model logits.
    with patch(
        "qvelt.posttrain.reasoning.GenerationEngine.generate", side_effect=generate
    ):
        manifest = build_reasoning_data(recipe, tmp_path / "selected")
    rows = list(
        iter_records(tmp_path / "selected/manifest.json", manifest_digest(manifest))
    )
    assert len(rows) == 1
    assert json.loads(rows[0].text)["messages"][-1]["reasoning"] == "Add one and one."
    report = json.loads((tmp_path / "selected/report.json").read_text())
    assert report["difficulty"]["easy"]["pass_at_k"]["1"] == 0.5
    assert report["difficulty"]["easy"]["pass_at_k"]["4"] == 1
    with pytest.raises(ValueError, match="approved"):
        assert recipe.problems is not None
        licensed_records(recipe.problems, False)
    with pytest.raises(ValueError):
        replace(recipe, samples=True)


def test_external_default_off_terms_and_permission(tmp_path: Path) -> None:
    text = json.dumps(
        {
            "schema_version": 2,
            "messages": [
                {"role": "user", "content": "1+1?"},
                {"role": "assistant", "content": "2", "reasoning": "Add."},
            ],
        }
    )
    source = write_raw_bundle(
        [RawDocument(text, "p", "sum", "en", "easy", "2")],
        tmp_path / "raw",
        authored_source("original-unit-teacher-fixture"),
        PERMISSION,
    )
    ref = ArtifactRef(str(tmp_path / "raw/manifest.json"), manifest_digest(source))
    terms_path = tmp_path / "terms.yaml"
    terms_path.write_text(
        "teacher: original-unit-test-fixture\nrevision: v1\nlicense_id: Apache-2.0\nterms_url: https://example.invalid/terms\ntraining_permitted: true\nreviewed: true\n"
    )
    terms = ArtifactRef(
        str(terms_path), hashlib.sha256(terms_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="off by default"):
        ingest_external_traces(
            ref, tmp_path / "off", approved_for_training=True, terms=terms
        )
    assert not (tmp_path / "off").exists()
    manifest = ingest_external_traces(
        ref,
        tmp_path / "on",
        approved_for_training=True,
        external_synthetic=True,
        terms=terms,
    )
    assert (
        len(
            list(iter_records(tmp_path / "on/manifest.json", manifest_digest(manifest)))
        )
        == 1
    )
    assert any(f.role == "teacher_terms" for f in manifest.files)
    terms_path.write_text(
        terms_path.read_text().replace("reviewed: true", "reviewed: false")
    )
    bad = replace(terms, sha256=hashlib.sha256(terms_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="reviewed"):
        ingest_external_traces(
            ref,
            tmp_path / "bad",
            approved_for_training=True,
            external_synthetic=True,
            terms=bad,
        )
    denied = write_raw_bundle(
        [RawDocument(text, "p", "sum", "en", "easy", "2")],
        tmp_path / "denied",
        authored_source("test-evaluation-only"),
        PERMISSION,
        allowed_uses=("evaluation",),
    )
    with pytest.raises(ValueError, match="permit training"):
        licensed_records(
            ArtifactRef(
                str(tmp_path / "denied/manifest.json"), manifest_digest(denied)
            ),
            True,
        )
