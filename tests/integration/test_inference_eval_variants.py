"""Actual matched HQQ evaluation receipts,ledger deltas and needle-budget controls."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from qvelt.eval.config import ArtifactRef, dump_eval, eval_digest, parse_eval
from qvelt.eval.ledger import build_results, comparison_key, export_receipt
from qvelt.eval.runner import run_evaluation
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.inference.needles import needle_cases
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]


def test_needles_exact_gold_budgets_and_determinism(tmp_path: Path) -> None:
    job = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    tokenizer, _ = load_tokenizer_bundle(Path(job.tokenizer.path), job.tokenizer.sha256)
    cases, report = needle_cases(tokenizer, lengths=(64, 128), samples=2)
    assert len(cases) == 12 and (cases, report) == needle_cases(
        tokenizer, lengths=(64, 128), samples=2
    )
    for case, row in zip(cases, report["cases"], strict=True):
        assert case.prompt.count("key=" + case.target + ". ") == 1
        assert (
            1 + len(tokenizer.encode(case.prompt))
            == row["prompt_tokens_with_bos"]
            <= row["context_budget"] - 8
        )
        assert case.prompt.endswith("\nkey? ")
    assert len({c.identifier for c in cases}) == len(cases)
    with pytest.raises(ValueError):
        needle_cases(tokenizer, lengths=(16,))


def test_hqq_reports_bound_storage_and_ledger_deltas(tmp_path: Path) -> None:
    pytest.importorskip("hqq")
    fixture = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    paths = []
    receipts: list[dict[str, Any]] = []
    for bits in (0, 4, 8):
        job = replace(
            fixture,
            protocol="qvelt-eval-v2",
            prompt_format="plain-v1",
            quantization_bits=bits,
        )
        assert parse_eval(dump_eval(job)) == job
        result = run_evaluation(
            job,
            tmp_path / f"eval-{bits}",
            ROOT,
            command=(
                "python",
                "scripts/inference_phase8_smoke.py",
                "--directory",
                "runs/reproduce",
            ),
        )
        manifest = ArtifactStore(tmp_path / f"eval-{bits}").read_manifest()
        path = tmp_path / f"receipt-{bits}.json"
        receipt = export_receipt(
            ArtifactRef(
                str(tmp_path / f"eval-{bits}/manifest.json"), manifest_digest(manifest)
            ),
            path,
            benchmark="original-en-ru-code",
        )
        paths.append(path)
        receipts.append(receipt)
        if bits:
            variant = result["metrics"]["model_variant"]
            assert (
                variant["quantization_bits"] == bits
                and variant["weight_tensor_bytes"] > 0
            )
            assert len(variant["packed_payload_sha256"]) == 64
    assert (
        comparison_key(receipts[0])
        == comparison_key(receipts[1])
        == comparison_key(receipts[2])
    )
    ledger = build_results(paths, tmp_path / "RESULTS.md")
    assert (
        "Matched weight-only quantization comparisons" in ledger
        and "perplexity_delta" in ledger
    )
    # New default fields do not alter old semantic digests.
    body = json.loads(json.dumps(asdict(fixture)))
    body.pop("quantization_bits")
    body.pop("context_extension")
    import yaml

    assert eval_digest(fixture) == eval_digest(parse_eval(yaml.safe_dump(body)))
