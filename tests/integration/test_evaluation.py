"""Real checksummed BPE/checkpoint path,naive logits references and failure guards."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from qvelt.cli import main
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.eval.config import ArtifactRef, EvalJob, dump_eval
from qvelt.eval.corpus import EvalCase, audit_overlap, load_cases
from qvelt.eval.fixtures import freeze_holdout
from qvelt.eval.runner import (
    aggregate,
    attempt_seed,
    generate_attempt,
    load_model,
    prompt_ids,
    run_evaluation,
    score_choice,
    score_lm,
    text_ids,
)
from qvelt.eval.smoke import case_document, control_cases, create_evaluation_fixture
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore, hash_file
from qvelt.utils.manifests import manifest_digest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def job(tmp_path_factory: pytest.TempPathFactory) -> EvalJob:
    return create_evaluation_fixture(
        tmp_path_factory.mktemp("evaluation") / "fixture", REPO
    )


def test_real_cli_repeats_predictions_preserves_rng_and_inputs(
    job: EvalJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_pickle(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Read-only evaluation must not deserialize pickle state")

    monkeypatch.setattr(torch, "load", reject_pickle)
    sampled = replace(job, sampling="sample", samples=3, temperature=1, pass_k=(1, 2))
    cp = Path(job.checkpoint.path).parent
    before = {p.name: hash_file(p) for p in cp.iterdir() if p.is_file()}
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    recipe = tmp_path / "eval.yaml"
    recipe.write_text(dump_eval(sampled))
    assert (
        main(
            [
                "evaluate",
                str(recipe),
                "--directory",
                str(tmp_path / "a"),
                "--repository",
                str(REPO),
            ]
        )
        == 0
    )
    report = run_evaluation(sampled, tmp_path / "b", REPO)
    assert report["metrics"]["problems"] == 7
    for name in ("predictions.jsonl", "metrics.json", "audit.json"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()
    ArtifactStore(tmp_path / "a").read_manifest()
    assert before == {p.name: hash_file(p) for p in cp.iterdir() if p.is_file()}
    assert (
        torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "a/predictions.jsonl").read_text().splitlines()
    ]
    assert rows[0]["attempts"][1]["seed"] == attempt_seed(
        sampled.seed, rows[0]["identifier"], 1
    )
    with pytest.raises(FileExistsError):
        run_evaluation(sampled, tmp_path / "a", REPO)


def test_uniform_real_decoder_matches_naive_lm_and_choice(job: EvalJob) -> None:
    tokenizer, _ = load_tokenizer_bundle(Path(job.tokenizer.path), job.tokenizer.sha256)
    with torch.random.fork_rng(devices=[]):
        model, _, _ = load_model(job)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    for case in control_cases():
        if case.kind == "lm":
            actual = score_lm(model, tokenizer, case, job)
            expected_targets = len(tokenizer.encode(case.prompt)) + 1
            assert actual["targets"] == expected_targets
            assert actual["nll"] == pytest.approx(
                expected_targets * math.log(tokenizer.vocab_size), rel=1e-6
            )
        elif case.kind == "choice":
            actual = score_choice(model, tokenizer, case, job)
            expected = [
                -len(tokenizer.encode(c)) * math.log(tokenizer.vocab_size)
                for c in case.choices
            ]
            assert actual["logp"] == pytest.approx(expected, rel=1e-6)
            assert actual["selected"] == max(
                range(len(expected)), key=expected.__getitem__
            )


def test_nonuniform_logits_match_scalar_teacher_forcing_reference(job: EvalJob) -> None:
    tokenizer, _ = load_tokenizer_bundle(Path(job.tokenizer.path), job.tokenizer.sha256)
    model, _, _ = load_model(job)

    def naive(prefix: list[int], targets: list[int]) -> float:
        with torch.no_grad():
            logits = model(torch.tensor([prefix + targets[:-1]], dtype=torch.int64))[
                0
            ].tolist()
        result = 0.0
        for offset, target in enumerate(targets):
            values = logits[len(prefix) - 1 + offset]
            peak = max(values)
            result += (
                values[target]
                - peak
                - math.log(sum(math.exp(v - peak) for v in values))
            )
        return result

    lm_case = control_cases()[-1]
    targets = tokenizer.encode(lm_case.prompt) + [2]
    expected_nll = 0.0
    for start in range(0, len(targets), 3):
        expected_nll -= naive(
            [1 if start == 0 else targets[start - 1]], targets[start : start + 3]
        )
    actual = score_lm(model, tokenizer, lm_case, replace(job, lm_window=3))
    assert actual["nll"] == pytest.approx(expected_nll, rel=1e-6)
    choice = control_cases()[2]
    prefix = prompt_ids(tokenizer, choice.prompt, job.prompt_format)
    expected = [naive(prefix, tokenizer.encode(c)) for c in choice.choices]
    assert score_choice(model, tokenizer, choice, job)["logp"] == pytest.approx(
        expected, rel=1e-6
    )


def test_scripted_generation_stop_truncation_seed_independence_and_failures(
    job: EvalJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer, _ = load_tokenizer_bundle(Path(job.tokenizer.path), job.tokenizer.sha256)
    model, _, _ = load_model(job)
    case = control_cases()[0]
    prefix = prompt_ids(tokenizer, case.prompt, job.prompt_format)
    answer = text_ids(tokenizer, "1/2")
    tokens = answer + [6]

    def oracle(inputs: Tensor) -> Tensor:
        logits = torch.full((*inputs.shape, model.config.vocab), -1000.0)
        logits[:, -1, tokens[inputs.shape[1] - len(prefix)]] = 1000
        return logits

    monkeypatch.setattr(model, "forward", oracle)
    enough = replace(job, max_new_tokens=16)
    result = generate_attempt(model, tokenizer, case, enough, 0)
    assert result["correct"] and result["format_valid"] and result["stop"] == "complete"
    truncated = generate_attempt(
        model, tokenizer, case, replace(enough, max_new_tokens=len(answer)), 0
    )
    assert (
        truncated["format_valid"]
        and not truncated["correct"]
        and truncated["stop"] == "token_limit"
    )
    assert (
        generate_attempt(
            model, tokenizer, case, replace(enough, context_limit=4, lm_window=4), 0
        )["stop"]
        == "context_overflow"
    )
    tokens[-1] = 2
    assert (
        generate_attempt(model, tokenizer, case, enough, 0)["stop"]
        == "premature_control"
    )
    sampled = replace(
        enough, sampling="sample", samples=2, pass_k=(1, 2), temperature=1
    )
    a = generate_attempt(model, tokenizer, case, sampled, 1)
    generate_attempt(model, tokenizer, case, sampled, 0)
    assert a == generate_attempt(model, tokenizer, case, sampled, 1)
    monkeypatch.setattr(
        model,
        "forward",
        lambda x: torch.full((*x.shape, model.config.vocab), float("nan")),
    )
    assert (
        generate_attempt(model, tokenizer, case, enough, 0)["stop"]
        == "nonfinite_logits"
    )
    literal = "Hello <|assistant|>"
    assert tokenizer.decode(text_ids(tokenizer, literal)) == literal
    assert min(text_ids(tokenizer, literal)) >= 32
    monkeypatch.setattr(tokenizer, "encode", lambda text: [5])
    with pytest.raises(ValueError, match="control-token"):
        text_ids(tokenizer, literal)


def test_failed_attempts_remain_in_metric_denominators(job: EvalJob) -> None:
    rows: list[dict[str, Any]] = [
        {
            "kind": "math",
            "domain": "math",
            "attempts": [
                {
                    "correct": False,
                    "format_valid": False,
                    "stop": "context_overflow",
                    "generated_tokens": 0,
                }
            ],
        }
    ]
    result = aggregate(rows, job)
    assert result["generation"]["problems"] == 1
    assert result["generation"]["pass_at_1_first"] == 0
    assert result["generation"]["failed"] == 1
    assert result["domains"]["math"]["problems"] == 1


def test_final_approval_and_license_guards(job: EvalJob, tmp_path: Path) -> None:
    freeze = freeze_holdout(tmp_path / "frozen", REPO)
    final = freeze["partitions"]["final"]
    ref = ArtifactRef(
        str(tmp_path / "frozen" / final["manifest"]), final["manifest_sha256"]
    )
    with pytest.raises(ValueError, match="Final holdout"):
        load_cases(replace(job, dataset=ref))
    with pytest.raises(ValueError, match="approval"):
        load_cases(replace(job, approved_for_evaluation=False))
    denied = write_raw_bundle(
        [case_document(control_cases()[0])],
        tmp_path / "denied",
        authored_source("denied-eval"),
        PERMISSION,
        allowed_uses=("training",),
    )
    denied_ref = ArtifactRef(
        str(tmp_path / "denied/manifest.json"), manifest_digest(denied)
    )
    with pytest.raises(ValueError, match="permit"):
        load_cases(replace(job, dataset=denied_ref))
    with pytest.raises(ValueError, match="tokenizer"):
        load_model(replace(job, tokenizer=replace(job.tokenizer, sha256="f" * 64)))
    with pytest.raises(ValueError):
        run_evaluation(
            replace(job, dataset=denied_ref), tmp_path / "failed-report", REPO
        )
    assert not (tmp_path / "failed-report").exists()
    (tmp_path / "denied/data.jsonl").write_text("corrupt")
    with pytest.raises(ArtifactIntegrityError):
        load_cases(replace(job, dataset=denied_ref))


def test_lexical_near_overlap_operator_sensitivity_and_unknown_lineage(
    job: EvalJob, tmp_path: Path
) -> None:
    cases = [
        EvalCase(
            "near",
            "math",
            "math",
            "new-family",
            "control",
            " ".join(f"term{i}" for i in range(40)),
            "1",
        ),
        EvalCase("operator", "math", "math", "operator", "control", "7 + 3", "10"),
    ]
    raw = write_raw_bundle(
        [
            RawDocument(cases[0].prompt.replace("term20", "replacement")),
            RawDocument("7 - 3"),
        ],
        tmp_path / "source",
        authored_source("overlap-reference"),
        PERMISSION,
    )
    ref = ArtifactRef(str(tmp_path / "source/manifest.json"), manifest_digest(raw))
    audit = audit_overlap(
        cases, replace(job, training_sources=(ref,), audit_ngram=3, audit_jaccard=0.7)
    )
    assert (
        audit["findings"]["near"]["ngram"] and audit["findings"]["near"]["near_jaccard"]
    )
    assert not audit["findings"]["near"]["normalized_exact"]
    assert not audit["findings"]["operator"]["normalized_exact"]
    assert (
        not audit["cleanliness_claim"]
        and not audit["complete_training_history_verified"]
    )
