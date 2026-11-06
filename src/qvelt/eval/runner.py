"""Read-only CPU FP32 evaluation;commit reports only after all inputs/checks pass."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from qvelt.config import ExperimentConfig, config_digest, dump_config
from qvelt.eval.code_eval import score_code
from qvelt.eval.config import EvalJob, dump_eval, eval_digest
from qvelt.eval.corpus import EvalCase, audit_overlap, load_cases
from qvelt.eval.diagnostics import (
    arithmetic_witnesses,
    canonical_answer,
    failure_mode,
    majority_vote,
)
from qvelt.eval.scoring import pass_at_k, perplexity, score_answer, wilson_interval
from qvelt.inference.artifacts import checked_payload, load_checkpoint_model
from qvelt.inference.config import GenerationOptions, TokenRequest
from qvelt.inference.generation import GenerationEngine
from qvelt.model.decoder import Decoder
from qvelt.posttrain.dpo import response_logps
from qvelt.tokenizer.core import QveltTokenizer
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.checkpoints import source_digest
from qvelt.utils.manifest_types import CheckpointInfo, Manifest, RunInfo
from qvelt.utils.manifests import manifest_bytes, manifest_digest
from qvelt.utils.provenance import capture_code, capture_environment


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


_payload = checked_payload


def load_model(job: EvalJob) -> tuple[Decoder, ExperimentConfig, Manifest]:
    return load_checkpoint_model(job.checkpoint, job.tokenizer.sha256)


def text_ids(tokenizer: QveltTokenizer, text: str) -> list[int]:
    tokens = tokenizer.encode(text)
    if any(t < 32 for t in tokens):
        raise ValueError(
            "Ordinary evaluation text cannot contain control-token spellings"
        )
    return tokens


def prompt_ids(tokenizer: QveltTokenizer, text: str, protocol: str) -> list[int]:
    ordinary = text_ids(tokenizer, text)
    return [1, 4, *ordinary, 6, 5] if protocol == "chat-v1" else [1, *ordinary]


def attempt_seed(seed: int, identifier: str, attempt: int) -> int:
    value = f"qvelt-eval-v1\0{seed}\0{identifier}\0{attempt}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63)


@torch.no_grad()
def generate_attempt(
    model: Decoder,
    tokenizer: QveltTokenizer,
    case: EvalCase,
    job: EvalJob,
    attempt: int,
) -> dict[str, Any]:
    prefix = prompt_ids(tokenizer, case.prompt, job.prompt_format)
    context = min(model.config.context, job.context_limit)
    seed = attempt_seed(job.seed, case.identifier, attempt) % (
        2**32 if job.protocol == "qvelt-eval-v2" else 2**63
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    actions: list[int] = []
    content: list[int] = []
    stop = "token_limit"
    text = ""
    native = job.protocol == "qvelt-eval-v2"
    if native and len(prefix) + job.max_new_tokens > context:
        stop = "context_overflow"
    elif native:
        request = TokenRequest(
            "evaluation",
            tuple(prefix),
            GenerationOptions(
                max_new_tokens=job.max_new_tokens,
                strategy=job.sampling,
                temperature=job.temperature,
                seed=seed % 2**32,
                context_limit=context,
                max_seconds=300,
                stop_tokens=(6,) if job.prompt_format == "chat-v1" else (2,),
            ),
        )
        output = GenerationEngine(model, tokenizer).generate((request,))[0]
        actions, text = list(output.token_ids), output.text
        stop = {
            "stop_token": "complete",
            "max_tokens": "token_limit",
            "nonfinite": "nonfinite_logits",
        }.get(output.reason, output.reason)
    for _ in range(0 if native else job.max_new_tokens):
        if len(prefix) + len(actions) > context:
            stop = "context_overflow"
            break
        logits = model(torch.tensor([prefix + actions], dtype=torch.int64))[
            0, -1
        ].float()
        if not bool(torch.isfinite(logits).all()):
            stop = "nonfinite_logits"
            break
        if job.sampling == "greedy":
            token = int(logits.argmax())
        else:
            probabilities = torch.softmax(logits / job.temperature, dim=-1)
            if not bool(torch.isfinite(probabilities).all()):
                stop = "nonfinite_logits"
                break
            token = int(torch.multinomial(probabilities, 1, generator=generator))
        actions.append(token)
        if token < 32:
            stop = (
                "complete"
                if token == (6 if job.prompt_format == "chat-v1" else 2)
                else "premature_control"
            )
            break
        content.append(token)
        text = tokenizer.decode(content)
        if len(text) > 8192:
            text, stop = text[:8192], "character_limit"
            break
    attempt_result = {
        "attempt": attempt,
        "seed": seed,
        "action_ids": actions,
        "generated_tokens": len(actions),
        "text": text,
        "stop": stop,
        **(
            score_code(text, case.target, complete=stop == "complete")
            if case.kind == "code"
            else score_answer(
                text,
                case.target,
                case.kind,
                job.extraction,
                complete=stop == "complete",
                atol=job.numeric_atol,
                rtol=job.numeric_rtol,
            )
        ),
    }
    if stop not in ("complete", "token_limit"):
        attempt_result["correct"] = False
    attempt_result["failure_mode"] = failure_mode(
        attempt_result, expected_method=case.expected_method
    )
    attempt_result["arithmetic_witnesses"] = arithmetic_witnesses(text)
    body: list[int] = []
    if tokenizer.control_version == 2 and actions.count(8) == actions.count(9) == 1:
        start, end = actions.index(8), actions.index(9)
        if end > start:
            body = actions[start + 1 : end]
    attempt_result["reasoning_tokens"] = len(body)
    attempt_result["reasoning_characters"] = len(tokenizer.decode(body))
    return attempt_result


@torch.no_grad()
def continuation_logp(model: Decoder, prefix: list[int], targets: list[int]) -> float:
    inputs = torch.tensor([prefix + targets[:-1]], dtype=torch.int64)
    labels = torch.tensor([[0] * (len(prefix) - 1) + targets], dtype=torch.int64)
    mask = torch.arange(inputs.shape[1]).unsqueeze(0) >= len(prefix) - 1
    return float(response_logps(model(inputs), labels, mask).item())


def score_choice(
    model: Decoder, tokenizer: QveltTokenizer, case: EvalCase, job: EvalJob
) -> dict[str, Any]:
    prefix = prompt_ids(tokenizer, case.prompt, job.prompt_format)
    choices = [text_ids(tokenizer, c) for c in case.choices]
    context = min(model.config.context, job.context_limit)
    if any(not c or len(prefix) + len(c) - 1 > context for c in choices):
        return {
            "failed": "context_overflow",
            "correct": False,
            "normalized_correct": False,
        }
    try:
        scores = [continuation_logp(model, prefix, c) for c in choices]
    except FloatingPointError:
        return {
            "failed": "nonfinite_logits",
            "correct": False,
            "normalized_correct": False,
        }
    lengths = (
        [len(c) for c in choices]
        if job.choice_normalization == "token"
        else [len(c) for c in case.choices]
    )
    if job.choice_normalization == "character-without-delimiter":
        if any(not c.startswith(" ") or len(c) < 2 for c in case.choices):
            raise ValueError(
                "Delimiter-excluded normalization needs one stored leading space"
            )
        lengths = [len(c) - 1 for c in case.choices]
    normalized = [s / n for s, n in zip(scores, lengths, strict=True)]
    selected = max(range(len(scores)), key=scores.__getitem__)
    selected_normalized = max(range(len(scores)), key=normalized.__getitem__)
    return {
        "failed": None,
        "correct": case.choices[selected] == case.target,
        "normalized_correct": case.choices[selected_normalized] == case.target,
        "logp": scores,
        "token_lengths": [len(c) for c in choices],
        "selected": selected,
        "selected_normalized": selected_normalized,
    }


def score_lm(
    model: Decoder, tokenizer: QveltTokenizer, case: EvalCase, job: EvalJob
) -> dict[str, Any]:
    targets = text_ids(tokenizer, case.prompt) + [2]
    window = min(job.lm_window, job.context_limit, model.config.context)
    nll = 0.0
    try:
        for start in range(0, len(targets), window):
            prefix = [1 if start == 0 else targets[start - 1]]
            nll -= continuation_logp(model, prefix, targets[start : start + window])
    except FloatingPointError:
        return {
            "failed": "nonfinite_logits",
            "targets": 0,
            "nll": 0.0,
            "expected_targets": len(targets),
        }
    return {
        "failed": None,
        "targets": len(targets),
        "nll": nll,
        "expected_targets": len(targets),
    }


def aggregate(rows: list[dict[str, Any]], job: EvalJob) -> dict[str, Any]:
    generated = [r for r in rows if r["kind"] in ("math", "numeric", "code")]
    choice = [r for r in rows if r["kind"] == "choice"]
    lm = [r for r in rows if r["kind"] == "lm"]
    first = sum(bool(r["attempts"][0]["correct"]) for r in generated)
    attempts = [a for r in generated for a in r["attempts"]]
    votes: dict[str, Any] = {}
    math_rows = [r for r in generated if r["kind"] != "code"]
    for k in job.pass_k:
        correct = ties = 0
        for row in math_rows:
            vote = majority_vote(
                [
                    canonical_answer(a, row["kind"], job.extraction)
                    for a in row["attempts"]
                ],
                k,
            )
            ties += vote["tied"]
            if vote["answer"] is not None:
                correct += (
                    (vote["answer"] == row["target"].replace(",", "").removesuffix("."))
                    if job.extraction == "gsm8k-strict-v1"
                    else bool(
                        score_answer(
                            vote["answer"],
                            row["target"],
                            row["kind"],
                            "whole-v1",
                            complete=True,
                            atol=job.numeric_atol,
                            rtol=job.numeric_rtol,
                        )["correct"]
                    )
                )
        votes[str(k)] = {
            "problems": len(math_rows),
            "correct": correct,
            "accuracy": correct / len(math_rows) if math_rows else None,
            "ties": ties,
            "wilson95_descriptive": wilson_interval(correct, len(math_rows))
            if math_rows
            else None,
        }
    domains: dict[str, dict[str, Any]] = {}
    for domain in sorted({r["domain"] for r in rows}):
        subset = [r for r in rows if r["domain"] == domain and r["kind"] != "lm"]
        c = sum(
            bool(
                r["attempts"][0]["correct"]
                if "attempts" in r
                else r["score"]["correct"]
            )
            for r in subset
        )
        domains[domain] = {
            "problems": len(subset),
            "correct": c,
            "accuracy": c / len(subset) if subset else None,
            "wilson95_descriptive": wilson_interval(c, len(subset)) if subset else None,
        }
        domain_lm = [r["score"] for r in lm if r["domain"] == domain]
        domain_targets = sum(r["targets"] for r in domain_lm)
        domain_nll = sum(r["nll"] for r in domain_lm)
        domain_failed = sum(bool(r["failed"]) for r in domain_lm)
        domains[domain]["lm"] = {
            "documents": len(domain_lm),
            "targets": domain_targets,
            "nll": domain_nll,
            "failed_documents": domain_failed,
            "perplexity": perplexity(domain_nll, domain_targets, domain_failed),
        }
    total_targets = sum(r["score"]["targets"] for r in lm)
    total_nll = sum(r["score"]["nll"] for r in lm)
    failed_lm = sum(bool(r["score"]["failed"]) for r in lm)
    languages: dict[str, Any] = {}
    for language in sorted({r.get("language", "und") for r in rows}):
        subset = [
            r
            for r in rows
            if r.get("language", "und") == language and r["kind"] != "lm"
        ]
        correct = sum(
            bool(
                r["attempts"][0]["correct"]
                if "attempts" in r
                else r["score"]["correct"]
            )
            for r in subset
        )
        languages[language] = {
            "problems": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else None,
            "wilson95_descriptive": wilson_interval(correct, len(subset))
            if subset
            else None,
        }
    return {
        "official_benchmark": False,
        "problems": len(rows),
        "families": dict(
            sorted(Counter(r.get("family", "unknown") for r in rows).items())
        ),
        "domains": domains,
        "languages": languages,
        "generation": {
            "problems": len(generated),
            "attempts": len(attempts),
            "first_correct": first,
            "pass_at_1_first": first / len(generated) if generated else None,
            "pass_at_k": {
                str(k): sum(
                    pass_at_k(
                        job.samples, sum(bool(a["correct"]) for a in r["attempts"]), k
                    )
                    for r in generated
                )
                / len(generated)
                if generated
                else None
                for k in job.pass_k
            },
            "invalid": sum(not a["format_valid"] for a in attempts),
            "truncated": sum(
                a["stop"] in ("token_limit", "character_limit", "context_overflow")
                for a in attempts
            ),
            "failed": sum(
                a["stop"]
                in ("context_overflow", "nonfinite_logits", "premature_control")
                for a in attempts
            ),
            "tokens": sum(a["generated_tokens"] for a in attempts),
            "average_completion_tokens": sum(a["generated_tokens"] for a in attempts)
            / len(attempts)
            if attempts
            else None,
            "average_reasoning_tokens": sum(
                a.get("reasoning_tokens", 0) for a in attempts
            )
            / len(attempts)
            if attempts
            else None,
            "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
            "failure_modes": dict(
                sorted(
                    Counter(
                        a.get("failure_mode", failure_mode(a)) for a in attempts
                    ).items()
                )
            ),
            "majority_vote_at_k": votes,
            "stop_counts": dict(sorted(Counter(a["stop"] for a in attempts).items())),
            "wilson95_descriptive": wilson_interval(first, len(generated))
            if generated
            else None,
        },
        "choice": {
            "problems": len(choice),
            "correct": sum(r["score"]["correct"] for r in choice),
            "normalized_correct": sum(r["score"]["normalized_correct"] for r in choice),
            "failed": sum(bool(r["score"]["failed"]) for r in choice),
            "accuracy": sum(r["score"]["correct"] for r in choice) / len(choice)
            if choice
            else None,
            "wilson95_descriptive": wilson_interval(
                sum(r["score"]["correct"] for r in choice), len(choice)
            )
            if choice
            else None,
            "normalization": job.choice_normalization,
        },
        "lm": {
            "documents": len(lm),
            "failed_documents": failed_lm,
            "targets": total_targets,
            "nll": total_nll,
            "perplexity": perplexity(total_nll, total_targets, failed_lm),
        },
        "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed.",
        "protocols": {
            "generation": "explicit-control-stop-v1",
            "choice": "continuation-sum-and-"
            + job.choice_normalization
            + "-mean;first-option-tie",
            "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
            "pass_at_k": "1-C(n-c,k)/C(n,k)",
        },
    }


def run_evaluation(
    job: EvalJob,
    directory: Path,
    repository: Path,
    *,
    command: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    dataset, cases, available = load_cases(job)
    audit = audit_overlap(cases, job)
    tokenizer, token_manifest = load_tokenizer_bundle(
        Path(job.tokenizer.path), job.tokenizer.sha256
    )
    previous_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rows: list[dict[str, Any]] = []
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            model, experiment, checkpoint = load_model(job)
            if tokenizer.vocab_size != model.config.vocab:
                raise ValueError("Checkpoint/tokenizer vocabularies disagree")
            for case in cases:
                row = asdict(case)
                if case.kind in ("math", "numeric", "code"):
                    row["attempts"] = [
                        generate_attempt(model, tokenizer, case, job, i)
                        for i in range(job.samples)
                    ]
                else:
                    row["score"] = (
                        score_choice if case.kind == "choice" else score_lm
                    )(model, tokenizer, case, job)
                rows.append(row)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(previous_threads)
    metrics = aggregate(rows, job)
    metrics.update(
        {
            "available_problems": available,
            "evaluated_prefix": len(cases),
            "evaluation_sha256": eval_digest(job),
        }
    )
    root, code = capture_code(repository)
    environment = capture_environment()
    config = replace(
        experiment,
        runtime=replace(
            experiment.runtime,
            device="cpu",
            precision="fp32",
            seed=job.seed,
            deterministic=True,
        ),
    )
    run_id = uuid.uuid4().hex
    store = ArtifactStore(create_directory(directory))
    if command is None:
        command = (
            "qvelt",
            "evaluate",
            os.path.relpath(directory / "evaluation.yaml", root),
            "--directory",
            os.path.relpath(directory.parent / (directory.name + "-reproduce"), root),
            "--repository",
            ".",
        )
    if not 1 <= len(command) <= 128 or any(
        not isinstance(part, str)
        or not 1 <= len(part) <= 2048
        or any(ord(c) < 32 for c in part)
        for part in command
    ):
        raise ValueError("Evaluation reproduction argv is invalid")
    files = [
        store.add_bytes("config.yaml", dump_config(config).encode(), role="config"),
        store.add_file("uv.lock", root / "uv.lock", role="dependency_lock"),
        store.add_bytes(
            "evaluation.yaml", dump_eval(job).encode(), role="evaluation_recipe"
        ),
        store.add_bytes(
            "predictions.jsonl", b"".join(_json(r) for r in rows), role="predictions"
        ),
        store.add_bytes("metrics.json", _json(metrics), role="metrics"),
        store.add_bytes("audit.json", _json(audit), role="overlap_audit"),
        store.add_bytes(
            "command.json",
            _json({"argv": command, "kind": "reproduction_command"}),
            role="reproduction_command",
        ),
        store.add_bytes(
            "provenance.json",
            _json(
                {
                    "source_sha256": source_digest(),
                    "code": asdict(code),
                    "environment": asdict(environment),
                    "device": "cpu",
                    "precision": "fp32",
                    "threads": 1,
                    "checkpoint_immutable": True,
                    "checkpoint_optimizer_step": checkpoint.metadata.optimizer_step
                    if isinstance(checkpoint.metadata, CheckpointInfo)
                    else None,
                }
            ),
            role="evaluation_provenance",
        ),
    ]
    for name, manifest in (
        ("checkpoint", checkpoint),
        ("tokenizer", token_manifest),
        ("dataset", dataset),
    ):
        files.append(
            store.add_bytes(
                name + ".manifest.json",
                manifest_bytes(manifest),
                role=name + "_manifest",
            )
        )
    for i, ref in enumerate(job.training_sources):
        audited = ArtifactStore(Path(ref.path).parent).read_manifest(
            Path(ref.path).name, expected_sha256=ref.sha256
        )
        files.append(
            store.add_bytes(
                f"audit-source-{i}.manifest.json",
                manifest_bytes(audited),
                role="audit_source_manifest",
            )
        )
    report = Manifest(
        kind="run",
        owner=run_id,
        created_at=datetime.now(UTC).isoformat(),
        metadata=RunInfo(run_id, config_digest(config), code, environment, job.seed, 1),
        files=tuple(files),
    )
    store.publish(report)
    return {"report_sha256": manifest_digest(report), "metrics": metrics}
