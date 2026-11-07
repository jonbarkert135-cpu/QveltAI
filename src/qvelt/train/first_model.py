"""Bounded documented RFC tiny pipeline;original fixtures,not a useful-model promise."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from qvelt.config import load_config
from qvelt.config.pretraining import TokenSchedule, dump_pretraining_job
from qvelt.eval.config import ArtifactRef, dump_eval
from qvelt.eval.ledger import build_results, export_receipt
from qvelt.eval.runner import run_evaluation
from qvelt.eval.suite import SuiteSettings, create_original_suite, suite_job
from qvelt.eval.suite_smoke import _ref
from qvelt.inference.artifacts import export_snapshot, load_snapshot
from qvelt.inference.benchmark import benchmark_server
from qvelt.inference.config import (
    EngineLimits,
    GenerationJob,
    GenerationOptions,
    TextRequest,
    dump_generation,
)
from qvelt.inference.huggingface import export_huggingface
from qvelt.posttrain.safety import build_safety_data
from qvelt.train.diagnostics import acceptance_recipe, create_acceptance_data
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.utils.artifacts import create_directory, hash_file
from qvelt.utils.checkpoints import latest_checkpoint, source_digest
from qvelt.utils.provenance import capture_code, capture_environment


def first_model(directory: Path, repository: Path, *, steps: int = 6) -> dict[str, Any]:
    if type(steps) is not int or not 4 <= steps <= 20:
        raise ValueError("The bounded tiny tutorial needs4..20 updates")
    started = time.perf_counter()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory.parent).free < 1024**3:
        raise ValueError(
            "Tiny tutorial needs at least1GiB free at the output filesystem;allow8GiB for the full environment"
        )
    root = create_directory(directory)
    repository = repository.resolve()
    data = create_acceptance_data(root / "data", repository)
    base = acceptance_recipe(data, repository, steps=steps)
    base = replace(
        base,
        keep_checkpoints=1,
        sample_every=0,
        layer_stats_every=0,
        validation_every=steps,
        domain_validation_batches=0,
        job=replace(
            base.job, training=replace(base.job.training, checkpoint_every=steps)
        ),
    )
    assert (
        base.job.experiment.model
        == load_config(repository / "configs/model/tiny.yaml").model
    )
    assert base.job.experiment.model.parameter_count == 19_338_560
    (root / "pretraining.yaml").write_text(dump_pretraining_job(base))
    stop = steps // 2
    train_pretraining(base, root / "pretraining", repository, stop_after=stop)
    resumed = train_pretraining(base, root / "pretraining", repository, resume=True)
    assert resumed["resumed_from"] == stop
    checkpoint = _ref(latest_checkpoint(root / "pretraining") / "manifest.json")
    tokenizer = _ref(root / "data/prepared/tokenizer/manifest.json")
    build_safety_data(tokenizer, root / "safety")
    sft = demo_recipe(
        root / "safety/prepared/manifest.json", repository, steps=3, accumulation=1
    )
    sft = replace(
        sft,
        objective="sft",
        initial_checkpoint=checkpoint.path,
        initial_sha256=checkpoint.sha256,
        schedule=TokenSchedule(0, 10**9, 1),
        keep_checkpoints=1,
        job=replace(
            sft.job,
            experiment=replace(sft.job.experiment, model=base.job.experiment.model),
            training=replace(sft.job.training, checkpoint_every=3, learning_rate=1e-4),
        ),
    )
    (root / "sft.yaml").write_text(dump_pretraining_job(sft))
    train_pretraining(sft, root / "sft", repository)
    final = _ref(latest_checkpoint(root / "sft") / "manifest.json")
    weights = Path(final.path).parent / "model.safetensors"
    before = hash_file(weights)
    settings = SuiteSettings(
        math_per_language=1,
        samples=1,
        max_new_tokens=4,
        context_limit=512,
        seed=20261107,
    )
    create_original_suite(settings, root / "evaluation-data")
    evaluation = suite_job(
        settings,
        final,
        tokenizer,
        _ref(root / "evaluation-data/manifest.json"),
        (_ref(data), _ref(root / "safety/prepared/manifest.json")),
    )
    (root / "evaluation.yaml").write_text(dump_eval(evaluation))
    command = (
        "python",
        "scripts/first_model.py",
        "--directory",
        os.path.relpath(root, repository),
        "--repository",
        ".",
        "--steps",
        str(steps),
    )
    report = run_evaluation(
        evaluation, root / "evaluation", repository, command=command
    )
    receipt = root / "receipts/tutorial.json"
    export_receipt(
        _ref(root / "evaluation/manifest.json"),
        receipt,
        benchmark="original-en-ru-code",
    )
    build_results([receipt], root / "RESULTS.md")
    export_snapshot(final, tokenizer, root / "snapshot", repository)
    snapshot_ref = _ref(root / "snapshot/manifest.json")
    export_huggingface(snapshot_ref, root / "hf", repository)
    from transformers import AutoModelForCausalLM

    native = load_snapshot(snapshot_ref)
    hf = AutoModelForCausalLM.from_pretrained(
        root / "hf",
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    tokens = torch.tensor([[1, *native.tokenizer.encode("Compute two plus three.")]])
    with torch.no_grad():
        expected, actual = native.model(tokens), hf(tokens).logits
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        delta = float((actual - expected).abs().max())
    job = GenerationJob(
        snapshot_ref,
        (
            TextRequest(
                "first",
                "Compute two plus three.",
                GenerationOptions(max_new_tokens=8, context_limit=512),
            ),
        ),
        limits=EngineLimits(max_context=512),
    )
    (root / "generate.yaml").write_text(dump_generation(job))
    http = benchmark_server(
        ArtifactRef(str(Path(snapshot_ref.path).resolve()), snapshot_ref.sha256),
        root / "http",
    )
    assert hash_file(weights) == before
    _, code = capture_code(repository)
    from dataclasses import asdict

    result = {
        "scope": "real RFC tiny on original nonce/math fixtures;not useful reasoning or safety acceptance",
        "command": list(command),
        "source_sha256": source_digest(),
        "code": asdict(code),
        "environment": asdict(capture_environment()),
        "model": asdict(base.job.experiment.model),
        "parameters": base.job.experiment.model.parameter_count,
        "base_updates": steps,
        "resumed_from": stop,
        "safety_sft_updates": 3,
        "trained_safety_certified": False,
        "checkpoint_sha256": final.sha256,
        "model_payload_sha256": before.sha256,
        "tokenizer_sha256": tokenizer.sha256,
        "native_snapshot_sha256": snapshot_ref.sha256,
        "hf_max_abs_error": delta,
        "eval_report_sha256": report["report_sha256"],
        "final_cases_scored": 0,
        "weights_unchanged_by_eval": True,
        "http": http,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    return result
