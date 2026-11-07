"""Reproducible tiny inference/export/quantization/context/real-HTTP acceptance."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from qvelt.config.pretraining import dump_pretraining_job
from qvelt.eval.config import ArtifactRef, dump_eval
from qvelt.eval.ledger import build_results, export_receipt
from qvelt.eval.runner import run_evaluation
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.eval.suite import SuiteSettings, create_original_suite, suite_job
from qvelt.eval.suite_smoke import _ref
from qvelt.inference.artifacts import export_snapshot, load_snapshot
from qvelt.inference.benchmark import benchmark_server
from qvelt.inference.config import EngineLimits, GenerationOptions, TokenRequest
from qvelt.inference.generation import GenerationEngine
from qvelt.inference.hf_checkpoint import export_checkpoint_huggingface
from qvelt.inference.huggingface import export_huggingface
from qvelt.inference.long_context import extend_context
from qvelt.inference.needles import create_needles
from qvelt.inference.quantization import export_quantized, load_quantized
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.checkpoints import latest_checkpoint, source_digest


def run_phase8_smoke(directory: Path, repository: Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory.parent.mkdir(parents=True, exist_ok=True)
    root = create_directory(directory)
    repository = repository.resolve()
    fixture = create_evaluation_fixture(root / "fixture", repository)
    training_path = root / "fixture/model-data/prepared/manifest.json"
    recipe = demo_recipe(training_path, repository, steps=3)
    recipe = replace(
        recipe,
        job=replace(
            recipe.job,
            experiment=replace(
                recipe.job.experiment,
                model=replace(recipe.job.experiment.model, context=256),
            ),
        ),
    )
    (root / "pretraining.yaml").write_text(dump_pretraining_job(recipe))
    train_pretraining(recipe, root / "training", repository)
    checkpoint_path = latest_checkpoint(root / "training")
    checkpoint = _ref(checkpoint_path / "manifest.json")
    before = hash_file(checkpoint_path / "model.safetensors")
    export_snapshot(checkpoint, fixture.tokenizer, root / "snapshot", repository)
    native_ref = _ref(root / "snapshot/manifest.json")
    snapshot = load_snapshot(native_ref)
    hf_errors = {}
    for factor in (1, 2):
        target = root / f"hf-{factor}"
        if factor == 1:
            export_huggingface(native_ref, target, repository)
        else:
            export_checkpoint_huggingface(
                checkpoint,
                fixture.tokenizer,
                target,
                repository,
                context_extension=factor,
                shard_bytes=65536,
            )
        hf = AutoModelForCausalLM.from_pretrained(
            target,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            target, local_files_only=True, trust_remote_code=False
        )
        assert tokenizer.encode(
            "Привет café 😀", add_special_tokens=False
        ) == snapshot.tokenizer.encode("Привет café 😀")
        native = extend_context(snapshot.model, factor)
        ids = (
            torch.arange(259 if factor == 2 else 31) % native.config.vocab
        ).unsqueeze(0)
        with torch.no_grad():
            expected = native(ids)
            actual = hf(ids, use_cache=True)
            torch.testing.assert_close(actual.logits, expected, atol=4e-6, rtol=4e-5)
            suffix = torch.tensor([[32, 33]])
            cached = hf(
                suffix, past_key_values=actual.past_key_values, use_cache=True
            ).logits
            full = native(torch.cat((ids, suffix), dim=1))[:, -2:]
            torch.testing.assert_close(cached, full, atol=4e-6, rtol=4e-5)
        hf_errors[str(factor)] = {
            "full_max_abs_error": float((actual.logits - expected).abs().max()),
            "cached_max_abs_error": float((cached - full).abs().max()),
            "export_manifest_sha256": _ref(target / "manifest.json").sha256,
        }
    quantized = {}
    for bits in (4, 8):
        target = root / f"hqq-{bits}"
        export_quantized(native_ref, target, repository, bits=bits)
        exported = _ref(target / "manifest.json")
        restored = load_quantized(exported)
        assert restored.model.config == snapshot.model.config
        quantized[str(bits)] = {
            "manifest_sha256": exported.sha256,
            "payload_sha256": next(
                file.sha256
                for file in restored.manifest.files
                if file.role == "quantized_weights"
            ),
        }
    engine = GenerationEngine(
        snapshot.model, snapshot.tokenizer, EngineLimits(max_context=256)
    )
    options = GenerationOptions(
        strategy="sample",
        temperature=0.8,
        seed=20261107,
        max_new_tokens=12,
        top_k=32,
        top_p=0.9,
        min_p=0.02,
        repetition_penalty=1.1,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        context_limit=256,
    )
    requests = (
        TokenRequest("a", (1, *snapshot.tokenizer.encode("2+2?")), options),
        TokenRequest(
            "b", (1, *snapshot.tokenizer.encode("Привет!")), replace(options, seed=81)
        ),
    )
    generated = engine.generate(requests)
    assert generated == engine.generate(requests)
    uncached = engine.generate(requests, use_cache=False)
    assert [(r.token_ids, r.text, r.reason) for r in generated] == [
        (r.token_ids, r.text, r.reason) for r in uncached
    ]
    events = list(engine.stream(requests))
    for completion in generated:
        assert (
            tuple(
                event.token_id
                for event in events
                if event.identifier == completion.identifier
                and event.token_id is not None
            )
            == completion.token_ids
        )
    settings = SuiteSettings(samples=1, seed=20261107)
    create_original_suite(settings, root / "mixed")
    create_needles(snapshot.tokenizer, root / "needles")
    base = suite_job(
        settings,
        checkpoint,
        fixture.tokenizer,
        _ref(root / "mixed/manifest.json"),
        (_ref(training_path),),
    )
    command = (
        "python",
        "scripts/inference_phase8_smoke.py",
        "--directory",
        os.path.relpath(root, repository),
        "--repository",
        ".",
    )
    reports, receipts = {}, []
    jobs = [
        (f"mixed-{bits}", replace(base, quantization_bits=bits), "original-en-ru-code")
        for bits in (0, 4, 8)
    ]
    jobs += [
        (
            f"needle-{factor}",
            replace(
                base,
                dataset=_ref(root / "needles/manifest.json"),
                context_limit=1024,
                context_extension=factor,
            ),
            "original-needle",
        )
        for factor in (1, 2, 4)
    ]
    for name, job, benchmark in jobs:
        (root / (name + ".yaml")).write_text(dump_eval(job))
        report = run_evaluation(
            job, root / ("eval-" + name), repository, command=command
        )
        path = root / "receipts" / (name + ".json")
        export_receipt(
            _ref(root / ("eval-" + name) / "manifest.json"), path, benchmark=benchmark
        )
        receipts.append(path)
        reports[name] = report
        if job.quantization_bits:
            assert (
                report["metrics"]["model_variant"]["packed_payload_sha256"]
                == quantized[str(job.quantization_bits)]["payload_sha256"]
            )
    build_results(receipts, root / "RESULTS.md")
    build_results(receipts, root / "RESULTS.md", check=True)
    http_benchmark = benchmark_server(
        ArtifactRef(str(Path(native_ref.path).resolve()), native_ref.sha256),
        root / "http",
    )
    assert hash_file(checkpoint_path / "model.safetensors") == before
    result = {
        "scope": "tiny three-update engineering model;no release quality or GPU throughput claim",
        "source_sha256": source_digest(),
        "command": list(command),
        "model_config": asdict(snapshot.model.config),
        "parameters": snapshot.model.config.parameter_count,
        "checkpoint_sha256": checkpoint.sha256,
        "tokenizer_sha256": fixture.tokenizer.sha256,
        "native_snapshot_sha256": native_ref.sha256,
        "hf": hf_errors,
        "hqq": quantized,
        "sampling_cached_uncached_stream_repeat_exact": True,
        "checkpoint_weights_unchanged": True,
        "final_cases_scored": 0,
        "http": http_benchmark,
        "evaluations": reports,
        "receipts": [p.name for p in receipts],
    }
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    ArtifactStore(root / "snapshot").verify(snapshot.manifest)
    return result
