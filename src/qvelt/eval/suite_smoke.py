"""Small actual training, checkpoint regression and provenance-bound mixed evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qvelt.eval.benchmarks import import_benchmark
from qvelt.eval.config import ArtifactRef, dump_eval
from qvelt.eval.ledger import build_results, export_receipt
from qvelt.eval.runner import run_evaluation
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.eval.suite import create_original_suite, load_suite_settings, suite_job
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.train.smoke import _equal
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.checkpoints import latest_checkpoint, source_digest
from qvelt.utils.manifests import manifest_digest


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(
        str(path), manifest_digest(ArtifactStore(path.parent).read_manifest(path.name))
    )


def run_suite_smoke(
    directory: Path,
    repository: Path,
    *,
    with_harness: bool = False,
    external_sources: Path | None = None,
) -> dict[str, Any]:
    directory.parent.mkdir(parents=True, exist_ok=True)
    root = create_directory(directory)
    repository = repository.resolve()
    settings = load_suite_settings(repository / "configs/eval/quick.yaml")
    fixture = create_evaluation_fixture(root / "fixture", repository)
    create_original_suite(settings, root / "mixed")
    training = root / "fixture/model-data/prepared/manifest.json"
    base = suite_job(
        settings,
        fixture.checkpoint,
        fixture.tokenizer,
        _ref(root / "mixed/manifest.json"),
        (_ref(training),),
    )
    policy = replace(
        base,
        samples=1,
        sampling="greedy",
        temperature=0,
        pass_k=(1,),
        max_cases=4,
        max_new_tokens=4,
    )
    policy_path = root / "regression.yaml"
    policy_path.write_text(dump_eval(policy))
    recipe = demo_recipe(training, repository, steps=3)
    recipe = replace(
        recipe,
        job=replace(
            recipe.job,
            experiment=replace(
                recipe.job.experiment,
                model=replace(
                    recipe.job.experiment.model, context=settings.context_limit
                ),
            ),
            training=replace(
                recipe.job.training,
                checkpoint_every=1,
                regression_recipe=str(policy_path),
                regression_sha256=hash_file(policy_path).sha256,
            ),
        ),
    )
    train_pretraining(recipe, root / "training", repository)
    train_pretraining(recipe, root / "replay", repository, stop_after=2)
    resumed = train_pretraining(recipe, root / "replay", repository, resume=True)
    assert resumed["resumed_from"] == 2
    a, b = (latest_checkpoint(root / name) for name in ("training", "replay"))
    _equal(
        load_file(str(a / "model.safetensors")), load_file(str(b / "model.safetensors"))
    )
    for name in ("optimizer.pt", "rng-rank-00000.pt"):
        _equal(
            torch.load(a / name, weights_only=True),
            torch.load(b / name, weights_only=True),
        )
    curves = [
        json.loads((root / name / "regression/curve-step-00000003.json").read_text())
        for name in ("training", "replay")
    ]
    assert [(r["step"], r["value"]) for r in curves[0]["points"]] == [
        (r["step"], r["value"]) for r in curves[1]["points"]
    ]
    final = replace(base, checkpoint=_ref(a / "manifest.json"))
    jobs = [
        (
            "base",
            replace(base, samples=1, sampling="greedy", temperature=0, pass_k=(1,)),
            "original-en-ru-code",
        ),
        (
            "final",
            replace(final, samples=1, sampling="greedy", temperature=0, pass_k=(1,)),
            "original-en-ru-code",
        ),
        ("sampled", final, "original-en-ru-code"),
    ]
    if external_sources is not None:
        for name, payload, license_name in (
            ("gsm8k", "gsm8k-test.jsonl", "gsm8k-LICENSE"),
            (
                "mmlu-abstract-algebra",
                "mmlu-abstract-algebra-test.parquet",
                "mmlu-LICENSE",
            ),
            ("arc-challenge", "arc-challenge-test.parquet", "arc-LICENSE"),
        ):
            source, license_path = (
                external_sources / payload,
                external_sources / license_name,
            )
            import_benchmark(
                name,
                ArtifactRef(str(source), hash_file(source).sha256),
                ArtifactRef(str(license_path), hash_file(license_path).sha256),
                root / name,
                count=16,
            )
            job = replace(
                jobs[1][1],
                dataset=_ref(root / name / "manifest.json"),
                choice_normalization="character-without-delimiter"
                if name == "arc-challenge"
                else "character",
                extraction="gsm8k-strict-v1" if name == "gsm8k" else "whole-v1",
            )
            jobs.append((name, job, name))
    command: tuple[str, ...] = (
        "python",
        "scripts/evaluation_suite_smoke.py",
        "--directory",
        os.path.relpath(root, repository),
        "--repository",
        ".",
    )
    if with_harness:
        command += ("--with-harness",)
    if external_sources is not None:
        command += (
            "--external-sources",
            os.path.relpath(external_sources.resolve(), repository),
        )
    evidence: dict[str, Any] = {}
    receipts = []
    before = hash_file(a / "model.safetensors")
    for name, job, benchmark in jobs:
        (root / (name + ".yaml")).write_text(dump_eval(job))
        report = run_evaluation(
            job, root / ("eval-" + name), repository, command=command
        )
        receipt_path = root / "receipts" / (name + ".json")
        export_receipt(
            _ref(root / ("eval-" + name) / "manifest.json"),
            receipt_path,
            benchmark=benchmark,
        )
        receipts.append(receipt_path)
        evidence[name] = report
    repeat = run_evaluation(final, root / "eval-repeat", repository, command=command)
    for filename in ("predictions.jsonl", "metrics.json", "audit.json"):
        assert (root / "eval-repeat" / filename).read_bytes() == (
            root / "eval-sampled" / filename
        ).read_bytes()
    assert repeat["metrics"] == evidence["sampled"]["metrics"]
    assert hash_file(a / "model.safetensors") == before
    for step in range(1, 4):
        receipt_path = root / "receipts" / f"regression-{step}.json"
        export_receipt(
            _ref(root / f"training/regression/step-{step:08d}/manifest.json"),
            receipt_path,
            benchmark="original-en-ru-code",
        )
        receipts.append(receipt_path)
    build_results(receipts, root / "RESULTS.md")
    build_results(receipts, root / "RESULTS.md", check=True)
    harness: dict[str, Any] = {"executed": False}
    if with_harness:
        from qvelt.eval.harness import run_harness_parity

        harness = run_harness_parity(jobs[1][1])
    result = {
        "scope": "engineering controls and explicitly limited public development subsets;not a quality promotion",
        "source_sha256": source_digest(),
        "command": list(command),
        "checkpoint_regression_steps": [1, 2, 3],
        "resumed_from": 2,
        "weights_optimizer_rng_exact": True,
        "curve_values_exact": True,
        "sampled_predictions_metrics_audit_exact": True,
        "weights_unchanged_by_eval": True,
        "final_cases_scored": 0,
        "harness": harness,
        "evaluations": evidence,
        "receipts": [p.name for p in receipts],
    }
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    return result
