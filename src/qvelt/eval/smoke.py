"""Untrained real BPE checkpoint,known controls and repeated heldout CPU scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from qvelt.data.demo import PERMISSION, authored_source, create_demo
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.eval.config import ArtifactRef, EvalJob, dump_eval
from qvelt.eval.corpus import EvalCase
from qvelt.eval.fixtures import freeze_holdout
from qvelt.eval.runner import run_evaluation
from qvelt.model.decoder import Decoder
from qvelt.train.pretraining import prepare_inputs
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.checkpoints import (
    initialize_training_run,
    save_checkpoint,
    source_digest,
)
from qvelt.utils.distributed import DistributedContext
from qvelt.utils.manifests import manifest_digest


def case_document(case: EvalCase) -> RawDocument:
    values = asdict(case)
    version = 2
    if case.language == "und" and case.expected_method is None and case.kind != "code":
        values.pop("language")
        values.pop("expected_method")
        version = 1
    return RawDocument(
        json.dumps({"schema_version": version, **values}, sort_keys=True),
        case.identifier,
        case.family,
        case.language if case.language != "und" else "",
    )


def control_cases() -> list[EvalCase]:
    return [
        EvalCase(
            "control-exact",
            "math",
            "math",
            "control-exact",
            "control",
            "Return the fraction equal to one half.",
            "1/2",
        ),
        EvalCase(
            "control-numeric",
            "numeric",
            "math",
            "control-decimal",
            "control",
            "Return one divided by four as a decimal.",
            "0.25",
        ),
        EvalCase(
            "control-choice-web",
            "choice",
            "web",
            "control-choice-web",
            "control",
            "A cup usually holds",
            " water",
            (" water", " a mountain"),
        ),
        EvalCase(
            "control-choice-code",
            "choice",
            "code",
            "control-choice-code",
            "control",
            "An integer in Python can be added to",
            " another integer",
            (" another integer", " a function definition"),
        ),
        EvalCase(
            "control-lm-web",
            "lm",
            "web",
            "control-lm-web",
            "control",
            "The green cup stands beside the window.",
            "",
        ),
        EvalCase(
            "control-lm-code",
            "lm",
            "code",
            "control-lm-code",
            "control",
            "def square(x):\n    return x * x\n",
            "",
        ),
        EvalCase(
            "control-lm-math",
            "lm",
            "math",
            "control-lm-math",
            "control",
            "The sum of two even integers is even.",
            "",
        ),
    ]


def create_evaluation_fixture(directory: Path, repository: Path) -> EvalJob:
    """No optimizer updates;tokenizer fits approved older original training text only."""
    root = create_directory(directory)
    create_demo(root / "model-data", repository, documents=8)
    recipe = demo_recipe(
        root / "model-data/prepared/manifest.json", repository, steps=1
    )
    job = replace(
        recipe.job,
        experiment=replace(
            recipe.job.experiment,
            model=replace(recipe.job.experiment.model, context=256),
        ),
    )
    inputs = prepare_inputs(replace(recipe, job=job))
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.experiment.runtime.seed)
        model = Decoder(job.experiment.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        initialize_training_run(
            job,
            root / "checkpoint",
            repository,
            context,
            data=inputs.data,
            tokenizer=inputs.tokenizer,
            application_recipe=b"scope: untrained-evaluation-fixture-v1\n",
        )
        cp = save_checkpoint(
            model,
            optimizer,
            job,
            root / "checkpoint",
            0,
            context,
            processed_tokens=0,
            application_state={"kind": "untrained-evaluation-fixture-v1"},
        )
    controls = control_cases()
    dataset = write_raw_bundle(
        [case_document(c) for c in controls],
        root / "controls",
        authored_source("evaluation-controls"),
        PERMISSION,
        allowed_uses=("evaluation", "decontamination"),
    )
    # Deliberate audit calibration;this bundle was never used to fit this model.
    canary = write_raw_bundle(
        [RawDocument(controls[0].prompt, "deliberate-overlap", controls[0].family)],
        root / "audit-calibration",
        authored_source("evaluation-overlap-calibration"),
        PERMISSION,
    )
    return EvalJob(
        ArtifactRef(
            str(cp / "manifest.json"),
            manifest_digest(ArtifactStore(cp).read_manifest()),
        ),
        ArtifactRef(
            str(root / "model-data/prepared/tokenizer/manifest.json"),
            manifest_digest(inputs.tokenizer),
        ),
        ArtifactRef(str(root / "controls/manifest.json"), manifest_digest(dataset)),
        training_sources=(
            ArtifactRef(
                str(root / "audit-calibration/manifest.json"), manifest_digest(canary)
            ),
        ),
        approved_for_evaluation=True,
        context_limit=256,
        max_new_tokens=4,
        lm_window=16,
    )


def run_evaluation_smoke(directory: Path, repository: Path) -> dict[str, Any]:
    root = create_directory(directory)
    registry = freeze_holdout(root / "heldout", repository)
    registered = json.loads(
        (repository / "docs/development/evidence/evaluation-frozen-v1.json").read_text()
    )
    if registry != registered:
        raise ValueError("Prospective heldout no longer matches its committed freeze")
    controls = create_evaluation_fixture(root / "fixture", repository)
    validation = registry["partitions"]["validation"]
    heldout = replace(
        controls,
        dataset=ArtifactRef(
            str(root / "heldout" / validation["manifest"]),
            validation["manifest_sha256"],
        ),
        training_sources=(),
        max_cases=200,
    )
    sampled = replace(
        controls, sampling="sample", temperature=1.0, samples=3, pass_k=(1, 2)
    )
    evidence = {}
    before = hash_file(Path(controls.checkpoint.path).parent / "model.safetensors")
    for name, job in (
        ("controls", controls),
        ("sampled", sampled),
        ("validation", heldout),
    ):
        (root / (name + ".yaml")).write_text(dump_eval(job))
        a = run_evaluation(job, root / (name + "-a"), repository)
        b = run_evaluation(job, root / (name + "-b"), repository)
        for file in ("predictions.jsonl", "metrics.json", "audit.json"):
            assert (root / (name + "-a") / file).read_bytes() == (
                root / (name + "-b") / file
            ).read_bytes()
        evidence[name] = {
            "report_a": a["report_sha256"],
            "report_b": b["report_sha256"],
            "metrics": a["metrics"],
            "predictions_metrics_audit_bit_exact": True,
        }
    assert (
        hash_file(Path(controls.checkpoint.path).parent / "model.safetensors") == before
    )
    audit = json.loads((root / "controls-a/audit.json").read_text())
    assert audit["findings"]["control-exact"]["normalized_exact"]
    assert not audit["cleanliness_claim"]
    result = {
        "scope": "untrained original BPE model;engineering acceptance only;calibration audit source never trained",
        "source_sha256": source_digest(),
        "checkpoint_sha256": controls.checkpoint.sha256,
        "tokenizer_sha256": controls.tokenizer.sha256,
        "checkpoint_weights_unchanged": True,
        "final_cases_scored": 0,
        "evaluations": evidence,
    }
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    return result
