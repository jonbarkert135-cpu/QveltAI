"""Bounded original-data process recovery, compilation and mini-scaling diagnostics."""

from __future__ import annotations

import hashlib
import json
import random
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from qvelt.config import load_config
from qvelt.config.pretraining import PretrainingJob, TokenSchedule, dump_pretraining_job
from qvelt.data.config import DataRecipe, MixStage, SourceSpec, dump_data_recipe
from qvelt.data.demo import HOLDOUT, PERMISSION, authored_source
from qvelt.data.math import generate_problems
from qvelt.data.pipeline import prepare_data
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.tokenizer.core import TokenizerSettings
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.train.report import merged_metrics
from qvelt.train.smoke import _equal
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.checkpoints import latest_checkpoint, source_digest
from qvelt.utils.manifest_types import CheckpointInfo, DataInfo
from qvelt.utils.manifests import manifest_digest


def create_acceptance_data(directory: Path, repository: Path) -> Path:
    """Train-only BPE32768 on original domain fixtures; never production approval."""
    root = create_directory(directory)
    rng = random.Random(20261104)
    sources = []
    problems = generate_problems(32, 20261104)
    for domain in ("web", "code", "math"):
        documents = []
        for index, problem in enumerate(problems):
            prefix = {
                "web": f"Original educational article {index}: numbers and algorithms describe precise processes.",
                "code": f"def add_{index}(x):\n    return x + {index}\n# Original fixture",
                "math": problem.question + "\nAnswer: " + problem.answer,
            }[domain]
            nonce = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz ", k=4096))
            documents.append(
                RawDocument(
                    prefix + "\nNonce: " + nonce,
                    f"{domain}-{index}",
                    f"{domain}-family-{index}",
                    "en",
                )
            )
        raw = write_raw_bundle(
            documents, root / domain, authored_source("phase5-" + domain), PERMISSION
        )
        sources.append(
            SourceSpec(
                domain, domain + "/manifest.json", manifest_digest(raw), True, True
            )
        )
    canary = write_raw_bundle(
        [RawDocument(HOLDOUT)],
        root / "holdout",
        authored_source("phase5-canary"),
        PERMISSION,
        allowed_uses=("evaluation", "decontamination"),
    )
    registry = (
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "demo",
                "required": ["canary"],
                "benchmarks": [
                    {
                        "name": "canary",
                        "manifest": "holdout/manifest.json",
                        "sha256": manifest_digest(canary),
                        "approved": True,
                        "fixture": True,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    (root / "benchmarks.json").write_bytes(registry)
    recipe = DataRecipe(
        tuple(sources),
        "benchmarks.json",
        hashlib.sha256(registry).hexdigest(),
        (MixStage(0, tuple((s.name, 1.0) for s in sources)),),
        context=16,
        validation_fraction=0.2,
        test_fraction=0.2,
        tokenizer=TokenizerSettings(vocab_size=32768, min_frequency=1),
    )
    path = root / "recipe.yaml"
    path.write_text(dump_data_recipe(recipe))
    manifest = prepare_data(path, root / "prepared", repository)
    if (
        not isinstance(manifest.metadata, DataInfo)
        or manifest.metadata.representation != "token_ids"
    ):
        raise AssertionError("Expected prepared fixture")
    return root / "prepared/manifest.json"


def acceptance_recipe(path: Path, repository: Path, steps: int = 200) -> PretrainingJob:
    base = demo_recipe(path, repository, steps=steps, accumulation=1)
    model = load_config(repository / "configs/model/tiny.yaml").model
    if model.vocab != base.job.experiment.model.vocab:
        raise ValueError("Acceptance must use the full RFC tiny vocabulary")
    return replace(
        base,
        job=replace(
            base.job,
            experiment=replace(base.job.experiment, model=model),
            training=replace(
                base.job.training, checkpoint_every=25, fused_optimizer=True
            ),
        ),
        schedule=TokenSchedule(0, 10**9, 1),
        validation_batches=1,
        validation_every=50,
        domain_validation_batches=1,
        sample_every=50,
        sample_prompts=("Compute 2 + 3.", "def add(x):"),
        sample_tokens=4,
        layer_stats_every=50,
        keep_checkpoints=2,
    )


def kill_resume_check(
    recipe: PretrainingJob, directory: Path, repository: Path, *, kill_after: int
) -> dict[str, Any]:
    """Actual SIGKILL then auto-resume; compare every loss and all committed state."""
    if (
        recipe.job.training.strategy != "single"
        or recipe.job.experiment.runtime.device != "cpu"
        or recipe.workers
    ):
        raise ValueError("Process acceptance is single CPU with no child data workers")
    if not 1 <= kill_after < recipe.job.training.steps - 1:
        raise ValueError("kill_after must precede the fixed final horizon")
    root = create_directory(directory)
    recipe_path = root / "pretraining.yaml"
    recipe_path.write_text(dump_pretraining_job(recipe))
    reference = train_pretraining(recipe, root / "continuous", repository)
    command = [
        sys.executable,
        "-m",
        "qvelt",
        "pretrain",
        str(recipe_path),
        "--directory",
        str(root / "replayed"),
        "--repository",
        str(repository),
        "--resume",
        "auto",
    ]
    observed = 0
    with (root / "killed-process.log").open("w") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, cwd=repository
        )
        try:
            deadline = time.monotonic() + 900
            while process.poll() is None:
                try:
                    records, _ = merged_metrics(root / "replayed")
                    observed = max(
                        (s for s, r in records.items() if "loss" in r), default=0
                    )
                except (OSError, json.JSONDecodeError):
                    pass
                if observed >= kill_after:
                    process.kill()
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Training did not reach the kill boundary")
                time.sleep(0.005)
            returncode = process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
    if returncode != -signal.SIGKILL:
        raise AssertionError(f"Expected actual SIGKILL, got process exit {returncode}")
    committed = ArtifactStore(latest_checkpoint(root / "replayed")).read_manifest()
    assert isinstance(committed.metadata, CheckpointInfo)
    with (root / "resumed-process.log").open("w") as stream:
        subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=repository,
            check=True,
            timeout=900,
        )
    left, _ = merged_metrics(root / "continuous")
    right, _ = merged_metrics(root / "replayed")
    steps = recipe.job.training.steps
    curve = [
        {"step": step, "continuous": left[step]["loss"], "resumed": right[step]["loss"]}
        for step in range(1, steps + 1)
    ]
    maximum = max(abs(row["continuous"] - row["resumed"]) for row in curve)
    assert maximum <= 1e-6, f"Loss replay difference {maximum}"
    a, b = (latest_checkpoint(root / name) for name in ("continuous", "replayed"))
    weights = load_file(str(a / "model.safetensors"))
    parameters = sum(value.numel() for value in weights.values())
    _equal(weights, load_file(str(b / "model.safetensors")))
    del weights
    for filename in ("optimizer.pt", "rng-rank-00000.pt"):
        _equal(
            torch.load(a / filename, weights_only=True, mmap=True),
            torch.load(b / filename, weights_only=True, mmap=True),
        )
    _equal(
        json.loads((a / "application-rank-00000.json").read_text()),
        json.loads((b / "application-rank-00000.json").read_text()),
    )
    result = {
        "scope": "Original-data CPU process recovery, not production/model quality",
        "source_sha256": source_digest(),
        "steps": steps,
        "parameters": parameters,
        "requested_kill_after": kill_after,
        "observed_kill_step": observed,
        "killed_returncode": returncode,
        "resumed_checkpoint_step": committed.metadata.optimizer_step,
        "maximum_loss_difference": maximum,
        "weights_bit_exact": True,
        "optimizer_bit_exact": True,
        "rng_bit_exact": True,
        "application_state_exact": True,
        "processed_tokens": reference["processed_tokens"],
        "loss_curve": curve,
        "first_validation": reference["initial_validation"],
        "final_validation": reference["final_validation"],
    }
    (root / "acceptance-result.json").write_text(
        json.dumps(result, sort_keys=True) + "\n"
    )
    return result


def scaling_sanity(
    path: Path, directory: Path, repository: Path, *, steps: int = 20
) -> dict[str, Any]:
    """Fit log(loss)=a+b*log(parameters) on four tiny diagnostics; no universal law."""
    root = create_directory(directory)
    rows = []
    for hidden in (16, 32, 48, 64):
        base = demo_recipe(path, repository, steps=steps, accumulation=1)
        model = replace(
            base.job.experiment.model,
            layers=2,
            hidden=hidden,
            heads=4,
            kv_heads=2,
            ffn=hidden * 3,
        )
        recipe = replace(
            base,
            schedule=TokenSchedule(0, 10**9, 1),
            validation_batches=2,
            keep_checkpoints=1,
            job=replace(
                base.job,
                experiment=replace(base.job.experiment, model=model),
                training=replace(base.job.training, learning_rate=0.001),
            ),
        )
        result = train_pretraining(recipe, root / f"hidden-{hidden}", repository)
        rows.append(
            {
                "hidden": hidden,
                "parameters": model.parameter_count,
                "tokens": result["processed_tokens"],
                "loss": result["final_validation"]["validation_probe_loss"],
            }
        )
    x = np.log(np.array([row["parameters"] for row in rows], dtype=np.float64))
    y = np.log(np.array([row["loss"] for row in rows], dtype=np.float64))
    slope, intercept = np.linalg.lstsq(
        np.stack([x, np.ones_like(x)], axis=1), y, rcond=None
    )[0]
    predicted = slope * x + intercept
    residual = float(np.square(y - predicted).sum())
    total = float(np.square(y - y.mean()).sum())
    output = {
        "scope": "Four short original-fixture runs; descriptive log-log fit, not a scaling-law prediction",
        "steps": steps,
        "learning_rate": 0.001,
        "rows": rows,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1 - residual / total if total else None,
        "red_flags": ["heldout loss increases with size in this short diagnostic"]
        if slope >= 0
        else [],
    }
    (root / "scaling-result.json").write_text(json.dumps(output, sort_keys=True) + "\n")
    return output


def compile_benchmark(
    path: Path, directory: Path, repository: Path, *, steps: int = 8
) -> dict[str, Any]:
    """Real optimizer runs with/without block compilation; isolate cold compile time."""
    if type(steps) is not int or steps < 3:
        raise ValueError("Compile benchmark requires at least three steps")
    root = create_directory(directory)
    rows = []
    for compiled in (False, True):
        recipe = demo_recipe(path, repository, steps=steps, accumulation=1)
        recipe = replace(
            recipe,
            schedule=TokenSchedule(0, 10**9, 1),
            validation_batches=1,
            keep_checkpoints=1,
            job=replace(
                recipe.job, training=replace(recipe.job.training, compile=compiled)
            ),
        )
        began = time.perf_counter()
        result = train_pretraining(
            recipe, root / ("compiled" if compiled else "eager"), repository
        )
        records, _ = merged_metrics(root / ("compiled" if compiled else "eager"))
        speeds = [
            r["tokens_per_second_per_device"]
            for step, r in sorted(records.items())
            if step > 2 and "tokens_per_second_per_device" in r
        ]
        rows.append(
            {
                "compiled": compiled,
                "wall_seconds_including_initial_compile_and_checkpoints": time.perf_counter()
                - began,
                "median_steady_tokens_per_second_per_device": float(np.median(speeds)),
                "estimated_mfu": None,
                "processed_tokens": result["processed_tokens"],
                "final_loss": result["final_validation"]["validation_probe_loss"],
            }
        )
    a, b = (latest_checkpoint(root / name) for name in ("eager", "compiled"))
    left, right = (
        load_file(str(a / "model.safetensors")),
        load_file(str(b / "model.safetensors")),
    )
    maximum = max(float((left[name] - right[name]).abs().max()) for name in left)
    torch.testing.assert_close(left, right, rtol=1e-3, atol=2e-5)
    output = {
        "scope": "Small CPU decoder optimizer benchmark; no GPU speedup/MFU claim",
        "steps": steps,
        "maximum_weight_difference": maximum,
        "rows": rows,
        "source_sha256": source_digest(),
    }
    (root / "compile-result.json").write_text(json.dumps(output, sort_keys=True) + "\n")
    return output
