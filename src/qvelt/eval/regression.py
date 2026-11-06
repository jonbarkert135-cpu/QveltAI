"""Pinned evaluation after each committed CPU checkpoint, with RNG-safe replay/plots."""

from __future__ import annotations

import io
import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qvelt.config import parse_yaml_mapping
from qvelt.config.training import TrainingJob
from qvelt.eval.config import ArtifactRef, EvalJob, eval_digest, load_eval, parse_eval
from qvelt.eval.corpus import load_cases
from qvelt.eval.runner import _json, _payload, run_evaluation
from qvelt.utils.artifacts import ArtifactStore, hash_file
from qvelt.utils.manifest_types import CheckpointInfo, Manifest
from qvelt.utils.manifests import manifest_digest


def regression_policy(job: TrainingJob, tokenizer: Manifest | None = None) -> EvalJob:
    path = job.training.regression_recipe
    if path is None or job.training.regression_sha256 is None:
        raise ValueError("No pinned checkpoint-regression recipe")
    if hash_file(Path(path)).sha256 != job.training.regression_sha256:
        raise ValueError("Checkpoint-regression recipe bytes changed")
    policy = load_eval(Path(path))
    if (
        policy.protocol != "qvelt-eval-v2"
        or policy.allow_final
        or not policy.approved_for_evaluation
        or policy.max_cases > 32
        or policy.samples != 1
        or policy.max_new_tokens > 32
        or job.training.strategy != "single"
        or job.experiment.runtime.device != "cpu"
        or job.experiment.runtime.precision != "fp32"
        or job.experiment.model.parameter_count > 20_000_000
    ):
        raise ValueError(
            "Regression needs bounded approved v2 CPU FP32 single-rank evaluation without final cases"
        )
    if tokenizer is not None and manifest_digest(tokenizer) != policy.tokenizer.sha256:
        raise ValueError("Regression tokenizer differs from actual training tokenizer")
    _, cases, _ = load_cases(policy)
    if not any(case.kind != "lm" for case in cases):
        raise ValueError("Regression curve requires an accuracy task")
    return policy


def _actual_training_source(directory: Path, checkpoint: Manifest) -> ArtifactRef:
    store = ArtifactStore(directory)
    with store.open_file("application.yaml") as stream:
        recipe = parse_yaml_mapping(stream.read(128 * 1024 + 1).decode())
    source_path: object
    if "data_manifest" not in recipe:
        from qvelt.config.pretraining import load_pretraining_job

        path, expected = recipe.get("sft_recipe"), recipe.get("sft_recipe_sha256")
        if not isinstance(path, str) or hash_file(Path(path)).sha256 != expected:
            raise ValueError("Cannot verify actual regression training-data lineage")
        parent = load_pretraining_job(Path(path))
        source_path = parent.data_manifest
    else:
        source_path = recipe["data_manifest"]
    if not isinstance(source_path, str) or not isinstance(
        checkpoint.metadata, CheckpointInfo
    ):
        raise ValueError("Unsupported training-data lineage for checkpoint regression")
    return ArtifactRef(source_path, checkpoint.metadata.data_sha256)


def _plot(directory: Path) -> None:
    rows = []
    for path in sorted(directory.glob("step-*/manifest.json")):
        store = ArtifactStore(path.parent)
        report = store.read_manifest()
        metrics = json.loads(_payload(store, report, "metrics", 2 * 1024**2))
        cp = json.loads(_payload(store, report, "checkpoint_manifest", 128 * 1024))
        value = metrics["generation"]["pass_at_1_first"]
        metric = "generation.pass_at_1_first"
        if value is None:
            value, metric = metrics["choice"]["accuracy"], "choice.accuracy"
        rows.append(
            {
                "step": cp["metadata"]["optimizer_step"],
                "checkpoint_sha256": manifest_digest(
                    ArtifactStore(path.parent).read_manifest(
                        "checkpoint.manifest.json", verify=False
                    )
                ),
                "report_sha256": manifest_digest(report),
                "metric": metric,
                "value": value,
            }
        )
    if (
        not rows
        or len({r["metric"] for r in rows}) != 1
        or any(r["value"] is None for r in rows)
    ):
        raise ValueError("Cannot plot mixed/empty regression metrics")
    prefix = directory / f"curve-step-{rows[-1]['step']:08d}"
    metadata = prefix.with_suffix(".json")
    image = prefix.with_suffix(".png")
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if (
            previous["points"] != rows
            or hash_file(image).sha256 != previous["plot_sha256"]
        ):
            raise ValueError("Existing regression curve changed")
        return
    if image.exists():
        from qvelt.utils.checkpoints import quarantine

        quarantine(image, "uncommitted regression plot")
    cache = directory / ".matplotlib"
    cache.mkdir(exist_ok=True)
    previous_cache = os.environ.get("MPLCONFIGDIR")
    if previous_cache is None:
        os.environ["MPLCONFIGDIR"] = str(cache)
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(6, 3.5))
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.plot([r["step"] for r in rows], [r["value"] for r in rows], marker="o")
        axes.set(
            xlabel="Optimizer step",
            ylabel=rows[0]["metric"],
            ylim=(-0.02, 1.02),
            title="Fixed checkpoint regression control",
        )
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png")
        ArtifactStore(directory).add_bytes(
            image.name, output.getvalue(), role="regression_plot"
        )
    finally:
        if previous_cache is None:
            os.environ.pop("MPLCONFIGDIR", None)
    ArtifactStore(directory).add_bytes(
        metadata.name,
        _json(
            {
                "protocol": "qvelt-checkpoint-curve-v1",
                "points": rows,
                "plot_sha256": hash_file(image).sha256,
            }
        ),
        role="regression_curve",
    )


def evaluate_checkpoint(
    directory: Path, step: int, job: TrainingJob, repository: Path
) -> dict[str, Any]:
    """Idempotent on exact identities; a failed eval never deletes a valid checkpoint."""
    policy = regression_policy(job)
    cp_path = directory / "checkpoints" / f"step-{step:08d}"
    checkpoint = ArtifactStore(cp_path).read_manifest()
    if (
        not isinstance(checkpoint.metadata, CheckpointInfo)
        or checkpoint.metadata.tokenizer_sha256 != policy.tokenizer.sha256
    ):
        raise ValueError("Regression checkpoint/tokenizer mismatch")
    source = _actual_training_source(directory, checkpoint)
    sources = policy.training_sources
    if source.sha256 not in {r.sha256 for r in sources}:
        sources = (*sources, source)
    policy = replace(
        policy,
        checkpoint=ArtifactRef(
            str(cp_path / "manifest.json"), manifest_digest(checkpoint)
        ),
        training_sources=sources,
    )
    root = directory / "regression"
    root.mkdir(exist_ok=True)
    output = root / f"step-{step:08d}"
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    try:
        if (output / "manifest.json").exists():
            store = ArtifactStore(output)
            report = store.read_manifest()
            saved = parse_eval(
                _payload(store, report, "evaluation_recipe", 128 * 1024).decode()
            )
            if eval_digest(saved) != eval_digest(policy):
                raise ValueError(
                    "Regression checkpoint/report/recipe identity mismatch"
                )
            result = {
                "report_sha256": manifest_digest(report),
                "metrics": json.loads(_payload(store, report, "metrics", 2 * 1024**2)),
            }
        else:
            if output.exists():
                from qvelt.utils.checkpoints import quarantine

                quarantine(output, "incomplete checkpoint evaluation")
            result = run_evaluation(policy, output, repository)
        _plot(root)
        return result
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
