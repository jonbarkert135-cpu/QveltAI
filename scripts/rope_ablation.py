"""Exploratory paired RoPE scale1/2 ablation using frozen M7.2 training inputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from qvelt.config import MAX_CONFIG_BYTES, config_digest, parse_config
from qvelt.config.pretraining import (
    PretrainingJob,
    dump_pretraining_job,
    parse_pretraining_job,
)
from qvelt.eval.config import ArtifactRef, dump_eval, parse_eval
from qvelt.eval.runner import _json, _payload, run_evaluation
from qvelt.eval.study import (
    checkpoint_ledger,
    comparison_status,
    necessary_gates,
    paired_summary,
    recipe_bytes,
    validate_lineage,
    verifier_controls,
)
from qvelt.eval.study_config import StudyEntry, load_study
from qvelt.eval.study_smoke import manifest_ref
from qvelt.model.decoder import Decoder
from qvelt.posttrain.grpo import policy_digest
from qvelt.train.pretraining import train_pretraining
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.checkpoints import latest_checkpoint, source_digest
from qvelt.utils.manifest_types import ArtifactFile, Manifest, RunInfo
from qvelt.utils.manifests import manifest_digest
from qvelt.utils.provenance import capture_code, capture_environment


def variant_recipe(
    original: PretrainingJob, parent: ArtifactRef | None = None
) -> PretrainingJob:
    if (
        original.job.experiment.model.rope_scale != 1
        or original.job.experiment.runtime.device != "cpu"
        or original.job.experiment.model.parameter_count > 1_000_000
    ):
        raise ValueError(
            "RoPE ablation needs original CPU scale1 model below1M parameters"
        )
    if (
        original.objective == "sft"
        and parent is None
        or original.objective == "pretraining"
        and (parent is not None or original.initial_checkpoint is not None)
    ):
        raise ValueError("RoPE variant needs its derived SFT parent and a fresh base")
    candidate = replace(
        original,
        job=replace(
            original.job,
            experiment=replace(
                original.job.experiment,
                model=replace(original.job.experiment.model, rope_scale=2),
            ),
        ),
    )
    if parent is not None:
        candidate = replace(
            candidate, initial_checkpoint=parent.path, initial_sha256=parent.sha256
        )
    before, after = asdict(original), asdict(candidate)
    assert before["job"]["experiment"]["model"].pop("rope_scale") == 1
    assert after["job"]["experiment"]["model"].pop("rope_scale") == 2
    if parent is not None:
        for key in ("initial_checkpoint", "initial_sha256"):
            before.pop(key)
            after.pop(key)
    if before != after:
        raise ValueError("RoPE ablation changes more than its declared factor")
    return candidate


def initial_weights(original: PretrainingJob, variant: PretrainingJob) -> str:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(original.job.experiment.runtime.seed)
        before = Decoder(original.job.experiment.model)
        torch.manual_seed(variant.job.experiment.runtime.seed)
        after = Decoder(variant.job.experiment.model)
        if any(
            not torch.equal(value, after.state_dict()[key])
            for key, value in before.state_dict().items()
        ):
            raise ValueError("RoPE arms do not share bit-exact initial parameters")
        return policy_digest(before)


def original_recipe(entry: StudyEntry) -> PretrainingJob:
    cp_store = ArtifactStore(Path(entry.checkpoint.path).parent)
    cp = cp_store.read_manifest(expected_sha256=entry.checkpoint.sha256)
    run_store = ArtifactStore(Path(entry.checkpoint.path).parent.parents[1])
    from qvelt.utils.manifest_types import CheckpointInfo

    if not isinstance(cp.metadata, CheckpointInfo):
        raise ValueError("Architecture input is not a checkpoint")
    run = run_store.read_manifest(expected_sha256=cp.metadata.run_sha256)
    return parse_pretraining_job(
        _payload(run_store, run, "application_recipe", MAX_CONFIG_BYTES).decode()
    )


def run_ablation(study_path: Path, directory: Path, repository: Path) -> dict[str, Any]:
    study = load_study(study_path)
    protocol = parse_eval(recipe_bytes(study.evaluation_recipe).decode())
    if protocol.allow_final or protocol.sampling != "greedy":
        raise ValueError("Architecture ablation requires greedy nonfinal evaluation")
    ledgers = [checkpoint_ledger(e, protocol.tokenizer.sha256) for e in study.entries]
    validate_lineage(ledgers)
    if any(r["training_source_sha256"] != source_digest() for r in ledgers):
        raise ValueError(
            "Architecture arms require unchanged model/trainer/evaluator source"
        )
    root = create_directory(directory)
    rows = []
    control = verifier_controls()
    for seed in sorted({e.seed for e in study.entries}):
        baseline = {e.role: e for e in study.entries if e.seed == seed}
        original_base = original_recipe(baseline["base"])
        original_sft = original_recipe(baseline["sft"])
        base = variant_recipe(original_base)
        initial = initial_weights(original_base, base)
        (root / f"base-{seed}.yaml").write_text(dump_pretraining_job(base))
        train_pretraining(base, root / f"base-{seed}", repository)
        base_cp = manifest_ref(
            latest_checkpoint(root / f"base-{seed}") / "manifest.json"
        )
        sft = variant_recipe(original_sft, base_cp)
        (root / f"sft-{seed}.yaml").write_text(dump_pretraining_job(sft))
        train_pretraining(sft, root / f"sft-{seed}", repository)
        sft_cp = manifest_ref(latest_checkpoint(root / f"sft-{seed}") / "manifest.json")
        changed = [
            checkpoint_ledger(StudyEntry(role, seed, ref), protocol.tokenizer.sha256)
            for role, ref in (("base", base_cp), ("sft", sft_cp))
        ]
        for ledger in changed:
            old = next(
                r for r in ledgers if r["seed"] == seed and r["role"] == ledger["role"]
            )
            if any(
                old[k] != ledger[k]
                for k in (
                    "training_source_sha256",
                    "data_sha256",
                    "training",
                    "optimizer_boundary",
                    "processed_targets_or_actions",
                )
            ):
                raise ValueError("RoPE paired training budget/source/data mismatch")
        predictions, metrics, measurements, reports = {}, {}, {}, {}
        for name, checkpoint in (
            ("control", baseline["sft"].checkpoint),
            ("variant", sft_cp),
        ):
            job = replace(protocol, checkpoint=checkpoint)
            recipe = root / f"{name}-{seed}.yaml"
            recipe.write_text(dump_eval(job))
            output = root / f"{name}-{seed}"
            reports[name] = run_evaluation(job, output, repository)
            predictions[name] = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text().splitlines()
            ]
            metrics[name] = reports[name]["metrics"]
            path = root / f"{name}-{seed}-resources.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qvelt.eval.resources",
                    "--recipe",
                    str(recipe),
                    "--output",
                    str(path),
                ],
                check=True,
                timeout=120,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            measurements[name] = json.loads(path.read_text())
            if (
                measurements[name]["checkpoint_sha256"] != checkpoint.sha256
                or measurements[name]["source_sha256"] != source_digest()
            ):
                raise ValueError("Architecture resource identity mismatch")
        paired = paired_summary(predictions["control"], predictions["variant"])
        gates = necessary_gates(
            paired,
            metrics["control"],
            metrics["variant"],
            measurements["control"],
            measurements["variant"],
            control,
        )
        rows.append(
            {
                "seed": seed,
                "initial_weights_bit_exact": True,
                "initial_weights_sha256": initial,
                "paired": paired,
                "gates": gates,
                "training": changed,
                "reports": reports,
                "resources": measurements,
            }
        )
    checkout, code = capture_code(repository)
    result = {
        "scope": "exploratory short-context RoPE interpolation ablation;not long-context extension or production promotion",
        "source_sha256": source_digest(),
        "script_sha256": hash_file(Path(__file__)).sha256,
        "code": asdict(code),
        "factor": {"model.rope_scale": [1.0, 2.0]},
        "seeds": rows,
        "status": comparison_status(rows),
        "automatic_promotion": False,
        "final_cases_scored": 0,
        "seed_scope": "Runtime seed controls both initialization and sampler ordering;within each paired arm both agree.",
    }
    store = ArtifactStore(root)
    first_seed = rows[0]["seed"]
    config_file = root / f"control-{first_seed}/config.yaml"
    config = parse_config(config_file.read_text())
    store.add_file("config.yaml", config_file, role="config")
    store.add_file("uv.lock", checkout / "uv.lock", role="dependency_lock")
    store.add_bytes("result.json", _json(result), role="ablation_result")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            info = hash_file(path)
            files.append(
                ArtifactFile(
                    relative,
                    {"config.yaml": "config", "uv.lock": "dependency_lock"}.get(
                        relative, "ablation_artifact"
                    ),
                    info.size_bytes,
                    info.sha256,
                )
            )
    run_id = uuid.uuid4().hex
    manifest = Manifest(
        kind="run",
        owner=run_id,
        created_at=datetime.now(UTC).isoformat(),
        metadata=RunInfo(
            run_id,
            config_digest(config),
            code,
            capture_environment(),
            config.runtime.seed,
            1,
        ),
        files=tuple(files),
    )
    store.publish(manifest)
    return {"ablation_sha256": manifest_digest(manifest), "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_ablation(args.study, args.directory, args.repository)
    print(
        json.dumps(
            {
                "ablation_sha256": result["ablation_sha256"],
                "source_sha256": result["result"]["source_sha256"],
                "script_sha256": result["result"]["script_sha256"],
                "status": result["result"]["status"],
                "seeds": [
                    {"seed": r["seed"], "paired": r["paired"], "gates": r["gates"]}
                    for r in result["result"]["seeds"]
                ],
                "automatic_promotion": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
