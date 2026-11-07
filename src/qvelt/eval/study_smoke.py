"""Prospective shared-data multi-initialization study,with one-factor DPO LR arm."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from qvelt.config.pretraining import TokenSchedule, dump_pretraining_job
from qvelt.data.demo import PERMISSION, authored_source, create_demo
from qvelt.data.records import iter_records, write_raw_bundle
from qvelt.eval.config import ArtifactRef, EvalJob, dump_eval
from qvelt.eval.corpus import EvalCase
from qvelt.eval.fixtures import freeze_holdout
from qvelt.eval.freeze_identity import verify_frozen_identity
from qvelt.eval.smoke import case_document, control_cases
from qvelt.eval.study import run_study
from qvelt.eval.study_config import StudyEntry, StudyJob, dump_study
from qvelt.posttrain.data import prepare_sft
from qvelt.posttrain.dpo import DPOJob, dump_dpo, train_dpo
from qvelt.posttrain.grpo_training import GRPOJob, dump_grpo, train_grpo
from qvelt.posttrain.preference_smoke import math_preferences
from qvelt.posttrain.preferences import prepare_preferences
from qvelt.posttrain.smoke import math_supervision
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.checkpoints import latest_checkpoint
from qvelt.utils.manifests import manifest_digest


def manifest_ref(path: Path) -> ArtifactRef:
    manifest = ArtifactStore(path.parent).read_manifest(path.name)
    return ArtifactRef(str(path.resolve()), manifest_digest(manifest))


def study_cases(primary: list[EvalCase]) -> list[EvalCase]:
    if len(primary) != 200 or any(
        c.kind != "math" or c.partition != "validation" or c.domain != "math"
        for c in primary
    ):
        raise ValueError("Study fixture requires all200 frozen validation math cases")
    variants = []
    for family in sorted({c.family for c in primary}):
        for case in [c for c in primary if c.family == family][:4]:
            for kind, prompt in (
                ("leading", "  " + case.prompt),
                ("trailing", case.prompt + "\n"),
            ):
                variants.append(
                    replace(
                        case,
                        identifier=f"robust-{case.identifier}-{kind}",
                        partition="control",
                        domain="robustness",
                        prompt=prompt,
                    )
                )
    if len(variants) != 40:
        raise ValueError("Study robustness fixture needs five full families")
    return [*primary, *variants, *control_cases()]


def prepare_study(
    directory: Path,
    repository: Path,
    *,
    seeds: tuple[int, ...] = (20261105, 20261106, 20261107),
) -> StudyJob:
    if (
        type(seeds) is not tuple
        or not 1 <= len(seeds) <= 3
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Study accepts1..3 distinct initialization seeds")
    root = create_directory(directory)
    frozen = freeze_holdout(root / "heldout", repository)
    registered = json.loads(
        (repository / "docs/development/evidence/evaluation-frozen-v1.json").read_text()
    )
    verify_frozen_identity(frozen, registered, root / "heldout")
    validation = frozen["partitions"]["validation"]
    primary = []
    for doc in iter_records(
        root / "heldout" / validation["manifest"], validation["manifest_sha256"]
    ):
        row = json.loads(doc.text)
        row.pop("schema_version")
        row["choices"] = tuple(row["choices"])
        primary.append(EvalCase(**row))
    suite = write_raw_bundle(
        [case_document(c) for c in study_cases(primary)],
        root / "suite",
        authored_source("promotion-validation-suite-v1"),
        PERMISSION,
        allowed_uses=("evaluation", "decontamination"),
    )
    (root / "prospective-plan.json").write_text(
        json.dumps(
            {
                "protocol": "qvelt-promotion-v1",
                "initialization_seeds": seeds,
                "suite_sha256": manifest_digest(suite),
                "base_updates": 3,
                "sft_updates": 80,
                "dpo_updates": 40,
                "dpo_learning_rates": [0.001, 0.0005],
                "grpo_windows": 8,
                "evaluation_max_tokens": 16,
                "final_cases_scored": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    create_demo(root / "base-data", repository, documents=32)
    tokenizer = manifest_ref(root / "base-data/prepared/tokenizer/manifest.json")
    write_raw_bundle(
        math_supervision(),
        root / "supervision",
        authored_source("sft-exact-math"),
        PERMISSION,
    )
    prepare_sft(
        root / "supervision/manifest.json",
        manifest_ref(root / "supervision/manifest.json").sha256,
        Path(tokenizer.path),
        tokenizer.sha256,
        root / "sft-data",
        context=128,
        approved_for_training=True,
    )
    preferences = write_raw_bundle(
        math_preferences(),
        root / "preference-source",
        authored_source("verifier-math-preferences"),
        PERMISSION,
    )
    pair_data = prepare_preferences(
        root / "preference-source/manifest.json",
        manifest_digest(preferences),
        Path(tokenizer.path),
        tokenizer.sha256,
        root / "pairs",
        context=128,
        approved_for_training=True,
    )
    entries = []
    for seed in seeds:
        run = create_directory(root / f"seed-{seed}")
        base = demo_recipe(
            root / "base-data/prepared/manifest.json", repository, steps=3
        )
        base = replace(
            base,
            job=replace(
                base.job,
                experiment=replace(
                    base.job.experiment,
                    model=replace(base.job.experiment.model, context=128),
                    runtime=replace(base.job.experiment.runtime, seed=seed),
                ),
            ),
        )
        train_pretraining(base, run / "base", repository)
        base_cp = manifest_ref(latest_checkpoint(run / "base") / "manifest.json")
        sft = demo_recipe(
            root / "sft-data/manifest.json", repository, steps=80, accumulation=1
        )
        sft = replace(
            sft,
            objective="sft",
            initial_checkpoint=base_cp.path,
            initial_sha256=base_cp.sha256,
            schedule=TokenSchedule(0, 10**9, 1),
            job=replace(
                sft.job,
                experiment=replace(
                    sft.job.experiment,
                    runtime=replace(sft.job.experiment.runtime, seed=seed),
                ),
                training=replace(sft.job.training, learning_rate=0.003),
            ),
        )
        sft_path = run / "sft.yaml"
        sft_path.write_text(dump_pretraining_job(sft))
        train_pretraining(sft, run / "sft", repository)
        sft_cp = manifest_ref(latest_checkpoint(run / "sft") / "manifest.json")
        for role, learning_rate in (("dpo", 0.001), ("dpo-half-lr", 0.0005)):
            job = replace(
                sft.job,
                training=replace(
                    sft.job.training,
                    steps=40,
                    learning_rate=learning_rate,
                    weight_decay=0,
                    checkpoint_every=20,
                ),
            )
            dpo = DPOJob(
                job,
                str(root / "pairs/manifest.json"),
                manifest_digest(pair_data),
                sft_cp.path,
                sft_cp.sha256,
            )
            (run / (role + ".yaml")).write_text(dump_dpo(dpo))
            train_dpo(dpo, run / role, repository)
            entries.append(
                StudyEntry(
                    role,
                    seed,
                    manifest_ref(latest_checkpoint(run / role) / "manifest.json"),
                )
            )
        grpo = GRPOJob(
            str(sft_path),
            hash_file(sft_path).sha256,
            sft_cp.path,
            sft_cp.sha256,
            steps=8,
            checkpoint_every=4,
            seed=seed,
        )
        (run / "grpo.yaml").write_text(dump_grpo(grpo))
        train_grpo(grpo, run / "grpo", repository)
        entries.extend(
            (
                StudyEntry("base", seed, base_cp),
                StudyEntry("sft", seed, sft_cp),
                StudyEntry(
                    "grpo",
                    seed,
                    manifest_ref(latest_checkpoint(run / "grpo") / "manifest.json"),
                ),
            )
        )
    template = EvalJob(
        entries[0].checkpoint,
        tokenizer,
        manifest_ref(root / "suite/manifest.json"),
        training_sources=tuple(
            manifest_ref(root / p)
            for p in (
                "base-data/en/manifest.json",
                "base-data/ru/manifest.json",
                "supervision/manifest.json",
                "preference-source/manifest.json",
            )
        ),
        approved_for_evaluation=True,
        context_limit=128,
        lm_window=64,
        max_cases=247,
        max_new_tokens=16,
    )
    path = root / "evaluation.yaml"
    path.write_text(dump_eval(template))
    study = StudyJob(
        ArtifactRef(str(path.resolve()), hash_file(path).sha256),
        tuple(sorted(entries, key=lambda e: (e.seed, e.role))),
    )
    (root / "study.yaml").write_text(dump_study(study))
    return study


def run_study_smoke(
    directory: Path,
    repository: Path,
    *,
    seeds: tuple[int, ...] = (20261105, 20261106, 20261107),
) -> dict[str, Any]:
    root = create_directory(directory)
    study = prepare_study(root / "training", repository, seeds=seeds)
    result = run_study(study, root / "analysis", repository)
    assert (
        not result["result"]["automatic_promotion"]
        and result["result"]["final_cases_scored"] == 0
    )
    # Summaries omit protected prompts/answers and machine-specific input paths.
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    return result
