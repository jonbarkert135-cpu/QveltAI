"""Fixed SFT->DPO->GRPO arithmetic mechanism control, with honest reward evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qvelt.config.pretraining import load_pretraining_job
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.eval.config import ArtifactRef
from qvelt.inference.artifacts import load_checkpoint_model
from qvelt.posttrain.control import arithmetic_control
from qvelt.posttrain.dpo import DPOJob, dump_dpo, train_dpo
from qvelt.posttrain.grpo import generate_rollout
from qvelt.posttrain.grpo_training import GRPOJob, dump_grpo, train_grpo
from qvelt.posttrain.preferences import prepare_preferences
from qvelt.posttrain.reasoning import ReasoningJob, build_reasoning_data, dump_reasoning
from qvelt.posttrain.smoke import run_sft_smoke
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.train.smoke import _equal
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.checkpoints import latest_checkpoint, source_digest
from qvelt.utils.manifests import manifest_digest


def run_phase6_smoke(directory: Path, repository: Path) -> dict[str, Any]:
    root = create_directory(directory)
    sft_result = run_sft_smoke(
        root / "sft", repository, steps=20, environment="arithmetic-control-v1"
    )
    sft = load_pretraining_job(root / "sft/sft.yaml")
    sft_cp = latest_checkpoint(root / "sft/continuous")
    sft_manifest = ArtifactStore(sft_cp).read_manifest()
    token_path = root / "sft/data/tokenizer/manifest.json"
    tokenizer_manifest = ArtifactStore(token_path.parent).read_manifest()
    tokenizer_sha = manifest_digest(tokenizer_manifest)
    tokenizer, _ = load_tokenizer_bundle(token_path, tokenizer_sha)
    report = json.loads((root / "sft/data/report.json").read_text())
    training_families = set(report["counts"]["train"]["families"])
    validation_families = set(report["counts"]["validation"]["families"])
    problems = arithmetic_control()
    pairs = [
        RawDocument(
            json.dumps(
                {
                    "schema_version": 1,
                    "prompt": p.question,
                    "chosen": p.answer,
                    "rejected": str(1 - Fraction(p.answer)),
                }
            ),
            p.identifier,
            p.family,
            p.language,
            p.difficulty,
            p.answer,
        )
        for p in problems
        if p.family in training_families
    ]
    source = write_raw_bundle(
        pairs,
        root / "pairs-source",
        authored_source("phase6-control-pairs"),
        PERMISSION,
    )
    data = prepare_preferences(
        root / "pairs-source/manifest.json",
        manifest_digest(source),
        token_path,
        tokenizer_sha,
        root / "pairs",
        context=128,
        approved_for_training=True,
    )
    dpo = DPOJob(
        replace(
            sft.job,
            training=replace(
                sft.job.training,
                steps=4,
                learning_rate=0.001,
                weight_decay=0,
                checkpoint_every=2,
            ),
        ),
        str(root / "pairs/manifest.json"),
        manifest_digest(data),
        str(sft_cp / "manifest.json"),
        manifest_digest(sft_manifest),
    )
    (root / "dpo.yaml").write_text(dump_dpo(dpo))
    dpo_result = train_dpo(dpo, root / "dpo", repository)
    dpo_cp = latest_checkpoint(root / "dpo")
    dpo_manifest = ArtifactStore(dpo_cp).read_manifest()
    grpo = GRPOJob(
        str(root / "sft/sft.yaml"),
        hash_file(root / "sft/sft.yaml").sha256,
        str(sft_cp / "manifest.json"),
        manifest_digest(sft_manifest),
        steps=60,
        learning_rate=0.001,
        groups=8,
        prompts=4,
        max_tokens=4,
        checkpoint_every=20,
        seed=20261106,
        policy_manifest=str(dpo_cp / "manifest.json"),
        policy_sha256=manifest_digest(dpo_manifest),
        environment="arithmetic-control-v1",
    )
    (root / "grpo.yaml").write_text(dump_grpo(grpo))
    validation = [p for p in problems if p.family in validation_families]
    if not validation or training_families & validation_families:
        raise ValueError("Control needs separate validation strata")

    def measure(path: Path) -> dict[str, Any]:
        manifest = ArtifactStore(path).read_manifest()
        model, _, _ = load_checkpoint_model(
            ArtifactRef(str(path / "manifest.json"), manifest_digest(manifest)),
            tokenizer_sha,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261107)
            rows = [
                generate_rollout(model, validation[i % len(validation)], tokenizer, 4)
                for i in range(64)
            ]
        return {
            "correct": sum(r.correctness for r in rows),
            "samples": 64,
            "reward": sum(r.reward for r in rows) / 64,
            "complete": sum(r.stop == "end_turn" for r in rows),
            "audit": [
                {
                    "question": validation[i % len(validation)].question,
                    "gold": validation[i % len(validation)].answer,
                    "problem_id": r.prompt_id,
                    "text": r.text,
                    "reward": r.reward,
                    "seed": r.seed,
                }
                for i, r in enumerate(rows)
            ],
        }

    before = measure(dpo_cp)
    result = train_grpo(grpo, root / "rl", repository)
    after_cp = latest_checkpoint(root / "rl")
    after = measure(after_cp)
    reasoning_source = write_raw_bundle(
        [
            RawDocument(
                p.question, p.identifier, p.family, p.language, p.difficulty, p.answer
            )
            for p in problems
            if p.family in training_families
        ],
        root / "reasoning-problems",
        authored_source("phase6-control-rejection-problems"),
        PERMISSION,
    )
    after_manifest = ArtifactStore(after_cp).read_manifest()
    reasoning = ReasoningJob(
        ArtifactRef(str(after_cp / "manifest.json"), manifest_digest(after_manifest)),
        ArtifactRef(str(token_path), tokenizer_sha),
        ArtifactRef(
            str(root / "reasoning-problems/manifest.json"),
            manifest_digest(reasoning_source),
        ),
        approved_for_training=True,
        count=64,
        samples=4,
        max_tokens=4,
    )
    (root / "reasoning.yaml").write_text(dump_reasoning(reasoning))
    build_reasoning_data(reasoning, root / "reasoning")
    rejection_report = json.loads((root / "reasoning/report.json").read_text())
    offline_path = next((root / "rl").glob("rollouts-*.jsonl"))
    offline = replace(
        grpo,
        mode="offline",
        rollout_file=str(offline_path),
        rollout_sha256=hash_file(offline_path).sha256,
    )
    (root / "offline.yaml").write_text(dump_grpo(offline))
    replay = train_grpo(offline, root / "offline", repository)
    offline_cp = latest_checkpoint(root / "offline")
    _equal(
        load_file(str(after_cp / "model.safetensors")),
        load_file(str(offline_cp / "model.safetensors")),
    )
    for filename in ("optimizer.pt", "rng-rank-00000.pt"):
        _equal(
            torch.load(after_cp / filename, weights_only=True),
            torch.load(offline_cp / filename, weights_only=True),
        )
    if result["losses_this_attempt"] != replay["losses_this_attempt"]:
        raise AssertionError("Online/offline loss curves differ")
    records = [
        json.loads(line)
        for p in (root / "rl").glob("metrics-*.jsonl")
        for line in p.read_text().splitlines()
    ]
    rewards = [r["reward"] for r in records]
    summary = {
        "schema_version": 1,
        "protocol": "qvelt-phase6-control-v1",
        "source_sha256": source_digest(),
        "scope": "easy two-answer arithmetic mechanism control; not general reasoning gains",
        "sft_steps": 20,
        "dpo_steps": 4,
        "rl_steps": 60,
        "seed": 20261106,
        "sft_train_initial": sft_result["continuous"]["initial_training"],
        "sft_train_final": sft_result["continuous"]["final_training"],
        "dpo": dpo_result,
        "rl": result,
        "validation_before": before,
        "validation_after": after,
        "training_reward_first10": sum(rewards[:10]) / 10,
        "training_reward_last10": sum(rewards[-10:]) / 10,
        "all_training_rewards": rewards,
        "reward_increased": after["reward"] > before["reward"],
        "online_offline_weights_adam_rng_losses_exact": True,
        "final800_scored": 0,
        "rejection_sampling": rejection_report,
    }
    (root / "result.json").write_text(
        json.dumps(summary, sort_keys=True, allow_nan=False) + "\n"
    )
    return summary
