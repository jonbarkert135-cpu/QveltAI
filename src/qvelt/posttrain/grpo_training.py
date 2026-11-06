"""Bounded original-fixture CPU GRPO with atomic rollout-window recovery."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from qvelt.config import (
    MAX_CONFIG_BYTES,
    ConfigError,
    _integer,
    _mapping,
    _number,
    parse_yaml_mapping,
)
from qvelt.config.pretraining import load_pretraining_job
from qvelt.data.math import generate_problems
from qvelt.eval.config import ArtifactRef
from qvelt.model.decoder import Decoder
from qvelt.posttrain.control import arithmetic_control
from qvelt.posttrain.dpo import initialize_sft_reference
from qvelt.posttrain.grpo import (
    generate_rollout,
    group_advantages,
    grpo_loss,
    policy_digest,
    rollout_logps,
)
from qvelt.posttrain.offline import read_rollout_file, replay_window
from qvelt.posttrain.verifier_audit import audit_verifiers
from qvelt.posttrain.verifiers import RewardPolicy
from qvelt.train.optim import make_optimizer
from qvelt.train.parallel import persistent_state_finite
from qvelt.train.pretraining import initialize_weights, prepare_inputs
from qvelt.utils.artifacts import ArtifactStore, hash_file
from qvelt.utils.checkpoints import (
    initialize_training_run,
    load_checkpoint,
    save_checkpoint,
)
from qvelt.utils.distributed import distributed_context
from qvelt.utils.manifest_types import digest
from qvelt.utils.manifests import _reject_constant, _unique_mapping, manifest_digest
from qvelt.utils.metrics import JsonMetricSink
from qvelt.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class GRPOJob:
    sft_recipe: str
    sft_recipe_sha256: str
    reference_manifest: str
    reference_sha256: str
    steps: int = 8
    learning_rate: float = 1e-6
    groups: int = 4
    prompts: int = 4
    max_tokens: int = 16
    clip: float = 0.2
    kl: float = 0.01
    checkpoint_every: int = 2
    seed: int = 20261105
    schema_version: int = 1
    format_weight: float = 0.0
    length_penalty: float = 0.0
    framed: bool = False
    mode: str = "online"
    rollout_file: str | None = None
    rollout_sha256: str | None = None
    policy_manifest: str | None = None
    policy_sha256: str | None = None
    environment: str = "original"

    def __post_init__(self) -> None:
        RewardPolicy(self.framed, self.format_weight, self.length_penalty)
        if self.environment not in ("original", "arithmetic-control-v1"):
            raise ConfigError("Unknown registered GRPO environment")
        if self.mode not in ("online", "offline") or (self.mode == "offline") != (
            self.rollout_file is not None
        ):
            raise ConfigError("Offline GRPO requires a pinned rollout file")
        for key, sha_key in (
            ("rollout_file", "rollout_sha256"),
            ("policy_manifest", "policy_sha256"),
        ):
            value, sha = getattr(self, key), getattr(self, sha_key)
            if (value is None) != (sha is None):
                raise ConfigError("Optional GRPO input requires both path and digest")
            if value is not None:
                ArtifactRef(value, sha)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigError("Only GRPO demo schema1 is supported")
        for key in ("sft_recipe", "reference_manifest"):
            value = getattr(self, key)
            if (
                not isinstance(value, str)
                or not 1 <= len(value) <= 2048
                or any(ord(c) < 32 for c in value)
            ):
                raise ConfigError("Invalid GRPO lineage path")
        for key in ("sft_recipe_sha256", "reference_sha256"):
            digest(getattr(self, key), "grpo." + key)
        for key, minimum, maximum in [
            ("steps", 1, 100),
            ("groups", 2, 8),
            ("prompts", 1, 8),
            ("max_tokens", 1, 64),
            ("checkpoint_every", 1, 100),
            ("seed", 0, 2**32 - 1),
        ]:
            _integer(
                getattr(self, key), "grpo." + key, minimum=minimum, maximum=maximum
            )
        object.__setattr__(
            self, "learning_rate", _number(self.learning_rate, "grpo.learning_rate")
        )
        if (
            type(self.clip) not in (float, int)
            or not 0 < self.clip < 1
            or type(self.kl) not in (float, int)
            or not 0 <= self.kl < 1
        ):
            raise ConfigError("GRPO needs clip in (0,1),KL in [0,1)")


def dump_grpo(recipe: GRPOJob) -> str:
    return yaml.safe_dump(asdict(recipe), sort_keys=True)


def load_grpo(path: Path) -> GRPOJob:
    with path.open("rb") as stream:
        text = stream.read(MAX_CONFIG_BYTES + 1)
    if len(text) > MAX_CONFIG_BYTES:
        raise ConfigError("GRPO recipe exceeds its byte limit")
    root = _mapping(
        parse_yaml_mapping(text.decode("utf-8-sig")),
        "grpo",
        set(GRPOJob.__dataclass_fields__),
    )
    if not {
        "sft_recipe",
        "sft_recipe_sha256",
        "reference_manifest",
        "reference_sha256",
    } <= set(root):
        raise ConfigError("GRPO needs pinned SFT recipe/reference identities")
    return GRPOJob(**cast(dict[str, Any], root))


def grpo_digest(recipe: GRPOJob) -> str:
    values = asdict(recipe)
    for key in ("sft_recipe", "reference_manifest", "rollout_file", "policy_manifest"):
        values.pop(key)
    return hashlib.sha256(
        json.dumps(
            values, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def train_grpo(
    recipe: GRPOJob,
    directory: Path,
    repository: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or type(resume) is not bool:
        raise ValueError("GRPO supports one CPU process and explicit resume only")
    if stop_after is not None and (
        type(stop_after) is not int or not 1 <= stop_after <= recipe.steps
    ):
        raise ValueError("Invalid GRPO stop boundary")
    path = Path(recipe.sft_recipe)
    if hash_file(path).sha256 != recipe.sft_recipe_sha256:
        raise ValueError("SFT recipe bytes disagree with GRPO lineage")
    parent = load_pretraining_job(path)
    if (
        parent.objective != "sft"
        or parent.job.experiment.runtime.device != "cpu"
        or parent.job.experiment.runtime.precision != "fp32"
        or parent.job.experiment.model.parameter_count > 1_000_000
    ):
        raise ValueError(
            "GRPO prototype requires original-demo CPU FP32 SFT below1M parameters"
        )
    inputs = prepare_inputs(parent)
    if parent.lora_rank:
        raise ValueError(
            "GRPO requires a full SFT reference; merge adapters explicitly first"
        )
    offline = (
        read_rollout_file(ArtifactRef(recipe.rollout_file, recipe.rollout_sha256))
        if recipe.mode == "offline" and recipe.rollout_file and recipe.rollout_sha256
        else None
    )
    if offline is not None and len(offline) < recipe.steps:
        raise ValueError("Offline file lacks required rollout windows")
    shaping = (
        RewardPolicy(recipe.framed, recipe.format_weight, recipe.length_penalty)
        if recipe.framed or recipe.format_weight or recipe.length_penalty
        else None
    )
    if inputs.scope != "demo" or inputs.text_tokenizer is None:
        raise ValueError(
            "Only the registered original demo RL environment is supported"
        )
    tokenizer = inputs.text_tokenizer
    store = ArtifactStore(Path(parent.data_manifest).parent)
    with store.open_file("report.json") as stream:
        body = stream.read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ValueError("SFT report exceeds its limit")
    report = json.loads(
        body, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
    )
    families = report["counts"]["train"]["families"]
    environment = (
        arithmetic_control()
        if recipe.environment == "arithmetic-control-v1"
        else generate_problems(32, 20261104, "en")
    )
    problems = [p for p in environment if p.family in families]
    validation = [
        p for p in environment if p.family in report["counts"]["validation"]["families"]
    ]
    if not problems or not validation:
        raise ValueError("RL needs distinct original training/validation families")
    job = replace(
        parent.job,
        experiment=replace(
            parent.job.experiment,
            runtime=replace(parent.job.experiment.runtime, seed=recipe.seed),
        ),
        training=replace(
            parent.job.training,
            steps=recipe.steps,
            batch_size=recipe.prompts * recipe.groups,
            learning_rate=recipe.learning_rate,
            grad_clip=1.0,
            weight_decay=0.0,
            warmup_steps=0,
            decay_steps=0,
            checkpoint_every=recipe.checkpoint_every,
            num_threads=1,
            strategy="single",
            compile=False,
            metrics_backend="json",
        ),
    )
    torch.set_num_threads(1)
    seed_everything(job.experiment.runtime)
    initialization = replace(
        parent,
        job=job,
        initial_checkpoint=recipe.reference_manifest,
        initial_sha256=recipe.reference_sha256,
    )
    identity = grpo_digest(recipe)
    with distributed_context("cpu") as context:
        model = Decoder(job.experiment.model, job.training.execution)
        initialize_sft_reference(
            model, initialization, manifest_digest(inputs.tokenizer)
        )
        reference = Decoder(job.experiment.model, job.training.execution)
        reference.load_state_dict(model.state_dict())
        reference.requires_grad_(False).eval()
        reference_digest = policy_digest(reference)
        if recipe.policy_manifest is not None:
            policy_store = ArtifactStore(Path(recipe.policy_manifest).parent)
            policy_store.read_manifest(
                Path(recipe.policy_manifest).name, expected_sha256=recipe.policy_sha256
            )
            policy_run = ArtifactStore(Path(recipe.policy_manifest).parents[2])
            policy_run.read_manifest()
            with policy_run.open_file("application.yaml") as stream:
                from qvelt.posttrain.dpo import dpo_digest, parse_dpo

                policy_recipe = parse_dpo(stream.read(MAX_CONFIG_BYTES + 1).decode())
            with policy_store.open_file("application-rank-00000.json") as stream:
                policy_state = json.loads(
                    stream.read(MAX_CONFIG_BYTES + 1),
                    object_pairs_hook=_unique_mapping,
                    parse_constant=_reject_constant,
                )
            if (
                policy_recipe.reference_sha256 != recipe.reference_sha256
                or policy_recipe.job.experiment.model != job.experiment.model
                or not isinstance(policy_state, dict)
                or policy_state.get("kind") != "qvelt-dpo-v1"
                or policy_state.get("recipe_sha256") != dpo_digest(policy_recipe)
            ):
                raise ValueError(
                    "GRPO initial DPO policy must descend from its pinned SFT reference"
                )
            initialize_weights(
                model,
                replace(
                    initialization,
                    initial_checkpoint=recipe.policy_manifest,
                    initial_sha256=recipe.policy_sha256,
                ),
                manifest_digest(inputs.tokenizer),
            )
        optimizer = make_optimizer(model, job.training)
        counters = {
            "tokens": 0,
            "updates": 0,
            "policy_updates": 0,
            "skipped": 0,
            "zero_groups": 0,
        }
        last_rollout = ""

        def state(step: int) -> dict[str, Any]:
            return {
                "kind": "qvelt-grpo-v1",
                "recipe_sha256": identity,
                "next_prompt": step * recipe.prompts,
                "counters": dict(counters),
                "last_rollout_sha256": last_rollout,
            }

        def validate(saved: dict[str, Any], step: int, tokens: int) -> None:
            nonlocal last_rollout
            values = saved.get("counters")
            if (
                not isinstance(values, dict)
                or set(values) != set(counters)
                or any(type(v) is not int or v < 0 for v in values.values())
                or values["tokens"] != tokens
                or values["updates"] + values["skipped"] != step
                or not values["policy_updates"] <= values["updates"] <= step
                or values["zero_groups"] > step * recipe.prompts
            ):
                raise ValueError("Invalid GRPO checkpoint counters")
            last = saved.get("last_rollout_sha256")
            if not isinstance(last, str):
                raise ValueError("Invalid rollout digest")
            digest(last, "grpo.last_rollout_sha256")
            counters.update(values)
            last_rollout = last
            if saved != state(step):
                raise ValueError("GRPO prompt/reference/recipe cursor mismatch")

        start = 0
        if resume:
            run = ArtifactStore(directory)
            run.read_manifest()
            with run.open_file("application.yaml") as stream:
                stored = _mapping(
                    parse_yaml_mapping(stream.read(MAX_CONFIG_BYTES + 1).decode()),
                    "grpo",
                    set(GRPOJob.__dataclass_fields__),
                )
            if grpo_digest(GRPOJob(**cast(dict[str, Any], stored))) != identity:
                raise ValueError("GRPO recipe/reference mismatch")
            start = load_checkpoint(
                model,
                optimizer,
                job,
                directory,
                context,
                application_validator=validate,
            )
        else:
            initialize_training_run(
                job,
                directory,
                repository,
                context,
                data=inputs.data,
                tokenizer=inputs.tokenizer,
                application_recipe=dump_grpo(recipe).encode(),
            )

        def evaluate() -> dict[str, float]:
            rows = [
                generate_rollout(
                    model,
                    p,
                    tokenizer,
                    recipe.max_tokens,
                    sample=False,
                    reward_policy=shaping,
                )
                for p in validation
            ]
            return {
                "correct": float(sum(r.correctness for r in rows)),
                "problems": float(len(rows)),
                "complete": float(sum(r.stop == "end_turn" for r in rows)),
            }

        initial = evaluate()
        verifier_controls = audit_verifiers()
        if not verifier_controls["passed"]:
            raise RuntimeError("Held-out verifier controls failed")
        (directory / "verifier-audit.json").write_text(
            json.dumps(verifier_controls, sort_keys=True) + "\n"
        )
        final = initial
        stop = recipe.steps if stop_after is None else stop_after
        if stop < start:
            raise ValueError("GRPO stop precedes restored cursor")
        losses = []
        receipts = []
        step = start
        attempt = uuid.uuid4().hex
        sink = JsonMetricSink(directory / f"metrics-{attempt}.jsonl")
        try:
            with (directory / f"rollouts-{attempt}.jsonl").open("x") as log:
                for index in range(start, stop):
                    fingerprint = policy_digest(model)
                    selected = [
                        problems[(index * recipe.prompts + i) % len(problems)]
                        for i in range(recipe.prompts)
                    ]
                    rows = (
                        replay_window(
                            offline[index],
                            index + 1,
                            selected,
                            recipe.groups,
                            model,
                            tokenizer,
                            reference_digest,
                            recipe.max_tokens,
                            shaping,
                        )
                        if offline is not None
                        else [
                            generate_rollout(
                                model,
                                p,
                                tokenizer,
                                recipe.max_tokens,
                                reward_policy=shaping,
                            )
                            for p in selected
                            for _ in range(recipe.groups)
                        ]
                    )
                    if policy_digest(model) != fingerprint:
                        raise RuntimeError(
                            "Policy mutated while collecting on-policy rollouts"
                        )
                    rewards = torch.tensor(
                        [r.reward for r in rows], dtype=torch.float32
                    ).reshape(recipe.prompts, recipe.groups)
                    advantages = group_advantages(rewards)
                    zero = int((rewards.var(dim=1, unbiased=False) == 0).sum())
                    counters["zero_groups"] += zero
                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    current, mask = rollout_logps(model, rows)
                    with torch.no_grad():
                        ref, _ = rollout_logps(reference, rows)
                    width = current.shape[-1]
                    old = torch.stack(
                        [
                            torch.nn.functional.pad(
                                torch.tensor(r.old_logps, dtype=torch.float32),
                                (0, width - len(r.actions)),
                            )
                            for r in rows
                        ]
                    )
                    shape = (recipe.prompts, recipe.groups, width)
                    loss = grpo_loss(
                        current.reshape(shape),
                        old.reshape(shape),
                        ref.reshape(shape),
                        advantages,
                        mask.reshape(shape),
                        clip=recipe.clip,
                        kl=recipe.kl,
                    )
                    torch.autograd.backward(loss)
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True, foreach=False
                    )
                    active = bool(norm > 0)
                    if active:
                        optimizer.step()
                        counters["updates"] += 1
                        if zero < recipe.prompts:
                            counters["policy_updates"] += 1
                    else:
                        counters["skipped"] += 1
                    if not persistent_state_finite(model, optimizer, context):
                        raise FloatingPointError("Nonfinite GRPO model/Adam state")
                    if policy_digest(reference) != reference_digest or any(
                        p.grad is not None for p in reference.parameters()
                    ):
                        raise RuntimeError("Frozen RL reference changed")
                    counters["tokens"] += int(mask.sum())
                    step = index + 1
                    receipt = {
                        "step": step,
                        "policy_sha256": fingerprint,
                        "reference_sha256": reference_digest,
                        "problems": [asdict(p) for p in selected],
                        "rows": [asdict(r) for r in rows],
                    }
                    serialized = json.dumps(
                        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    last_rollout = hashlib.sha256(serialized.encode()).hexdigest()
                    receipts.append(last_rollout)
                    log.write(serialized + "\n")
                    log.flush()
                    losses.append(float(loss.detach()))
                    delta = ref.detach() - current.detach()
                    sampled_kl = float(
                        (
                            torch.where(mask, delta.expm1() - delta, 0).sum(-1)
                            / mask.sum(-1)
                        ).mean()
                    )
                    sink.log(
                        step,
                        {
                            "loss": losses[-1],
                            "reward": float(rewards.mean()),
                            "sampled_kl": sampled_kl,
                            "correctness": sum(r.correctness for r in rows) / len(rows),
                            "format_reward": sum(r.format_reward for r in rows)
                            / len(rows),
                            "length_cost": sum(r.length_cost for r in rows) / len(rows),
                            "zero_variance_groups": float(zero),
                            "grad_norm": float(norm),
                            "optimizer_updates": float(counters["updates"]),
                            "policy_updates": float(counters["policy_updates"]),
                            "skipped_windows": float(counters["skipped"]),
                            "truncated": float(sum(r.stop == "length" for r in rows)),
                            "complete": float(sum(r.stop == "end_turn" for r in rows)),
                            "processed_targets": float(counters["tokens"]),
                        },
                    )
                    if step % recipe.checkpoint_every == 0 or step == stop:
                        if not audit_verifiers()["passed"]:
                            raise RuntimeError(
                                "Held-out verifier controls failed during training"
                            )
                        final = evaluate()
                        save_checkpoint(
                            model,
                            optimizer,
                            job,
                            directory,
                            step,
                            context,
                            processed_tokens=counters["tokens"],
                            application_state=state(step),
                        )
        finally:
            sink.close()
        return {
            "scope": "Original exact-answer CPU outcome-GRPO prototype,not reasoning improvement",
            "optimizer_step": step,
            "resumed_from": start,
            "counters": counters,
            "losses_this_attempt": losses,
            "rollout_sha256_this_attempt": receipts,
            "initial_validation": initial,
            "final_validation": final,
            "reference_frozen": True,
            "verifier_audit": {
                key: verifier_controls[key] for key in ("protocol", "cases", "passed")
            },
        }
