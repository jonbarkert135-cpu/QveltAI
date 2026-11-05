"""Original sigmoid DPO and same-topology CPU optimization with a frozen SFT reference."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from qvelt.config import (
    MAX_CONFIG_BYTES,
    ConfigError,
    _mapping,
    _number,
    parse_yaml_mapping,
)
from qvelt.config.pretraining import (
    PretrainingJob,
    TokenSchedule,
    parse_pretraining_job,
)
from qvelt.config.training import TrainingJob, parse_training_job
from qvelt.model.decoder import Decoder
from qvelt.posttrain.chat import assistant_mask
from qvelt.posttrain.preferences import EncodedPair, load_preferences
from qvelt.train.optim import learning_rate, make_optimizer
from qvelt.train.parallel import persistent_state_finite
from qvelt.train.pretraining import initialize_weights
from qvelt.utils.artifacts import ArtifactStore
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


def response_logps(logits: Tensor, labels: Tensor, response_mask: Tensor) -> Tensor:
    """SUM log p(response|prompt), including end-turn; never length-normalize DPO."""
    if (
        logits.ndim != 3
        or not logits.is_floating_point()
        or labels.dtype != torch.int64
        or response_mask.dtype != torch.bool
        or logits.shape[:2] != labels.shape
        or labels.shape != response_mask.shape
        or logits.shape[0] == 0
        or logits.shape[1] == 0
        or logits.shape[2] == 0
        or logits.device != labels.device
        or labels.device != response_mask.device
    ):
        raise ValueError(
            "Response logits/labels/mask have incompatible shapes/dtypes/devices"
        )
    if bool(((labels < 0) | (labels >= logits.shape[-1])).any()) or not bool(
        response_mask.any(dim=1).all()
    ):
        raise ValueError(
            "Response targets must be in vocabulary and nonempty per sequence"
        )
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite response logits")
    values = (
        F.log_softmax(
            logits, dim=-1, dtype=torch.promote_types(logits.dtype, torch.float32)
        )
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    return torch.where(response_mask, values, 0).sum(dim=1)


def dpo_loss(policy: Tensor, reference: Tensor, beta: float) -> Tensor:
    """Per-pair softplus(-beta*(policy log-ratio - frozen-reference log-ratio))."""
    beta = _number(beta, "dpo.beta")
    if (
        policy.ndim != 2
        or policy.shape[1] != 2
        or policy.shape[0] == 0
        or policy.shape != reference.shape
        or policy.device != reference.device
        or not policy.is_floating_point()
        or not reference.is_floating_point()
    ):
        raise ValueError("DPO expects nonempty [pairs,chosen/rejected] floating logps")
    if not bool(torch.isfinite(policy).all() & torch.isfinite(reference).all()):
        raise FloatingPointError("Nonfinite DPO log probabilities")
    dtype = torch.promote_types(
        torch.promote_types(policy.dtype, reference.dtype), torch.float32
    )
    rewards = beta * (policy.to(dtype) - reference.detach().to(dtype))
    return F.softplus(rewards[:, 1] - rewards[:, 0])


@dataclass(frozen=True)
class DPOJob:
    job: TrainingJob
    data_manifest: str
    data_sha256: str
    reference_manifest: str
    reference_sha256: str
    beta: float = 0.1
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.job, TrainingJob)
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ConfigError("DPO needs a validated training job and schema1")
        object.__setattr__(self, "beta", _number(self.beta, "dpo.beta"))
        for name in ("data_manifest", "reference_manifest"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not 1 <= len(value) <= 2048
                or any(ord(c) < 32 for c in value)
            ):
                raise ConfigError("Invalid DPO manifest path")
        for name in ("data_sha256", "reference_sha256"):
            digest(getattr(self, name), "dpo." + name)
        t = self.job.training
        runtime = self.job.experiment.runtime
        if (
            t.strategy != "single"
            or runtime.device != "cpu"
            or runtime.precision != "fp32"
            or t.compile
            or t.metrics_backend != "json"
        ):
            raise ConfigError(
                "DPO currently supports single-process CPU FP32/JSON,compile=false"
            )


def dump_dpo(job: DPOJob) -> str:
    return yaml.safe_dump(asdict(job), sort_keys=True, allow_unicode=True)


def parse_dpo(text: str) -> DPOJob:
    root = _mapping(parse_yaml_mapping(text), "dpo", set(DPOJob.__dataclass_fields__))
    required = {
        "job",
        "data_manifest",
        "data_sha256",
        "reference_manifest",
        "reference_sha256",
    }
    if not required <= set(root):
        raise ConfigError("DPO recipe needs job,data and reference manifest identities")
    values = cast(dict[str, Any], dict(root))
    job = parse_training_job(yaml.safe_dump(values.pop("job")))
    return DPOJob(job, **values)


def load_dpo(path: Path) -> DPOJob:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("DPO recipe exceeds byte limit")
    return parse_dpo(data.decode("utf-8-sig"))


def dpo_digest(recipe: DPOJob) -> str:
    values = asdict(recipe)
    for key in ("data_manifest", "reference_manifest"):
        values.pop(key)
    return hashlib.sha256(
        json.dumps(
            values, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def initialize_sft_reference(
    model: Decoder, initialization: PretrainingJob, tokenizer_sha: str
) -> None:
    """Verify SFT application provenance and load finite weights without optimizer state."""
    if initialization.initial_checkpoint is None:
        raise ValueError("SFT reference checkpoint is required")
    reference_path = Path(initialization.initial_checkpoint)
    reference_store = ArtifactStore(reference_path.parent)
    reference_store.read_manifest(
        reference_path.name, expected_sha256=initialization.initial_sha256
    )
    with reference_store.open_file("application-rank-00000.json") as stream:
        body = stream.read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ValueError("SFT reference application state exceeds its limit")
    application = json.loads(
        body, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
    )
    if (
        not isinstance(application, dict)
        or application.get("kind") not in ("packed-sft-v1", "packed-sft-v2")
        or not isinstance(application.get("recipe_yaml"), str)
        or parse_pretraining_job(application["recipe_yaml"]).objective != "sft"
    ):
        raise ValueError("Reference must be a manifested SFT checkpoint")
    initialize_weights(model, initialization, tokenizer_sha)


def pair_logps(model: Decoder, pairs: list[EncodedPair]) -> tuple[Tensor, int]:
    if not pairs:
        raise ValueError("Empty preference batch")
    frames = [frame for pair in pairs for frame in (pair.chosen, pair.rejected)]
    length = max(len(frame) - 1 for frame in frames)
    ids = torch.zeros(
        (len(frames), length), dtype=torch.int64, device=model.embedding.weight.device
    )
    labels = torch.zeros_like(ids)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    mask = torch.zeros_like(valid)
    for index, frame in enumerate(frames):
        n = len(frame) - 1
        ids[index, :n] = torch.tensor(frame[:-1], device=ids.device)
        labels[index, :n] = torch.tensor(frame[1:], device=ids.device)
        valid[index, :n] = True
        mask[index, :n] = torch.tensor(assistant_mask(list(frame)), device=ids.device)
    values = response_logps(model(ids, valid_mask=valid), labels, mask)
    return values.reshape(len(pairs), 2), int(mask.sum())


@torch.no_grad()
def evaluate_pairs(
    model: Decoder, reference: Decoder, pairs: list[EncodedPair], recipe: DPOJob
) -> dict[str, float]:
    if not pairs:
        raise ValueError("Cannot evaluate empty preference split")
    training = model.training
    model.eval()
    totals = {
        "dpo_loss": 0.0,
        "reward_margin": 0.0,
        "relative_accuracy": 0.0,
        "ties": 0.0,
    }
    try:
        batch = recipe.job.training.batch_size
        for index in range(0, len(pairs), batch):
            selected = pairs[index : index + batch]
            policy, _ = pair_logps(model, selected)
            ref, _ = pair_logps(reference, selected)
            reward = recipe.beta * (policy - ref)
            margin = reward[:, 0] - reward[:, 1]
            values = {
                "dpo_loss": dpo_loss(policy, ref, recipe.beta),
                "reward_margin": margin,
                "relative_accuracy": (margin > 0).float(),
                "ties": (margin == 0).float(),
            }
            for key, value in values.items():
                totals[key] += float(value.sum())
    finally:
        model.train(training)
    return {key: value / len(pairs) for key, value in totals.items()}


def train_dpo(
    recipe: DPOJob,
    directory: Path,
    repository: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Fixed cyclic pairs; atomic model/Adam/RNG/cursor replay at full update boundaries."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Distributed DPO is not implemented")
    if (
        type(resume) is not bool
        or stop_after is not None
        and (
            type(stop_after) is not int
            or not 1 <= stop_after <= recipe.job.training.steps
        )
    ):
        raise ValueError("Invalid DPO resume/stop boundary")
    job = recipe.job
    torch.set_num_threads(job.training.num_threads)
    seed_everything(job.experiment.runtime)
    data, tokenizer, pairs = load_preferences(
        Path(recipe.data_manifest),
        recipe.data_sha256,
        job.experiment.model.vocab,
        job.training.sequence_length,
    )
    training = [pair for pair in pairs if pair.split == "train"]
    validation = [pair for pair in pairs if pair.split == "validation"]
    identity = dpo_digest(recipe)
    with distributed_context("cpu") as context:
        model = Decoder(job.experiment.model, job.training.execution)
        initialization = PretrainingJob(
            replace(job, training=replace(job.training, warmup_steps=0, decay_steps=0)),
            recipe.data_manifest,
            recipe.data_sha256,
            TokenSchedule(0, 1, 1),
            initial_checkpoint=recipe.reference_manifest,
            initial_sha256=recipe.reference_sha256,
        )
        initialize_sft_reference(model, initialization, manifest_digest(tokenizer))
        reference = Decoder(job.experiment.model, job.training.execution)
        reference.load_state_dict(model.state_dict())
        reference.requires_grad_(False).eval()
        optimizer = make_optimizer(model, job.training)
        per_pair = [
            sum(assistant_mask(list(frame)))
            for p in training
            for frame in (p.chosen, p.rejected)
        ]
        per_pair = [sum(per_pair[i : i + 2]) for i in range(0, len(per_pair), 2)]
        batch = job.training.batch_size

        def token_count(cursor: int) -> int:
            epochs, tail = divmod(cursor, len(training))
            return epochs * sum(per_pair) + sum(per_pair[:tail])

        def state(step: int) -> dict[str, Any]:
            cursor = step * batch
            return {
                "kind": "qvelt-dpo-v1",
                "recipe_sha256": identity,
                "next_pair": cursor,
                "processed_targets": token_count(cursor),
            }

        def validate(saved: dict[str, Any], step: int, tokens: int) -> None:
            if saved != state(step) or tokens != token_count(step * batch):
                raise ValueError("DPO cursor/reference/pair identity mismatch")

        start = 0
        if resume:
            store = ArtifactStore(directory)
            store.read_manifest()
            with store.open_file("application.yaml") as stream:
                stored = parse_dpo(stream.read(MAX_CONFIG_BYTES + 1).decode())
            if dpo_digest(stored) != identity:
                raise ValueError("DPO recipe/reference/pair mismatch")
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
                data=data,
                tokenizer=tokenizer,
                application_recipe=dump_dpo(recipe).encode(),
            )
        initial_train = evaluate_pairs(model, reference, training, recipe)
        initial_validation = evaluate_pairs(model, reference, validation, recipe)
        final_train, final_validation = initial_train, initial_validation
        losses = []
        limit = job.training.steps if stop_after is None else stop_after
        if limit < start:
            raise ValueError("DPO stop precedes restored step")
        step = start
        sink = JsonMetricSink(directory / f"metrics-{uuid.uuid4().hex}.jsonl")
        try:
            sink.log(
                start,
                {"initial_train_" + k: v for k, v in initial_train.items()}
                | {"initial_validation_" + k: v for k, v in initial_validation.items()},
            )
            for index in range(start, limit):
                selected = [
                    training[(index * batch + i) % len(training)] for i in range(batch)
                ]
                optimizer.zero_grad(set_to_none=True)
                model.train()
                with torch.no_grad():
                    ref, _ = pair_logps(reference, selected)
                policy, _ = pair_logps(model, selected)
                loss = dpo_loss(policy, ref, recipe.beta).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite DPO loss")
                torch.autograd.backward(loss)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    job.training.grad_clip,
                    error_if_nonfinite=True,
                    foreach=False,
                )
                lr = learning_rate(
                    job.training, index * batch, batch, job.training.steps * batch
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                if not persistent_state_finite(model, optimizer, context):
                    raise FloatingPointError("Nonfinite DPO model/Adam state")
                if any(p.grad is not None for p in reference.parameters()):
                    raise RuntimeError(
                        "Frozen reference unexpectedly received gradients"
                    )
                step = index + 1
                losses.append(float(loss.detach()))
                values = {
                    "loss": losses[-1],
                    "grad_norm": float(norm),
                    "learning_rate": lr,
                    "processed_pairs": float(step * batch),
                    "processed_targets": float(token_count(step * batch)),
                }
                if step % job.training.checkpoint_every == 0 or step == limit:
                    final_train = evaluate_pairs(model, reference, training, recipe)
                    final_validation = evaluate_pairs(
                        model, reference, validation, recipe
                    )
                    values.update(
                        {"train_" + k: v for k, v in final_train.items()}
                        | {"validation_" + k: v for k, v in final_validation.items()}
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        job,
                        directory,
                        step,
                        context,
                        processed_tokens=token_count(step * batch),
                        application_state=state(step),
                    )
                sink.log(step, values)
        finally:
            sink.close()
        return {
            "scope": "original demo CPU DPO; offline pair scores,not generation quality",
            "optimizer_step": step,
            "resumed_from": start,
            "processed_pairs": step * batch,
            "processed_targets": token_count(step * batch),
            "initial_train": initial_train,
            "final_train": final_train,
            "initial_validation": initial_validation,
            "final_validation": final_validation,
            "losses_this_attempt": losses,
            "reference_parameters": sum(p.numel() for p in reference.parameters()),
            "reference_frozen": not any(
                p.requires_grad or p.grad is not None for p in reference.parameters()
            ),
        }
