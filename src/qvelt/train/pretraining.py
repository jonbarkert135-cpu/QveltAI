"""Manifest-pinned packed-data optimization with atomic consumer/optimizer replay."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from qvelt.config.pretraining import (
    PretrainingJob,
    dump_pretraining_job,
    parse_pretraining_job,
    pretraining_digest,
)
from qvelt.config.training import parse_training_job
from qvelt.data.config import MixStage, load_data_recipe
from qvelt.data.loader import LoaderConfig, PackedDataset, ResumableLoader
from qvelt.model.decoder import Decoder
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.train.accumulation import (
    accumulate_gradients,
    batch_loss,
    device_batch,
    token_learning_rate,
)
from qvelt.train.optim import flops_per_token, make_optimizer
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore
from qvelt.utils.checkpoints import (
    compatibility,
    initialize_training_run,
    load_checkpoint,
    save_checkpoint,
)
from qvelt.utils.distributed import DistributedContext, distributed_context
from qvelt.utils.manifest_types import CheckpointInfo, DataInfo, Manifest
from qvelt.utils.manifests import _reject_constant, _unique_mapping, manifest_digest
from qvelt.utils.metrics import JsonMetricSink, MetricSink, optional_sink
from qvelt.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class PreparedInputs:
    training: PackedDataset
    validation: PackedDataset
    stages: tuple[MixStage, ...]
    data: Manifest
    tokenizer: Manifest
    scope: str


def _read_json(store: ArtifactStore, name: str) -> dict[str, Any]:
    with store.open_file(name) as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Training metadata exceeds1MiB")
    value = json.loads(
        data, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError("Training metadata must be a mapping")
    return value


def prepare_inputs(recipe: PretrainingJob) -> PreparedInputs:
    path = Path(recipe.data_manifest)
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(path.name, expected_sha256=recipe.data_sha256)
    if (
        not isinstance(manifest.metadata, DataInfo)
        or manifest.metadata.representation != "token_ids"
    ):
        raise ValueError("Pretraining requires actual packed token data")
    text_tokenizer, tokenizer = load_tokenizer_bundle(
        path.parent / "tokenizer/manifest.json", manifest.metadata.tokenizer_sha256
    )
    report = _read_json(store, "report.json")
    data_recipe = load_data_recipe(path.parent / "recipe.yaml")
    scope = report.get("scope")
    if scope not in ("demo", "production") or scope != data_recipe.purpose:
        raise ValueError("Invalid dataset preparation scope")
    if (
        scope == "demo"
        and (not recipe.allow_demo or report.get("production_ready") is not False)
        or scope == "production"
        and report.get("production_ready") is not True
    ):
        raise ValueError(
            "Dataset is not approved for this training scope; demo requires allow_demo=true"
        )
    training = PackedDataset(path, "train", recipe.data_sha256)
    validation = PackedDataset(path, "validation", recipe.data_sha256)
    if training.assistant_only != (recipe.objective == "sft") or (
        recipe.objective == "sft" and report.get("objective") != "assistant-v1"
    ):
        raise ValueError("Packed dataset and requested objective disagree")
    if (
        training.vocab != recipe.job.experiment.model.vocab
        or training.vocab != text_tokenizer.vocab_size
        or _read_json(store, "layout.json")["control_version"]
        != text_tokenizer.control_version
        or training.context != recipe.job.training.sequence_length
        or training.context > recipe.job.experiment.model.context
    ):
        raise ValueError(
            "Model vocabulary/sequence length disagrees with actual tokenizer/data"
        )
    return PreparedInputs(
        training, validation, data_recipe.mixing, manifest, tokenizer, cast(str, scope)
    )


def initialize_weights(
    model: nn.Module, recipe: PretrainingJob, tokenizer_sha: str
) -> None:
    """Load only verified model tensors; SFT starts fresh optimizer/RNG/counters."""
    if recipe.initial_checkpoint is None:
        return
    path = Path(recipe.initial_checkpoint)
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(path.name, expected_sha256=recipe.initial_sha256)
    if (
        not isinstance(manifest.metadata, CheckpointInfo)
        or manifest.metadata.tokenizer_sha256 != tokenizer_sha
    ):
        raise ValueError("Initial checkpoint tokenizer identity mismatch")
    records = {file.path: file for file in manifest.files}
    if (
        "model.safetensors" not in records
        or records["model.safetensors"].role != "model"
        or "job.yaml" not in records
    ):
        raise ValueError("Initialization requires a full Qvelt model checkpoint")
    with store.open_file("job.yaml") as stream:
        data = stream.read(128 * 1024 + 1)
    if (
        len(data) > 128 * 1024
        or parse_training_job(data.decode()).experiment.model
        != recipe.job.experiment.model
    ):
        raise ValueError("Initial checkpoint architecture mismatch")
    state = load_file(str(path.parent / "model.safetensors"), device="cpu")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Nonfinite initial model weights")
    model.load_state_dict(state, strict=True)


def _loader_settings(
    recipe: PretrainingJob, context: DistributedContext, *, validation: bool = False
) -> LoaderConfig:
    return LoaderConfig(
        batch_size=recipe.job.training.batch_size,
        seed=(recipe.job.experiment.runtime.seed + (17 if validation else 0)) % 2**32,
        rank=context.rank,
        world_size=context.world_size,
        workers=recipe.workers,
        prefetch=recipe.prefetch,
        max_batches=recipe.validation_batches if validation else recipe.batches,
        attention_mode=recipe.attention_mode,
    )


@torch.no_grad()
def validation_probe(
    model: nn.Module,
    recipe: PretrainingJob,
    inputs: PreparedInputs,
    context: DistributedContext,
) -> dict[str, float]:
    """A fixed weighted heldout probe; it neither advances train state nor model RNG."""
    training = model.training
    try:
        model.eval()
        devices = [context.local_rank] if context.device.type == "cuda" else []
        with (
            torch.random.fork_rng(devices=devices),
            ResumableLoader(
                inputs.validation,
                _loader_settings(recipe, context, validation=True),
                inputs.stages,
            ) as loader,
        ):
            total = torch.zeros((), dtype=torch.float64, device=context.device)
            count = torch.zeros((), dtype=torch.int64, device=context.device)
            for _ in range(recipe.validation_batches):
                raw = context.collective(
                    lambda: next(loader), "Validation data failure"
                )
                batch = context.collective(
                    partial(device_batch, raw, context.device),
                    "Validation transfer failure",
                )
                loss, targets = context.collective(
                    partial(
                        batch_loss,
                        model,
                        batch,
                        recipe.job.experiment.runtime.precision,
                        context.device,
                    ),
                    "Validation forward failure",
                )
                if (
                    int(context.sum(torch.isfinite(loss).to(torch.int64)).item())
                    != context.world_size
                ):
                    raise FloatingPointError("Nonfinite validation loss")
                total += loss.double()
                count += targets
            global_count = int(context.sum(count).item())
            if global_count <= 0:
                raise FloatingPointError("Empty validation probe")
            return {
                "validation_probe_loss": float(context.sum(total).item())
                / global_count,
                "validation_probe_targets": float(global_count),
            }
    finally:
        model.train(training)


def train_pretraining(
    recipe: PretrainingJob,
    directory: Path,
    repository: Path,
    *,
    base: Path = Path("."),
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Complete only optimizer windows; checkpoints commit every rank's consumer state."""
    path = (base / recipe.data_manifest).resolve()
    recipe = replace(
        recipe,
        data_manifest=str(path),
        initial_checkpoint=str((base / recipe.initial_checkpoint).resolve())
        if recipe.initial_checkpoint
        else None,
    )
    job = recipe.job
    final = job.training.steps if stop_after is None else stop_after
    if type(final) is not int or not 1 <= final <= job.training.steps:
        raise ValueError("stop_after outside fixed optimizer-step horizon")
    with distributed_context(job.experiment.runtime.device) as context:
        if (job.training.strategy == "single") != (context.world_size == 1):
            raise ValueError("single requires one process; ddp requires multiple ranks")
        if (
            recipe.batches
            * job.training.batch_size
            * job.training.sequence_length
            * context.world_size
            > 2**63 - 1
        ):
            raise ValueError("Global token capacity exceeds int64 counters")
        previous_threads = torch.get_num_threads()
        sinks: list[MetricSink] = []
        loader: ResumableLoader | None = None
        try:
            torch.set_num_threads(job.training.num_threads)
            context.collective(
                lambda: seed_everything(job.experiment.runtime),
                "Runtime/seed setup failure",
            )
            inputs = context.collective(
                lambda: prepare_inputs(recipe), "Dataset/tokenizer preflight failure"
            )
            identity = pretraining_digest(recipe)
            digests = [identity if context.rank == 0 else None]
            if context.world_size > 1:
                dist.broadcast_object_list(digests, src=0, device=context.device)
            if (
                int(
                    context.sum(
                        torch.tensor(int(identity == digests[0]), device=context.device)
                    ).item()
                )
                != context.world_size
            ):
                raise ValueError("Ranks disagree on pretraining recipe")
            if resume:

                def verify_run() -> None:
                    store = ArtifactStore(directory)
                    store.read_manifest()
                    with store.open_file("application.yaml") as stream:
                        stored = parse_pretraining_job(
                            stream.read(128 * 1024 + 1).decode()
                        )
                    if pretraining_digest(stored) != identity or _read_json(
                        store, "compatibility.json"
                    ) != compatibility(job, context):
                        raise ArtifactIntegrityError(
                            "Pretraining recipe/source/software/topology mismatch"
                        )

                context.collective(verify_run, "Cannot resume this pretraining run")
            else:
                initialize_training_run(
                    job,
                    directory,
                    repository,
                    context,
                    data=inputs.data,
                    tokenizer=inputs.tokenizer,
                    application_recipe=dump_pretraining_job(recipe).encode(),
                )
            model: nn.Module = Decoder(job.experiment.model, job.training.execution).to(
                context.device
            )
            if not resume:
                context.collective(
                    lambda: initialize_weights(
                        model, recipe, manifest_digest(inputs.tokenizer)
                    ),
                    "Initial model failure",
                )
            if context.world_size > 1:
                model = DistributedDataParallel(
                    model,
                    device_ids=[context.local_rank]
                    if context.device.type == "cuda"
                    else None,
                    broadcast_buffers=False,
                )
            optimizer = make_optimizer(model, job.training)
            settings = _loader_settings(recipe, context)
            progress: dict[str, Any] = {"processed_tokens": 0}

            def restore_application(
                state: dict[str, Any], step: int, tokens: int
            ) -> None:
                if set(state) != {
                    "kind",
                    "recipe_yaml",
                    "recipe_sha256",
                    "loader",
                    "processed_tokens",
                    "data_sha256",
                    "tokenizer_sha256",
                }:
                    raise ArtifactIntegrityError(
                        "Unknown application checkpoint fields"
                    )
                if (
                    state["kind"]
                    != (
                        "packed-sft-v1"
                        if recipe.objective == "sft"
                        else "packed-pretraining-v1"
                    )
                    or state["recipe_sha256"] != identity
                    or pretraining_digest(parse_pretraining_job(state["recipe_yaml"]))
                    != identity
                    or state["data_sha256"] != recipe.data_sha256
                    or state["tokenizer_sha256"] != manifest_digest(inputs.tokenizer)
                    or type(state["processed_tokens"]) is not int
                    or state["processed_tokens"] != tokens
                    or not 0
                    < tokens
                    <= min(step * recipe.accumulation_steps, recipe.batches)
                    * settings.batch_size
                    * job.training.sequence_length
                    * context.world_size
                    or state["loader"]["step"]
                    != min(step * recipe.accumulation_steps, recipe.batches)
                ):
                    raise ArtifactIntegrityError(
                        "Application recipe/data/counter mismatch"
                    )
                validated = ResumableLoader(
                    inputs.training, settings, inputs.stages, state["loader"]
                )
                validated.close()
                progress.update(state)

            start = (
                load_checkpoint(
                    model,
                    optimizer,
                    job,
                    directory,
                    context,
                    application_validator=restore_application,
                )
                if resume
                else 0
            )
            if final < start:
                raise ValueError("Cannot move pretraining backwards")
            loader = ResumableLoader(
                inputs.training, settings, inputs.stages, progress.get("loader")
            )
            attempt = uuid.uuid4().hex

            def start_logs() -> None:
                store = ArtifactStore(directory)
                store.add_bytes(
                    f"attempt-{attempt}.yaml",
                    dump_pretraining_job(recipe).encode(),
                    role="attempt_recipe",
                )
                sinks.append(JsonMetricSink(directory / f"metrics-{attempt}.jsonl"))
                extra = optional_sink(job.training.metrics_backend, directory)
                if extra is not None:
                    sinks.append(extra)

            context.rank_zero(start_logs)

            def log(step: int, values: Mapping[str, float | None]) -> None:
                def write_log() -> None:
                    for sink in sinks:
                        sink.log(step, values)

                context.rank_zero(write_log)

            initial = validation_probe(model, recipe, inputs, context)
            initial_training = (
                validation_probe(
                    model, recipe, replace(inputs, validation=inputs.training), context
                )
                if recipe.objective == "sft"
                else None
            )
            final_training = initial_training
            log(
                start,
                {
                    **initial,
                    **(
                        {
                            "training_probe_loss": initial_training[
                                "validation_probe_loss"
                            ],
                            "training_probe_targets": initial_training[
                                "validation_probe_targets"
                            ],
                        }
                        if initial_training
                        else {}
                    ),
                },
            )
            tokens = cast(int, progress["processed_tokens"])
            step = saved = start
            last_validation = initial
            for index in range(start, final):
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                began = time.perf_counter()
                window = min(
                    recipe.accumulation_steps,
                    recipe.batches - cast(int, loader.state_dict()["step"]),
                )
                stats = accumulate_gradients(
                    model,
                    lambda: next(loader),
                    window,
                    context,
                    job.experiment.runtime.precision,
                )
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    job.training.grad_clip,
                    error_if_nonfinite=True,
                    foreach=False,
                )
                lr = token_learning_rate(
                    job.training.learning_rate, recipe.schedule, tokens, stats.targets
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                tokens += stats.targets
                step = index + 1
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                seconds = torch.tensor(
                    time.perf_counter() - began,
                    dtype=torch.float64,
                    device=context.device,
                )
                if context.world_size > 1:
                    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
                throughput = stats.targets / float(seconds)
                estimated_mfu = (
                    flops_per_token(job.experiment.model, job.training.sequence_length)
                    * throughput
                    / (job.training.peak_tflops_per_gpu * context.world_size * 1e12)
                    if job.training.peak_tflops_per_gpu
                    else None
                )
                values = {
                    "loss": stats.loss,
                    "learning_rate": lr,
                    "gradient_norm": float(norm),
                    "tokens_per_second": throughput,
                    "clipped": float(float(norm) > job.training.grad_clip),
                    "processed_tokens": float(tokens),
                    "consumed_microbatches": float(loader.state_dict()["step"]),
                    "estimated_mfu": estimated_mfu,
                }
                if step % recipe.validation_every == 0 or step == final:
                    last_validation = validation_probe(model, recipe, inputs, context)
                    values.update(last_validation)
                if recipe.objective == "sft" and step == final:
                    final_training = validation_probe(
                        model,
                        recipe,
                        replace(inputs, validation=inputs.training),
                        context,
                    )
                    values.update(
                        {
                            "training_probe_loss": final_training[
                                "validation_probe_loss"
                            ],
                            "training_probe_targets": final_training[
                                "validation_probe_targets"
                            ],
                        }
                    )
                log(step, values)
                if step % job.training.checkpoint_every == 0 or step == final:
                    state = {
                        "kind": "packed-sft-v1"
                        if recipe.objective == "sft"
                        else "packed-pretraining-v1",
                        "recipe_yaml": dump_pretraining_job(recipe),
                        "recipe_sha256": identity,
                        "loader": loader.state_dict(),
                        "processed_tokens": tokens,
                        "data_sha256": recipe.data_sha256,
                        "tokenizer_sha256": manifest_digest(inputs.tokenizer),
                    }
                    save_checkpoint(
                        model,
                        optimizer,
                        job,
                        directory,
                        step,
                        context,
                        processed_tokens=tokens,
                        application_state=state,
                    )
                    saved = step
            assert saved == step
            return {
                "rank": context.rank,
                "world_size": context.world_size,
                "optimizer_step": step,
                "processed_tokens": tokens,
                "scope": inputs.scope,
                "data_sha256": recipe.data_sha256,
                "tokenizer_sha256": manifest_digest(inputs.tokenizer),
                "consumed_microbatches": loader.state_dict()["step"],
                "initial_validation": initial,
                "final_validation": last_validation,
                **(
                    {
                        "initial_training": initial_training,
                        "final_training": final_training,
                    }
                    if initial_training
                    else {}
                ),
            }
        finally:
            try:
                if loader is not None:
                    context.collective(
                        loader.close, "A rank could not close its data loader"
                    )

                def close_logs() -> None:
                    for sink in sinks:
                        sink.close()

                context.rank_zero(close_logs)
            finally:
                torch.set_num_threads(previous_threads)
