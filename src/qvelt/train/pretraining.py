"""Manifest-pinned packed-data optimization with atomic consumer/optimizer replay."""

from __future__ import annotations

import json
import logging
import math
import signal
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import nn

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
from qvelt.tokenizer.core import QveltTokenizer
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.train.accumulation import (
    accumulate_gradients,
    batch_loss,
    device_batch,
    token_learning_rate,
)
from qvelt.train.evaluation import domain_validation, generate_samples
from qvelt.train.optim import flops_per_token, make_optimizer
from qvelt.train.parallel import (
    full_scalar,
    is_sharded,
    parallelize,
    persistent_state_finite,
    unwrap,
)
from qvelt.train.stability import (
    ActivationRecorder,
    SpikeMonitor,
    layer_gradient_norms,
)
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore
from qvelt.utils.checkpoints import (
    committed_steps,
    compatibility,
    initialize_training_run,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    select_valid_checkpoint,
)
from qvelt.utils.distributed import DistributedContext, distributed_context
from qvelt.utils.manifest_types import CheckpointInfo, DataInfo, Manifest
from qvelt.utils.manifests import _reject_constant, _unique_mapping, manifest_digest
from qvelt.utils.metrics import JsonMetricSink, MetricSink, optional_sink
from qvelt.utils.reproducibility import seed_everything

LOGGER = logging.getLogger("qvelt")
APPLICATION_KIND = "packed-pretraining-v2"
APPLICATION_FIELDS = {
    "kind",
    "recipe_yaml",
    "recipe_sha256",
    "loader",
    "processed_tokens",
    "data_sha256",
    "tokenizer_sha256",
    "counters",
    "spike_monitor",
}


@dataclass(frozen=True)
class PreparedInputs:
    training: PackedDataset
    validation: PackedDataset
    stages: tuple[MixStage, ...]
    data: Manifest
    tokenizer: Manifest
    scope: str
    text_tokenizer: QveltTokenizer | None = None


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
        training,
        validation,
        data_recipe.mixing,
        manifest,
        tokenizer,
        cast(str, scope),
        text_tokenizer,
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


class StopSignal:
    """Record SIGTERM; the loop checkpoints at the next optimizer boundary."""

    def __init__(self) -> None:
        self.received: str | None = None

    def __call__(self, number: int, _frame: object) -> None:
        self.received = signal.Signals(number).name


@contextmanager
def graceful_stop() -> Iterator[StopSignal]:
    handler = StopSignal()
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield handler
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _shared(context: DistributedContext, action: Callable[[], Any]) -> Any:
    """Run on rank0 (errors propagate to every rank) and broadcast the result."""
    box = [context.rank_zero(action)]
    if context.world_size > 1:
        dist.broadcast_object_list(box, src=0, device=context.device)
    return box[0]


def _resume_mode(resume: bool | str) -> str:
    if resume is False:
        return "fresh"
    if resume is True or resume == "latest":
        return "latest"
    if resume == "auto":
        return "auto"
    raise ValueError("resume must be False, True/'latest' or 'auto'")


def train_pretraining(
    recipe: PretrainingJob,
    directory: Path,
    repository: Path,
    *,
    base: Path = Path("."),
    resume: bool | str = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Complete only optimizer windows; checkpoints commit every rank's consumer state.

    ``resume``: False (new run), True/"latest" (strict: newest committed checkpoint,
    corrupt latest is an error) or "auto" (new run if the directory has no run
    manifest, else the newest *verified* checkpoint, quarantining corrupt ones).
    """
    path = (base / recipe.data_manifest).resolve()
    recipe = replace(
        recipe,
        data_manifest=str(path),
        initial_checkpoint=str((base / recipe.initial_checkpoint).resolve())
        if recipe.initial_checkpoint
        else None,
    )
    job = recipe.job
    mode = _resume_mode(resume)
    final = job.training.steps if stop_after is None else stop_after
    if type(final) is not int or not 1 <= final <= job.training.steps:
        raise ValueError("stop_after outside fixed optimizer-step horizon")
    with (
        graceful_stop() as stop_signal,
        distributed_context(job.experiment.runtime.device) as context,
    ):
        if (job.training.strategy == "single") != (context.world_size == 1):
            raise ValueError(
                "single requires one process; ddp/fsdp require multiple ranks"
            )
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
            if mode == "auto":
                exists = bool(
                    _shared(context, lambda: (directory / "manifest.json").exists())
                )
                mode = "auto" if exists else "fresh"
                LOGGER.info("auto-resume: %s", "resuming" if exists else "new run")
            if mode != "fresh":

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
            chosen = (
                _shared(context, lambda: select_valid_checkpoint(directory))
                if mode == "auto"
                else None
            )
            decoder = Decoder(job.experiment.model, job.training.execution).to(
                context.device
            )
            if mode == "fresh" or mode == "auto" and chosen is None:
                context.collective(
                    lambda: initialize_weights(
                        decoder, recipe, manifest_digest(inputs.tokenizer)
                    ),
                    "Initial model failure",
                )
            model = parallelize(decoder, job, context)
            sharded = is_sharded(model)
            optimizer = make_optimizer(model, job.training)
            settings = _loader_settings(recipe, context)
            progress: dict[str, Any] = {"processed_tokens": 0}
            counters = {"nonfinite_windows": 0, "rollbacks": 0, "optimizer_updates": 0}
            rollback_updates = {"value": 0}
            monitor = SpikeMonitor(recipe.spike_threshold, recipe.spike_warmup_windows)

            def check_identity(state: dict[str, Any]) -> None:
                if set(state) != APPLICATION_FIELDS:
                    raise ArtifactIntegrityError(
                        "Unknown application checkpoint fields"
                    )
                if (
                    state["kind"]
                    != (
                        "packed-sft-v2"
                        if recipe.objective == "sft"
                        else APPLICATION_KIND
                    )
                    or state["recipe_sha256"] != identity
                    or pretraining_digest(parse_pretraining_job(state["recipe_yaml"]))
                    != identity
                    or state["data_sha256"] != recipe.data_sha256
                    or state["tokenizer_sha256"] != manifest_digest(inputs.tokenizer)
                ):
                    raise ArtifactIntegrityError("Application recipe/data mismatch")

            def restore_application(
                state: dict[str, Any], step: int, tokens: int
            ) -> None:
                check_identity(state)
                if (
                    type(state["processed_tokens"]) is not int
                    or state["processed_tokens"] != tokens
                    or not 0
                    <= tokens
                    <= min(step * recipe.accumulation_steps, recipe.batches)
                    * settings.batch_size
                    * job.training.sequence_length
                    * context.world_size
                    or state["loader"]["step"]
                    != min(step * recipe.accumulation_steps, recipe.batches)
                    or (tokens == 0) != (step == 0)
                    or not isinstance(state["counters"], dict)
                    or set(state["counters"]) != set(counters)
                    or any(
                        type(v) is not int or v < 0 for v in state["counters"].values()
                    )
                ):
                    raise ArtifactIntegrityError(
                        "Application recipe/data/counter mismatch"
                    )
                validated = ResumableLoader(
                    inputs.training, settings, inputs.stages, state["loader"]
                )
                validated.close()
                try:
                    monitor.load_state_dict(state["spike_monitor"])
                except (TypeError, ValueError) as error:
                    raise ArtifactIntegrityError(str(error)) from error
                counters.update(state["counters"])
                progress.update(state)

            def rollback_application(
                state: dict[str, Any], _step: int, _tokens: int
            ) -> None:
                # Weights/Adam/RNG come from the checkpoint; cursor, token count,
                # counters and spike statistics keep moving forward (data skipped).
                check_identity(state)
                value = state["counters"].get("optimizer_updates")
                if type(value) is not int or value < 0:
                    raise ArtifactIntegrityError("Invalid rollback optimizer counter")
                rollback_updates["value"] = value

            start = 0
            if mode == "latest":
                start = load_checkpoint(
                    model,
                    optimizer,
                    job,
                    directory,
                    context,
                    application_validator=restore_application,
                )
            elif mode == "auto":
                if chosen is None:
                    LOGGER.warning(
                        "auto-resume: no valid checkpoint; restarting this run at 0"
                    )
                else:
                    start = load_checkpoint(
                        model,
                        optimizer,
                        job,
                        directory,
                        context,
                        application_validator=restore_application,
                        checkpoint=Path(chosen),
                    )
                    LOGGER.warning("auto-resume: restored %s", chosen)
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

            def evaluate() -> dict[str, float | None]:
                values: dict[str, float | None] = dict(
                    validation_probe(model, recipe, inputs, context)
                )
                if recipe.domain_validation_batches:
                    values.update(
                        domain_validation(
                            model,
                            inputs.validation,
                            batches=recipe.domain_validation_batches,
                            batch_size=job.training.batch_size,
                            attention_mode=recipe.attention_mode,
                            precision=job.experiment.runtime.precision,
                            context=context,
                        )
                    )
                return values

            def write_samples(step: int) -> None:
                assert inputs.text_tokenizer is not None
                samples = generate_samples(
                    model,
                    inputs.text_tokenizer,
                    recipe.sample_prompts,
                    tokens=recipe.sample_tokens,
                    context_length=job.training.sequence_length,
                    precision=job.experiment.runtime.precision,
                    context=context,
                )

                def append() -> None:
                    with (directory / f"samples-{attempt}.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        for sample in samples:
                            stream.write(
                                json.dumps({"step": step, **sample}, sort_keys=True)
                                + "\n"
                            )

                context.rank_zero(append)

            def application_state(tokens: int) -> dict[str, Any]:
                assert loader is not None
                return {
                    "kind": "packed-sft-v2"
                    if recipe.objective == "sft"
                    else APPLICATION_KIND,
                    "recipe_yaml": dump_pretraining_job(recipe),
                    "recipe_sha256": identity,
                    "loader": loader.state_dict(),
                    "processed_tokens": tokens,
                    "data_sha256": recipe.data_sha256,
                    "tokenizer_sha256": manifest_digest(inputs.tokenizer),
                    "counters": dict(counters),
                    "spike_monitor": monitor.state_dict(),
                }

            initial = evaluate()
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
            if recipe.sample_every and start == 0:
                write_samples(0)
            tokens = cast(int, progress["processed_tokens"])
            step = saved = start
            if (
                recipe.nonfinite_policy == "rollback"
                and start == 0
                and not committed_steps(directory)
            ):
                save_checkpoint(
                    model,
                    optimizer,
                    job,
                    directory,
                    0,
                    context,
                    sharded=sharded,
                    processed_tokens=0,
                    application_state=application_state(0),
                )
            last_validation = initial
            interrupted: str | None = None
            spikes_at_start = monitor.spikes
            if recipe.layer_stats_every and job.training.compile:
                LOGGER.warning(
                    "layer stats under compile: gradient norms only (no hooks)"
                )
            horizon = recipe.schedule.decay_start_tokens + recipe.schedule.decay_tokens
            for index in range(start, final):
                if tokens >= horizon:
                    # RFC-002: no update after the token horizon D; a resumed or
                    # over-provisioned step cap ends cleanly here.
                    LOGGER.warning("Token horizon reached at step %d; done", step)
                    break
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                began = time.perf_counter()
                window = min(
                    recipe.accumulation_steps,
                    recipe.batches - cast(int, loader.state_dict()["step"]),
                )
                window_start = loader.state_dict()
                stats_step = bool(
                    recipe.layer_stats_every
                    and (index + 1) % recipe.layer_stats_every == 0
                )
                recorder = (
                    ActivationRecorder(unwrap(model))
                    if stats_step and not job.training.compile
                    else None
                )
                with recorder.recording() if recorder else nullcontext():
                    stats = accumulate_gradients(
                        model,
                        lambda: next(loader),
                        window,
                        context,
                        job.experiment.runtime.precision,
                        z_weight=recipe.z_loss,
                        stop_on_nonfinite=recipe.nonfinite_policy == "abort",
                        target_budget=(horizon - tokens)
                        if horizon - tokens
                        < window
                        * settings.batch_size
                        * job.training.sequence_length
                        * context.world_size
                        else None,
                    )
                values: dict[str, float | None] = {}
                healthy = stats.finite
                norm: float | None = None
                if healthy:
                    if stats_step:
                        values.update(
                            layer_gradient_norms(
                                unwrap(model), context, sharded=sharded
                            )
                        )
                        if recorder is not None:
                            values.update(recorder.reduce(context))
                    norm = full_scalar(
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            job.training.grad_clip,
                            error_if_nonfinite=False,
                            foreach=False,
                        )
                    )
                    if not math.isfinite(norm):
                        if recipe.nonfinite_policy == "abort":
                            raise FloatingPointError("Nonfinite gradient norm")
                        healthy, norm = False, None
                        model.zero_grad(set_to_none=True)
                lr = token_learning_rate(
                    job.training.learning_rate, recipe.schedule, tokens, stats.targets
                )
                if healthy:
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    healthy = persistent_state_finite(model, optimizer, context)
                    if not healthy:
                        if recipe.nonfinite_policy != "rollback":
                            raise FloatingPointError(
                                "Nonfinite persistent optimizer/model state; rollback required"
                            )
                        norm = None
                    else:
                        counters["optimizer_updates"] += 1
                tokens += stats.targets
                step = index + 1
                if not healthy:
                    counters["nonfinite_windows"] += 1

                    failure = {
                        "step": step,
                        "data_sha256": recipe.data_sha256,
                        "window_start": window_start,
                        "window_end": loader.state_dict(),
                        "skipped_targets": stats.targets,
                    }
                    context.rank_zero(
                        partial(
                            ArtifactStore(directory).add_bytes,
                            f"failure-{attempt}-{step:08d}.json",
                            (json.dumps(failure, sort_keys=True) + "\n").encode(),
                            role="failure_diagnostic",
                        )
                    )
                    values["nonfinite_window"] = 1.0
                    LOGGER.error(
                        "NONFINITE window at step %d (%d/%d allowed): update skipped, "
                        "%d targets of data skipped",
                        step,
                        counters["nonfinite_windows"],
                        recipe.max_nonfinite_windows,
                        stats.targets,
                    )
                    if counters["nonfinite_windows"] > recipe.max_nonfinite_windows:
                        raise FloatingPointError(
                            "Nonfinite window budget exhausted; inspect the run"
                        )
                    if recipe.nonfinite_policy == "rollback":
                        target = _shared(
                            context,
                            partial(select_valid_checkpoint, directory, before=step),
                        )
                        if target is None:
                            raise FloatingPointError(
                                "Nonfinite window and no committed checkpoint to roll "
                                "back to"
                            )
                        restored = load_checkpoint(
                            model,
                            optimizer,
                            job,
                            directory,
                            context,
                            application_validator=rollback_application,
                            checkpoint=Path(target),
                        )
                        counters["rollbacks"] += 1
                        counters["optimizer_updates"] = rollback_updates["value"]
                        values["rollback_to_step"] = float(restored)
                        LOGGER.error(
                            "ROLLBACK at step %d: weights/optimizer restored from "
                            "step %d; data cursor kept (window skipped)",
                            step,
                            restored,
                        )
                elif monitor.threshold:
                    score, spike = monitor.update(stats.loss)
                    values["loss_zscore"] = score
                    values["loss_spike"] = float(spike)
                    if spike:
                        LOGGER.warning(
                            "LOSS SPIKE at step %d: loss %.5f, z=%.1f over EMA %.5f",
                            step,
                            stats.loss,
                            score,
                            monitor.mean,
                        )
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
                values.update(
                    {
                        "loss": stats.loss if healthy else None,
                        "learning_rate": lr,
                        "gradient_norm": norm,
                        "tokens_per_second": throughput,
                        "tokens_per_second_per_device": throughput / context.world_size,
                        "clipped": float(norm > job.training.grad_clip)
                        if norm is not None
                        else None,
                        "processed_tokens": float(tokens),
                        "consumed_microbatches": float(loader.state_dict()["step"]),
                        "estimated_mfu": estimated_mfu,
                        "nonfinite_windows": float(counters["nonfinite_windows"]),
                        "rollbacks": float(counters["rollbacks"]),
                        "optimizer_updates": float(counters["optimizer_updates"]),
                    }
                )
                if stats.z_loss is not None and healthy:
                    values["z_loss"] = stats.z_loss
                stopping = int(
                    context.sum(
                        torch.tensor(
                            int(stop_signal.received is not None),
                            device=context.device,
                        )
                    ).item()
                )
                if stopping:
                    interrupted = stop_signal.received or "SIGTERM"
                complete = tokens >= horizon
                if step % recipe.validation_every == 0 or step == final or complete:
                    last_validation = evaluate()
                    values.update(last_validation)
                if recipe.sample_every and step % recipe.sample_every == 0:
                    write_samples(step)
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
                values = {
                    key: value if value is None or math.isfinite(value) else None
                    for key, value in values.items()
                }
                log(step, values)
                if (
                    step % job.training.checkpoint_every == 0
                    or step == final
                    or interrupted
                    or complete
                ):
                    save_checkpoint(
                        model,
                        optimizer,
                        job,
                        directory,
                        step,
                        context,
                        sharded=sharded,
                        processed_tokens=tokens,
                        application_state=application_state(tokens),
                    )
                    saved = step
                    if recipe.keep_checkpoints:
                        context.rank_zero(
                            lambda: prune_checkpoints(
                                directory, recipe.keep_checkpoints
                            )
                        )
                if interrupted:
                    LOGGER.warning(
                        "Stopped by %s after committing step %d", interrupted, step
                    )
                    break
            assert saved == step
            return {
                "rank": context.rank,
                "world_size": context.world_size,
                "strategy": job.training.strategy,
                "resumed_from": start,
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
                "interrupted": interrupted,
                "schedule_complete": tokens >= horizon,
                "nonfinite_windows": counters["nonfinite_windows"],
                "rollbacks": counters["rollbacks"],
                "optimizer_updates": counters["optimizer_updates"],
                "loss_spikes_this_attempt": monitor.spikes - spikes_at_start,
                "committed_checkpoints": len(committed_steps(directory))
                if context.rank == 0
                else None,
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
