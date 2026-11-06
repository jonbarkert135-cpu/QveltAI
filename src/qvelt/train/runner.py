"""Real RFC-decoder synthetic training; not a real-data pretraining pipeline."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from qvelt.config.training import TrainingJob
from qvelt.data.synthetic import synthetic_batch
from qvelt.model.decoder import Decoder
from qvelt.model.loss import causal_loss
from qvelt.train.optim import flops_per_token, learning_rate, make_optimizer
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore
from qvelt.utils.checkpoints import (
    compatibility,
    initialize_training_run,
    load_checkpoint,
    save_checkpoint,
)
from qvelt.utils.distributed import distributed_context
from qvelt.utils.metrics import JsonMetricSink, MetricSink, optional_sink
from qvelt.utils.reproducibility import seed_everything


def train_synthetic(
    job: TrainingJob,
    directory: Path,
    repository: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, int | str]:
    """Completed-step checkpoints support exact replay on the tested CPU topology."""
    import json

    with distributed_context(job.experiment.runtime.device) as context:
        config = job.training
        if (config.strategy == "single") != (context.world_size == 1):
            raise ValueError(
                "single requires one process; ddp/fsdp require torchrun with multiple ranks"
            )
        if config.strategy == "fsdp" and context.device.type != "cuda":
            raise ValueError(
                "FSDP2 training requires CUDA; CPU DCP checks do not validate FSDP2"
            )
        final = config.steps if stop_after is None else stop_after
        if not 1 <= final <= config.steps:
            raise ValueError("stop_after must lie inside the configured step horizon")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(config.num_threads)
        seed_everything(job.experiment.runtime)
        if resume:
            ArtifactStore(directory).read_manifest()
            saved = json.loads((directory / "compatibility.json").read_text())
            if saved != compatibility(job, context):
                raise ArtifactIntegrityError(
                    "Run/config/software/topology compatibility mismatch"
                )
        else:
            initialize_training_run(job, directory, repository, context)
        model: nn.Module = Decoder(job.experiment.model, execution=config.execution).to(
            context.device
        )
        from qvelt.train.parallel import parallelize

        assert isinstance(model, Decoder)
        model = parallelize(model, job, context)
        optimizer = make_optimizer(model, config)
        start = (
            load_checkpoint(
                model, optimizer, job, directory, context, repository=repository
            )
            if resume
            else 0
        )
        if final < start:
            raise ValueError("A resumed job cannot move backwards")
        sinks: list[MetricSink] = []

        def start_logs() -> None:
            attempt = uuid.uuid4().hex
            sinks.append(JsonMetricSink(directory / f"metrics-{attempt}.jsonl"))
            sink = optional_sink(config.metrics_backend, directory)
            if sink is not None:
                sinks.append(sink)

        step = last_saved = start
        update_tokens = config.batch_size * config.sequence_length * context.world_size
        total_tokens = config.steps * update_tokens
        try:
            context.rank_zero(start_logs)
            for index in range(start, final):
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                began = time.perf_counter()
                batch = synthetic_batch(
                    index,
                    config.batch_size,
                    config.sequence_length,
                    job.experiment.model.vocab,
                    rank=context.rank,
                    world_size=context.world_size,
                    device=context.device,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    context.device.type,
                    dtype=torch.bfloat16,
                    enabled=job.experiment.runtime.precision == "bf16",
                ):
                    logits = model(batch.inputs)
                    loss_sum, local_count = causal_loss(logits, batch.targets)
                global_count = context.sum(local_count)
                finite = context.sum(torch.isfinite(loss_sum).to(torch.int64))
                if (
                    int(finite.item()) != context.world_size
                    or int(global_count.item()) == 0
                ):
                    raise FloatingPointError(
                        "Invalid/nonfinite global training loss or empty target window"
                    )
                # DDP averages gradients: n/global_count recovers the global mean.
                torch.autograd.backward(loss_sum * (context.world_size / global_count))
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                    error_if_nonfinite=True,
                    foreach=False,
                )
                norm_value = float(
                    norm.full_tensor().item()
                    if isinstance(norm, DTensor)
                    else norm.item()
                )
                lr = learning_rate(
                    config, index * update_tokens, update_tokens, total_tokens
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                elapsed = torch.tensor(
                    time.perf_counter() - began,
                    dtype=torch.float64,
                    device=context.device,
                )
                if context.world_size > 1:
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                seconds = float(elapsed.item())
                loss = float((context.sum(loss_sum) / global_count).item())
                tokens_per_second = update_tokens / seconds
                estimate = (
                    flops_per_token(job.experiment.model, config.sequence_length)
                    * tokens_per_second
                )
                mfu = (
                    estimate / (config.peak_tflops_per_gpu * context.world_size * 1e12)
                    if config.peak_tflops_per_gpu
                    else None
                )
                step = index + 1
                values = {
                    "loss": loss,
                    "learning_rate": lr,
                    "gradient_norm": norm_value,
                    "clipped": float(norm_value > config.grad_clip),
                    "tokens_per_second": tokens_per_second,
                    "estimated_mfu": mfu,
                    "processed_tokens": float(step * update_tokens),
                }

                def log(
                    current_step: int = step,
                    current_values: dict[str, float | None] = values,
                ) -> None:
                    for sink in sinks:
                        sink.log(current_step, current_values)

                context.rank_zero(log)
                if step % config.checkpoint_every == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        job,
                        directory,
                        step,
                        context,
                        sharded=config.strategy == "fsdp",
                        repository=repository,
                    )
                    last_saved = step
            if step != last_saved:
                save_checkpoint(
                    model,
                    optimizer,
                    job,
                    directory,
                    step,
                    context,
                    sharded=config.strategy == "fsdp",
                    repository=repository,
                )
            return {
                "rank": context.rank,
                "optimizer_step": step,
                "processed_tokens": step * update_tokens,
                "world_size": context.world_size,
                "run_directory": str(directory),
            }
        finally:

            def close_logs() -> None:
                for sink in sinks:
                    sink.close()

            context.rank_zero(close_logs)
            torch.set_num_threads(previous_threads)
