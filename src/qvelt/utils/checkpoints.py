"""Manifest-committed local checkpoints and same-topology synthetic replay."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import random
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file as load_tensor_file
from safetensors.torch import save as save_tensors
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load as distributed_load
from torch.distributed.checkpoint.state_dict_saver import save as distributed_save

from qvelt.config import config_digest, dump_config
from qvelt.config.training import (
    TrainingJob,
    dump_training_job,
    parse_training_job,
    training_job_digest,
)
from qvelt.utils.artifacts import (
    ArtifactIntegrityError,
    ArtifactStore,
    _sync_directory,
    atomic_write,
    create_directory,
    hash_file,
)
from qvelt.utils.distributed import DistributedContext
from qvelt.utils.manifest_types import ArtifactFile, CheckpointInfo, Manifest, RunInfo
from qvelt.utils.manifests import manifest_bytes, manifest_digest, parse_manifest
from qvelt.utils.provenance import capture_code, capture_environment


def _json(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )


def source_digest() -> str:
    """Identify loaded Python source; a dirty flag alone cannot establish replay."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(
            str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0"
        )
    return digest.hexdigest()


def compatibility(job: TrainingJob, context: DistributedContext) -> dict[str, object]:
    return {
        "job_sha256": training_job_digest(job),
        "source_sha256": source_digest(),
        "environment_sha256": hashlib.sha256(
            _json(asdict(capture_environment()))
        ).hexdigest(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": context.device.type,
        "world_size": context.world_size,
        "threads": torch.get_num_threads(),
        "cuda_build": torch.version.cuda,
        "cudnn": cast(Callable[[], int | None], torch.backends.cudnn.version)(),
        "device_name": torch.cuda.get_device_name(context.device)
        if context.device.type == "cuda"
        else platform.machine(),
    }


def initialize_training_run(
    job: TrainingJob, directory: Path, repository: Path, context: DistributedContext
) -> None:
    def initialize() -> None:
        root, code = capture_code(repository)
        environment = capture_environment()
        store = ArtifactStore(create_directory(directory))
        run_id = uuid.uuid4().hex
        files = [
            store.add_bytes(
                "config.yaml", dump_config(job.experiment).encode(), role="config"
            ),
            store.add_bytes(
                "job.yaml", dump_training_job(job).encode(), role="training_recipe"
            ),
            store.add_file("uv.lock", root / "uv.lock", role="dependency_lock"),
            store.add_bytes(
                "environment.json", _json(asdict(environment)), role="environment"
            ),
            store.add_bytes(
                "pip-freeze.txt",
                "".join(
                    f"{p.name}=={p.version}\n" for p in environment.packages
                ).encode(),
                role="package_pins",
            ),
            store.add_bytes(
                "data-spec.json",
                _json(
                    {
                        "algorithm": "qvelt-cyclic-token-fixture-v1",
                        "world_size": context.world_size,
                        "batch_size": job.training.batch_size,
                        "sequence_length": job.training.sequence_length,
                        "vocab": job.experiment.model.vocab,
                    }
                ),
                role="data_spec",
            ),
            store.add_bytes(
                "tokenizer-spec.json",
                _json(
                    {
                        "mapping": "numeric-symbols-only-no-text-tokenizer",
                        "vocab": job.experiment.model.vocab,
                        "reserved": 32,
                    }
                ),
                role="tokenizer_spec",
            ),
            store.add_bytes(
                "compatibility.json",
                _json(compatibility(job, context)),
                role="compatibility",
            ),
        ]
        info = RunInfo(
            run_id,
            config_digest(job.experiment),
            code,
            environment,
            job.experiment.runtime.seed,
            context.world_size,
        )
        store.publish(
            Manifest("run", run_id, datetime.now(UTC).isoformat(), info, tuple(files))
        )
        create_directory(directory / "checkpoints")

    context.rank_zero(initialize)


def _read(store: ArtifactStore, name: str, limit: int = 128 * 1024 * 1024) -> bytes:
    with store.open_file(name) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ArtifactIntegrityError("Checkpoint metadata/state exceeds its load limit")
    return data


def _torch_bytes(value: object) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _rng_state(device: torch.device) -> dict[str, object]:
    numpy = cast(
        tuple[str, NDArray[np.uint32], int, int, float],
        np.random.get_state(legacy=True),
    )
    return {
        "python": random.getstate(),
        "numpy_name": numpy[0],
        "numpy_keys": torch.tensor(numpy[1].astype(np.int64)),
        "numpy_position": numpy[2],
        "numpy_has_gauss": numpy[3],
        "numpy_cached": numpy[4],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(
        (
            state["numpy_name"],
            state["numpy_keys"].numpy().astype(np.uint32),
            state["numpy_position"],
            state["numpy_has_gauss"],
            state["numpy_cached"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def export_safetensors(
    model: nn.Module, path: Path, context: DistributedContext
) -> None:
    """All ranks gather only when export is requested; rank0 commits CPU tensors."""
    state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )

    def write() -> None:
        data = save_tensors(
            {
                name: cast(Tensor, value).detach().cpu().contiguous()
                for name, value in state.items()
            }
        )

        def writer(stream: Any) -> None:
            stream.write(data)

        atomic_write(path, writer)

    context.rank_zero(write)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    job: TrainingJob,
    directory: Path,
    step: int,
    context: DistributedContext,
    *,
    sharded: bool = False,
) -> Path:
    """Save at an optimizer boundary; only manifest.json commits the directory."""
    run_store = ArtifactStore(directory)
    run = run_store.read_manifest()
    run_files = {record.role: record for record in run.files}
    checkpoint = directory / "checkpoints" / f"step-{step:08d}"
    context.rank_zero(lambda: create_directory(checkpoint))
    store = ArtifactStore(checkpoint)
    if sharded:
        distributed_save(
            {"model": get_model_state_dict(model)},
            checkpoint_id=checkpoint / "model",
            no_dist=context.world_size == 1,
        )
        distributed_save(
            {"optimizer": get_optimizer_state_dict(model, optimizer)},
            checkpoint_id=checkpoint / "optimizer",
            no_dist=context.world_size == 1,
        )
    else:
        export_safetensors(model, checkpoint / "model.safetensors", context)
        context.rank_zero(
            lambda: store.add_bytes(
                "optimizer.pt", _torch_bytes(optimizer.state_dict()), role="optimizer"
            )
        )
    store.add_bytes(
        f"rng-rank-{context.rank:05d}.pt",
        _torch_bytes(_rng_state(context.device)),
        role="rng",
    )
    context.barrier()

    def publish() -> None:
        count = (
            step
            * job.training.batch_size
            * job.training.sequence_length
            * context.world_size
        )
        store.add_bytes(
            "config.yaml", dump_config(job.experiment).encode(), role="config"
        )
        store.add_bytes(
            "job.yaml", dump_training_job(job).encode(), role="training_recipe"
        )
        store.add_bytes(
            "scheduler.json",
            _json({"optimizer_step": step, "processed_tokens": count}),
            role="scheduler",
        )
        store.add_bytes(
            "sampler.json",
            _json(
                {
                    "next_step": step,
                    "compatibility": compatibility(job, context),
                    "sharded": sharded,
                }
            ),
            role="sampler",
        )
        store.add_bytes("run.manifest.json", manifest_bytes(run), role="run_manifest")
        store.add_file(
            "pip-freeze.txt", directory / "pip-freeze.txt", role="package_pins"
        )
        records = []
        for path in sorted(checkpoint.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(checkpoint).as_posix()
            if path.is_symlink():
                raise ArtifactIntegrityError("Checkpoint symlinks are forbidden")
            info = hash_file(path)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            if relative.startswith("model/") or relative == "model.safetensors":
                role = "model"
            elif relative.startswith("optimizer/") or relative == "optimizer.pt":
                role = "optimizer"
            elif relative.startswith("rng-rank-"):
                role = "rng"
            else:
                role = {
                    "config.yaml": "config",
                    "job.yaml": "training_recipe",
                    "scheduler.json": "scheduler",
                    "sampler.json": "sampler",
                    "run.manifest.json": "run_manifest",
                    "pip-freeze.txt": "package_pins",
                }[relative]
            records.append(ArtifactFile(relative, role, info.size_bytes, info.sha256))
        for parent in sorted(
            (p for p in checkpoint.rglob("*") if p.is_dir()), reverse=True
        ):
            _sync_directory(parent)
        _sync_directory(checkpoint)
        metadata = CheckpointInfo(
            manifest_digest(run),
            config_digest(job.experiment),
            run_files["data_spec"].sha256,
            run_files["tokenizer_spec"].sha256,
            step,
            count,
            context.world_size,
            job.experiment.runtime.precision,
        )
        store.publish(
            Manifest(
                "checkpoint",
                run.owner,
                datetime.now(UTC).isoformat(),
                metadata,
                tuple(records),
            )
        )

    context.rank_zero(publish)
    return checkpoint


def latest_checkpoint(directory: Path) -> Path:
    """Ignore incomplete directories; never silently fall back from corrupt latest."""
    candidates = []
    for path in (directory / "checkpoints").glob("step-*"):
        if path.is_symlink():
            raise ArtifactIntegrityError("Checkpoint root may not be a symlink")
        if (path / "manifest.json").exists():
            manifest = ArtifactStore(path).read_manifest(verify=False)
            if not isinstance(manifest.metadata, CheckpointInfo):
                raise ArtifactIntegrityError("Expected checkpoint metadata")
            candidates.append((manifest.metadata.optimizer_step, path))
    if not candidates:
        raise FileNotFoundError("No committed checkpoint exists")
    return max(candidates, key=lambda pair: pair[0])[1]


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    job: TrainingJob,
    directory: Path,
    context: DistributedContext,
) -> int:
    checkpoint = latest_checkpoint(directory)
    store = ArtifactStore(checkpoint)
    manifest = store.read_manifest()
    metadata = manifest.metadata
    if not isinstance(metadata, CheckpointInfo):
        raise ArtifactIntegrityError("Expected checkpoint metadata")
    run = ArtifactStore(directory).read_manifest()
    parse_manifest(
        _read(store, "run.manifest.json"), expected_sha256=manifest_digest(run)
    )
    stored_job = parse_training_job(_read(store, "job.yaml").decode("utf-8"))
    if training_job_digest(stored_job) != training_job_digest(job):
        raise ArtifactIntegrityError(
            "Stored training recipe disagrees with requested job"
        )
    lineage = {record.role: record.sha256 for record in run.files}
    if (
        metadata.data_sha256 != lineage["data_spec"]
        or metadata.tokenizer_sha256 != lineage["tokenizer_spec"]
        or manifest.owner != run.owner
    ):
        raise ArtifactIntegrityError("Checkpoint data/tokenizer/owner lineage mismatch")
    sampler = json.loads(_read(store, "sampler.json"))
    schedule = json.loads(_read(store, "scheduler.json"))
    expected = (
        metadata.optimizer_step
        * job.training.batch_size
        * job.training.sequence_length
        * context.world_size
    )
    if (
        metadata.run_sha256 != manifest_digest(run)
        or metadata.config_sha256 != config_digest(job.experiment)
        or sampler["compatibility"] != compatibility(job, context)
        or sampler["next_step"] != metadata.optimizer_step
        or schedule
        != {"optimizer_step": metadata.optimizer_step, "processed_tokens": expected}
        or metadata.processed_tokens != expected
        or metadata.world_size != context.world_size
        or metadata.optimizer_step > job.training.steps
    ):
        raise ArtifactIntegrityError(
            "Checkpoint/config/software/topology or sampler/schedule mismatch"
        )
    optimizer.zero_grad(set_to_none=True)
    if sampler["sharded"]:
        model_state = get_model_state_dict(model)
        optimizer_state = get_optimizer_state_dict(model, optimizer)
        distributed_load(
            {"model": model_state},
            checkpoint_id=checkpoint / "model",
            no_dist=context.world_size == 1,
        )
        distributed_load(
            {"optimizer": optimizer_state},
            checkpoint_id=checkpoint / "optimizer",
            no_dist=context.world_size == 1,
        )
        set_model_state_dict(model, model_state)
        set_optimizer_state_dict(model, optimizer, optimizer_state)
    else:
        set_model_state_dict(
            model,
            cast(
                Any,
                load_tensor_file(
                    str(checkpoint / "model.safetensors"), device=str(context.device)
                ),
            ),
        )
        optimizer.load_state_dict(
            torch.load(
                checkpoint / "optimizer.pt",
                mmap=True,
                weights_only=True,
                map_location=context.device,
            )
        )
    rng = torch.load(
        io.BytesIO(_read(store, f"rng-rank-{context.rank:05d}.pt")),
        weights_only=True,
        map_location="cpu",
    )
    _restore_rng(rng, context.device)
    return metadata.optimizer_step
