"""Manifest-committed local checkpoints and same-topology synthetic replay."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import random
import re
import shutil
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
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactStore,
    _sync_directory,
    atomic_write,
    create_directory,
    hash_file,
)
from qvelt.utils.distributed import DistributedContext
from qvelt.utils.manifest_types import (
    ArtifactFile,
    CheckpointInfo,
    DataInfo,
    Manifest,
    RunInfo,
    TokenizerInfo,
    uint,
)
from qvelt.utils.manifests import (
    _reject_constant,
    _unique_mapping,
    manifest_bytes,
    manifest_digest,
    parse_manifest,
)
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
    job: TrainingJob,
    directory: Path,
    repository: Path,
    context: DistributedContext,
    *,
    data: Manifest | None = None,
    tokenizer: Manifest | None = None,
    application_recipe: bytes | None = None,
) -> None:
    if any(v is not None for v in (data, tokenizer, application_recipe)):
        if (
            data is None
            or tokenizer is None
            or application_recipe is None
            or not isinstance(data.metadata, DataInfo)
            or data.metadata.representation != "token_ids"
            or not isinstance(tokenizer.metadata, TokenizerInfo)
            or data.metadata.tokenizer_sha256 != manifest_digest(tokenizer)
        ):
            raise ValueError(
                "Actual data/tokenizer lineage and application recipe must agree"
            )
    if job.training.regression_recipe is not None:
        from qvelt.eval.regression import regression_policy

        if tokenizer is None or application_recipe is None:
            raise ValueError("Checkpoint regression requires actual BPE training data")
        regression_policy(job, tokenizer)

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
                "compatibility.json",
                _json(compatibility(job, context)),
                role="compatibility",
            ),
        ]
        if (
            data is not None
            and tokenizer is not None
            and application_recipe is not None
        ):
            files.extend(
                (
                    store.add_bytes(
                        "data.manifest.json", manifest_bytes(data), role="data_manifest"
                    ),
                    store.add_bytes(
                        "tokenizer.manifest.json",
                        manifest_bytes(tokenizer),
                        role="tokenizer_manifest",
                    ),
                    store.add_bytes(
                        "application.yaml",
                        application_recipe,
                        role="application_recipe",
                    ),
                )
            )
        else:
            files.extend(
                [
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
                ]
            )
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


def _run_lineage(store: ArtifactStore, run: Manifest) -> tuple[str, str]:
    roles = {f.role: f for f in run.files}
    for role in ("data_manifest", "tokenizer_manifest", "data_spec", "tokenizer_spec"):
        if sum(f.role == role for f in run.files) > 1:
            raise ArtifactIntegrityError("Duplicate data/tokenizer lineage role")
    actual = {"data_manifest", "tokenizer_manifest"} & set(roles)
    if actual:
        if actual != {"data_manifest", "tokenizer_manifest"} or {
            "data_spec",
            "tokenizer_spec",
        } & set(roles):
            raise ArtifactIntegrityError("Ambiguous data/tokenizer lineage")
        data = parse_manifest(_read(store, roles["data_manifest"].path, 1024 * 1024))
        tokenizer = parse_manifest(
            _read(store, roles["tokenizer_manifest"].path, 1024 * 1024)
        )
        if (
            not isinstance(data.metadata, DataInfo)
            or data.metadata.representation != "token_ids"
            or not isinstance(tokenizer.metadata, TokenizerInfo)
            or data.metadata.tokenizer_sha256 != manifest_digest(tokenizer)
        ):
            raise ArtifactIntegrityError("Actual data/tokenizer lineage mismatch")
        return manifest_digest(data), manifest_digest(tokenizer)
    if not {"data_spec", "tokenizer_spec"} <= set(roles):
        raise ArtifactIntegrityError("Missing data/tokenizer lineage")
    return roles["data_spec"].sha256, roles["tokenizer_spec"].sha256


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


def _sharded_optimizer_state(
    model: nn.Module, optimizer: torch.optim.Optimizer, step: int
) -> dict[str, Any]:
    """DCP materializes empty Adam state with an LR0 step; initial t must stay0."""
    initial = step == 0 and not optimizer.state
    result = get_optimizer_state_dict(model, optimizer)
    if initial:
        for state in optimizer.state.values():
            counter = state.get("step")
            if isinstance(counter, Tensor):
                counter.zero_()
        result = get_optimizer_state_dict(model, optimizer)
    return result


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    job: TrainingJob,
    directory: Path,
    step: int,
    context: DistributedContext,
    *,
    sharded: bool = False,
    processed_tokens: int | None = None,
    application_state: dict[str, Any] | None = None,
    repository: Path | None = None,
) -> Path:
    """Save at an optimizer boundary; only manifest.json commits the directory."""
    if job.training.regression_recipe is not None and repository is None:
        raise ValueError("Checkpoint regression requires the actual repository")
    run_store = ArtifactStore(directory)
    run = run_store.read_manifest()
    data_sha, tokenizer_sha = _run_lineage(run_store, run)
    if (processed_tokens is None) != (application_state is None):
        raise ValueError(
            "Application cursor and processed-token count must be saved together"
        )
    if processed_tokens is not None:
        uint(processed_tokens, "checkpoint.processed_tokens")
    checkpoint = directory / "checkpoints" / f"step-{step:08d}"

    def create() -> None:
        if checkpoint.exists() and not (checkpoint / "manifest.json").exists():
            # A manifest-less directory is an uncommitted write (e.g. SIGKILL
            # mid-save); move it aside instead of failing or overwriting.
            quarantine(checkpoint, "incomplete checkpoint replaced by a new save")
        create_directory(checkpoint)

    context.rank_zero(create)
    store = ArtifactStore(checkpoint)
    if sharded:
        distributed_save(
            {"model": get_model_state_dict(model)},
            checkpoint_id=checkpoint / "model",
            no_dist=context.world_size == 1,
        )
        distributed_save(
            {"optimizer": _sharded_optimizer_state(model, optimizer, step)},
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

    def save_rank() -> None:
        store.add_bytes(
            f"rng-rank-{context.rank:05d}.pt",
            _torch_bytes(_rng_state(context.device)),
            role="rng",
        )
        if application_state is not None:
            body = _json(application_state)
            if len(body) > 1024 * 1024:
                raise ValueError("Application state exceeds1MiB")
            store.add_bytes(
                f"application-rank-{context.rank:05d}.json", body, role="application"
            )

    context.collective(save_rank, "A rank could not save its checkpoint state")
    context.barrier()

    def publish() -> None:
        count = (
            processed_tokens
            if processed_tokens is not None
            else (
                step
                * job.training.batch_size
                * job.training.sequence_length
                * context.world_size
            )
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
                    **({"application": True} if application_state is not None else {}),
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
            elif relative.startswith("application-rank-"):
                role = "application"
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
            data_sha,
            tokenizer_sha,
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
    if job.training.regression_recipe is not None:
        from qvelt.eval.regression import evaluate_checkpoint

        assert repository is not None
        context.rank_zero(lambda: evaluate_checkpoint(directory, step, job, repository))
    return checkpoint


_STEP = re.compile(r"step-(\d{8,})")
LOGGER = logging.getLogger("qvelt")


def quarantine(path: Path, reason: str) -> Path:
    """Move a checkpoint directory under checkpoints/quarantine; never delete it."""
    if path.is_symlink() or path.parent.is_symlink():
        raise ArtifactIntegrityError("Checkpoint symlinks are forbidden")
    root = path.parent / "quarantine"
    if root.is_symlink():
        raise ArtifactIntegrityError("Quarantine symlinks are forbidden")
    root.mkdir(mode=0o700, exist_ok=True)
    target = root / f"{path.name}-{uuid.uuid4().hex[:8]}"
    path.rename(target)
    _sync_directory(path.parent)
    _sync_directory(root)
    LOGGER.error("CHECKPOINT QUARANTINED %s -> %s: %s", path.name, target, reason)
    return target


def committed_steps(directory: Path) -> list[tuple[int, Path]]:
    """Committed step directories, newest first, from names (no hashing)."""
    found = []
    if (directory / "checkpoints").is_symlink():
        raise ArtifactIntegrityError("Checkpoint root may not be a symlink")
    for path in (directory / "checkpoints").glob("step-*"):
        if path.is_symlink():
            raise ArtifactIntegrityError("Checkpoint symlinks are forbidden")
        match = _STEP.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            if (path / "manifest.json").exists():
                found.append((int(match.group(1)), path))
    return sorted(found, reverse=True)


def select_valid_checkpoint(
    directory: Path, *, before: int | None = None
) -> Path | None:
    """Newest checkpoint whose manifest and every file hash verify.

    Incomplete (manifest-less) and corrupt newer checkpoints are quarantined with
    a loud error log, then the next older one is tried. Returns None if none is
    valid. Semantic (recipe/software) mismatches are *not* handled here: those
    still fail closed in load_checkpoint.
    """
    committed_steps(directory)  # Reject symlinks before any quarantine mutation.
    for path in (directory / "checkpoints").glob("step-*"):
        if (
            _STEP.fullmatch(path.name)
            and path.is_dir()
            and not (path / "manifest.json").exists()
        ):
            if before is None:
                quarantine(path, "incomplete checkpoint (no manifest)")
    for step, path in committed_steps(directory):
        if before is not None and step >= before:
            continue
        try:
            manifest = ArtifactStore(path).read_manifest()
            if (
                not isinstance(manifest.metadata, CheckpointInfo)
                or manifest.metadata.optimizer_step != step
            ):
                raise ArtifactIntegrityError("Checkpoint step/metadata mismatch")
        except (ArtifactError, OSError, ValueError) as error:
            quarantine(path, f"{type(error).__name__}: {error}")
            continue
        return path
    return None


def prune_checkpoints(directory: Path, keep: int) -> list[Path]:
    """Delete committed checkpoints beyond the newest ``keep`` (0 keeps all)."""
    if keep <= 0:
        return []
    removed = []
    for _, path in committed_steps(directory)[keep:]:
        (path / "manifest.json").unlink()  # uncommit first, then remove payload
        _sync_directory(path)
        shutil.rmtree(path)
        removed.append(path)
    return removed


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
    *,
    application_validator: Callable[[dict[str, Any], int, int], None] | None = None,
    checkpoint: Path | None = None,
    repository: Path | None = None,
) -> int:
    if job.training.regression_recipe is not None and repository is None:
        raise ValueError("Checkpoint regression resume requires the repository")
    if checkpoint is None:
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
    data_sha, tokenizer_sha = _run_lineage(ArtifactStore(directory), run)
    if (
        metadata.data_sha256 != data_sha
        or metadata.tokenizer_sha256 != tokenizer_sha
        or manifest.owner != run.owner
    ):
        raise ArtifactIntegrityError("Checkpoint data/tokenizer/owner lineage mismatch")
    sampler = json.loads(
        _read(store, "sampler.json", 1024 * 1024),
        object_pairs_hook=_unique_mapping,
        parse_constant=_reject_constant,
    )
    schedule = json.loads(
        _read(store, "scheduler.json", 1024 * 1024),
        object_pairs_hook=_unique_mapping,
        parse_constant=_reject_constant,
    )
    has_application = sampler.get("application", False)
    if type(has_application) is not bool or has_application != (
        application_validator is not None
    ):
        raise ArtifactIntegrityError(
            "Checkpoint requires its matching application-state loader"
        )
    if has_application:

        def validate_application() -> None:
            state = json.loads(
                _read(store, f"application-rank-{context.rank:05d}.json", 1024 * 1024),
                object_pairs_hook=_unique_mapping,
                parse_constant=_reject_constant,
            )
            if not isinstance(state, dict):
                raise ArtifactIntegrityError("Invalid checkpoint application state")
            assert application_validator is not None
            application_validator(
                state, metadata.optimizer_step, metadata.processed_tokens
            )

        context.collective(validate_application, "Invalid checkpoint application state")
    expected = (
        metadata.processed_tokens
        if has_application
        else (
            metadata.optimizer_step
            * job.training.batch_size
            * job.training.sequence_length
            * context.world_size
        )
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
        or metadata.precision != job.experiment.runtime.precision
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
    if job.training.regression_recipe is not None:
        from qvelt.eval.regression import evaluate_checkpoint

        assert repository is not None
        evaluate_checkpoint(directory, metadata.optimizer_step, job, repository)
    return metadata.optimizer_step
