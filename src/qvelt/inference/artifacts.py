"""Bounded data-only CPU checkpoint loading and self-contained native snapshots."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load as load_tensors

from qvelt.config import ExperimentConfig, config_digest, dump_config, parse_config
from qvelt.config.training import parse_training_job
from qvelt.eval.config import ArtifactRef
from qvelt.model.decoder import Decoder
from qvelt.tokenizer.core import QveltTokenizer
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.manifest_types import CheckpointInfo, Manifest, RunInfo, TokenizerInfo
from qvelt.utils.manifests import (
    _reject_constant,
    _unique_mapping,
    manifest_bytes,
    parse_manifest,
)
from qvelt.utils.provenance import capture_code, capture_environment


def checked_payload(
    store: ArtifactStore, manifest: Manifest, role: str, limit: int
) -> bytes:
    records = [f for f in manifest.files if f.role == role]
    if len(records) != 1 or records[0].size_bytes > limit:
        raise ValueError("Model artifact requires one bounded " + role + " payload")
    record = records[0]
    with store.open_file(record.path) as stream:
        data = stream.read(limit + 1)
    if (
        len(data) != record.size_bytes
        or hashlib.sha256(data).hexdigest() != record.sha256
    ):
        raise ValueError("Model payload changed during read")
    return data


def _model(config: ExperimentConfig, data: bytes) -> Decoder:
    if config.model.parameter_count > 20_000_000:
        raise ValueError("Model exceeds CPU inference capacity")
    weights = load_tensors(data)
    if any(
        t.dtype != torch.float32 or not bool(torch.isfinite(t).all())
        for t in weights.values()
    ):
        raise ValueError("CPU reference requires finite FP32 weights")
    with (
        torch.inference_mode(False),
        torch.random.fork_rng(devices=[]),
        torch.device("cpu"),
    ):
        model = Decoder(config.model).float().eval()
        model.load_state_dict(weights, strict=True)
    return model


def load_checkpoint_model(
    checkpoint_ref: ArtifactRef, tokenizer_sha: str
) -> tuple[Decoder, ExperimentConfig, Manifest]:
    path = Path(checkpoint_ref.path)
    store = ArtifactStore(path.parent)
    checkpoint = store.read_manifest(
        path.name, expected_sha256=checkpoint_ref.sha256, verify=False
    )
    if (
        not isinstance(checkpoint.metadata, CheckpointInfo)
        or checkpoint.metadata.world_size != 1
        or checkpoint.metadata.tokenizer_sha256 != tokenizer_sha
        or sum(f.size_bytes for f in checkpoint.files) > 512 * 1024**2
    ):
        raise ValueError(
            "Model needs a bounded single-rank checkpoint and matching tokenizer"
        )
    store.verify(checkpoint)
    training = parse_training_job(
        checked_payload(store, checkpoint, "training_recipe", 128 * 1024).decode()
    )
    config = training.experiment
    if config_digest(config) != checkpoint.metadata.config_sha256:
        raise ValueError("Checkpoint recipe/config disagree")
    # Reject oversized models before parsing tensor storage or allocating a decoder.
    if config.model.parameter_count > 20_000_000:
        raise ValueError("Model exceeds CPU inference capacity")
    model = _model(config, checked_payload(store, checkpoint, "model", 128 * 1024**2))
    return model, config, checkpoint


@dataclass(frozen=True)
class InferenceSnapshot:
    model: Decoder
    tokenizer: QveltTokenizer
    config: ExperimentConfig
    manifest: Manifest


def export_snapshot(
    checkpoint: ArtifactRef,
    tokenizer_ref: ArtifactRef,
    directory: Path,
    repository: Path,
) -> Manifest:
    _, config, source = load_checkpoint_model(checkpoint, tokenizer_ref.sha256)
    token_path = Path(tokenizer_ref.path)
    token_store = ArtifactStore(token_path.parent)
    token_meta = token_store.read_manifest(
        token_path.name, expected_sha256=tokenizer_ref.sha256, verify=False
    )
    if sum(f.size_bytes for f in token_meta.files) > 64 * 1024**2:
        raise ValueError("Tokenizer bundle exceeds export capacity")
    tokenizer, token_meta = load_tokenizer_bundle(token_path, tokenizer_ref.sha256)
    if tokenizer.vocab_size != config.model.vocab:
        raise ValueError("Model/tokenizer vocabularies disagree")
    cp_store = ArtifactStore(Path(checkpoint.path).parent)
    config = replace(
        config, runtime=replace(config.runtime, device="cpu", precision="fp32")
    )
    descriptor = {
        "schema_version": 1,
        "format": "qvelt-inference-v1",
        "device": "cpu",
        "dtype": "float32",
        "checkpoint_sha256": checkpoint.sha256,
        "tokenizer_sha256": tokenizer_ref.sha256,
        "source_config_sha256": source.metadata.config_sha256
        if isinstance(source.metadata, CheckpointInfo)
        else None,
        "parameters": config.model.parameter_count,
        "release_status": "engineering_snapshot_not_model_release",
    }
    root, code = capture_code(repository)
    environment = capture_environment()
    owner = uuid.uuid4().hex
    store = ArtifactStore(create_directory(directory))
    files = [
        store.add_bytes("config.yaml", dump_config(config).encode(), role="config"),
        store.add_file("uv.lock", root / "uv.lock", role="dependency_lock"),
        store.add_bytes(
            "model.safetensors",
            checked_payload(cp_store, source, "model", 128 * 1024**2),
            role="model",
        ),
        store.add_bytes(
            "tokenizer.json",
            checked_payload(token_store, token_meta, "tokenizer", 64 * 1024**2),
            role="tokenizer",
        ),
        store.add_bytes(
            "source.config.yaml",
            checked_payload(cp_store, source, "config", 128 * 1024),
            role="source_config",
        ),
        store.add_bytes(
            "source.checkpoint.json",
            manifest_bytes(source),
            role="source_checkpoint_manifest",
        ),
        store.add_bytes(
            "source.tokenizer.json",
            manifest_bytes(token_meta),
            role="source_tokenizer_manifest",
        ),
        store.add_bytes(
            "inference.json",
            (json.dumps(descriptor, sort_keys=True) + "\n").encode(),
            role="inference_descriptor",
        ),
        store.add_file("LICENSE", root / "LICENSE", role="code_license"),
        store.add_file("NOTICE", root / "NOTICE", role="code_notice"),
    ]
    manifest = Manifest(
        kind="run",
        owner=owner,
        created_at=datetime.now(UTC).isoformat(),
        metadata=RunInfo(
            owner, config_digest(config), code, environment, config.runtime.seed, 1
        ),
        files=tuple(files),
    )
    store.publish(manifest)
    return manifest


def load_snapshot(reference: ArtifactRef) -> InferenceSnapshot:
    path = Path(reference.path)
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(
        path.name, expected_sha256=reference.sha256, verify=False
    )
    required = {
        "config",
        "dependency_lock",
        "model",
        "tokenizer",
        "source_config",
        "source_checkpoint_manifest",
        "source_tokenizer_manifest",
        "inference_descriptor",
        "code_license",
        "code_notice",
    }
    if (
        not isinstance(manifest.metadata, RunInfo)
        or len(manifest.files) != len(required)
        or {f.role for f in manifest.files} != required
        or sum(f.size_bytes for f in manifest.files) > 256 * 1024**2
    ):
        raise ValueError("Invalid bounded inference snapshot roles/type/capacity")
    store.verify(manifest)
    descriptor = json.loads(
        checked_payload(store, manifest, "inference_descriptor", 4096),
        object_pairs_hook=_unique_mapping,
        parse_constant=_reject_constant,
    )
    keys = {
        "schema_version",
        "format",
        "device",
        "dtype",
        "checkpoint_sha256",
        "tokenizer_sha256",
        "source_config_sha256",
        "parameters",
        "release_status",
    }
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != keys
        or type(descriptor["schema_version"]) is not int
        or descriptor["schema_version"] != 1
        or descriptor["format"] != "qvelt-inference-v1"
        or descriptor["device"] != "cpu"
        or descriptor["dtype"] != "float32"
        or descriptor["release_status"] != "engineering_snapshot_not_model_release"
    ):
        raise ValueError("Unsupported inference snapshot protocol/device/dtype")
    source = parse_manifest(
        checked_payload(store, manifest, "source_checkpoint_manifest", 1024**2),
        expected_sha256=descriptor["checkpoint_sha256"],
    )
    token_meta = parse_manifest(
        checked_payload(store, manifest, "source_tokenizer_manifest", 1024**2),
        expected_sha256=descriptor["tokenizer_sha256"],
    )
    original = parse_config(
        checked_payload(store, manifest, "source_config", 128 * 1024).decode()
    )
    config = parse_config(
        checked_payload(store, manifest, "config", 128 * 1024).decode()
    )
    if (
        not isinstance(source.metadata, CheckpointInfo)
        or not isinstance(token_meta.metadata, TokenizerInfo)
        or source.metadata.world_size != 1
        or original.runtime.precision != source.metadata.precision
        or source.metadata.tokenizer_sha256 != descriptor["tokenizer_sha256"]
        or source.metadata.config_sha256 != descriptor["source_config_sha256"]
        or config_digest(original) != source.metadata.config_sha256
        or replace(
            original, runtime=replace(original.runtime, device="cpu", precision="fp32")
        )
        != config
        or type(descriptor["parameters"]) is not int
        or descriptor["parameters"] != config.model.parameter_count
    ):
        raise ValueError(
            "Inference snapshot source/config/tokenizer identities disagree"
        )
    for role, origin in (
        ("model", source),
        ("tokenizer", token_meta),
        ("source_config", source),
    ):
        actual = next(f for f in manifest.files if f.role == role)
        expected = [
            f
            for f in origin.files
            if f.role == ("config" if role == "source_config" else role)
        ]
        if (
            len(expected) != 1
            or actual.sha256 != expected[0].sha256
            or actual.size_bytes != expected[0].size_bytes
        ):
            raise ValueError("Inference snapshot payload differs from its source")
    tokenizer = QveltTokenizer.from_bytes(
        checked_payload(store, manifest, "tokenizer", 64 * 1024**2)
    )
    if (
        tokenizer.vocab_size != config.model.vocab
        or tokenizer.vocab_size != token_meta.metadata.vocab_size
        or token_meta.metadata.reserved_tokens != 32
    ):
        raise ValueError("Inference tokenizer vocabulary/control policy mismatch")
    model = _model(config, checked_payload(store, manifest, "model", 128 * 1024**2))
    return InferenceSnapshot(model, tokenizer, config, manifest)
