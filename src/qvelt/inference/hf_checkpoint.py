"""Streaming builtin HF export of dense checkpoints,without the CPU decoder size cap."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from qvelt.config import ModelConfig, config_digest
from qvelt.config.training import parse_training_job
from qvelt.eval.config import ArtifactRef
from qvelt.inference.artifacts import checked_payload
from qvelt.inference.huggingface import _write_export, split_rotary_rows
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, hash_file
from qvelt.utils.manifest_types import CheckpointInfo, Manifest


def weight_layout(config: ModelConfig) -> dict[str, tuple[str, tuple[int, ...], int]]:
    """Native name -> HF name,expected shape,rotary head count (zero for no rotation)."""
    d, k, f = (
        config.hidden,
        config.kv_heads * (config.hidden // config.heads),
        config.ffn,
    )
    layout = {
        "embedding.weight": ("model.embed_tokens.weight", (config.vocab, d), 0),
        "norm.weight": ("model.norm.weight", (d,), 0),
    }
    for i in range(config.layers):
        for source, target, shape, heads in (
            ("attention_norm.weight", "input_layernorm.weight", (d,), 0),
            ("ffn_norm.weight", "post_attention_layernorm.weight", (d,), 0),
            ("attention.query.weight", "self_attn.q_proj.weight", (d, d), config.heads),
            (
                "attention.key.weight",
                "self_attn.k_proj.weight",
                (k, d),
                config.kv_heads,
            ),
            ("attention.value.weight", "self_attn.v_proj.weight", (k, d), 0),
            ("attention.output.weight", "self_attn.o_proj.weight", (d, d), 0),
            ("ffn.gate.weight", "mlp.gate_proj.weight", (f, d), 0),
            ("ffn.up.weight", "mlp.up_proj.weight", (f, d), 0),
            ("ffn.down.weight", "mlp.down_proj.weight", (d, f), 0),
        ):
            layout[f"blocks.{i}.{source}"] = (
                f"model.layers.{i}.{target}",
                shape,
                heads,
            )
    return layout


def export_checkpoint_huggingface(
    checkpoint: ArtifactRef,
    tokenizer_ref: ArtifactRef,
    directory: Path,
    repository: Path,
    *,
    context_extension: int = 1,
    shard_bytes: int = 128 * 1024**2,
) -> Manifest:
    """Export finite FP32 dense single/DDP weights;tensorwise conversion,never load pickle.

    DCP/FSDP shards require explicit consolidation before this command. The cap
    admits RFC tiers through large;execution of large exports needs corresponding disk/RAM.
    """
    if type(context_extension) is not int or context_extension not in (1, 2, 4):
        raise ValueError("HF context factors are1/2/4")
    if type(shard_bytes) is not int or not 1024 <= shard_bytes <= 512 * 1024**2:
        raise ValueError("HF shard budget must be1KiB..512MiB")
    path = Path(checkpoint.path)
    store = ArtifactStore(path.parent)
    source = store.read_manifest(
        path.name, expected_sha256=checkpoint.sha256, verify=False
    )
    if (
        not isinstance(source.metadata, CheckpointInfo)
        or source.metadata.tokenizer_sha256 != tokenizer_ref.sha256
        or sum(record.size_bytes for record in source.files) > 32 * 1024**3
    ):
        raise ValueError("Checkpoint type,tokenizer or capacity invalid")
    records = [record for record in source.files if record.role == "model"]
    if (
        len(records) != 1
        or not records[0].path.endswith(".safetensors")
        or records[0].size_bytes > 8 * 1024**3
    ):
        raise ValueError(
            "HF export requires one dense safetensors model;consolidate DCP explicitly"
        )
    store.verify(source)
    config = parse_training_job(
        checked_payload(store, source, "training_recipe", 128 * 1024).decode()
    ).experiment
    if (
        config_digest(config) != source.metadata.config_sha256
        or config.model.parameter_count > 2_000_000_000
    ):
        raise ValueError("Checkpoint config mismatch or export capacity exceeded")
    token_path = Path(tokenizer_ref.path)
    token_store = ArtifactStore(token_path.parent)
    token_manifest = token_store.read_manifest(
        token_path.name, expected_sha256=tokenizer_ref.sha256, verify=False
    )
    if sum(record.size_bytes for record in token_manifest.files) > 64 * 1024**2:
        raise ValueError("Tokenizer exceeds export capacity")
    tokenizer, _ = load_tokenizer_bundle(token_path, tokenizer_ref.sha256)
    if tokenizer.vocab_size != config.model.vocab:
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")
    layout = weight_layout(config.model)
    weight_path = path.parent / records[0].path

    def weights() -> Iterator[tuple[str, Tensor]]:
        with safe_open(str(weight_path), framework="pt", device="cpu") as reader:  # type: ignore[no-untyped-call]
            if set(reader.keys()) != set(layout):
                raise ValueError(
                    "Checkpoint keys differ;unmerged adapters are not exportable"
                )
            # Validate every shape before materializing any tensor.
            for name, (_, shape, _) in layout.items():
                view = reader.get_slice(name)
                if tuple(view.get_shape()) != shape or view.get_dtype() != "F32":
                    raise ValueError(
                        "Checkpoint shape/dtype differs from the RFC dense layout"
                    )
            for name, (target, _, heads) in layout.items():
                tensor = reader.get_tensor(name)
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("HF export requires finite weights")
                yield target, split_rotary_rows(tensor, heads) if heads else tensor
        observed = hash_file(weight_path)
        if (
            observed.sha256 != records[0].sha256
            or observed.size_bytes != records[0].size_bytes
        ):
            raise ValueError("Checkpoint changed during conversion")

    config = replace(
        config,
        model=replace(
            config.model,
            context=config.model.context * context_extension,
            rope_scale=config.model.rope_scale * context_extension,
        ),
    )
    return _write_export(
        config,
        tokenizer,
        source,
        checkpoint,
        directory,
        repository,
        weights(),
        context_extension,
        shard_bytes=shard_bytes,
    )
