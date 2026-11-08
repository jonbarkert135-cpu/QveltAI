"""Data-only builtin Llama export with explicit adjacent-to-split RoPE conversion."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from safetensors.torch import save
from torch import Tensor

from qvelt.config import ExperimentConfig, config_digest, dump_config
from qvelt.eval.config import ArtifactRef
from qvelt.inference.artifacts import InferenceSnapshot, load_snapshot
from qvelt.tokenizer.core import QveltTokenizer
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.manifest_types import Manifest, RunInfo
from qvelt.utils.manifests import manifest_bytes
from qvelt.utils.provenance import capture_code, capture_environment


def split_rotary_rows(weight: Tensor, heads: int) -> Tensor:
    """[even0,odd0,even1,odd1,...] -> [even0,even1,...,odd0,odd1,...]."""
    if (
        weight.ndim != 2
        or type(heads) is not int
        or heads < 1
        or weight.shape[0] % (2 * heads)
    ):
        raise ValueError("Projection requires complete even rotary heads")
    return (
        weight.reshape(heads, -1, 2, weight.shape[1])
        .transpose(1, 2)
        .reshape_as(weight)
        .contiguous()
    )


def llama_weights(snapshot: InferenceSnapshot) -> dict[str, Tensor]:
    model = snapshot.model
    config = model.config
    result = {
        "model.embed_tokens.weight": model.embedding.weight.detach().contiguous(),
        "model.norm.weight": model.norm.weight.detach().contiguous(),
    }
    for i in range(config.layers):
        native = f"blocks.{i}."
        exported = f"model.layers.{i}."
        mapping = {
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "attention.query.weight": "self_attn.q_proj.weight",
            "attention.key.weight": "self_attn.k_proj.weight",
            "attention.value.weight": "self_attn.v_proj.weight",
            "attention.output.weight": "self_attn.o_proj.weight",
            "ffn.gate.weight": "mlp.gate_proj.weight",
            "ffn.up.weight": "mlp.up_proj.weight",
            "ffn.down.weight": "mlp.down_proj.weight",
        }
        state = model.state_dict()
        for source, target in mapping.items():
            tensor = state[native + source].detach()
            if source == "attention.query.weight":
                tensor = split_rotary_rows(tensor, config.heads)
            elif source == "attention.key.weight":
                tensor = split_rotary_rows(tensor, config.kv_heads)
            result[exported + target] = tensor.contiguous()
    return result


def chat_template(controls: tuple[str, ...], version: int) -> str:
    """Conventional HF special parsing;validate the same explicit chat framing."""
    if version not in (1, 2):
        raise ValueError("Tokenizer v3 chat export awaits schema3 validation")
    return (
        "{%- set controls = " + json.dumps(controls) + " -%}"
        "{%- set reasoning_allowed = "
        + ("true" if version == 2 else "false")
        + " -%}"
        + """
{%- if not messages or messages|length > 64 -%}{{- raise_exception('Chat message count invalid') -}}{%- endif -%}
{{- bos_token -}}{%- set ns = namespace(expected='user') -%}
{%- for message in messages -%}
{%- set role = message['role'] -%}
{%- if role not in ['system','user','assistant'] -%}{{- raise_exception('Unsupported role') -}}{%- endif -%}
{%- if role == 'system' -%}
{%- if not loop.first -%}{{- raise_exception('System must be first') -}}{%- endif -%}
{%- else -%}
{%- if role != ns.expected -%}{{- raise_exception('Roles must alternate') -}}{%- endif -%}
{%- set ns.expected = 'assistant' if role == 'user' else 'user' -%}
{%- endif -%}
{%- if not message['content'].strip() -%}{{- raise_exception('Empty message') -}}{%- endif -%}
{%- for control in controls -%}
{%- if control in message['content'] or control in (message.get('reasoning') or '') -%}
{{- raise_exception('Reserved control spelling in chat content') -}}{%- endif -%}
{%- endfor -%}
{{- '<|' + role + '|>' -}}
{%- if message.get('reasoning') is not none -%}
{%- if not reasoning_allowed or role != 'assistant' or not message['reasoning'].strip() -%}
{{- raise_exception('Invalid reasoning field/tokenizer') -}}{%- endif -%}
{{- '<think>' + message['reasoning'] + '</think>' -}}
{%- endif -%}
{{- message['content'] + '<|end_turn|>' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{%- if messages[-1]['role'] != 'user' -%}{{- raise_exception('Generation must follow user') -}}{%- endif -%}
{{- '<|assistant|>' -}}
{%- else -%}
{%- if messages[-1]['role'] != 'assistant' -%}{{- raise_exception('Closed chat must end with assistant') -}}{%- endif -%}
{{- eos_token -}}
{%- endif -%}
"""
    )


def export_huggingface(
    reference: ArtifactRef,
    directory: Path,
    repository: Path,
    *,
    context_extension: int = 1,
) -> Manifest:
    snapshot = load_snapshot(reference)
    if context_extension != 1:
        from qvelt.inference.long_context import extend_context

        model_copy = extend_context(snapshot.model, context_extension)
        snapshot = replace(
            snapshot,
            model=model_copy,
            config=replace(snapshot.config, model=model_copy.config),
        )
    return _write_export(
        snapshot.config,
        snapshot.tokenizer,
        snapshot.manifest,
        reference,
        directory,
        repository,
        llama_weights(snapshot).items(),
        context_extension,
    )


def _write_export(
    native_config: ExperimentConfig,
    tokenizer: QveltTokenizer,
    source: Manifest,
    reference: ArtifactRef,
    directory: Path,
    repository: Path,
    weights: Iterable[tuple[str, Tensor]],
    context_extension: int,
    *,
    shard_bytes: int = 128 * 1024**2,
) -> Manifest:
    model = native_config.model
    rope: dict[str, object] = {"rope_theta": model.rope_theta, "rope_type": "default"}
    if model.rope_scale != 1:
        rope.update(rope_type="linear", factor=model.rope_scale)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": model.vocab,
        "hidden_size": model.hidden,
        "intermediate_size": model.ffn,
        "num_hidden_layers": model.layers,
        "num_attention_heads": model.heads,
        "num_key_value_heads": model.kv_heads,
        "max_position_embeddings": model.context,
        "rms_norm_eps": model.norm_eps,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "rope_parameters": rope,
        "rope_theta": model.rope_theta,
        "rope_scaling": None
        if model.rope_scale == 1
        else {"rope_type": "linear", "factor": model.rope_scale},
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "dtype": "float32",
        "use_cache": True,
    }
    token_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": model.context,
        "bos_token": tokenizer.control_tokens[1],
        "eos_token": tokenizer.control_tokens[2],
        "pad_token": tokenizer.control_tokens[0],
        "additional_special_tokens": list(tokenizer.control_tokens[3:]),
        "clean_up_tokenization_spaces": False,
        "padding_side": "left",
        "chat_template": chat_template(
            tokenizer.control_tokens, tokenizer.control_version
        ),
    }
    root, code = capture_code(repository)
    owner = uuid.uuid4().hex
    store = ArtifactStore(create_directory(directory))
    from tokenizers import Tokenizer, processors

    hf_tokenizer = Tokenizer.from_str(tokenizer.to_bytes().decode())
    hf_tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A", special_tokens=[("<|bos|>", 1)]
    )
    files = [
        store.add_bytes(
            "qvelt_config.yaml", dump_config(native_config).encode(), role="config"
        ),
    ]
    # Bound conversion memory by a shard plus the largest individual tensor.
    pending: dict[str, Tensor] = {}
    weight_map: dict[str, str] = {}
    total, current, number = 0, 0, 0

    def flush() -> None:
        nonlocal number, current
        if not pending:
            return
        number += 1
        name = f"model-{number:05d}.safetensors"
        files.append(
            store.add_bytes(
                name, save(pending, metadata={"format": "pt"}), role="hf_weights"
            )
        )
        weight_map.update({key: name for key in pending})
        pending.clear()
        current = 0

    for name, tensor in weights:
        size = tensor.numel() * tensor.element_size()
        if current and current + size > shard_bytes:
            flush()
        pending[name] = tensor
        current += size
        total += size
    flush()
    if not weight_map:
        raise ValueError("HF export requires weights")
    files.append(
        store.add_bytes(
            "model.safetensors.index.json",
            (
                json.dumps(
                    {"metadata": {"total_size": total}, "weight_map": weight_map},
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
            role="hf_weights_index",
        )
    )
    for name, body, role in (
        ("config.json", config, "hf_config"),
        ("tokenizer_config.json", token_config, "hf_tokenizer_config"),
        (
            "special_tokens_map.json",
            {
                k: token_config[k]
                for k in (
                    "bos_token",
                    "eos_token",
                    "pad_token",
                    "additional_special_tokens",
                )
            },
            "hf_special_tokens",
        ),
        (
            "generation_config.json",
            {
                "bos_token_id": 1,
                "eos_token_id": [2, 6],
                "pad_token_id": 0,
                "do_sample": False,
            },
            "hf_generation_config",
        ),
        (
            "conversion.json",
            {
                "format": "qvelt-hf-llama-v1",
                "source_manifest_sha256": reference.sha256,
                "qk_layout": "adjacent-to-split-rows",
                "context_extension": context_extension,
                "head": "tied-embedding",
                "release_status": "engineering_export_not_model_release",
            },
            "hf_conversion",
        ),
    ):
        files.append(
            store.add_bytes(
                name,
                (json.dumps(body, sort_keys=True, ensure_ascii=False) + "\n").encode(),
                role=role,
            )
        )
    files.extend(
        (
            store.add_bytes(
                "tokenizer.json", hf_tokenizer.to_str().encode(), role="hf_tokenizer"
            ),
            store.add_bytes(
                "source.manifest.json",
                manifest_bytes(source),
                role="source_model_manifest",
            ),
            store.add_file("uv.lock", root / "uv.lock", role="dependency_lock"),
            store.add_file("LICENSE", repository / "LICENSE", role="code_license"),
            store.add_file("NOTICE", repository / "NOTICE", role="code_notice"),
        )
    )
    manifest = Manifest(
        "run",
        owner,
        datetime.now(UTC).isoformat(),
        RunInfo(
            owner,
            config_digest(native_config),
            code,
            capture_environment(),
            native_config.runtime.seed,
            1,
        ),
        tuple(files),
    )
    store.publish(manifest)
    return manifest
