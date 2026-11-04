"""Prepare licensed conversational raw bundles as family-isolated SFT shards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qvelt.data.config import DataRecipe, MixStage, SourceSpec, dump_data_recipe
from qvelt.data.pipeline import document_split
from qvelt.data.records import iter_records
from qvelt.data.shards import ShardWriter
from qvelt.posttrain.chat import encode_chat, parse_messages
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.manifest_types import DataInfo, Manifest
from qvelt.utils.manifests import _reject_constant, _unique_mapping, manifest_bytes


def prepare_sft(
    source: Path,
    source_sha256: str,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    directory: Path,
    *,
    context: int = 128,
    approved_for_training: bool = False,
) -> Manifest:
    """Demo-scope preparation only; no production contamination clearance implied."""
    raw_store = ArtifactStore(source.parent)
    raw = raw_store.read_manifest(source.name, expected_sha256=source_sha256)
    if not isinstance(raw.metadata, DataInfo) or raw.metadata.representation != "raw":
        raise ValueError("SFT requires a licensed raw conversation bundle")
    licenses = [f for f in raw.files if f.role == "license"]
    policies = [f for f in raw.files if f.role == "usage_policy"]
    if not licenses or len(policies) != 1 or policies[0].size_bytes > 4096:
        raise ValueError("SFT requires source license and explicit usage policy")
    with raw_store.open_file(policies[0].path) as stream:
        policy = json.loads(
            stream.read(4097),
            object_pairs_hook=_unique_mapping,
            parse_constant=_reject_constant,
        )
    if (
        not isinstance(policy, dict)
        or set(policy) != {"allowed_uses"}
        or not isinstance(policy["allowed_uses"], list)
        or "training" not in policy["allowed_uses"]
    ):
        raise ValueError("Source does not permit training")
    if approved_for_training is not True:
        raise ValueError(
            "SFT source requires explicit training approval recorded in DATA_CARD"
        )
    registry = (
        json.dumps(
            {"schema_version": 1, "purpose": "demo", "required": [], "benchmarks": []},
            sort_keys=True,
        )
        + "\n"
    ).encode()
    tokenizer, token_manifest = load_tokenizer_bundle(tokenizer_path, tokenizer_sha256)
    recipe = DataRecipe(
        (SourceSpec("supervision", str(source.resolve()), source_sha256, True, True),),
        "benchmarks.json",
        hashlib.sha256(registry).hexdigest(),
        (MixStage(0, (("supervision", 1.0),)),),
        context=context,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    store = ArtifactStore(create_directory(directory))
    files = [
        store.add_bytes("benchmarks.json", registry, role="benchmark_registry"),
        store.add_bytes(
            "source.manifest.json", manifest_bytes(raw), role="source_manifest"
        ),
        store.add_bytes(
            "recipe.yaml", dump_data_recipe(recipe).encode(), role="recipe"
        ),
    ]
    for index, record in enumerate(licenses):
        files.append(
            store.add_file(
                f"source-license-{index}.txt",
                source.parent / record.path,
                role="license",
            )
        )
    files.append(
        store.add_file(
            "usage.json", source.parent / policies[0].path, role="usage_policy"
        )
    )
    for record in token_manifest.files:
        files.append(
            store.add_file(
                "tokenizer/" + record.path,
                tokenizer_path.parent / record.path,
                role="tokenizer_bundle",
            )
        )
    files.append(
        store.add_bytes(
            "tokenizer/manifest.json",
            manifest_bytes(token_manifest),
            role="tokenizer_bundle",
        )
    )
    sources: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    total_documents = total_tokens = 0
    for split in ("train", "validation", "test"):
        writer = ShardWriter(
            store,
            f"shards/supervision/{split}",
            tokenizer.vocab_size,
            context,
            65536,
            whole_documents=True,
        )
        families: set[str] = set()
        targets = truncated_count = 0
        try:
            for index, document in enumerate(iter_records(source, source_sha256)):
                if not document.family:
                    raise ValueError("SFT requires explicit problem family metadata")
                key = hashlib.sha256(("family:" + document.family).encode()).hexdigest()
                if document_split(key, recipe) != split:
                    continue
                tokens, truncated = encode_chat(
                    parse_messages(document.text), tokenizer, context + 1
                )
                from qvelt.posttrain.chat import assistant_mask

                targets += sum(assistant_mask(tokens))
                truncated_count += truncated
                writer.add(
                    tokens,
                    index,
                    {
                        "id": document.identifier,
                        "family": document.family,
                        "truncated": truncated,
                        "language": document.language,
                    },
                )
                families.add(document.family)
                total_documents += 1
                total_tokens += len(tokens)
            if writer.documents:
                sources[split] = writer.finish()
                files.extend(writer.files)
            counts[split] = {
                "families": sorted(families),
                "assistant_targets": targets,
                "truncated": truncated_count,
                "documents": writer.documents,
            }
        finally:
            writer.close()
    if not {"train", "validation"} <= set(sources):
        raise ValueError("SFT requires nonempty disjoint train/validation families")
    layout = {
        "schema_version": 2,
        "context": context,
        "vocab": tokenizer.vocab_size,
        "control_version": tokenizer.control_version,
        "tokenizer_sha256": tokenizer_sha256,
        "sources": {"supervision": sources},
        "mixing": [asdict(s) for s in recipe.mixing],
    }
    report = {
        "scope": "demo",
        "production_ready": False,
        "objective": "assistant-v1",
        "chat_schema_version": 1,
        "counts": counts,
        "source_sha256": source_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "limitations": [
            "Family isolation is not a public benchmark contamination audit",
            "Right truncation may shorten an assistant response; outer EOS is masked",
        ],
    }
    for name, value in (("layout.json", layout), ("report.json", report)):
        files.append(
            store.add_bytes(
                name,
                (json.dumps(value, sort_keys=True) + "\n").encode(),
                role="packing_layout" if name == "layout.json" else "curation_report",
            )
        )
    manifest = Manifest(
        "data",
        "sft-data",
        datetime.now(UTC).isoformat(),
        DataInfo(
            "token_ids",
            raw.metadata.sources,
            total_documents,
            total_tokens,
            tokenizer_sha256,
        ),
        tuple(files),
    )
    store.publish(manifest)
    return manifest
