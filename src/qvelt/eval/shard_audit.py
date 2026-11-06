"""Decode checksummed train-only document frames, without joining packed documents."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from qvelt.data.loader import PackedDataset
from qvelt.data.records import RawDocument
from qvelt.data.shards import token_dtype
from qvelt.eval.config import ArtifactRef
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifest_types import DataInfo, Manifest
from qvelt.utils.manifests import _reject_constant, _unique_mapping


def _json(data: bytes) -> Any:
    return json.loads(
        data, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
    )


def shard_documents(
    reference: ArtifactRef,
) -> tuple[Manifest, Iterator[RawDocument], dict[str, Any]]:
    path = Path(reference.path)
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(
        path.name, expected_sha256=reference.sha256, verify=False
    )
    if (
        not isinstance(manifest.metadata, DataInfo)
        or manifest.metadata.representation != "token_ids"
        or manifest.metadata.documents > 100000
        or sum(f.size_bytes for f in manifest.files) > 128 * 1024**2
    ):
        raise ValueError("Shard audit requires bounded prepared token data")
    store.verify(manifest)
    policies = [f for f in manifest.files if f.role == "usage_policy"]
    if not policies or not any(f.role == "license" for f in manifest.files):
        raise ValueError(
            "Shard audit needs explicit copied training/decontamination permission and license"
        )
    for record in policies:
        if record.size_bytes > 4096:
            raise ValueError("Shard audit usage policy exceeds limit")
        with store.open_file(record.path) as stream:
            policy = _json(stream.read(4097))
        if (
            not isinstance(policy, dict)
            or set(policy) != {"allowed_uses"}
            or not isinstance(policy["allowed_uses"], list)
            or any(not isinstance(use, str) for use in policy["allowed_uses"])
            or not {"training", "decontamination"} <= set(policy["allowed_uses"])
        ):
            raise ValueError(
                "Source does not permit training-shard decontamination audit"
            )
    tokenizer, _ = load_tokenizer_bundle(
        path.parent / "tokenizer/manifest.json", manifest.metadata.tokenizer_sha256
    )
    if not any(f.path == "layout.json" for f in manifest.files):
        from qvelt.posttrain.preferences import load_preferences

        pair_record = next((f for f in manifest.files if f.path == "pairs.json"), None)
        if pair_record is None or pair_record.size_bytes > 64 * 1024**2:
            raise ValueError("Unsupported tokenized training source format")
        with store.open_file(pair_record.path) as stream:
            pair_body = _json(stream.read(64 * 1024**2 + 1))
        if not isinstance(pair_body, dict) or type(pair_body.get("context")) is not int:
            raise ValueError("Invalid preference context")
        _, _, pairs = load_preferences(
            path, reference.sha256, tokenizer.vocab_size, pair_body["context"]
        )
        coverage_pairs: dict[str, Any] = {
            "representation": "token_ids",
            "storage_format": "encoded_preferences",
            "split": "train",
            "documents": 0,
            "tokens": 0,
            "tokenizer_sha256": manifest.metadata.tokenizer_sha256,
            "document_boundaries_verified": False,
        }

        def preference_documents() -> Iterator[RawDocument]:
            for pair in pairs:
                if pair.split != "train":
                    continue
                for label, frame in (
                    ("chosen", pair.chosen),
                    ("rejected", pair.rejected),
                ):
                    coverage_pairs["documents"] += 1
                    coverage_pairs["tokens"] += len(frame)
                    yield RawDocument(
                        tokenizer.decode(frame[1:-1]),
                        pair.identifier + "-" + label,
                        pair.family,
                    )
            coverage_pairs["document_boundaries_verified"] = True

        return manifest, preference_documents(), coverage_pairs
    dataset = PackedDataset(path, "train", reference.sha256)
    if dataset.vocab != tokenizer.vocab_size:
        raise ValueError("Shard audit tokenizer vocabulary mismatch")
    coverage: dict[str, Any] = {
        "representation": "token_ids",
        "split": "train",
        "documents": 0,
        "tokens": 0,
        "shard_bytes": 0,
        "tokenizer_sha256": manifest.metadata.tokenizer_sha256,
        "document_boundaries_verified": False,
    }

    def documents() -> Iterator[RawDocument]:
        for source in dataset.sources.values():
            cursors = [0] * len(source["shards"])
            count = 0
            last_document = -1
            with store.open_file(source["document_metadata"]) as stream:
                while line := stream.readline(65537):
                    if len(line) > 65536:
                        raise ValueError("Shard document metadata exceeds line limit")
                    row = _json(line)
                    if not isinstance(row, dict) or not {
                        "document",
                        "shard",
                        "start",
                        "tokens",
                    } <= set(row):
                        raise ValueError("Invalid shard document coordinates")
                    for key in ("document", "shard", "start", "tokens"):
                        if type(row[key]) is not int or row[key] < 0:
                            raise ValueError("Invalid shard coordinate integer")
                    shard_index, start, length = (
                        row["shard"],
                        row["start"],
                        row["tokens"],
                    )
                    if (
                        shard_index >= len(cursors)
                        or start != cursors[shard_index]
                        or not 2 <= length <= 131072
                        or row["document"] <= last_document
                    ):
                        raise ValueError(
                            "Shard frames must be bounded,ordered and cover storage exactly"
                        )
                    shard = source["shards"][shard_index]
                    if start + length > shard["tokens"]:
                        raise ValueError("Document exceeds token shard")
                    dtype = token_dtype(dataset.vocab)
                    with store.open_file(shard["path"]) as tokens_file:
                        tokens_file.seek(start * dtype.itemsize)
                        raw = tokens_file.read(length * dtype.itemsize)
                    if len(raw) != length * dtype.itemsize:
                        raise ValueError("Short shard read")
                    tokens = np.frombuffer(raw, dtype=dtype).astype(np.int64).tolist()
                    if (
                        tokens[0] != 1
                        or tokens[-1] != 2
                        or any(
                            t in (0, 1, 2) or t >= dataset.vocab for t in tokens[1:-1]
                        )
                    ):
                        raise ValueError("Invalid stored document frame")
                    cursors[shard_index] += length
                    last_document = row["document"]
                    count += 1
                    coverage["documents"] += 1
                    coverage["tokens"] += length
                    yield RawDocument(
                        tokenizer.decode(tokens[1:-1]),
                        str(row.get("id", row["document"])),
                        str(row.get("family", "")),
                    )
            if count != source["documents"] or any(
                c != s["tokens"] for c, s in zip(cursors, source["shards"], strict=True)
            ):
                raise ValueError("Incomplete/duplicated train document coverage")
            coverage["shard_bytes"] += sum(
                s["tokens"] * token_dtype(dataset.vocab).itemsize
                for s in source["shards"]
            )
        coverage["document_boundaries_verified"] = True

    return manifest, documents(), coverage
