"""Licensed single-turn preference pairs, explicit token frames and family splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qvelt.data.records import iter_records
from qvelt.posttrain.chat import Message, assistant_mask, encode_chat
from qvelt.tokenizer.core import QveltTokenizer
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.manifest_types import ArtifactFile, DataInfo, Manifest
from qvelt.utils.manifests import (
    _reject_constant,
    _unique_mapping,
    manifest_bytes,
    manifest_digest,
)


def _read(text: str) -> Any:
    return json.loads(
        text, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
    )


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    family: str
    identifier: str

    def __post_init__(self) -> None:
        for name in ("prompt", "chosen", "rejected"):
            Message("user" if name == "prompt" else "assistant", getattr(self, name))
        for name in ("family", "identifier"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not 1 <= len(value) <= 512
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("Preference pairs need bounded identifiers/families")
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("Preference answers must differ")


def encode_pair(
    pair: PreferencePair, tokenizer: QveltTokenizer, context: int
) -> tuple[list[int], list[int]]:
    frames = []
    for response in (pair.chosen, pair.rejected):
        tokens, truncated = encode_chat(
            (Message("user", pair.prompt), Message("assistant", response)),
            tokenizer,
            context + 1,
        )
        if truncated:
            raise ValueError(
                "Preference truncation would change the decision; shorten the source or increase context"
            )
        frames.append(tokens)
    if frames[0] == frames[1]:
        raise ValueError("Tokenized preference responses must differ")
    return frames[0], frames[1]


def family_split(family: str) -> str:
    """Fixed seed20261105: split entire families before any optimization."""
    value = (
        int.from_bytes(
            hashlib.sha256(("20261105:family:" + family).encode()).digest()[:8], "big"
        )
        / 2**64
    )
    return "train" if value < 0.6 else "validation" if value < 0.8 else "test"


def _training_permission(
    store: ArtifactStore, manifest: Manifest
) -> list[ArtifactFile]:
    policies = [f for f in manifest.files if f.role == "usage_policy"]
    licenses = [f for f in manifest.files if f.role == "license"]
    if len(policies) != 1 or policies[0].size_bytes > 4096 or not licenses:
        raise ValueError("Preferences require license and explicit usage policy")
    with store.open_file(policies[0].path) as stream:
        policy = _read(stream.read(4097).decode())
    if (
        not isinstance(policy, dict)
        or set(policy) != {"allowed_uses"}
        or not isinstance(policy["allowed_uses"], list)
        or "training" not in policy["allowed_uses"]
    ):
        raise ValueError("Preference source does not permit training")
    return [*licenses, *policies]


def prepare_preferences(
    source: Path,
    source_sha: str,
    tokenizer_path: Path,
    tokenizer_sha: str,
    directory: Path,
    *,
    context: int = 128,
    approved_for_training: bool = False,
) -> Manifest:
    """Prepare bounded demo pairs; rights approval is explicit, never inferred."""
    if approved_for_training is not True or type(context) is not int or context < 4:
        raise ValueError(
            "Preference source needs explicit training approval and context"
        )
    raw_store = ArtifactStore(source.parent)
    raw = raw_store.read_manifest(source.name, expected_sha256=source_sha)
    if (
        not isinstance(raw.metadata, DataInfo)
        or raw.metadata.representation != "raw"
        or not 1 <= raw.metadata.documents <= 100000
    ):
        raise ValueError("Preferences need a bounded raw source")
    permissions = _training_permission(raw_store, raw)
    tokenizer, token_manifest = load_tokenizer_bundle(tokenizer_path, tokenizer_sha)
    rows = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for document in iter_records(source, source_sha):
        row = _read(document.text)
        if (
            not isinstance(row, dict)
            or set(row) != {"schema_version", "prompt", "chosen", "rejected"}
            or type(row["schema_version"]) is not int
            or row["schema_version"] != 1
        ):
            raise ValueError("Invalid preference document schema")
        pair = PreferencePair(
            row["prompt"],
            row["chosen"],
            row["rejected"],
            document.family,
            document.identifier,
        )
        if pair.identifier in seen_ids or pair.prompt in seen_prompts:
            raise ValueError("Duplicate or contradictory preference prompt/identifier")
        seen_ids.add(pair.identifier)
        seen_prompts.add(pair.prompt)
        chosen, rejected = encode_pair(pair, tokenizer, context)
        rows.append(
            {
                "identifier": pair.identifier,
                "family": pair.family,
                "split": family_split(pair.family),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    if not {"train", "validation"} <= {row["split"] for row in rows}:
        raise ValueError(
            "Preference fixtures require distinct train/validation families"
        )
    store = ArtifactStore(create_directory(directory))
    files = [
        store.add_bytes(
            "source.manifest.json", manifest_bytes(raw), role="source_manifest"
        ),
        store.add_bytes(
            "pairs.json",
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "scope": "demo",
                        "context": context,
                        "pairs": rows,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
            role="data",
        ),
    ]
    for index, record in enumerate(permissions):
        files.append(
            store.add_file(
                f"source/{index}-{record.path}",
                source.parent / record.path,
                role=record.role,
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
    tokens = sum(len(row[k]) - 1 for row in rows for k in ("chosen", "rejected"))
    manifest = Manifest(
        "data",
        "qvelt-preferences",
        "2026-11-05T00:00:00Z",
        DataInfo(
            "token_ids",
            raw.metadata.sources,
            len(rows),
            tokens,
            manifest_digest(token_manifest),
        ),
        tuple(files),
    )
    store.publish(manifest)
    return manifest


@dataclass(frozen=True)
class EncodedPair:
    identifier: str
    family: str
    split: str
    chosen: tuple[int, ...]
    rejected: tuple[int, ...]


def load_preferences(
    path: Path, expected_sha: str, vocab: int, context: int
) -> tuple[Manifest, Manifest, list[EncodedPair]]:
    """Verify every artifact byte and the response/prompt/frame semantics."""
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(path.name, expected_sha256=expected_sha)
    if (
        not isinstance(manifest.metadata, DataInfo)
        or manifest.metadata.representation != "token_ids"
    ):
        raise ValueError("Expected prepared token preference data")
    _training_permission(store, manifest)
    tokenizer, token_manifest = load_tokenizer_bundle(
        path.parent / "tokenizer/manifest.json", manifest.metadata.tokenizer_sha256
    )
    if tokenizer.vocab_size != vocab:
        raise ValueError("Preference tokenizer vocabulary mismatch")
    records = [f for f in manifest.files if f.role == "data"]
    if (
        len(records) != 1
        or records[0].path != "pairs.json"
        or records[0].size_bytes > 64 * 1024**2
    ):
        raise ValueError("Preference payload must be unique and at most64MiB")
    with store.open_file("pairs.json") as stream:
        body = _read(stream.read(64 * 1024**2 + 1).decode())
    if (
        not isinstance(body, dict)
        or set(body) != {"schema_version", "scope", "context", "pairs"}
        or type(body["schema_version"]) is not int
        or body["schema_version"] != 1
        or body["scope"] != "demo"
        or type(body["context"]) is not int
        or body["context"] != context
        or not isinstance(body["pairs"], list)
        or not 1 <= len(body["pairs"]) <= 100000
    ):
        raise ValueError("Invalid prepared preference schema/scope/context")
    pairs = []
    identifiers: set[str] = set()
    prompts: set[tuple[int, ...]] = set()
    for row in body["pairs"]:
        if not isinstance(row, dict) or set(row) != set(
            EncodedPair.__dataclass_fields__
        ):
            raise ValueError("Invalid encoded preference pair")
        for name in ("identifier", "family"):
            if (
                not isinstance(row[name], str)
                or not 1 <= len(row[name]) <= 512
                or any(ord(c) < 32 for c in row[name])
            ):
                raise ValueError("Invalid preference identifier/family")
        if (
            row["split"] != family_split(row["family"])
            or row["identifier"] in identifiers
        ):
            raise ValueError("Invalid preference split or duplicate identifier")
        prefixes = []
        for name in ("chosen", "rejected"):
            frame = row[name]
            if (
                not isinstance(frame, list)
                or not 4 <= len(frame) <= context + 1
                or any(type(t) is not int or not 0 <= t < vocab for t in frame)
            ):
                raise ValueError("Invalid preference token frame")
            mask = assistant_mask(frame, control_version=tokenizer.control_version)
            if frame[-2] != 6:
                raise ValueError(
                    "Preference responses must retain their complete end-turn"
                )
            boundary = mask.index(True) + 1
            # This prepared format supports one user turn and one final answer.
            if frame[:boundary].count(4) != 1 or frame.count(5) != 1 or 3 in frame:
                raise ValueError(
                    "Preferences require single-turn user/assistant frames"
                )
            prefixes.append(tuple(frame[:boundary]))
        if (
            prefixes[0] != prefixes[1]
            or prefixes[0] in prompts
            or row["chosen"] == row["rejected"]
        ):
            raise ValueError("Mismatched prompt or duplicate/identical preference")
        identifiers.add(row["identifier"])
        prompts.add(prefixes[0])
        pairs.append(
            EncodedPair(
                row["identifier"],
                row["family"],
                row["split"],
                tuple(row["chosen"]),
                tuple(row["rejected"]),
            )
        )
    if (
        len(pairs) != manifest.metadata.documents
        or sum(len(getattr(p, k)) - 1 for p in pairs for k in ("chosen", "rejected"))
        != manifest.metadata.tokens
        or not {"train", "validation"} <= {p.split for p in pairs}
    ):
        raise ValueError("Preference counts or heldout families disagree")
    return manifest, token_manifest, pairs
