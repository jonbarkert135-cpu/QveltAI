"""Raw text manifests with optional aligned, checksummed document metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from qvelt.tokenizer.core import _validate_text
from qvelt.tokenizer.workflow import iter_documents
from qvelt.utils.artifacts import (
    ArtifactIntegrityError,
    ArtifactStore,
    create_directory,
)
from qvelt.utils.manifest_types import DataInfo, Manifest, SourceIdentity
from qvelt.utils.manifests import _reject_constant, _unique_mapping


@dataclass(frozen=True)
class RawDocument:
    text: str
    identifier: str = ""
    family: str = ""
    language: str = ""
    difficulty: str = ""
    answer: str = ""

    def __post_init__(self) -> None:
        _validate_text(self.text)
        for key in ["identifier", "family", "language", "difficulty", "answer"]:
            value = getattr(self, key)
            if (
                not isinstance(value, str)
                or len(value) > 512
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("Invalid bounded document metadata")


def write_raw_bundle(
    documents: Iterable[RawDocument],
    directory: Path,
    source: SourceIdentity | tuple[SourceIdentity, ...],
    permission: bytes,
    *,
    allowed_uses: tuple[str, ...] = ("training", "evaluation", "decontamination"),
) -> Manifest:
    """Author/export one versioned source; preserve text and metadata row alignment."""
    import tempfile

    if not allowed_uses or not set(allowed_uses) <= {
        "training",
        "evaluation",
        "decontamination",
    }:
        raise ValueError("Invalid explicit raw source usage policy")
    store = ArtifactStore(create_directory(directory))
    count = 0
    with tempfile.TemporaryFile() as data, tempfile.TemporaryFile() as metadata:
        for doc in documents:
            row = asdict(doc)
            row.pop("text")
            data.write(
                (
                    json.dumps(
                        {"text": doc.text}, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                ).encode()
            )
            metadata.write(
                (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
            )
            count += 1
        data.seek(0)
        metadata.seek(0)
        files = (
            store.add_stream("data.jsonl", data, role="data"),
            store.add_stream("documents.jsonl", metadata, role="document_metadata"),
            store.add_bytes("PERMISSION.txt", permission, role="license"),
            store.add_bytes(
                "usage.json",
                (
                    json.dumps({"allowed_uses": list(allowed_uses)}, sort_keys=True)
                    + "\n"
                ).encode(),
                role="usage_policy",
            ),
        )
    sources = (source,) if isinstance(source, SourceIdentity) else source
    manifest = Manifest(
        "data",
        sources[0].name,
        "2026-10-31T00:00:00Z",
        DataInfo("raw", sources, count, None, None),
        files,
    )
    store.publish(manifest)
    return manifest


def iter_records(path: Path, expected: str) -> Iterator[RawDocument]:
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(path.name, expected_sha256=expected)
    records = [f for f in manifest.files if f.role == "document_metadata"]
    if len(records) > 1:
        raise ValueError("At most one aligned document metadata file is supported")
    if not records:
        for index, text in enumerate(iter_documents(path, expected)):
            yield RawDocument(text, str(index))
        return
    record = records[0]
    digest = hashlib.sha256()
    size = 0
    with store.open_file(record.path) as stream:
        for text in iter_documents(path, expected):
            line = stream.readline(min(32769, record.size_bytes - size + 1))
            size += len(line)
            digest.update(line)
            if len(line) > 32768 or size > record.size_bytes:
                raise ValueError("Document metadata exceeds its size limit")
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_mapping,
                    parse_constant=_reject_constant,
                )
                if not isinstance(row, dict) or set(row) != {
                    "identifier",
                    "family",
                    "language",
                    "difficulty",
                    "answer",
                }:
                    raise ValueError("Invalid metadata fields")
                yield RawDocument(text, **row)
            except (ValueError, UnicodeError, TypeError) as error:
                raise ValueError("Invalid aligned document metadata") from error
        if (
            stream.read(1)
            or size != record.size_bytes
            or digest.hexdigest() != record.sha256
        ):
            raise ArtifactIntegrityError(
                "Aligned document metadata count/hash mismatch"
            )
