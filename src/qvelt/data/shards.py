"""Compact little-endian token shards and document-aware packed block indices."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from typing import Any

import numpy as np

from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifest_types import ArtifactFile, identifier

SEGMENT_DTYPE = np.dtype(
    [("shard", "<u4"), ("start", "<u8"), ("pairs", "<u4"), ("document", "<u8")]
)
BLOCK_DTYPE = np.dtype([("first", "<u8"), ("segments", "<u4"), ("valid", "<u4")])


def token_dtype(vocab: int) -> np.dtype[Any]:
    if type(vocab) is not int or not 288 <= vocab <= 2**32:
        raise ValueError("Invalid vocabulary for token shards")
    return np.dtype("<u2" if vocab <= 65536 else "<u4")


class ShardWriter:
    """Store original document token frames once; reconstruct padded arrays later."""

    def __init__(
        self,
        store: ArtifactStore,
        prefix: str,
        vocab: int,
        context: int,
        shard_tokens: int,
        *,
        whole_documents: bool = False,
    ) -> None:
        identifier(prefix.replace("/", "-"), "shard.prefix")
        if any(
            type(v) is not int or not 1 <= v < 2**31 for v in (context, shard_tokens)
        ):
            raise ValueError("Invalid shard packing dimensions")
        self.store, self.prefix, self.vocab, self.context, self.target = (
            store,
            prefix,
            vocab,
            context,
            shard_tokens,
        )
        if type(whole_documents) is not bool:
            raise ValueError("whole_documents must be bool")
        self.whole_documents = whole_documents
        self.dtype = token_dtype(vocab)
        self.tokens = tempfile.TemporaryFile(dir=store.root)
        self.segments = tempfile.TemporaryFile(dir=store.root)
        self.blocks = tempfile.TemporaryFile(dir=store.root)
        self.metadata = tempfile.TemporaryFile(dir=store.root)
        self.shards: list[dict[str, object]] = []
        self.files: list[ArtifactFile] = []
        self.token_count = self.shard_count = self.segment_count = self.block_count = (
            self.used
        ) = self.first = self.current_segments = self.documents = self.pairs = 0
        self.last_document = -1

    def _flush_tokens(self) -> None:
        if not self.token_count:
            return
        self.tokens.seek(0)
        name = f"{self.prefix}/tokens-{self.shard_count:05d}.bin"
        self.files.append(self.store.add_stream(name, self.tokens, role="data"))
        self.shards.append({"path": name, "tokens": self.token_count})
        self.tokens.seek(0)
        self.tokens.truncate()
        self.shard_count += 1
        self.token_count = 0

    def _flush_block(self) -> None:
        if not self.used:
            return
        np.array(
            [(self.first, self.current_segments, self.used)], dtype=BLOCK_DTYPE
        ).tofile(self.blocks)
        self.first = self.segment_count
        self.current_segments = self.used = 0
        self.block_count += 1

    def add(
        self, tokens: list[int], document: int, metadata: Mapping[str, object]
    ) -> None:
        if (
            len(tokens) < 2
            or tokens[0] != 1
            or tokens[-1] != 2
            or any(type(t) is not int or not 0 <= t < self.vocab for t in tokens)
            or any(t in (0, 1, 2) for t in tokens[1:-1])
        ):
            raise ValueError("Documents require valid BOS/text/EOS token frames")
        if type(document) is not int or not self.last_document < document < 2**63:
            raise ValueError(
                "Document identities must increase within each source/split"
            )
        if {"document", "shard", "start", "tokens"} & set(metadata):
            raise ValueError("Document metadata cannot replace storage coordinates")
        if self.whole_documents:
            if len(tokens) - 1 > self.context:
                raise ValueError("Whole document exceeds packing context")
            if self.used + len(tokens) - 1 > self.context:
                self._flush_block()
        self.last_document = document
        if self.token_count and self.token_count + len(tokens) > self.target:
            self._flush_tokens()
        offset = self.token_count
        np.asarray(tokens, dtype=self.dtype).tofile(self.tokens)
        self.token_count += len(tokens)
        self.metadata.write(
            (
                json.dumps(
                    {
                        "document": document,
                        "shard": self.shard_count,
                        "start": offset,
                        "tokens": len(tokens),
                        **metadata,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode()
        )
        position = 0
        while position < len(tokens) - 1:
            count = min(self.context - self.used, len(tokens) - 1 - position)
            np.array(
                [(self.shard_count, offset + position, count, document)],
                dtype=SEGMENT_DTYPE,
            ).tofile(self.segments)
            self.segment_count += 1
            self.current_segments += 1
            self.used += count
            self.pairs += count
            position += count
            if self.used == self.context:
                self._flush_block()
        self.documents += 1

    def finish(self) -> dict[str, object]:
        if not self.documents:
            raise ValueError("Cannot finish an empty token source")
        self._flush_tokens()
        self._flush_block()
        index_files = {}
        for role, stream, name in [
            ("segment_index", self.segments, "segments.bin"),
            ("block_index", self.blocks, "blocks.bin"),
            ("document_metadata", self.metadata, "documents.jsonl"),
        ]:
            stream.seek(0)
            path = f"{self.prefix}/{name}"
            self.files.append(self.store.add_stream(path, stream, role=role))
            index_files[role] = path
        return {
            "prefix": self.prefix,
            "dtype": self.dtype.str,
            "context": self.context,
            "vocab": self.vocab,
            "documents": self.documents,
            "segments": self.segment_count,
            "blocks": self.block_count,
            "valid_targets": self.pairs,
            "shards": self.shards,
            **index_files,
        }

    def close(self) -> None:
        for stream in [self.tokens, self.segments, self.blocks, self.metadata]:
            stream.close()
