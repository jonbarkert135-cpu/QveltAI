"""Ordered prefetched memmap batches with consumer-committed mixed-source cursors."""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from qvelt.data.config import MixStage
from qvelt.data.shards import BLOCK_DTYPE, SEGMENT_DTYPE, token_dtype
from qvelt.utils.artifacts import (
    ArtifactIntegrityError,
    ArtifactStore,
    atomic_write,
    hash_file,
)
from qvelt.utils.manifest_types import DataInfo, artifact_path, identifier, uint
from qvelt.utils.manifests import _reject_constant, _unique_mapping, manifest_digest


class PackedDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self, path: Path, split: str = "train", expected_sha256: str | None = None
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown packed-data split")
        self.split = split
        self.root = path.parent
        store = ArtifactStore(self.root)
        manifest = store.read_manifest(
            path.name, expected_sha256=expected_sha256, verify=False
        )
        if (
            not isinstance(manifest.metadata, DataInfo)
            or manifest.metadata.representation != "token_ids"
        ):
            raise ValueError("Expected tokenized data manifest")
        self.identity = manifest_digest(manifest)
        self.records = {file.path: file for file in manifest.files}
        layout_file = self.records.get("layout.json")
        if layout_file is None:
            raise ValueError("Dataset is missing its layout")
        if layout_file.size_bytes > 1024 * 1024:
            raise ValueError("Dataset layout exceeds its size limit")
        with store.open_file("layout.json") as stream:
            data = stream.read(1024 * 1024 + 1)
        if (
            len(data) != layout_file.size_bytes
            or hashlib.sha256(data).hexdigest() != layout_file.sha256
        ):
            raise ArtifactIntegrityError("Dataset layout hash mismatch")
        layout = json.loads(
            data, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
        )
        if (
            not isinstance(layout, dict)
            or set(layout)
            != {
                "schema_version",
                "context",
                "vocab",
                "control_version",
                "tokenizer_sha256",
                "sources",
                "mixing",
            }
            or type(layout["schema_version"]) is not int
            or layout["schema_version"] not in (1, 2)
            or type(layout["control_version"]) is not int
            or layout["control_version"] not in (1, 2)
            or layout["tokenizer_sha256"] != manifest.metadata.tokenizer_sha256
        ):
            raise ValueError("Unsupported/mismatched dataset layout")
        self.assistant_only = layout["schema_version"] == 2
        self.context = uint(layout["context"], "packing.context", 1, 2**31 - 1)
        self.vocab = uint(layout["vocab"], "packing.vocab", 288, 2**32)
        if (
            not isinstance(layout["sources"], dict)
            or not 1 <= len(layout["sources"]) <= 64
        ):
            raise ValueError("Invalid packed source list")
        for name, parts in layout["sources"].items():
            identifier(name, "packing.source")
            if not isinstance(parts, dict) or set(parts) - {
                "train",
                "validation",
                "test",
            }:
                raise ValueError("Invalid packed split list")
            for source in parts.values():
                if not isinstance(source, dict) or set(source) != {
                    "prefix",
                    "dtype",
                    "context",
                    "vocab",
                    "documents",
                    "segments",
                    "blocks",
                    "valid_targets",
                    "shards",
                    "segment_index",
                    "block_index",
                    "document_metadata",
                }:
                    raise ValueError("Invalid source descriptor")
                for field in ("documents", "segments", "blocks", "valid_targets"):
                    uint(source[field], "packing." + field, 1)
                if (
                    source["dtype"] != token_dtype(self.vocab).str
                    or type(source["context"]) is not int
                    or source["context"] != self.context
                    or type(source["vocab"]) is not int
                    or source["vocab"] != self.vocab
                ):
                    raise ValueError("Packing dimensions/dtype mismatch")
                artifact_path(source["prefix"])
                for field in ("segment_index", "block_index", "document_metadata"):
                    artifact_path(source[field])
                    if source[field] not in self.records:
                        raise ValueError("Unregistered packed index")
                if not isinstance(source["shards"], list) or not source["shards"]:
                    raise ValueError("Missing source token shards")
                for shard in source["shards"]:
                    if not isinstance(shard, dict) or set(shard) != {"path", "tokens"}:
                        raise ValueError("Invalid shard descriptor")
                    artifact_path(shard["path"])
                    uint(shard["tokens"], "shard.tokens", 2)
                    if shard["path"] not in self.records:
                        raise ValueError("Unregistered token shard")
        self.all_names = set(layout["sources"])
        self.sources = {
            name: parts[split]
            for name, parts in sorted(layout["sources"].items())
            if split in parts and parts[split]["blocks"] > 0
        }
        if not self.sources:
            raise ValueError("No packed blocks in the requested split")
        self.names = list(self.sources)
        self.offsets = [0]
        for source in self.sources.values():
            self.offsets.append(self.offsets[-1] + source["blocks"])
        self._cache: OrderedDict[tuple[str, str], np.memmap[Any, Any]] = OrderedDict()
        self._verified: dict[str, tuple[int, ...]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        state["_verified"] = {}
        return state

    def __len__(self) -> int:
        return self.offsets[-1]

    def _map(self, name: str, dtype: np.dtype[Any], count: int) -> np.memmap[Any, Any]:
        record = self.records.get(name)
        if record is None or record.size_bytes != count * dtype.itemsize or count < 1:
            raise ArtifactIntegrityError("Invalid binary index/shard size")
        store = ArtifactStore(self.root)
        # Resolve every contained path through the established no-symlink policy.
        with store.open_file(name) as stream:
            stat = os.fstat(stream.fileno())
        stamp = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if name in self._verified and self._verified[name] != stamp:
            raise ArtifactIntegrityError("Prepared data changed during iteration")
        if name not in self._verified:
            if hash_file(self.root / name).sha256 != record.sha256:
                raise ArtifactIntegrityError("Prepared shard checksum mismatch")
            self._verified[name] = stamp
        key = (name, dtype.str + str(dtype.descr))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        value = np.memmap(self.root / name, mode="r", dtype=dtype, shape=(count,))
        self._cache[key] = value
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("Packed block index out of range")
        source_index = bisect.bisect_right(self.offsets, index) - 1
        source = self.sources[self.names[source_index]]
        local = index - self.offsets[source_index]
        blocks = self._map(source["block_index"], BLOCK_DTYPE, source["blocks"])
        block = blocks[local]
        segments = self._map(source["segment_index"], SEGMENT_DTYPE, source["segments"])
        first, count = int(block["first"]), int(block["segments"])
        if first + count > len(segments) or not 0 < int(block["valid"]) <= self.context:
            raise ArtifactIntegrityError("Invalid packed block descriptor")
        inputs = np.zeros(self.context, dtype=np.int64)
        labels = inputs.copy()
        positions = inputs.copy()
        documents = np.full(self.context, -1, dtype=np.int64)
        valid = np.zeros(self.context, dtype=np.bool_)
        loss_mask = valid.copy()
        cursor = 0
        for item in segments[first : first + count]:
            shard_index, start, pairs, document = map(
                int, (item["shard"], item["start"], item["pairs"], item["document"])
            )
            if (
                not 0 <= shard_index < len(source["shards"])
                or pairs < 1
                or cursor + pairs > self.context
            ):
                raise ArtifactIntegrityError("Invalid segment descriptor")
            shard = source["shards"][shard_index]
            data = self._map(shard["path"], token_dtype(self.vocab), shard["tokens"])
            if start + pairs + 1 > len(data):
                raise ArtifactIntegrityError("Segment exceeds token shard")
            frame = np.array(data[start : start + pairs + 1], dtype=np.int64)
            if bool((frame >= self.vocab).any()):
                raise ArtifactIntegrityError("Shard token exceeds vocabulary")
            part = slice(cursor, cursor + pairs)
            inputs[part] = frame[:-1]
            labels[part] = frame[1:]
            positions[part] = np.arange(pairs)
            documents[part] = document
            valid[part] = True
            if self.assistant_only:
                from qvelt.posttrain.chat import assistant_mask

                loss_mask[part] = assistant_mask(frame.tolist())
            else:
                loss_mask[part] = True
            cursor += pairs
        if cursor != int(block["valid"]):
            raise ArtifactIntegrityError("Block target count mismatch")
        if bool(((labels == 0) | (labels == 1))[valid].any()):
            raise ArtifactIntegrityError(
                "Cross-document BOS or padding target leaked into a valid segment"
            )
        return {
            "input_ids": torch.from_numpy(inputs),
            "labels": torch.from_numpy(labels),
            "positions": torch.from_numpy(positions),
            "document_ids": torch.from_numpy(documents),
            "valid_mask": torch.from_numpy(valid),
            "loss_mask": torch.from_numpy(loss_mask),
            "sample_index": torch.tensor(index),
            "source_index": torch.tensor(source_index),
        }


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 2
    seed: int = 20261031
    rank: int = 0
    world_size: int = 1
    workers: int = 0
    prefetch: int = 2
    max_batches: int = 100
    attention_mode: str = "document"

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for key, value in asdict(self).items()
            if key != "attention_mode"
        ):
            raise ValueError("Loader settings must be integers")
        if self.attention_mode not in ("document", "causal"):
            raise ValueError("Invalid attention boundary mode")
        if (
            min(self.batch_size, self.world_size, self.prefetch, self.max_batches) < 1
            or not 0 <= self.rank < self.world_size
            or not 0 <= self.workers <= 16
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("Invalid loader settings")


def _integer_hash(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).digest()[:8],
        "big",
    )


def permuted_index(index: int, length: int, seed: int, source: str) -> int:
    epoch, position = divmod(index, length)
    a = _integer_hash(seed, source, epoch, "a") % length or 1
    while math.gcd(a, length) != 1:
        a = (a + 1) % length or 1
    b = _integer_hash(seed, source, epoch, "b") % length
    return (a * position + b) % length


class _PlannedBatches(Sampler[list[int]]):
    def __init__(self, owner: ResumableLoader) -> None:
        self.owner = owner

    def __iter__(self) -> Iterator[list[int]]:
        state = copy.deepcopy(self.owner.state)
        while state["step"] < self.owner.config.max_batches:
            keys, state = self.owner.plan(state)
            yield keys

    def __len__(self) -> int:
        return max(0, self.owner.config.max_batches - int(self.owner.state["step"]))


def _numpy_collate(samples: list[dict[str, Tensor]]) -> dict[str, np.ndarray[Any, Any]]:
    """Plain NumPy IPC avoids shared-Tensor handle races on early worker shutdown."""
    return {
        key: np.stack([np.asarray(sample[key]) for sample in samples])
        for key in samples[0]
    }


class ResumableLoader:
    """Only delivered batches advance state; sampler/worker prefetch is speculative."""

    def __init__(
        self,
        dataset: PackedDataset,
        config: LoaderConfig,
        stages: tuple[MixStage, ...],
        state: dict[str, Any] | None = None,
    ) -> None:
        self.dataset, self.config, self.stages = dataset, config, stages
        if (
            not stages
            or stages[0].start_step != 0
            or any(
                a.start_step >= b.start_step
                for a, b in zip(stages, stages[1:], strict=False)
            )
        ):
            raise ValueError("Invalid stage ordering")
        for stage in stages:
            if set(name for name, _ in stage.weights) != dataset.all_names:
                raise ValueError("Mix stage must weight exactly the dataset's sources")
            if (
                sum(weight for name, weight in stage.weights if name in dataset.sources)
                <= 0
            ):
                raise ValueError("No positive stage mass for this split")
        identity = {
            "data": dataset.identity,
            "split": dataset.split,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "rank": config.rank,
            "world_size": config.world_size,
            "attention_mode": config.attention_mode,
            "stages": [asdict(s) for s in stages],
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.state: dict[str, Any] = (
            {
                "fingerprint": self.fingerprint,
                "step": 0,
                "counts": {name: 0 for name in dataset.names},
            }
            if state is None
            else copy.deepcopy(state)
        )
        if (
            set(self.state) != {"fingerprint", "step", "counts"}
            or self.state["fingerprint"] != self.fingerprint
            or type(self.state["step"]) is not int
            or not 0 <= self.state["step"] <= config.max_batches
        ):
            raise ValueError("Incompatible or invalid loader state")
        counts = self.state["counts"]
        if (
            not isinstance(counts, dict)
            or set(counts) != set(dataset.names)
            or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values())
            != self.state["step"] * config.world_size * config.batch_size
        ):
            raise ValueError("Invalid committed source cursors")
        kwargs: dict[str, Any] = {
            "num_workers": config.workers,
            "batch_sampler": _PlannedBatches(self),
            "in_order": True,
            "generator": torch.Generator().manual_seed(config.seed),
            "collate_fn": _numpy_collate,
        }
        if config.workers:
            kwargs.update(
                multiprocessing_context="spawn", prefetch_factor=config.prefetch
            )
        self._loader = DataLoader(dataset, **kwargs)
        self._iterator: Iterator[dict[str, np.ndarray[Any, Any]]] | None = None

    def plan(self, state: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
        step = state["step"]
        counts = dict(state["counts"])
        keys = []
        stage = max(
            (s for s in self.stages if s.start_step <= step), key=lambda s: s.start_step
        )
        weights = [
            (name, weight)
            for name, weight in stage.weights
            if name in self.dataset.sources and weight > 0
        ]
        total = math.fsum(weight for _, weight in weights)
        for slot in range(self.config.world_size * self.config.batch_size):
            uniform = (
                _integer_hash(self.config.seed, step, slot, "source") >> 11
            ) * 2**-53
            threshold = uniform * total
            accumulated = 0.0
            source = weights[-1][0]
            for name, weight in weights:
                accumulated += weight
                if threshold < accumulated:
                    source = name
                    break
            index = self.dataset.names.index(source)
            length = self.dataset.sources[source]["blocks"]
            key = self.dataset.offsets[index] + permuted_index(
                counts[source], length, self.config.seed, source
            )
            counts[source] += 1
            if slot // self.config.batch_size == self.config.rank:
                keys.append(key)
        return keys, {
            "fingerprint": self.fingerprint,
            "step": step + 1,
            "counts": counts,
        }

    def __iter__(self) -> ResumableLoader:
        return self

    def __next__(self) -> dict[str, Tensor]:
        if self.state["step"] >= self.config.max_batches:
            raise StopIteration
        if self._iterator is None:
            self._iterator = cast(
                Iterator[dict[str, np.ndarray[Any, Any]]], iter(self._loader)
            )
        keys, next_state = self.plan(self.state)
        batch = {
            key: torch.from_numpy(value) for key, value in next(self._iterator).items()
        }
        if batch["sample_index"].tolist() != keys:
            raise RuntimeError("Worker ordering disagrees with the committed plan")
        if self.config.attention_mode == "causal":
            batch["segment_ids"] = batch["document_ids"]
            batch["document_ids"] = torch.where(batch["valid_mask"], 0, -1)
            batch["positions"] = torch.where(
                batch["valid_mask"], torch.arange(self.dataset.context), 0
            )
        self.state = next_state
        return batch

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def save_state(self, path: Path) -> None:
        body = json.dumps(self.state, sort_keys=True, separators=(",", ":")).encode()
        data = (
            json.dumps(
                {"state": self.state, "sha256": hashlib.sha256(body).hexdigest()},
                sort_keys=True,
            ).encode()
            + b"\n"
        )

        def write(stream: Any) -> None:
            stream.write(data)

        atomic_write(path, write)

    @staticmethod
    def read_state(path: Path) -> dict[str, Any]:
        with path.open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("Loader state exceeds its size limit")
        raw = json.loads(data, object_pairs_hook=_unique_mapping)
        body = json.dumps(raw["state"], sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(body).hexdigest() != raw["sha256"]:
            raise ValueError("Loader state checksum mismatch")
        return cast(dict[str, Any], raw["state"])

    def close(self) -> None:
        # Non-persistent DataLoader workers are owned by this iterator; dropping
        # it invokes PyTorch's normal shutdown even with prefetched work pending.
        self._iterator = None

    def __enter__(self) -> ResumableLoader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
