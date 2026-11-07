"""Restricted data-class DCP metadata and weights-only optimizer byte fields.

Pinned PyTorch2.13.0 FileSystemReader/DefaultLoadPlanner use unrestricted pickle
for these two paths. Model tensors already use weights_only=True upstream.
Distributed ranks/storage must still be trusted;this is not a hostile-file sandbox.
"""

from __future__ import annotations

import io
import pickle
from collections import OrderedDict
from pathlib import Path, PosixPath, WindowsPath
from typing import Any

import torch
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader, _StorageInfo
from torch.distributed.checkpoint.metadata import (
    _MEM_FORMAT_ENCODING,
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import ReadItem

from qvelt.utils.manifest_types import artifact_path


def _strided_layout(name: str) -> torch.layout:
    if name != "torch.strided":
        raise pickle.UnpicklingError(
            "Only dense strided checkpoint tensors are supported"
        )
    return torch.strided


class _MetadataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # Every permitted callable only constructs data;never import a payload name.
        allowed: dict[tuple[str, str], Any] = {
            (item.__module__, item.__name__): item
            for item in (
                Metadata,
                MetadataIndex,
                StorageMeta,
                TensorProperties,
                TensorStorageMetadata,
                BytesStorageMetadata,
                ChunkStorageMetadata,
                _MEM_FORMAT_ENCODING,
                _StorageInfo,
                PosixPath,
                WindowsPath,
                Path,
                OrderedDict,
            )
        }
        allowed[("torch", "Size")] = torch.Size
        allowed[("torch.serialization", "_get_layout")] = _strided_layout
        for dtype in (
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "int8",
            "uint8",
            "int16",
            "int32",
            "int64",
            "bool",
            "complex64",
            "complex128",
        ):
            allowed[("torch", dtype)] = getattr(torch, dtype)
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unsupported DCP metadata global")
        return allowed[(module, name)]

    def persistent_load(self, pid: object) -> Any:
        raise pickle.UnpicklingError("External pickle references are forbidden")


def restricted_metadata(
    data: bytes, registered_files: dict[str, int] | None = None
) -> Metadata:
    if not 1 <= len(data) <= 16 * 1024**2:
        raise ValueError("DCP metadata exceeds its16MiB limit")
    stream = io.BytesIO(data)
    metadata = _MetadataOnlyUnpickler(stream).load()
    if not isinstance(metadata, Metadata) or stream.read(1):
        raise ValueError("Invalid DCP metadata type or trailing payload")
    if not isinstance(metadata.storage_data, dict):
        raise ValueError("Invalid DCP storage index")
    for entry in metadata.storage_data.values():
        if not isinstance(entry, _StorageInfo):
            raise ValueError("Invalid DCP storage entry")
        path = artifact_path(entry.relative_path)
        if (
            type(entry.offset) is not int
            or type(entry.length) is not int
            or not 0 <= entry.offset <= 2**63 - 1
            or not 0 <= entry.length <= 2**63 - 1 - entry.offset
        ):
            raise ValueError("Invalid DCP storage extent")
        if registered_files is not None and (
            path not in registered_files
            or entry.offset + entry.length > registered_files[path]
        ):
            raise ValueError("DCP storage extent is not registered")
    return metadata


class SafeFileSystemReader(FileSystemReader):
    def __init__(
        self, path: Path, *, registered_files: dict[str, int] | None = None
    ) -> None:
        super().__init__(path)
        self.registered_files = registered_files

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        path = self._get_metadata_path(kwargs.get("rank"))
        with self.fs.create_stream(path, "rb") as stream:
            metadata = restricted_metadata(
                stream.read(16 * 1024**2 + 1), self.registered_files
            )
        if metadata.storage_meta is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        return metadata


class SafeLoadPlanner(DefaultLoadPlanner):
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        state = torch.load(value, map_location="cpu", weights_only=True)
        if self.flatten_state_dict:
            set_element(
                self.original_state_dict, self.mappings[read_item.dest_index.fqn], state
            )
        else:
            self.state_dict[read_item.dest_index.fqn] = state
