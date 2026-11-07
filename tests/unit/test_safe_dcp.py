"""Real DCP tensor/byte reload and rejection of executable metadata/byte pickle."""

import io
import os
import pickle
import shlex
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.state_dict_saver import save

from qvelt.utils.safe_dcp import (
    SafeFileSystemReader,
    SafeLoadPlanner,
    restricted_metadata,
)


class Payload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> Any:
        return os.system, ("touch " + shlex.quote(str(self.marker)),)


def test_real_dcp_dense_state_and_optimizer_bytes(tmp_path: Path) -> None:
    source: dict[str, Any] = {
        "weight": torch.arange(12).reshape(3, 4),
        "lr": 0.1,
        "groups": [{"names": ["weight"], "betas": (0.9, 0.95)}],
    }
    # These two upstream single-rank/TypedStorage warnings are known library behavior.
    with pytest.warns(UserWarning):
        save(source, checkpoint_id=tmp_path, no_dist=True)
    target: dict[str, Any] = {
        "weight": torch.zeros_like(source["weight"]),
        "lr": 0.0,
        "groups": [{"names": ["weight"], "betas": (0.0, 0.0)}],
    }
    with pytest.warns(UserWarning, match="single process"):
        load(
            target,
            storage_reader=SafeFileSystemReader(tmp_path),
            planner=SafeLoadPlanner(),
            no_dist=True,
        )
    assert torch.equal(target["weight"], source["weight"])
    assert target["lr"] == source["lr"] and target["groups"] == source["groups"]
    with pytest.raises(ValueError, match="trailing"):
        restricted_metadata((tmp_path / ".metadata").read_bytes() + b"unexpected")


def test_executable_metadata_is_never_called(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    with pytest.raises(pickle.UnpicklingError, match="global"):
        restricted_metadata(pickle.dumps(Payload(marker)))
    assert not marker.exists()
    with pytest.raises(ValueError, match="type"):
        restricted_metadata(pickle.dumps({"ordinary": True}))
    with pytest.raises(ValueError, match="limit"):
        restricted_metadata(b"")


def test_byte_loader_is_explicitly_weights_only(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    value = io.BytesIO()
    torch.save(Payload(marker), value)
    value.seek(0)
    with pytest.raises(pickle.UnpicklingError):
        SafeLoadPlanner().load_bytes(None, value)  # type: ignore[arg-type]
    assert not marker.exists()


@pytest.mark.parametrize(
    "path,offset,length",
    [
        ("../outside", 0, 1),
        ("/outside", 0, 1),
        ("safe.distcp", -1, 1),
        ("safe.distcp", 0, 11),
        ("unregistered.distcp", 0, 1),
        ("safe.distcp", True, 1),
        ("safe.distcp", 2**63, 1),
    ],
)
def test_metadata_storage_is_bounded_by_manifest(
    path: str, offset: int, length: int
) -> None:
    metadata = Metadata(
        state_dict_metadata={},
        storage_data={"entry": _StorageInfo(path, offset, length)},
    )
    with pytest.raises(ValueError):
        restricted_metadata(pickle.dumps(metadata), {"safe.distcp": 10})


def test_metadata_accepts_registered_extent() -> None:
    metadata = Metadata(
        state_dict_metadata={},
        storage_data={"entry": _StorageInfo("safe.distcp", 2, 8)},
    )
    assert (
        restricted_metadata(pickle.dumps(metadata), {"safe.distcp": 10}).storage_data
        == metadata.storage_data
    )
