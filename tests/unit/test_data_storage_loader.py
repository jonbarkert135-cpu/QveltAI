"""Compact shard correctness against un-packed document pairs and exact cursor replay."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from qvelt.data.config import MixStage, load_data_recipe
from qvelt.data.demo import authored_source
from qvelt.data.loader import (
    LoaderConfig,
    PackedDataset,
    ResumableLoader,
    permuted_index,
)
from qvelt.data.shards import BLOCK_DTYPE, SEGMENT_DTYPE, ShardWriter, token_dtype
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore
from qvelt.utils.manifest_types import ArtifactFile, DataInfo, Manifest
from qvelt.utils.manifests import manifest_bytes


def packed_fixture(root: Path, vocab: int = 320) -> tuple[Path, list[list[int]]]:
    root.mkdir()
    store = ArtifactStore(root)
    frames = [[1, 32, 33, 2], [1, *range(40, 53), 2], [1, vocab - 1, 2]]
    writer = ShardWriter(store, "shards/test/train", vocab, context=5, shard_tokens=8)
    try:
        for index, frame in enumerate(frames):
            writer.add(frame, index, {"family": str(index)})
        descriptor = writer.finish()
    finally:
        writer.close()
    layout = {
        "schema_version": 1,
        "context": 5,
        "vocab": vocab,
        "control_version": 2,
        "tokenizer_sha256": "a" * 64,
        "sources": {"test": {"train": descriptor}},
        "mixing": [],
    }
    layout_file = store.add_bytes(
        "layout.json", json.dumps(layout).encode(), role="packing_layout"
    )
    manifest = Manifest(
        "data",
        "storage-test",
        "2026-10-31T00:00:00Z",
        DataInfo(
            "token_ids",
            (authored_source("storage"),),
            3,
            sum(map(len, frames)),
            "a" * 64,
        ),
        (*writer.files, layout_file),
    )
    store.publish(manifest)
    return root / "manifest.json", frames


@pytest.mark.parametrize("vocab", [320, 65536, 65537, 100000])
def test_shards_reconstruct_each_original_pair_exactly_once(
    tmp_path: Path, vocab: int
) -> None:
    path, frames = packed_fixture(tmp_path / "packed", vocab)
    dataset = PackedDataset(path)
    actual: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(frames))}
    padding = 0
    for index in range(len(dataset)):
        batch = dataset[index]
        assert batch["input_ids"].dtype == torch.int64
        valid = batch["valid_mask"]
        assert torch.equal(valid, batch["loss_mask"])
        padding += int((~valid).sum())
        assert bool((batch["document_ids"][~valid] == -1).all())
        for t in torch.where(valid)[0].tolist():
            doc = int(batch["document_ids"][t])
            actual[doc].append((int(batch["input_ids"][t]), int(batch["labels"][t])))
            if t == 0 or batch["document_ids"][t] != batch["document_ids"][t - 1]:
                assert batch["positions"][t] == 0
    expected = {
        i: list(zip(frame[:-1], frame[1:], strict=True))
        for i, frame in enumerate(frames)
    }
    assert actual == expected
    assert padding > 0
    assert BLOCK_DTYPE.itemsize == 16 and SEGMENT_DTYPE.itemsize == 24
    assert token_dtype(vocab).itemsize == (2 if vocab <= 65536 else 4)
    with pytest.raises(IndexError):
        dataset[-1]
    assert all(not array.flags.writeable for array in dataset._cache.values())


def test_shard_integrity_and_mutation_detection(tmp_path: Path) -> None:
    path, _ = packed_fixture(tmp_path / "packed")
    dataset = PackedDataset(path)
    dataset[0]
    token = path.parent / dataset.sources["test"]["shards"][0]["path"]
    data = bytearray(token.read_bytes())
    data[-1] ^= 1
    token.write_bytes(data)
    with pytest.raises(ArtifactIntegrityError):
        dataset[0]
    with pytest.raises(ArtifactIntegrityError):
        PackedDataset(path)[0]


@pytest.mark.parametrize("bad", ["dtype", "index", "bool", "layout-bool"])
def test_malformed_packing_layout_rejected_before_mapping(
    tmp_path: Path, bad: str
) -> None:
    path, _ = packed_fixture(tmp_path / "packed")
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest()
    layout = json.loads((path.parent / "layout.json").read_bytes())
    source = layout["sources"]["test"]["train"]
    if bad == "dtype":
        source["dtype"] = "<u8"
    elif bad == "index":
        source["block_index"] = "../../escape"
    elif bad == "bool":
        source["blocks"] = True
    else:
        layout["schema_version"] = True
    body = json.dumps(layout).encode()
    (path.parent / "layout.json").write_bytes(body)
    import hashlib

    records = tuple(
        ArtifactFile(f.path, f.role, len(body), hashlib.sha256(body).hexdigest())
        if f.path == "layout.json"
        else f
        for f in manifest.files
    )
    path.write_bytes(manifest_bytes(replace(manifest, files=records)))
    with pytest.raises(ValueError):
        PackedDataset(path)


def test_writer_rejects_invalid_frame_and_metadata(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    writer = ShardWriter(store, "tokens", 320, 8, 16)
    try:
        with pytest.raises(ValueError):
            writer.add([1, 0, 2], 0, {})
        with pytest.raises(ValueError):
            writer.add([1, 32, 2], 0, {"shard": 900})
        with pytest.raises(ValueError):
            writer.finish()
        writer.add([1, 32, 2], 0, {})
        with pytest.raises(ValueError):
            writer.add([1, 32, 2], 0, {})
    finally:
        writer.close()


def test_per_source_affine_epochs_are_bijections() -> None:
    for length in (1, 2, 7, 12, 64, 99):
        for epoch in range(3):
            indices = [
                permuted_index(epoch * length + i, length, 17, "source")
                for i in range(length)
            ]
            assert sorted(indices) == list(range(length))


def test_consumer_cursor_replay_stages_and_topology(
    prepared_demo: Path, tmp_path: Path
) -> None:
    path = prepared_demo / "prepared/manifest.json"
    dataset = PackedDataset(path)
    stages = load_data_recipe(prepared_demo / "recipe.yaml").mixing
    config = LoaderConfig(batch_size=3, max_batches=10)
    with ResumableLoader(dataset, config, stages) as loader:
        for _ in range(3):
            next(loader)
        loader.save_state(tmp_path / "cursor.json")
        state = loader.state_dict()
        tail = list(loader)
    with ResumableLoader(
        PackedDataset(path),
        config,
        stages,
        ResumableLoader.read_state(tmp_path / "cursor.json"),
    ) as restarted:
        replay = list(restarted)
    assert len(tail) == len(replay) == 7
    for old, new in zip(tail, replay, strict=True):
        assert old.keys() == new.keys()
        assert all(torch.equal(old[key], new[key]) for key in old)
    with pytest.raises(ValueError):
        ResumableLoader(dataset, replace(config, world_size=2), stages, state)
    tampered = dict(state)
    tampered["counts"] = {name: 0 for name in dataset.names}
    with pytest.raises(ValueError):
        ResumableLoader(dataset, config, stages, tampered)
    zero_then_ru = (
        MixStage(0, (("math-en", 1.0), ("math-ru", 0.0))),
        MixStage(2, (("math-en", 0.0), ("math-ru", 1.0)), "anneal"),
    )
    with ResumableLoader(
        dataset, replace(config, max_batches=4), zero_then_ru
    ) as loader:
        names = [dataset.names[int(next(loader)["source_index"][0])] for _ in range(4)]
    assert names == ["math-en", "math-en", "math-ru", "math-ru"]


def test_rank_plans_use_disjoint_source_cursors(prepared_demo: Path) -> None:
    dataset = PackedDataset(prepared_demo / "prepared/manifest.json")
    stages = (MixStage(0, (("math-en", 1.0), ("math-ru", 0.0))),)
    rank_keys = []
    for rank in (0, 1):
        with ResumableLoader(
            dataset,
            LoaderConfig(batch_size=2, rank=rank, world_size=2, max_batches=2),
            stages,
        ) as loader:
            rank_keys.append(
                [i for batch in loader for i in batch["sample_index"].tolist()]
            )
    assert not set(rank_keys[0]) & set(rank_keys[1])


def test_explicit_cross_document_attention_option(prepared_demo: Path) -> None:
    dataset = PackedDataset(prepared_demo / "prepared/manifest.json")
    stages = load_data_recipe(prepared_demo / "recipe.yaml").mixing
    with ResumableLoader(
        dataset, LoaderConfig(max_batches=1, attention_mode="causal"), stages
    ) as loader:
        batch = next(loader)
    assert "segment_ids" in batch
    assert bool((batch["document_ids"][batch["valid_mask"]] == 0).all())
    expected = torch.arange(dataset.context).expand(2, -1)
    assert torch.equal(
        batch["positions"][batch["valid_mask"]], expected[batch["valid_mask"]]
    )


def test_saved_cursor_checksum_and_settings(
    tmp_path: Path, prepared_demo: Path
) -> None:
    dataset = PackedDataset(prepared_demo / "prepared/manifest.json")
    stages = load_data_recipe(prepared_demo / "recipe.yaml").mixing
    with ResumableLoader(dataset, LoaderConfig(max_batches=1), stages) as loader:
        loader.save_state(tmp_path / "state.json")
    raw: dict[str, Any] = json.loads((tmp_path / "state.json").read_bytes())
    raw["state"]["step"] += 1
    (tmp_path / "state.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        ResumableLoader.read_state(tmp_path / "state.json")
    with pytest.raises(ValueError):
        LoaderConfig(workers=True)
    with pytest.raises(ValueError):
        LoaderConfig(attention_mode="invalid")


def test_loader_does_not_consume_model_rng(prepared_demo: Path) -> None:
    dataset = PackedDataset(prepared_demo / "prepared/manifest.json")
    stages = load_data_recipe(prepared_demo / "recipe.yaml").mixing
    before = torch.get_rng_state().clone()
    with ResumableLoader(dataset, LoaderConfig(max_batches=1), stages) as loader:
        next(loader)
    assert torch.equal(torch.get_rng_state(), before)


def test_cursor_cannot_cross_training_validation_split(prepared_demo: Path) -> None:
    path = prepared_demo / "prepared/manifest.json"
    stages = load_data_recipe(prepared_demo / "recipe.yaml").mixing
    config = LoaderConfig(max_batches=3)
    with ResumableLoader(PackedDataset(path, "train"), config, stages) as train:
        next(train)
        state = train.state_dict()
    with pytest.raises(ValueError, match="Incompatible"):
        ResumableLoader(PackedDataset(path, "validation"), config, stages, state)
