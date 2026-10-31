"""Real offline curation/packing workflow, split isolation and prefetched crash replay."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from qvelt.cli import main
from qvelt.config import ModelConfig
from qvelt.data.benchmark import benchmark_loader
from qvelt.data.config import load_data_recipe
from qvelt.data.demo import HOLDOUT, PERMISSION, authored_source
from qvelt.data.loader import LoaderConfig, PackedDataset, ResumableLoader
from qvelt.data.pipeline import prepare_data, prepare_recipe
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.model.decoder import Decoder
from qvelt.model.loss import causal_loss
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifest_types import DataInfo
from qvelt.utils.manifests import manifest_digest


def test_offline_raw_to_batches_and_frozen_splits(prepared_demo: Path) -> None:
    path = prepared_demo / "prepared/manifest.json"
    manifest = ArtifactStore(path.parent).read_manifest()
    report = json.loads((path.parent / "report.json").read_bytes())
    assert not report["production_ready"]
    assert report["counts"]["input"] > report["counts"]["retained"] > 100
    for reason in (
        "exact_duplicate",
        "near_duplicate",
        "benchmark_overlap",
        "toxic_phrase_quarantine",
    ):
        assert report["counts"][reason] > 0
    assert report["redactions"]["email"] > 0 and report["redactions"]["phone"] > 0
    assert isinstance(manifest.metadata, DataInfo)
    tokenizer, token_manifest = load_tokenizer_bundle(
        path.parent / "tokenizer/manifest.json", manifest.metadata.tokenizer_sha256
    )
    assert manifest.metadata.tokenizer_sha256 == manifest_digest(token_manifest)
    train_lineage = ArtifactStore(path.parent / "tokenizer").read_manifest(
        "training-data.manifest.json", verify=False
    )
    assert isinstance(train_lineage.metadata, DataInfo)
    assert train_lineage.metadata.documents == report["counts"]["train"]
    families: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        families[split] = set()
        dataset = PackedDataset(path, split)
        for file in (path.parent / "shards").glob(f"*/{split}/documents.jsonl"):
            for row in file.read_text().splitlines():
                families[split].add(json.loads(row)["family"])
        for index in range(len(dataset)):
            batch = dataset[index]
            assert batch["input_ids"].shape == batch["labels"].shape == (64,)
            for doc in set(batch["document_ids"][batch["valid_mask"]].tolist()):
                selected = batch["input_ids"][
                    (batch["document_ids"] == doc) & batch["valid_mask"]
                ].tolist()
                # No protected question enters token shards, even ordinary text.
                assert HOLDOUT not in tokenizer.decode(selected)
    assert not families["train"] & families["validation"]
    assert not families["train"] & families["test"]
    assert not families["validation"] & families["test"]
    assert len(set.union(*families.values())) > 5
    assert (
        main(
            [
                "data",
                "verify",
                str(path),
                "--expected-sha256",
                manifest_digest(manifest),
            ]
        )
        == 0
    )


def test_rebuild_has_identical_binary_and_tokenizer_payloads(
    prepared_demo: Path, tmp_path: Path
) -> None:
    manifest = prepare_data(
        prepared_demo / "recipe.yaml",
        tmp_path / "rebuilt",
        Path(__file__).resolve().parents[2],
    )
    before = ArtifactStore(prepared_demo / "prepared").read_manifest()
    old = {
        f.path: f.sha256
        for f in before.files
        if f.path.endswith(".bin") or f.path == "tokenizer/tokenizer.json"
    }
    new = {
        f.path: f.sha256
        for f in manifest.files
        if f.path.endswith(".bin") or f.path == "tokenizer/tokenizer.json"
    }
    assert old and new == old
    # Timestamps/provenance/report timings are descriptive, so entire manifests may differ.


@pytest.mark.parametrize("first_workers,next_workers", [(2, 0), (0, 2)])
def test_prefetch_simulated_crash_resumes_consumer_position(
    prepared_demo: Path, tmp_path: Path, first_workers: int, next_workers: int
) -> None:
    path = prepared_demo / "prepared/manifest.json"
    recipe = load_data_recipe(prepared_demo / "recipe.yaml")
    config = LoaderConfig(
        batch_size=3, workers=first_workers, prefetch=3, max_batches=9
    )
    with ResumableLoader(PackedDataset(path), config, recipe.mixing) as live:
        for _ in range(3):
            next(live)
        live.save_state(tmp_path / "cursor.json")
        assert live.state_dict()["step"] == 3
        expected = list(live)
    # Only the consumer cursor, not speculative sampler/worker progress, survives.
    with ResumableLoader(
        PackedDataset(path),
        replace(config, workers=next_workers, prefetch=1),
        recipe.mixing,
        ResumableLoader.read_state(tmp_path / "cursor.json"),
    ) as restarted:
        actual = list(restarted)
    assert len(expected) == len(actual) == 6
    assert all(
        torch.equal(a[key], b[key])
        for a, b in zip(expected, actual, strict=True)
        for key in a
    )


def test_packed_batch_runs_real_decoder_backward(prepared_demo: Path) -> None:
    path = prepared_demo / "prepared/manifest.json"
    dataset = PackedDataset(path)
    recipe = load_data_recipe(prepared_demo / "recipe.yaml")
    with ResumableLoader(dataset, LoaderConfig(max_batches=1), recipe.mixing) as loader:
        batch = next(loader)
    model = Decoder(
        ModelConfig(
            layers=1,
            hidden=32,
            heads=4,
            kv_heads=2,
            ffn=96,
            vocab=dataset.vocab,
            context=dataset.context,
        )
    )
    logits = model(
        batch["input_ids"],
        batch["positions"],
        batch["document_ids"],
        batch["valid_mask"],
    )
    total, count = causal_loss(logits, batch["labels"], batch["loss_mask"])
    mean = total / count
    cast(Callable[[], None], mean.backward)()
    assert torch.isfinite(mean)
    assert model.embedding.weight.grad is not None
    assert bool(torch.isfinite(model.embedding.weight.grad).all())


def test_missing_production_benchmarks_and_eval_only_source_blocked(
    prepared_demo: Path, tmp_path: Path
) -> None:
    recipe = load_data_recipe(prepared_demo / "recipe.yaml")
    sources = tuple(replace(s, fixture=False) for s in recipe.sources)
    with pytest.raises(ValueError):
        prepare_recipe(
            replace(recipe, sources=sources, purpose="production"),
            prepared_demo,
            tmp_path / "blocked",
            Path("."),
        )
    assert not (tmp_path / "blocked").exists()
    raw = write_raw_bundle(
        [
            RawDocument(
                "This explicitly approved evaluation-only question must never train a tokenizer."
            )
        ],
        tmp_path / "eval",
        authored_source("eval-only"),
        PERMISSION,
        allowed_uses=("evaluation", "decontamination"),
    )
    changed = (
        replace(
            recipe.sources[0],
            manifest=str(tmp_path / "eval/manifest.json"),
            sha256=manifest_digest(raw),
        ),
        recipe.sources[1],
    )
    with pytest.raises(ValueError, match="does not permit training"):
        prepare_recipe(
            replace(recipe, sources=changed),
            prepared_demo,
            tmp_path / "eval-blocked",
            Path("."),
        )
    assert not (tmp_path / "eval-blocked").exists()


def test_cpu_benchmark_reports_actual_count_and_no_gpu_claim(
    prepared_demo: Path,
) -> None:
    report = benchmark_loader(
        prepared_demo / "prepared/manifest.json",
        workers=0,
        batches=2,
        warmup=1,
        batch_size=2,
    )
    assert report["valid_targets_per_second"] > 0
    assert 0 < report["valid_targets"] <= report["capacity_tokens"] == 256
    assert report["measured_batches"] == 2
    assert "not GPU" in report["scope"]


def test_data_smoke_cli_entrypoint(tmp_path: Path) -> None:
    from qvelt.data.smoke import run_data_smoke

    result = run_data_smoke(
        tmp_path / "smoke", Path(__file__).resolve().parents[2], documents=48
    )
    assert result["replay_bit_exact"] and result["resumed_batches"] == 7
    assert result["verification"]["blocks_checked"] > 0
    assert not result["production_ready"]
