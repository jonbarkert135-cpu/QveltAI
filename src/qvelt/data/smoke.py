"""Offline end-to-end data acceptance with consumer-cursor crash simulation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from qvelt.data.cli import run
from qvelt.data.config import load_data_recipe
from qvelt.data.demo import create_demo
from qvelt.data.loader import LoaderConfig, PackedDataset, ResumableLoader
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest


def run_data_smoke(
    directory: Path, repository: Path, documents: int = 96
) -> dict[str, Any]:
    import argparse

    manifest = create_demo(directory, repository, documents)
    path = directory / "prepared/manifest.json"
    verification = run(
        argparse.Namespace(
            action="verify", path=path, expected_sha256=manifest_digest(manifest)
        )
    )
    stages = load_data_recipe(directory / "recipe.yaml").mixing
    config = LoaderConfig(batch_size=3, workers=2, prefetch=3, max_batches=10)
    with ResumableLoader(PackedDataset(path), config, stages) as live:
        for _ in range(3):
            next(live)
        live.save_state(directory / "cursor-step-3.json")
        expected = list(live)
    with ResumableLoader(
        PackedDataset(path),
        replace(config, workers=0),
        stages,
        ResumableLoader.read_state(directory / "cursor-step-3.json"),
    ) as restarted:
        actual = list(restarted)
        final = restarted.state_dict()
    equal = len(actual) == len(expected) == 7 and all(
        torch.equal(a[key], b[key])
        for a, b in zip(actual, expected, strict=True)
        for key in a
    )
    if not equal:
        raise RuntimeError(
            "Prefetched crash replay differs from uninterrupted batch delivery"
        )
    report = json.loads((path.parent / "report.json").read_bytes())
    store = ArtifactStore(directory)
    result = {
        "dataset_sha256": manifest_digest(manifest),
        "verification": verification,
        "replay_bit_exact": equal,
        "saved_consumer_step": 3,
        "resumed_batches": len(actual),
        "final_step": final["step"],
        "curation_counts": report["counts"],
        "production_ready": False,
    }
    store.add_bytes(
        "smoke-result.json",
        (json.dumps(result, sort_keys=True) + "\n").encode(),
        role="smoke_result",
    )
    return result
