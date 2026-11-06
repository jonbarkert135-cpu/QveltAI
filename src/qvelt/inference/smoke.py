"""Actual native export,fresh-process reload and batched cached/uncached parity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from qvelt.eval.config import ArtifactRef
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.inference.artifacts import load_checkpoint_model, load_snapshot
from qvelt.inference.config import (
    GenerationJob,
    GenerationOptions,
    TextRequest,
    dump_generation,
)
from qvelt.inference.generation import GenerationEngine
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.checkpoints import source_digest
from qvelt.utils.manifests import manifest_digest


def _cli(args: list[str]) -> str:
    process = subprocess.run(
        [sys.executable, "-m", "qvelt", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    return process.stdout


def run_inference_smoke(directory: Path, repository: Path) -> dict[str, Any]:
    root = create_directory(directory)
    fixture = create_evaluation_fixture(root / "source", repository)
    exported = json.loads(
        _cli(
            [
                "export",
                fixture.checkpoint.path,
                "--checkpoint-sha256",
                fixture.checkpoint.sha256,
                "--tokenizer-manifest",
                fixture.tokenizer.path,
                "--tokenizer-sha256",
                fixture.tokenizer.sha256,
                "--directory",
                str(root / "snapshot"),
                "--repository",
                str(repository),
            ]
        )
    )
    manifest = ArtifactStore(root / "snapshot").read_manifest()
    assert exported["sha256"] == manifest_digest(manifest)
    reference = ArtifactRef(
        str(root / "snapshot/manifest.json"), manifest_digest(manifest)
    )
    snapshot = load_snapshot(reference)
    model, config, _ = load_checkpoint_model(
        fixture.checkpoint, fixture.tokenizer.sha256
    )
    generator = torch.Generator().manual_seed(20261106)
    maximum_delta = 0.0
    with torch.no_grad():
        for length in (1, 3, 17, 61):
            probe_tokens = torch.randint(
                32, config.model.vocab, (2, length), generator=generator
            )
            before, after = model(probe_tokens), snapshot.model(probe_tokens)
            maximum_delta = max(maximum_delta, float((before - after).abs().max()))
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    original_engine = GenerationEngine(model, snapshot.tokenizer)
    engine = GenerationEngine(snapshot.model, snapshot.tokenizer)
    receipts = {}
    for strategy in ("greedy", "sample"):
        options = GenerationOptions(
            strategy=strategy,
            temperature=0 if strategy == "greedy" else 0.8,
            max_new_tokens=16,
        )
        requests = tuple(
            TextRequest(name, prompt, replace(options, seed=20261106 + i))
            for i, (name, prompt) in enumerate(
                (
                    ("ascii", "Return 2 + 3."),
                    ("unicode", "Привет 😀 café"),
                    ("longer", "Write one short original mathematical statement."),
                )
            )
        )
        tokens = engine.encode_requests(requests)
        original = original_engine.generate(tokens)
        cached = engine.generate(tokens)
        plain = engine.generate(tokens, use_cache=False)
        assert original == cached
        assert [(r.token_ids, r.text, r.reason) for r in cached] == [
            (r.token_ids, r.text, r.reason) for r in plain
        ]
        recipe = GenerationJob(reference, requests)
        path = root / (strategy + ".yaml")
        path.write_text(dump_generation(recipe))
        expected = json.loads(json.dumps([asdict(r) for r in cached]))
        assert json.loads(_cli(["generate", str(path)])) == expected
        events = list(engine.stream(tokens))
        done = {e.identifier: e for e in events if e.done}
        assert all(
            done[r.identifier].text == r.text and done[r.identifier].reason == r.reason
            for r in cached
        )
        # A standalone snapshot must not resolve any original checkpoint/data files.
        (root / "source").rename(root / "source-hidden")
        try:
            assert json.loads(_cli(["generate", str(path)])) == expected
            streamed = [
                json.loads(line)
                for line in _cli(["generate", str(path), "--stream"]).splitlines()
            ]
            assert streamed == [asdict(e) for e in events]
        finally:
            (root / "source-hidden").rename(root / "source")
        receipts[strategy] = {
            "requests": len(requests),
            "sampled_tokens": sum(len(r.token_ids) for r in cached),
            "cached_uncached_tokens_text_stops_equal": True,
            "export_reload_equal": True,
            "cli_api_equal": True,
            "stream_events_equal": True,
            "source_hidden_reload_equal": True,
        }
    result = {
        "scope": "Untrained original BPE CPU engineering snapshot;not reasoning quality or serving performance",
        "source_sha256": source_digest(),
        "snapshot_sha256": reference.sha256,
        "checkpoint_sha256": fixture.checkpoint.sha256,
        "tokenizer_sha256": fixture.tokenizer.sha256,
        "parameters": config.model.parameter_count,
        "maximum_export_reload_logit_delta": maximum_delta,
        "logit_shapes": [[2, n, config.model.vocab] for n in (1, 3, 17, 61)],
        "protocols": receipts,
        "device": "cpu",
        "precision": "fp32",
        "batch": 3,
        "torch": str(torch.__version__),
        "final_cases_scored": 0,
        "quantization_verified": False,
        "external_runtime_verified": False,
    }
    ArtifactStore(root).add_bytes(
        "result.json",
        (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode(),
        role="inference_acceptance",
    )
    return result
