"""Real native exports,independent checkpoint lineage and strict snapshot guards."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from qvelt.eval.config import ArtifactRef
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.inference.artifacts import export_snapshot, load_snapshot
from qvelt.inference.config import (
    GenerationJob,
    GenerationOptions,
    TextRequest,
    TokenRequest,
    dump_generation,
)
from qvelt.inference.generation import GenerationEngine
from qvelt.inference.smoke import run_inference_smoke
from qvelt.model.cache import parameter_stamp
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore, hash_file
from qvelt.utils.manifests import manifest_bytes, manifest_digest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[ArtifactRef, Path]:
    root = tmp_path_factory.mktemp("inference")
    fixture = create_evaluation_fixture(root / "source", REPO)
    manifest = export_snapshot(
        fixture.checkpoint, fixture.tokenizer, root / "snapshot", REPO
    )
    return ArtifactRef(
        str(root / "snapshot/manifest.json"), manifest_digest(manifest)
    ), root


def test_export_relocates_without_sources_and_matches_fresh_cli(
    exported: tuple[ArtifactRef, Path], tmp_path: Path
) -> None:
    reference, source = exported
    relocated = tmp_path / "copied"
    shutil.copytree(Path(reference.path).parent, relocated)
    reference = replace(reference, path=str(relocated / "manifest.json"))
    loaded = load_snapshot(reference)
    engine = GenerationEngine(loaded.model, loaded.tokenizer)
    requests = (
        TextRequest("one", "Привет 😀", GenerationOptions(max_new_tokens=4)),
        TextRequest(
            "two", "Original longer prompt.", GenerationOptions(max_new_tokens=4)
        ),
    )
    expected = engine.generate(engine.encode_requests(requests))
    path = tmp_path / "generate.yaml"
    path.write_text(dump_generation(GenerationJob(reference, requests)))
    original = source / "source"
    hidden = source / "hidden-source"
    original.rename(hidden)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "qvelt", "generate", str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        actual = json.loads(process.stdout)
    finally:
        hidden.rename(original)
    assert [r["token_ids"] for r in actual] == [list(r.token_ids) for r in expected]
    assert [r["text"] for r in actual] == [r.text for r in expected]
    assert not any(
        f.role in ("optimizer", "rng", "sampler") for f in loaded.manifest.files
    )


def test_load_never_unpickles_preserves_rng_and_normal_parameters(
    exported: tuple[ArtifactRef, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Inference must not deserialize training pickle")

    monkeypatch.setattr(torch, "load", reject)
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        snapshot = load_snapshot(exported[0])
        assert parameter_stamp(snapshot.model)
        result = GenerationEngine(snapshot.model, snapshot.tokenizer).generate(
            [TokenRequest("test", (1,), GenerationOptions(max_new_tokens=3))]
        )
        assert result and torch.is_inference_mode_enabled()
    assert torch.equal(rng, torch.get_rng_state())


def rewrite_descriptor(
    reference: ArtifactRef, root: Path, change: dict[str, Any]
) -> ArtifactRef:
    shutil.copytree(Path(reference.path).parent, root)
    descriptor = root / "inference.json"
    value = json.loads(descriptor.read_text())
    value.update(change)
    descriptor.write_text(json.dumps(value))
    store = ArtifactStore(root)
    manifest = store.read_manifest(verify=False)
    info = hash_file(descriptor)
    files = tuple(
        replace(f, sha256=info.sha256, size_bytes=info.size_bytes)
        if f.role == "inference_descriptor"
        else f
        for f in manifest.files
    )
    changed = replace(manifest, files=files)
    (root / "manifest.json").write_bytes(manifest_bytes(changed))
    return ArtifactRef(str(root / "manifest.json"), manifest_digest(changed))


@pytest.mark.parametrize(
    "change",
    [
        {"format": "gguf"},
        {"schema_version": True},
        {"schema_version": 2},
        {"dtype": "int8"},
        {"unknown": "flag"},
        {"parameters": True},
        {"parameters": 1},
    ],
)
def test_valid_hashes_do_not_bypass_protocol_or_shape_checks(
    exported: tuple[ArtifactRef, Path], tmp_path: Path, change: dict[str, Any]
) -> None:
    ref = rewrite_descriptor(exported[0], tmp_path / "bad", change)
    with pytest.raises(ValueError):
        load_snapshot(ref)


def test_corruption_missing_marker_and_ambiguous_roles_fail(
    exported: tuple[ArtifactRef, Path], tmp_path: Path
) -> None:
    root = tmp_path / "bad"
    shutil.copytree(Path(exported[0].path).parent, root)
    reference = replace(exported[0], path=str(root / "manifest.json"))
    (root / "model.safetensors").write_bytes(b"not a tensor file")
    with pytest.raises(ArtifactIntegrityError):
        load_snapshot(reference)
    (root / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_snapshot(reference)
    root2 = tmp_path / "ambiguous"
    shutil.copytree(Path(exported[0].path).parent, root2)
    manifest = ArtifactStore(root2).read_manifest()
    changed = replace(
        manifest,
        files=tuple(
            replace(f, role="model") if f.role == "tokenizer" else f
            for f in manifest.files
        ),
    )
    (root2 / "manifest.json").write_bytes(manifest_bytes(changed))
    with pytest.raises(ValueError, match="roles"):
        load_snapshot(
            ArtifactRef(str(root2 / "manifest.json"), manifest_digest(changed))
        )


def test_wrong_tokenizer_aborts_before_destination(
    exported: tuple[ArtifactRef, Path], tmp_path: Path
) -> None:
    root = exported[1]
    cp = root / "source/checkpoint/checkpoints/step-00000000/manifest.json"
    meta = ArtifactStore(cp.parent).read_manifest()
    token = root / "source/model-data/prepared/tokenizer/manifest.json"
    with pytest.raises(ValueError, match="tokenizer"):
        export_snapshot(
            ArtifactRef(str(cp), manifest_digest(meta)),
            ArtifactRef(str(token), "f" * 64),
            tmp_path / "denied",
            REPO,
        )
    assert not (tmp_path / "denied").exists()


def test_standalone_workflow_checks_cli_stream_and_export_references(
    tmp_path: Path,
) -> None:
    result = run_inference_smoke(tmp_path / "workflow", REPO)
    assert result["maximum_export_reload_logit_delta"] == 0
    assert result["final_cases_scored"] == 0
    assert all(
        p["cached_uncached_tokens_text_stops_equal"]
        and p["stream_events_equal"]
        and p["source_hidden_reload_equal"]
        for p in result["protocols"].values()
    )
