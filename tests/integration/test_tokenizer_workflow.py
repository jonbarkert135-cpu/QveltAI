"""Real local corpus artifacts, semantic failures and independent CLI processes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

import pytest

from qvelt.tokenizer.core import TokenizerError, TokenizerSettings, TokenizerV3Settings
from qvelt.tokenizer.recipe import TokenizerRecipe, load_tokenizer_recipe
from qvelt.tokenizer.workflow import (
    evaluate_tokenizer,
    iter_documents,
    load_tokenizer_bundle,
    train_tokenizer_bundle,
)
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore, hash_file
from qvelt.utils.manifest_types import (
    ArtifactFile,
    DataInfo,
    Manifest,
    ManifestError,
    SourceIdentity,
    TokenizerInfo,
)
from qvelt.utils.manifests import manifest_bytes, manifest_digest, parse_manifest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/tokenizer"
RECIPE = load_tokenizer_recipe(ROOT / "configs/tokenizer_smoke.yaml")
HELDOUT_ID = manifest_digest(
    parse_manifest((FIXTURES / "heldout.manifest.json").read_bytes())
)


def test_v3_bundle_roundtrip_and_deterministic_payload(tmp_path: Path) -> None:
    recipe = TokenizerRecipe(
        RECIPE.training_data_sha256,
        schema_version=2,
        tokenizer=TokenizerV3Settings(512),
    )
    digests = []
    for name in ("first-v3", "second-v3"):
        path = tmp_path / name
        manifest = train_tokenizer_bundle(
            recipe, FIXTURES / "train.manifest.json", path, ROOT
        )
        tokenizer, loaded = load_tokenizer_bundle(
            path / "manifest.json", manifest_digest(manifest)
        )
        assert loaded == manifest
        assert isinstance(loaded.metadata, TokenizerInfo)
        assert loaded.metadata.reserved_tokens == 128
        assert tokenizer.control_version == 3
        assert tokenizer.vocab_size == 512
        assert load_tokenizer_recipe(path / "recipe.yaml") == recipe
        metrics = evaluate_tokenizer(
            tokenizer, FIXTURES / "heldout.manifest.json", HELDOUT_ID
        )
        assert metrics.documents == 12 and metrics.tokens > 0
        digests.append(
            hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
        )
    assert digests[0] == digests[1]
    # Timestamp/provenance-bearing manifests are not claimed to have equal hashes.


def test_v3_bundle_rejects_legacy_recipe_relabelling(tmp_path: Path) -> None:
    recipe = TokenizerRecipe(
        RECIPE.training_data_sha256,
        schema_version=2,
        tokenizer=TokenizerV3Settings(512),
    )
    path = tmp_path / "v3"
    train_tokenizer_bundle(recipe, FIXTURES / "train.manifest.json", path, ROOT)
    raw = (
        (path / "recipe.yaml")
        .read_bytes()
        .replace(b"schema_version: 2", b"schema_version: 1")
    )
    rewrite_file(path / "manifest.json", "recipe.yaml", raw)
    with pytest.raises(TokenizerError, match="metadata"):
        load_tokenizer_bundle(path / "manifest.json")


def test_v3_cli_uses_explicit_recipe_version(tmp_path: Path) -> None:
    from qvelt.tokenizer.recipe import dump_tokenizer_recipe

    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        dump_tokenizer_recipe(
            TokenizerRecipe(
                RECIPE.training_data_sha256,
                schema_version=2,
                tokenizer=TokenizerV3Settings(512),
            )
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qvelt",
            "tokenizer",
            "train",
            str(recipe_path),
            "--data-manifest",
            str(FIXTURES / "train.manifest.json"),
            "--directory",
            str(tmp_path / "cli-v3"),
            "--repository",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    output = json.loads(result.stdout)
    assert output["metadata"]["reserved_tokens"] == 128
    tokenizer, _ = load_tokenizer_bundle(
        tmp_path / "cli-v3/manifest.json", output["sha256"]
    )
    assert tokenizer.control_version == 3


def corpus(tmp_path: Path, parts: dict[str, bytes], documents: int) -> tuple[Path, str]:
    directory = tmp_path / "raw"
    directory.mkdir()
    store = ArtifactStore(directory)
    files = tuple(
        store.add_bytes(name, data, role="data") for name, data in parts.items()
    )
    source = SourceIdentity(
        "authored-test",
        "fixture-v1",
        "LicenseRef-TestFixture",
        "https://example.invalid/license",
    )
    manifest = Manifest(
        "data",
        "fixture",
        "2026-10-31T00:00:00Z",
        DataInfo("raw", (source,), documents, None, None),
        files,
    )
    store.publish(manifest)
    return directory / "manifest.json", manifest_digest(manifest)


def rewrite_file(path: Path, name: str, payload: bytes) -> None:
    """Intentionally rewrite bytes AND the hash to test semantic validation."""
    manifest = parse_manifest(path.read_bytes())
    (path.parent / name).write_bytes(payload)
    files = tuple(
        ArtifactFile(f.path, f.role, len(payload), hashlib.sha256(payload).hexdigest())
        if f.path == name
        else f
        for f in manifest.files
    )
    path.write_bytes(manifest_bytes(replace(manifest, files=files)))


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    train_tokenizer_bundle(
        RECIPE, FIXTURES / "train.manifest.json", tmp_path / "output", ROOT
    )
    return tmp_path / "output/manifest.json"


def test_bundle_lineage_and_identity(bundle: Path) -> None:
    tokenizer, manifest = load_tokenizer_bundle(bundle)
    assert isinstance(manifest.metadata, TokenizerInfo)
    assert manifest.metadata.training_data_sha256 == RECIPE.training_data_sha256
    assert manifest.metadata.vocab_size == tokenizer.vocab_size <= 512
    assert len(manifest.files) == 5
    copied = parse_manifest(
        (bundle.parent / "training-data.manifest.json").read_bytes()
    )
    assert manifest_digest(copied) == RECIPE.training_data_sha256
    load_tokenizer_bundle(bundle, manifest_digest(manifest))
    with pytest.raises(ManifestError, match="identity"):
        load_tokenizer_bundle(bundle, "0" * 64)
    provenance = json.loads((bundle.parent / "provenance.json").read_bytes())
    assert provenance["code"]["lock_sha256"] == hash_file(ROOT / "uv.lock").sha256


def test_ordered_documents(tmp_path: Path) -> None:
    path, identity = corpus(
        tmp_path,
        {
            "z.jsonl": b'{"text":"last"}\n',
            "a.jsonl": b'{"text":"first\\nline"}\n{"text":"second"}',
        },
        3,
    )
    assert list(iter_documents(path, identity)) == ["first\nline", "second", "last"]


@pytest.mark.parametrize(
    "data",
    [
        b"{}\n",
        b"[]\n",
        b'{"text":null}\n',
        b'{"text":NaN}\n',
        b'{"text":"a","extra":1}\n',
        b'{"text":"a","text":"b"}\n',
        b"\xff\n",
        b"\n",
        b'{"text":"\\ud800"}\n',
        b'{"text":"a"}\ntrailing',
        b"[" * 1100,
        b'{"text":"' + b"a" * (1024 * 1024 + 1) + b'"}',
    ],
    ids=lambda data: "invalid-" + hashlib.sha256(data).hexdigest()[:8],
)
def test_invalid_jsonl_has_no_committed_output(tmp_path: Path, data: bytes) -> None:
    path, identity = corpus(tmp_path, {"data.jsonl": data}, 1)
    with pytest.raises(TokenizerError):
        train_tokenizer_bundle(
            TokenizerRecipe(identity, tokenizer=TokenizerSettings(288)),
            path,
            tmp_path / "out",
            ROOT,
        )
    assert not (tmp_path / "out/manifest.json").exists()


def test_document_count_and_empty_training_fail(tmp_path: Path) -> None:
    path, identity = corpus(tmp_path, {"data.jsonl": b'{"text":"a"}\n'}, 2)
    with pytest.raises(ArtifactIntegrityError, match="count"):
        list(iter_documents(path, identity))
    with pytest.raises(ArtifactIntegrityError, match="count"):
        train_tokenizer_bundle(TokenizerRecipe(identity), path, tmp_path / "out", ROOT)
    assert not (tmp_path / "out/manifest.json").exists()


def test_byte_change_after_preverification_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, identity = corpus(tmp_path, {"data.jsonl": b'{"text":"a"}\n'}, 1)
    original = ArtifactStore.open_file

    @contextmanager
    def changed(self: ArtifactStore, relative: str) -> Iterator[BinaryIO]:
        (self.root / relative).write_bytes(b'{"text":"b"}\n')
        with original(self, relative) as stream:
            yield stream

    monkeypatch.setattr(ArtifactStore, "open_file", changed)
    with pytest.raises(ArtifactIntegrityError, match="Consumed"):
        list(iter_documents(path, identity))


def test_growing_input_does_not_follow_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, identity = corpus(tmp_path, {"data.jsonl": b'{"text":"a"}\n'}, 1)
    original = ArtifactStore.open_file

    @contextmanager
    def growing(self: ArtifactStore, relative: str) -> Iterator[BinaryIO]:
        with (self.root / relative).open("ab") as stream:
            stream.write(b'{"text":"b"}\n' * 100)
        with original(self, relative) as stream:
            yield stream

    monkeypatch.setattr(ArtifactStore, "open_file", growing)
    with pytest.raises(ArtifactIntegrityError, match="grew"):
        list(iter_documents(path, identity))


def test_existing_output_survives(bundle: Path) -> None:
    before = bundle.read_bytes()
    with pytest.raises(FileExistsError):
        train_tokenizer_bundle(
            RECIPE, FIXTURES / "train.manifest.json", bundle.parent, ROOT
        )
    assert bundle.read_bytes() == before


def test_corruption_fails_byte_verification(bundle: Path) -> None:
    (bundle.parent / "tokenizer.json").write_bytes(b"{}")
    with pytest.raises(ArtifactIntegrityError, match="size/hash"):
        load_tokenizer_bundle(bundle)


@pytest.mark.parametrize(
    "change", ["digit_policy", "data_identity", "vocab_ceiling", "owner"]
)
def test_semantic_recipe_mismatches(bundle: Path, change: str) -> None:
    import yaml

    recipe = yaml.safe_load((bundle.parent / "recipe.yaml").read_text())
    if change == "digit_policy":
        recipe["tokenizer"]["isolate_digits"] = False
    if change == "data_identity":
        recipe["training_data_sha256"] = "b" * 64
    if change == "vocab_ceiling":
        recipe["tokenizer"]["vocab_size"] = 288
    if change == "owner":
        recipe["name"] = "other"
    rewrite_file(bundle, "recipe.yaml", yaml.safe_dump(recipe).encode())
    with pytest.raises(TokenizerError, match="disagrees"):
        load_tokenizer_bundle(bundle)


def test_metadata_vocab_is_checked(bundle: Path) -> None:
    manifest = parse_manifest(bundle.read_bytes())
    assert isinstance(manifest.metadata, TokenizerInfo)
    bundle.write_bytes(
        manifest_bytes(
            replace(manifest, metadata=replace(manifest.metadata, vocab_size=32768))
        )
    )
    with pytest.raises(TokenizerError, match="metadata"):
        load_tokenizer_bundle(bundle)


def test_provenance_links_checked(bundle: Path) -> None:
    provenance = json.loads((bundle.parent / "provenance.json").read_bytes())
    provenance["code"]["lock_sha256"] = "0" * 64
    rewrite_file(bundle, "provenance.json", json.dumps(provenance).encode())
    with pytest.raises(TokenizerError, match="provenance"):
        load_tokenizer_bundle(bundle)


def test_exact_reference_metrics(tmp_path: Path) -> None:
    from qvelt.tokenizer.core import train_tokenizer

    path, identity = corpus(
        tmp_path,
        {"data.jsonl": '{"text":"Aéя"}\n{"text":""}\n{"text":"12"}\n'.encode()},
        3,
    )
    tokenizer = train_tokenizer(["a"], TokenizerSettings(288))
    metrics = evaluate_tokenizer(tokenizer, path, identity)
    # Independent UTF-8 alphabet oracle: each byte has exactly one symbol.
    assert (
        metrics.documents,
        metrics.utf8_bytes,
        metrics.characters,
        metrics.tokens,
    ) == (3, 7, 5, 7)
    assert metrics.bytes_per_token == 1.0


def test_heldout_roundtrips_and_metrics(bundle: Path) -> None:
    tokenizer, _ = load_tokenizer_bundle(bundle)
    result = evaluate_tokenizer(
        tokenizer, FIXTURES / "heldout.manifest.json", HELDOUT_ID
    )
    assert result.documents == 12
    assert result.tokens > 0 and result.utf8_bytes > 0


def cli(*args: str, hash_seed: str = "123") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "qvelt", "tokenizer", *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_independent_process_recipe_train_verify_evaluate(tmp_path: Path) -> None:
    outputs = []
    for name, seed in [("first", "1"), ("second", "234567")]:
        directory = tmp_path / name
        run = cli(
            "train",
            str(ROOT / "configs/tokenizer_smoke.yaml"),
            "--data-manifest",
            str(FIXTURES / "train.manifest.json"),
            "--directory",
            str(directory),
            "--repository",
            str(ROOT),
            hash_seed=seed,
        )
        assert run.returncode == 0, run.stderr
        outputs.append((directory, json.loads(run.stdout)))
    assert (outputs[0][0] / "tokenizer.json").read_bytes() == (
        outputs[1][0] / "tokenizer.json"
    ).read_bytes()
    directory, output = outputs[0]
    verified = cli(
        "verify",
        str(directory / "manifest.json"),
        "--expected-sha256",
        output["sha256"],
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["verified"]
    result = cli(
        "evaluate",
        str(directory / "manifest.json"),
        "--data-manifest",
        str(FIXTURES / "heldout.manifest.json"),
        "--data-sha256",
        HELDOUT_ID,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["roundtrips_verified"]
    resolved = cli("resolve", str(ROOT / "configs/tokenizer_smoke.yaml"))
    assert resolved.returncode == 0 and RECIPE.training_data_sha256 in resolved.stdout
    wrong = cli(
        "verify", str(directory / "manifest.json"), "--expected-sha256", "0" * 64
    )
    assert wrong.returncode == 1
    (directory / "tokenizer.json").write_bytes(b"corrupt")
    broken = cli("verify", str(directory / "manifest.json"))
    assert broken.returncode == 1 and "Traceback" not in broken.stderr


def test_empty_corpus_cannot_publish(tmp_path: Path) -> None:
    path, identity = corpus(tmp_path, {"empty.jsonl": b""}, 0)
    with pytest.raises(TokenizerError, match="nonempty"):
        train_tokenizer_bundle(TokenizerRecipe(identity), path, tmp_path / "out", ROOT)
    assert not (tmp_path / "out/manifest.json").exists()


def test_record_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import qvelt.tokenizer.workflow as workflow

    path, identity = corpus(tmp_path, {"data.jsonl": b'{"text":"long value"}\n'}, 1)
    monkeypatch.setattr(workflow, "MAX_JSONL_BYTES", 12)
    with pytest.raises(ArtifactIntegrityError, match="size limit"):
        list(iter_documents(path, identity))


def test_input_type_and_extension(tmp_path: Path) -> None:
    path, identity = corpus(tmp_path, {"data.txt": b'{"text":"text"}\n'}, 1)
    with pytest.raises(TokenizerError, match="JSONL"):
        list(iter_documents(path, identity))
    manifest = parse_manifest(path.read_bytes())
    assert isinstance(manifest.metadata, DataInfo)
    changed = replace(
        manifest,
        metadata=replace(
            manifest.metadata,
            representation="token_ids",
            tokens=1,
            tokenizer_sha256="a" * 64,
        ),
    )
    path.write_bytes(manifest_bytes(changed))
    with pytest.raises(TokenizerError, match="raw-data"):
        list(iter_documents(path, manifest_digest(changed)))


def test_required_role_is_unique(bundle: Path) -> None:
    manifest = parse_manifest(bundle.read_bytes())
    path = bundle.parent / "duplicate.json"
    payload = (bundle.parent / "tokenizer.json").read_bytes()
    path.write_bytes(payload)
    changed = replace(
        manifest,
        files=manifest.files
        + (
            ArtifactFile(
                path.name,
                "tokenizer",
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            ),
        ),
    )
    bundle.write_bytes(manifest_bytes(changed))
    with pytest.raises(TokenizerError, match="exactly one tokenizer"):
        load_tokenizer_bundle(bundle)


def test_recorded_tokenizer_semantics_checked(bundle: Path) -> None:
    data = json.loads((bundle.parent / "tokenizer.json").read_bytes())
    data["normalizer"] = {"type": "Lowercase"}
    rewrite_file(bundle, "tokenizer.json", json.dumps(data).encode())
    with pytest.raises(TokenizerError, match="policy"):
        load_tokenizer_bundle(bundle)


def test_empty_evaluation_denominator(tmp_path: Path) -> None:
    from qvelt.tokenizer.core import train_tokenizer

    path, identity = corpus(tmp_path, {"empty.jsonl": b""}, 0)
    metrics = evaluate_tokenizer(train_tokenizer(["x"]), path, identity)
    assert metrics.documents == 0 and metrics.tokens == 0
    assert metrics.bytes_per_token is None
