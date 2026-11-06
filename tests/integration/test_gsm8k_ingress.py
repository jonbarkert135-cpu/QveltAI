"""Offline source-pinning/publication oracle using an explicitly original fixture."""

import hashlib
import json
from pathlib import Path

import pytest

from qvelt.data.records import iter_records
from qvelt.eval.config import ArtifactRef
from qvelt.posttrain import gsm8k
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest


def test_pinned_ingress_publishes_complete_raw_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only this isolated test replaces pinned hashes; production accepts exactly
    # the publisher bytes measured in DATA_CARD, verified separately in acceptance.
    payload = tmp_path / "data.jsonl"
    payload.write_text(
        json.dumps(
            {
                "question": "Original unit fixture: 8+9?",
                "answer": "Eight plus nine.\n#### 17",
            }
        )
        + "\n"
    )
    license_file = tmp_path / "permission"
    license_file.write_text("Original unit-only source permission fixture.")
    phash = hashlib.sha256(payload.read_bytes()).hexdigest()
    lhash = hashlib.sha256(license_file.read_bytes()).hexdigest()
    monkeypatch.setattr(gsm8k, "TRAIN_SHA256", phash)
    monkeypatch.setattr(gsm8k, "LICENSE_SHA256", lhash)
    manifest = gsm8k.import_gsm8k_training(
        ArtifactRef(str(payload), phash),
        ArtifactRef(str(license_file), lhash),
        tmp_path / "out",
        count=1,
    )
    observed = ArtifactStore(tmp_path / "out").read_manifest(
        expected_sha256=manifest_digest(manifest)
    )
    rows = list(iter_records(tmp_path / "out/manifest.json", manifest_digest(observed)))
    assert len(rows) == 1 and rows[0].answer == "17"
    assert (
        json.loads((tmp_path / "out/import.json").read_text())["payload_sha256"]
        == phash
    )
