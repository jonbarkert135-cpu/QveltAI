"""Pinned original GSM8K train-question ingress; no Socratic or test-data ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.eval.config import ArtifactRef
from qvelt.posttrain.verifiers import rational
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.checkpoints import source_digest
from qvelt.utils.manifest_types import Manifest, SourceIdentity
from qvelt.utils.manifests import _reject_constant, _unique_mapping

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
TRAIN_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
LICENSE_SHA256 = "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b"


def parse_training_rows(data: bytes, count: int) -> list[RawDocument]:
    """Extract numeric labels only; original free-text solutions are not proof-checked."""
    if type(count) is not int or not 1 <= count <= 64 or len(data) > 8 * 1024**2:
        raise ValueError("GSM8K development ingress is bounded to1..64 questions/8MiB")
    lines = data.splitlines()
    if len(lines) < count:
        raise ValueError("GSM8K payload lacks requested rows")
    records = []
    for line in lines[:count]:
        row = json.loads(
            line, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
        )
        if (
            not isinstance(row, dict)
            or set(row) != {"question", "answer"}
            or any(
                not isinstance(value, str) or not 1 <= len(value) <= 8192
                for value in row.values()
            )
            or len(row["answer"].split("####")) != 2
        ):
            raise ValueError("Invalid original GSM8K question/answer fields")
        answer = row["answer"].split("####")[1].strip().replace(",", "")
        rational(answer)
        question = row["question"]
        identifier = hashlib.sha256(question.encode()).hexdigest()
        records.append(
            RawDocument(
                question, identifier, "gsm8k-original-train", "en", "unrated", answer
            )
        )
    if len({r.identifier for r in records}) != len(records):
        raise ValueError("Duplicate GSM8K source problems")
    return records


def import_gsm8k_training(
    payload: ArtifactRef, license_file: ArtifactRef, directory: Path, *, count: int = 16
) -> Manifest:
    """Only the exact DATA_CARD-reviewed train/license bytes are accepted."""
    values = []
    for ref, expected, limit in (
        (payload, TRAIN_SHA256, 8 * 1024**2),
        (license_file, LICENSE_SHA256, 8192),
    ):
        path = Path(ref.path)
        with ArtifactStore(path.parent).open_file(path.name) as stream:
            data = stream.read(limit + 1)
        if (
            len(data) > limit
            or ref.sha256 != expected
            or hashlib.sha256(data).hexdigest() != expected
        ):
            raise ValueError(
                "GSM8K bytes must match the pinned reviewed original train/license snapshot"
            )
        values.append(data)
    records = parse_training_rows(values[0], count)
    source = SourceIdentity(
        "gsm8k-original-train-development",
        REVISION,
        "MIT",
        f"https://github.com/openai/grade-school-math/blob/{REVISION}/LICENSE",
    )
    root = create_directory(directory)
    original = write_raw_bundle(records, root / "source", source, values[1])
    store = ArtifactStore(root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "qvelt-gsm8k-original-train-v1",
        "revision": REVISION,
        "source_sha256": source_digest(),
        "payload_sha256": TRAIN_SHA256,
        "payload_bytes": len(values[0]),
        "payload_rows": len(values[0].splitlines()),
        "license_sha256": LICENSE_SHA256,
        "imported_problems": len(records),
        "difficulty": "unrated",
        "gold_status": "publisher_numeric_label_not_step_verification",
        "scope": "bounded development problems; no optimization or production clearance",
        "source_url": f"https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/grade_school_math/data/train.jsonl",
    }
    report_file = store.add_bytes(
        "import.json",
        (json.dumps(report, sort_keys=True) + "\n").encode(),
        role="import_report",
    )
    source_manifest = store.add_file(
        "source.manifest.json", root / "source/manifest.json", role="source_manifest"
    )
    manifest = replace(
        original,
        files=(
            *(replace(f, path="source/" + f.path) for f in original.files),
            report_file,
            source_manifest,
        ),
    )
    store.publish(manifest)
    return manifest
