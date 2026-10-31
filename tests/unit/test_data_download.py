"""Offline HTTPS response doubles exercise checksum, byte budgets and source-use gates."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from qvelt.data.download import DownloadSpec, _fetch, acquire_source, https_url
from qvelt.data.records import iter_records
from qvelt.utils.manifest_types import DataInfo
from qvelt.utils.manifests import manifest_digest


class Response(io.BytesIO):
    def geturl(self) -> str:
        return "https://publisher.invalid/immutable.jsonl"


def spec(payload: bytes, license_text: bytes) -> DownloadSpec:
    return DownloadSpec(
        "test-publisher",
        "immutable-v1",
        "https://publisher.invalid/immutable.jsonl",
        hashlib.sha256(payload).hexdigest(),
        "MIT",
        "https://publisher.invalid/LICENSE",
        hashlib.sha256(license_text).hexdigest(),
        True,
        "evaluation",
        "gsm8k-questions",
        100000,
    )


def test_offline_acquisition_extracts_only_questions_and_keeps_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b'{"question":"An original offline adapter question?","answer":"A gold answer that never trains"}\n'
    license_text = b"MIT: original test-only response fixture"
    recipe = spec(data, license_text)

    def urlopen(url: str, **kwargs: Any) -> Response:
        return Response(license_text if url.endswith("LICENSE") else data)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(asdict(recipe)))
    manifest = acquire_source(path, tmp_path / "acquired")
    assert isinstance(manifest.metadata, DataInfo) and manifest.metadata.documents == 1
    records = list(
        iter_records(tmp_path / "acquired/raw/manifest.json", manifest_digest(manifest))
    )
    assert (
        records[0].text == "An original offline adapter question?"
        and not records[0].answer
    )
    assert (tmp_path / "acquired/raw/PERMISSION.txt").read_bytes() == license_text
    uses = json.loads((tmp_path / "acquired/raw/usage.json").read_bytes())[
        "allowed_uses"
    ]
    assert uses == ["evaluation", "decontamination"]
    assert "training" not in uses


def test_download_rejects_wrong_hash_and_oversize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: Response(b"abcdef")
    )
    with pytest.raises(ValueError, match="checksum"):
        _fetch("https://publisher.invalid/data", "0" * 64, 10, tmp_path / "wrong")
    with pytest.raises(ValueError, match="budget"):
        _fetch(
            "https://publisher.invalid/data",
            hashlib.sha256(b"abcdef").hexdigest(),
            3,
            tmp_path / "oversized",
        )
    recipe = spec(b"{}", b"LICENSE")
    with pytest.raises(ValueError, match="not approved"):
        replace(recipe, approved=False)
    with pytest.raises(ValueError, match="evaluation"):
        replace(recipe, use="training")


@pytest.mark.parametrize(
    "url",
    [
        "http://host/a",
        "https://user:password@host/a",
        "https://host/a?token=secret",
        "file:///tmp/source",
        "https://host/a#fragment",
    ],
)
def test_source_urls_do_not_embed_credentials_or_plain_http(url: str) -> None:
    with pytest.raises(ValueError):
        https_url(url)
