"""Approved, bounded, immutable HTTPS JSONL acquisition; no scraping or discovery."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from qvelt.config import MAX_CONFIG_BYTES, parse_yaml_mapping
from qvelt.data.config import decode
from qvelt.data.records import RawDocument, write_raw_bundle
from qvelt.utils.artifacts import create_directory
from qvelt.utils.manifest_types import Manifest, SourceIdentity, digest, identifier
from qvelt.utils.manifests import _reject_constant, _unique_mapping


def https_url(value: str) -> None:
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or len(value) > 2048
    ):
        raise ValueError(
            "Source URLs require plain HTTPS without credentials/query/fragment"
        )


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    revision: str
    url: str
    sha256: str
    license_id: str
    license_url: str
    license_sha256: str
    approved: bool
    use: str
    format: str
    max_bytes: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.name, "download.name")
        SourceIdentity(self.name, self.revision, self.license_id, self.license_url)
        digest(self.sha256, "download.sha256")
        digest(self.license_sha256, "download.license_sha256")
        https_url(self.url)
        https_url(self.license_url)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported download spec version")
        if type(self.approved) is not bool or not self.approved:
            raise ValueError("Dataset download is not approved")
        if self.use not in ("training", "evaluation") or self.format not in (
            "text-jsonl",
            "gsm8k-questions",
        ):
            raise ValueError("Unsupported source use/format; no implicit converter")
        if self.format == "gsm8k-questions" and self.use != "evaluation":
            raise ValueError("This GSM8K adapter is approved only for evaluation")
        if type(self.max_bytes) is not int or not 1 <= self.max_bytes <= 2**40:
            raise ValueError("Invalid download size bound")


def load_download_spec(path: Path) -> DownloadSpec:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError("Download spec exceeds its size limit")
    return decode(DownloadSpec, parse_yaml_mapping(data.decode("utf-8")))


def _fetch(url: str, sha256: str, maximum: int, destination: Path) -> None:
    """Honor inherited proxies/TLS trust, bound bytes and verify before parsing."""
    https_url(url)
    checksum = hashlib.sha256()
    size = 0
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        destination.open("xb") as stream,
    ):
        https_url(response.geturl())
        while block := response.read(min(1024 * 1024, maximum - size + 1)):
            size += len(block)
            if size > maximum:
                raise ValueError("Source exceeded its approved download budget")
            checksum.update(block)
            stream.write(block)
    if checksum.hexdigest() != sha256:
        raise ValueError("Downloaded source checksum mismatch")


def acquire_source(path: Path, directory: Path) -> Manifest:
    spec = load_download_spec(path)
    root = create_directory(directory)
    with tempfile.TemporaryDirectory(prefix=".acquire-", dir=root) as scratch_name:
        scratch = Path(scratch_name)
        license_path, source_path = scratch / "LICENSE", scratch / "source.jsonl"
        _fetch(spec.license_url, spec.license_sha256, 1024 * 1024, license_path)
        _fetch(spec.url, spec.sha256, spec.max_bytes, source_path)

        def records() -> Iterator[RawDocument]:
            with source_path.open("rb") as stream:
                index = 0
                while line := stream.readline(6 * 1024 * 1024 + 129):
                    if len(line) > 6 * 1024 * 1024 + 128:
                        raise ValueError("Source JSONL row exceeds its size limit")
                    row = json.loads(
                        line,
                        object_pairs_hook=_unique_mapping,
                        parse_constant=_reject_constant,
                    )
                    expected = (
                        {"text"}
                        if spec.format == "text-jsonl"
                        else {"question", "answer"}
                    )
                    if not isinstance(row, dict) or set(row) != expected:
                        raise ValueError(
                            "Source JSONL fields disagree with approved format"
                        )
                    text = (
                        row["text"] if spec.format == "text-jsonl" else row["question"]
                    )
                    yield RawDocument(text, str(index))
                    index += 1

        identity = SourceIdentity(
            spec.name, spec.revision, spec.license_id, spec.license_url
        )
        uses = (
            ("training", "evaluation", "decontamination")
            if spec.use == "training"
            else ("evaluation", "decontamination")
        )
        manifest = write_raw_bundle(
            records(),
            root / "raw",
            identity,
            license_path.read_bytes(),
            allowed_uses=uses,
        )
    (root / "acquisition.yaml").write_text(
        yaml.safe_dump(asdict(spec), sort_keys=True), encoding="utf-8"
    )
    return manifest
