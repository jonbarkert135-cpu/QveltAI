"""Local corpus-to-tokenizer bundles with byte-verified, ordered JSONL input."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from qvelt.tokenizer.core import (
    CONTROL_TOKENS,
    MAX_DOCUMENT_BYTES,
    MAX_TOKENIZER_BYTES,
    QveltTokenizer,
    TokenizerError,
    _validate_text,
    train_tokenizer,
)
from qvelt.tokenizer.recipe import (
    TokenizerRecipe,
    dump_tokenizer_recipe,
    parse_tokenizer_recipe,
    tokenizer_recipe_digest,
)
from qvelt.utils.artifacts import (
    ArtifactIntegrityError,
    ArtifactStore,
    create_directory,
)
from qvelt.utils.manifest_types import DataInfo, Manifest, TokenizerInfo
from qvelt.utils.manifests import manifest_bytes, parse_manifest
from qvelt.utils.provenance import capture_code, capture_environment

MAX_JSONL_BYTES = 6 * MAX_DOCUMENT_BYTES + 128


def _raw_data(
    path: Path, expected_sha256: str, *, verify: bool = True
) -> tuple[ArtifactStore, Manifest]:
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(
        path.name, expected_sha256=expected_sha256, verify=verify
    )
    if (
        not isinstance(manifest.metadata, DataInfo)
        or manifest.metadata.representation != "raw"
    ):
        raise TokenizerError("Tokenizer input requires a raw-data manifest")
    if any(
        not file.path.endswith(".jsonl")
        for file in manifest.files
        if file.role == "data"
    ):
        raise TokenizerError("Raw tokenizer data must be JSONL")
    return store, manifest


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerError("JSONL records must not have duplicate fields")
        result[key] = value
    return result


def iter_documents(path: Path, expected_sha256: str) -> Iterator[str]:
    """Yield manifest-path/line order, verifying the bytes consumed and row count.

    Files are preverified by ArtifactStore and hashed again while read. Reaching
    iterator exhaustion is required before claiming successful corpus validation.
    No license declaration is interpreted as automatic data approval.
    """
    store, manifest = _raw_data(path, expected_sha256)
    count = 0
    for record in manifest.files:
        if record.role != "data":
            continue
        size, digest = 0, hashlib.sha256()
        with store.open_file(record.path) as stream:
            while True:
                line = stream.readline(
                    min(MAX_JSONL_BYTES + 1, record.size_bytes - size + 1)
                )
                if not line:
                    break
                size += len(line)
                if size > record.size_bytes or len(line) > MAX_JSONL_BYTES:
                    raise ArtifactIntegrityError(
                        "JSONL file grew or record exceeds size limit"
                    )
                digest.update(line)
                try:
                    row: object = json.loads(
                        line.decode("utf-8"), object_pairs_hook=_unique
                    )
                except (ValueError, UnicodeError, RecursionError) as error:
                    raise TokenizerError("Invalid UTF-8 JSONL text record") from error
                if not isinstance(row, dict) or set(row) != {"text"}:
                    raise TokenizerError(
                        "JSONL records must contain exactly one text field"
                    )
                text = row["text"]
                _validate_text(text)
                count += 1
                yield text
        if size != record.size_bytes or digest.hexdigest() != record.sha256:
            raise ArtifactIntegrityError(
                "Consumed JSONL bytes disagree with data manifest"
            )
    metadata = manifest.metadata
    if not isinstance(metadata, DataInfo) or count != metadata.documents:
        raise ArtifactIntegrityError(
            "JSONL document count disagrees with data manifest"
        )


def train_tokenizer_bundle(
    recipe: TokenizerRecipe,
    data_manifest: Path,
    directory: Path,
    repository: Path,
) -> Manifest:
    """Train a fresh BPE and publish config/data/provenance lineage last.

    Failed operations may leave uncommitted files. Existing output directories
    are never reused. Source/package states are observations, not execution locks.
    """
    _, source = _raw_data(data_manifest, recipe.training_data_sha256, verify=False)
    root, code = capture_code(repository)
    environment = capture_environment()
    store = ArtifactStore(create_directory(directory))
    tokenizer = train_tokenizer(
        iter_documents(data_manifest, recipe.training_data_sha256), recipe.tokenizer
    )
    provenance = (
        json.dumps(
            {
                "code": asdict(code),
                "environment": asdict(environment),
                "recipe_sha256": tokenizer_recipe_digest(recipe),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    files = (
        store.add_bytes("tokenizer.json", tokenizer.to_bytes(), role="tokenizer"),
        store.add_bytes(
            "recipe.yaml", dump_tokenizer_recipe(recipe).encode(), role="recipe"
        ),
        store.add_bytes(
            "training-data.manifest.json",
            manifest_bytes(source),
            role="training_manifest",
        ),
        store.add_bytes("provenance.json", provenance, role="provenance"),
        store.add_file("uv.lock", root / "uv.lock", role="dependency_lock"),
    )
    if files[-1].sha256 != code.lock_sha256:
        raise ArtifactIntegrityError(
            "Dependency lock changed during tokenizer training"
        )
    manifest = Manifest(
        kind="tokenizer",
        owner=recipe.name,
        created_at=datetime.now(UTC).isoformat(),
        metadata=TokenizerInfo(
            "byte_bpe",
            tokenizer.vocab_size,
            len(CONTROL_TOKENS),
            recipe.training_data_sha256,
        ),
        files=files,
    )
    store.publish(manifest)
    return manifest


def load_tokenizer_bundle(
    path: Path, expected_sha256: str | None = None
) -> tuple[QveltTokenizer, Manifest]:
    """Verify a complete schema-1 workflow bundle, including data/recipe linkage."""
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(path.name, expected_sha256=expected_sha256)
    metadata = manifest.metadata
    if not isinstance(metadata, TokenizerInfo):
        raise TokenizerError("Expected a tokenizer bundle")

    def read(role: str, limit: int) -> bytes:
        matches = [file for file in manifest.files if file.role == role]
        if len(matches) != 1:
            raise TokenizerError(f"Tokenizer bundle requires exactly one {role} file")
        record = matches[0]
        if record.size_bytes > limit:
            raise TokenizerError("Tokenizer bundle file exceeds its size limit")
        with store.open_file(record.path) as stream:
            data = stream.read(limit + 1)
        if (
            len(data) != record.size_bytes
            or hashlib.sha256(data).hexdigest() != record.sha256
        ):
            raise ArtifactIntegrityError("Tokenizer bundle file changed during load")
        return data

    tokenizer = QveltTokenizer.from_bytes(read("tokenizer", MAX_TOKENIZER_BYTES))
    try:
        recipe = parse_tokenizer_recipe(read("recipe", 128 * 1024).decode("utf-8"))
    except UnicodeError as error:
        raise TokenizerError("Tokenizer recipe must be UTF-8") from error
    source = parse_manifest(
        read("training_manifest", 1024 * 1024),
        expected_sha256=metadata.training_data_sha256,
    )
    if (
        not isinstance(source.metadata, DataInfo)
        or source.metadata.representation != "raw"
    ):
        raise TokenizerError("Tokenizer lineage requires raw training data")
    if (
        recipe.training_data_sha256 != metadata.training_data_sha256
        or metadata.vocab_size != tokenizer.vocab_size
        or metadata.reserved_tokens != len(CONTROL_TOKENS)
        or manifest.owner != recipe.name
        or tokenizer.vocab_size > recipe.tokenizer.vocab_size
    ):
        raise TokenizerError(
            "Tokenizer metadata disagrees with recipe or learned vocabulary"
        )
    # Check the exact selected pre-tokenizer rather than inferring it from a
    # handful of digit examples that could pass for different learned merges.
    from qvelt.tokenizer.core import _backend

    learned = json.loads(tokenizer.to_bytes())
    requested = json.loads(_backend(recipe.tokenizer.isolate_digits).to_str())
    if learned["pre_tokenizer"] != requested["pre_tokenizer"]:
        raise TokenizerError("Tokenizer digit policy disagrees with recipe")
    # Provenance is descriptive; validate only the links asserted by this workflow.
    try:
        provenance = json.loads(
            read("provenance", 1024 * 1024), object_pairs_hook=_unique
        )
        locks = [file for file in manifest.files if file.role == "dependency_lock"]
        if (
            len(locks) != 1
            or provenance["code"]["lock_sha256"] != locks[0].sha256
            or provenance["recipe_sha256"] != tokenizer_recipe_digest(recipe)
        ):
            raise TokenizerError("Tokenizer provenance links disagree")
    except (ValueError, KeyError, TypeError, RecursionError) as error:
        raise TokenizerError("Invalid tokenizer provenance links") from error
    return tokenizer, manifest


@dataclass(frozen=True)
class TokenizationMetrics:
    """Count-based coverage probe; ratios have explicit aggregate denominators."""

    documents: int
    utf8_bytes: int
    characters: int
    tokens: int

    @property
    def bytes_per_token(self) -> float | None:
        return self.utf8_bytes / self.tokens if self.tokens else None


def evaluate_tokenizer(
    tokenizer: QveltTokenizer, data_manifest: Path, expected_data_sha256: str
) -> TokenizationMetrics:
    """Require whole-text round trips and count held-out text without retaining it."""
    documents = utf8_bytes = characters = tokens = 0
    for text in iter_documents(data_manifest, expected_data_sha256):
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids) != text or any(i < len(CONTROL_TOKENS) for i in ids):
            raise TokenizerError(
                "Tokenizer failed plain-text coverage or control isolation"
            )
        documents += 1
        utf8_bytes += len(text.encode("utf-8"))
        characters += len(text)
        tokens += len(ids)
    return TokenizationMetrics(documents, utf8_bytes, characters, tokens)
