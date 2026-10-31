"""Stream approved sources through disk curation, train-only BPE and compact shards."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qvelt.data.cleaning import Cleaner
from qvelt.data.config import DataRecipe, dump_data_recipe, load_data_recipe
from qvelt.data.decontamination import BenchmarkRegistry, Decontaminator
from qvelt.data.dedup import DiskDeduplicator
from qvelt.data.records import RawDocument, iter_records, write_raw_bundle
from qvelt.data.shards import ShardWriter
from qvelt.tokenizer.recipe import TokenizerRecipe
from qvelt.tokenizer.workflow import load_tokenizer_bundle, train_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.manifest_types import ArtifactFile, DataInfo, Manifest, SourceIdentity
from qvelt.utils.manifests import (
    _reject_constant,
    _unique_mapping,
    manifest_bytes,
    manifest_digest,
)


def document_split(key: str, recipe: DataRecipe) -> str:
    """A family has one split across languages/sources; absent family uses clean text."""
    value = (
        int.from_bytes(
            hashlib.sha256(f"{recipe.seed}:{key}".encode()).digest()[:8], "big"
        )
        / 2**64
    )
    if value < recipe.test_fraction:
        return "test"
    if value < recipe.test_fraction + recipe.validation_fraction:
        return "validation"
    return "train"


def _json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def prepare_data(recipe_path: Path, directory: Path, repository: Path) -> Manifest:
    recipe = load_data_recipe(recipe_path)
    return prepare_recipe(recipe, recipe_path.parent, directory, repository)


def prepare_recipe(
    recipe: DataRecipe, base: Path, directory: Path, repository: Path
) -> Manifest:
    """Publish manifest last; failure leaves an incomplete output, never a usable bundle."""
    started = time.perf_counter()
    registry = BenchmarkRegistry.load(
        base / recipe.benchmark_registry, recipe.benchmark_sha256, recipe.purpose
    )
    inputs = []
    identities: list[SourceIdentity] = []
    for source in recipe.sources:
        path = base / source.manifest
        manifest = ArtifactStore(path.parent).read_manifest(
            path.name, expected_sha256=source.sha256
        )
        if (
            not isinstance(manifest.metadata, DataInfo)
            or manifest.metadata.representation != "raw"
        ):
            raise ValueError("All preparation inputs must be registered raw text")
        policies = [f for f in manifest.files if f.role == "usage_policy"]
        if len(policies) > 1 or (recipe.purpose == "production" and not policies):
            raise ValueError("Production sources require one explicit usage policy")
        if policies:
            record = policies[0]
            if record.size_bytes > 4096:
                raise ValueError("Source usage policy exceeds its size limit")
            with ArtifactStore(path.parent).open_file(record.path) as stream:
                policy = json.loads(
                    stream.read(4097),
                    object_pairs_hook=_unique_mapping,
                    parse_constant=_reject_constant,
                )
            if (
                not isinstance(policy, dict)
                or set(policy) != {"allowed_uses"}
                or not isinstance(policy["allowed_uses"], list)
                or "training" not in policy["allowed_uses"]
            ):
                raise ValueError("Source usage policy does not permit training")
        inputs.append((source, path, manifest))
        for identity in manifest.metadata.sources:
            if identity not in identities:
                identities.append(identity)
    store = ArtifactStore(create_directory(directory))
    files: list[ArtifactFile] = []
    for source, path, manifest in inputs:
        files.append(
            store.add_bytes(
                f"sources/{source.name}.manifest.json",
                manifest_bytes(manifest),
                role="source_manifest",
            )
        )
        for i, record in enumerate(f for f in manifest.files if f.role == "license"):
            with ArtifactStore(path.parent).open_file(record.path) as stream:
                files.append(
                    store.add_stream(
                        f"sources/{source.name}-license-{i}.txt", stream, role="license"
                    )
                )
    files.append(
        store.add_file("benchmarks.json", registry.path, role="benchmark_registry")
    )
    files.append(
        store.add_bytes("recipe.yaml", dump_data_recipe(recipe).encode(), role="recipe")
    )
    counts: Counter[str] = Counter()
    redactions: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    overlaps: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {s.name: Counter() for s in recipe.sources}
    with tempfile.TemporaryDirectory(
        prefix=".curation-", dir=store.root
    ) as scratch_name:
        scratch = Path(scratch_name)
        with (
            closing(sqlite3.connect(scratch / "curation.sqlite")) as connection,
            tempfile.TemporaryFile(dir=scratch) as audit,
        ):
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute("PRAGMA temp_store=FILE")
            dedup = DiskDeduplicator(connection, recipe.dedup, recipe.seed)
            decontam = Decontaminator(connection, registry, recipe.ngram_size)
            cleaner = Cleaner(recipe.cleaning, recipe.seed)
            for source, path, _ in inputs:
                for index, raw in enumerate(iter_records(path, source.sha256)):
                    counts["input"] += 1
                    source_counts[source.name]["input"] += 1
                    uid = hashlib.sha256(
                        f"{source.name}:{raw.identifier or index}".encode()
                    ).hexdigest()
                    clean = cleaner.clean(raw.text)
                    redactions.update(clean.redactions)
                    languages[clean.language] += 1
                    reason = clean.rejection or dedup.duplicate(clean.text)
                    hits = set() if reason else decontam.matches(clean.text)
                    if hits:
                        reason = "benchmark_overlap"
                        overlaps.update(hits)
                    if reason:
                        counts[reason] += 1
                        source_counts[source.name][reason] += 1
                        audit.write(
                            _json(
                                {
                                    "id": uid,
                                    "source": source.name,
                                    "reason": reason,
                                    "benchmarks": sorted(hits),
                                }
                            )
                        )
                        continue
                    family = hashlib.sha256(
                        (
                            "family:" + raw.family
                            if raw.family
                            else "text:" + clean.text
                        ).encode()
                    ).hexdigest()
                    split = document_split(family, recipe)
                    metadata = {
                        "id": uid,
                        "family": family,
                        "source": source.name,
                        "language": clean.language,
                        "language_probability": clean.language_probability,
                        "declared_language": raw.language
                        if re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z]{2,8})?", raw.language)
                        else "unknown",
                        "difficulty": raw.difficulty
                        if raw.difficulty in ("easy", "medium", "hard")
                        else "unspecified",
                        "heuristic_score": clean.heuristic_score,
                    }
                    dedup.add(clean.text, _json(metadata).decode(), split)
                    counts["retained"] += 1
                    counts[split] += 1
                    source_counts[source.name]["retained"] += 1
                    if counts["input"] % 256 == 0:
                        connection.commit()
            connection.commit()
            if not counts["train"]:
                raise ValueError(
                    "Curation produced no train documents; refusing to train a tokenizer"
                )

            def train_documents() -> Iterator[RawDocument]:
                for text, metadata in connection.execute(
                    "SELECT text,metadata FROM documents WHERE split='train' ORDER BY id"
                ):
                    meta = json.loads(metadata)
                    yield RawDocument(
                        text,
                        meta["id"],
                        meta["family"],
                        meta["declared_language"],
                        meta["difficulty"],
                    )

            permission = _json(
                {
                    "scope": "Derived train-only text; original licenses remain applicable",
                    "sources": [asdict(s) for s in identities],
                }
            )
            training = write_raw_bundle(
                train_documents(), scratch / "train", tuple(identities), permission
            )
            token_manifest = train_tokenizer_bundle(
                TokenizerRecipe(manifest_digest(training), tokenizer=recipe.tokenizer),
                scratch / "train/manifest.json",
                store.root / "tokenizer",
                repository,
            )
            tokenizer, _ = load_tokenizer_bundle(
                store.root / "tokenizer/manifest.json", manifest_digest(token_manifest)
            )
            for path in sorted((store.root / "tokenizer").iterdir()):
                from qvelt.utils.artifacts import hash_file

                info = hash_file(path)
                files.append(
                    ArtifactFile(
                        "tokenizer/" + path.name,
                        "tokenizer_bundle",
                        info.size_bytes,
                        info.sha256,
                    )
                )
            layout_sources: dict[str, dict[str, Any]] = {
                s.name: {} for s in recipe.sources
            }
            stored_tokens = targets = 0
            for source in recipe.sources:
                for split in ("train", "validation", "test"):
                    cursor = connection.execute(
                        "SELECT id,text,metadata FROM documents WHERE split=? AND json_extract(metadata,'$.source')=? ORDER BY id",
                        (split, source.name),
                    )
                    writer = ShardWriter(
                        store,
                        f"shards/{source.name}/{split}",
                        tokenizer.vocab_size,
                        recipe.context,
                        recipe.shard_tokens,
                    )
                    try:
                        for identifier, text, metadata_json in cursor:
                            frame = [
                                tokenizer.control_id("<|bos|>"),
                                *tokenizer.encode(text),
                                tokenizer.control_id("<|eos|>"),
                            ]
                            writer.add(frame, identifier, json.loads(metadata_json))
                            stored_tokens += len(frame)
                            targets += len(frame) - 1
                        if writer.documents:
                            layout_sources[source.name][split] = writer.finish()
                            files.extend(writer.files)
                    finally:
                        writer.close()
            audit.seek(0)
            files.append(
                store.add_stream("rejections.jsonl", audit, role="curation_audit")
            )
            scratch_bytes = sum(
                p.stat().st_size for p in scratch.rglob("*") if p.is_file()
            )
            report = {
                "scope": recipe.purpose,
                "production_ready": recipe.purpose == "production",
                "counts": dict(counts),
                "sources": source_counts,
                "redactions": dict(redactions),
                "detected_languages": dict(languages),
                "benchmark_overlaps": dict(overlaps),
                "benchmark_questions": decontam.questions,
                "benchmark_registry_sha256": registry.sha256,
                "train_only_tokenizer_documents": counts["train"],
                "tokenizer_sha256": manifest_digest(token_manifest),
                "stored_tokens_including_boundaries": stored_tokens,
                "valid_targets": targets,
                "scratch_snapshot_bytes": scratch_bytes,
                "elapsed_seconds_before_publish": time.perf_counter() - started,
                "limitations": [
                    "Pattern PII and phrase toxicity are incomplete",
                    "Probabilistic LSH can miss duplicates",
                    "N-grams do not exclude semantic/translated leaks",
                    "Demo fixtures do not scan public benchmarks",
                ]
                if recipe.purpose == "demo"
                else ["Pattern PII, LSH and n-gram recall require external audits"],
            }
            layout = {
                "schema_version": 1,
                "context": recipe.context,
                "vocab": tokenizer.vocab_size,
                "control_version": tokenizer.control_version,
                "tokenizer_sha256": manifest_digest(token_manifest),
                "sources": layout_sources,
                "mixing": [asdict(s) for s in recipe.mixing],
            }
            files.append(
                store.add_bytes("layout.json", _json(layout), role="packing_layout")
            )
            files.append(
                store.add_bytes("report.json", _json(report), role="curation_report")
            )
    manifest = Manifest(
        "data",
        "curated-data",
        datetime.now(UTC).isoformat(),
        DataInfo(
            "token_ids",
            tuple(identities),
            counts["retained"],
            stored_tokens,
            manifest_digest(token_manifest),
        ),
        tuple(files),
    )
    store.publish(manifest)
    return manifest
