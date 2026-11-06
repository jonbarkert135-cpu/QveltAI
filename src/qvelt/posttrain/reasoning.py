"""Own-checkpoint rejection sampling and explicit, licensed local trace ingestion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import yaml

from qvelt.config import MAX_CONFIG_BYTES, _integer, _mapping, parse_yaml_mapping
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.math import generate_problems
from qvelt.data.records import RawDocument, iter_records, write_raw_bundle
from qvelt.eval.config import ArtifactRef
from qvelt.inference.artifacts import load_checkpoint_model
from qvelt.inference.config import GenerationOptions, TokenRequest
from qvelt.inference.generation import GenerationEngine
from qvelt.posttrain.chat import parse_messages
from qvelt.posttrain.verifiers import AnswerSpec, extract_answer, verify_answer
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, create_directory
from qvelt.utils.checkpoints import source_digest
from qvelt.utils.manifest_types import DataInfo, Manifest
from qvelt.utils.manifests import manifest_bytes, manifest_digest

PERMISSIVE_LICENSES = frozenset(
    ("Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "CC-BY-4.0")
)


@dataclass(frozen=True)
class ReasoningJob:
    checkpoint: ArtifactRef
    tokenizer: ArtifactRef
    problems: ArtifactRef | None = None
    approved_for_training: bool = False
    count: int = 16
    samples: int = 4
    max_tokens: int = 64
    seed: int = 20261106
    framed: bool = False
    verifier: str = "exact"
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    diversity_jaccard: float = 0.9
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.checkpoint, ArtifactRef)
            or not isinstance(self.tokenizer, ArtifactRef)
            or self.problems is not None
            and not isinstance(self.problems, ArtifactRef)
        ):
            raise ValueError(
                "Reasoning inputs require checksum-bound artifact references"
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Only reasoning schema1 is supported")
        if (
            type(self.approved_for_training) is not bool
            or type(self.framed) is not bool
        ):
            raise ValueError("Explicit approval/framing booleans required")
        for name, low, high in (
            ("count", 1, 64),
            ("samples", 1, 8),
            ("max_tokens", 1, 128),
            ("seed", 0, 2**32 - 1),
        ):
            _integer(
                getattr(self, name), "reasoning." + name, minimum=low, maximum=high
            )
        AnswerSpec("0", self.verifier, self.absolute_tolerance, self.relative_tolerance)
        if (
            type(self.diversity_jaccard) not in (int, float)
            or not math.isfinite(self.diversity_jaccard)
            or not 0 < self.diversity_jaccard <= 1
        ):
            raise ValueError("Diversity Jaccard threshold must be in (0,1]")


def load_reasoning(path: Path) -> ReasoningJob:
    with path.open("rb") as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError("Reasoning recipe exceeds byte limit")
    body = _mapping(
        parse_yaml_mapping(data.decode("utf-8-sig")),
        "reasoning",
        {f.name for f in fields(ReasoningJob)},
    )
    for name in ("checkpoint", "tokenizer", "problems"):
        if name in body and body[name] is not None:
            ref = _mapping(body[name], name, {"path", "sha256"})
            if set(ref) != {"path", "sha256"}:
                raise ValueError("Artifact reference requires path and sha256")
            body[name] = ArtifactRef(**cast(dict[str, Any], ref))
    if not {"checkpoint", "tokenizer"} <= set(body):
        raise ValueError("Reasoning requires own checkpoint/tokenizer")
    return ReasoningJob(**cast(dict[str, Any], body))


def licensed_records(
    reference: ArtifactRef, approved_for_training: bool
) -> tuple[list[RawDocument], Manifest]:
    """Read <=64 reviewed local problems/traces; no dataset download or rights inference."""
    if approved_for_training is not True:
        raise ValueError("Source revision must be approved for training in DATA_CARD")
    path = Path(reference.path)
    store = ArtifactStore(path.parent)
    manifest = store.read_manifest(
        path.name, expected_sha256=reference.sha256, verify=False
    )
    if (
        not isinstance(manifest.metadata, DataInfo)
        or manifest.metadata.representation != "raw"
        or manifest.metadata.documents > 64
        or sum(f.size_bytes for f in manifest.files) > 4 * 1024**2
        or any(
            s.license_id not in PERMISSIVE_LICENSES for s in manifest.metadata.sources
        )
    ):
        raise ValueError(
            "Only bounded raw bundles with reviewed permissive source licenses are supported"
        )
    store.verify(manifest)
    licenses = [f for f in manifest.files if f.role == "license"]
    policies = [f for f in manifest.files if f.role == "usage_policy"]
    if not licenses or len(policies) != 1 or policies[0].size_bytes > 4096:
        raise ValueError("Source needs license and usage evidence")
    with store.open_file(policies[0].path) as stream:
        policy = parse_yaml_mapping(stream.read(4097).decode())
    if (
        set(policy) != {"allowed_uses"}
        or not isinstance(policy["allowed_uses"], list)
        or "training" not in policy["allowed_uses"]
    ):
        raise ValueError("Source does not permit training")
    records = list(iter_records(path, reference.sha256))
    if (
        not records
        or len(records) != manifest.metadata.documents
        or any(
            not all((r.identifier, r.family, r.difficulty, r.answer))
            or len(r.text) > 4096
            for r in records
        )
        or len({r.identifier for r in records}) != len(records)
    ):
        raise ValueError(
            "Problems require unique IDs, families, difficulties and gold answers"
        )
    return records, manifest


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased 1-C(n-c,k)/C(n,k) for n independent samples under a fixed protocol."""
    if (
        any(type(value) is not int for value in (n, correct, k))
        or not 0 <= correct <= n <= 8
        or not 1 <= k <= n
    ):
        raise ValueError("Invalid bounded pass@k counts")
    return float(
        1
        - Fraction(
            math.comb(n - correct, k) if n - correct >= k else 0, math.comb(n, k)
        )
    )


def diverse(text: str, kept: list[str], threshold: float) -> bool:
    """Lexical whitespace/case/token-set diversity, never a semantic novelty claim."""

    def normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    current = normalize(text)
    words = set(re.findall(r"\w+|[^\w\s]", current))
    for previous in kept:
        other = normalize(previous)
        tokens = set(re.findall(r"\w+|[^\w\s]", other))
        if (
            current == other
            or len(words & tokens) / max(1, len(words | tokens)) >= threshold
        ):
            return False
    return True


def _document(
    problem: RawDocument, text: str, answer: str, framed: bool, index: int
) -> RawDocument:
    reasoning = None
    if framed and text.strip().startswith("<think>"):
        reasoning = text.strip()[7:].split("</think>", 1)[0].strip()
    assistant: dict[str, str] = {
        "role": "assistant",
        "content": "Final answer: " + answer if framed else answer,
    }
    if reasoning is not None:
        assistant["reasoning"] = reasoning
    payload = {
        "schema_version": 2 if reasoning else 1,
        "messages": [{"role": "user", "content": problem.text}, assistant],
    }
    return RawDocument(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        problem.identifier + f"-{index}",
        problem.family,
        problem.language,
        problem.difficulty,
        problem.answer,
    )


def _publish(
    directory: Path,
    documents: list[RawDocument],
    audits: list[dict[str, Any]],
    report: dict[str, Any],
    sources: tuple[Manifest, ...] = (),
    evidence: tuple[tuple[str, bytes, str], ...] = (),
) -> Manifest:
    root = create_directory(directory)
    source_ids = tuple(
        s
        for m in sources
        if isinstance(m.metadata, DataInfo)
        for s in m.metadata.sources
    )
    permission = (
        PERMISSION
        if not sources
        else b"Derived answer-verified traces; retain the original source permissions and terms recorded in this bundle. No new license is granted for source material.\n"
    )
    selected = write_raw_bundle(
        documents,
        root / "selected",
        source_ids or (authored_source("reasoning-rejection-v1"),),
        permission,
    )
    store = ArtifactStore(root)
    files = [
        store.add_file(
            "selected.manifest.json",
            root / "selected/manifest.json",
            role="selection_manifest",
        )
    ]
    # Keep actual data in one place while making this root a normal raw bundle.
    from dataclasses import replace

    files.extend(replace(f, path="selected/" + f.path) for f in selected.files)
    files.extend(
        store.add_bytes(name, data, role=role) for name, data, role in evidence
    )
    for i, original in enumerate(sources):
        files.append(
            store.add_bytes(
                f"source-{i}.manifest.json",
                manifest_bytes(original),
                role="source_manifest",
            )
        )
        reference = report["source_paths"][i]
        original_store = ArtifactStore(Path(reference).parent)
        for j, record in enumerate(original.files):
            if record.role == "license":
                with original_store.open_file(record.path) as stream:
                    files.append(
                        store.add_stream(
                            f"source-{i}-license-{j}.txt", stream, role="license"
                        )
                    )
    report = {k: v for k, v in report.items() if k != "source_paths"}
    files.append(
        store.add_bytes(
            "audit.jsonl",
            b"".join(
                (
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                ).encode()
                for row in audits
            ),
            role="reasoning_audit",
        )
    )
    files.append(
        store.add_bytes(
            "report.json",
            (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode(),
            role="reasoning_report",
        )
    )
    manifest = Manifest(
        "data", selected.owner, selected.created_at, selected.metadata, tuple(files)
    )
    store.publish(manifest)
    return manifest


def build_reasoning_data(job: ReasoningJob, directory: Path) -> Manifest:
    if job.problems is None:
        problems = [
            RawDocument(
                p.question, p.identifier, p.family, p.language, p.difficulty, p.answer
            )
            for p in generate_problems(job.count, job.seed)
        ]
        sources: tuple[Manifest, ...] = ()
    else:
        problems, source = licensed_records(job.problems, job.approved_for_training)
        if len(problems) > job.count:
            raise ValueError("Source exceeds declared problem budget")
        sources = (source,)
    specs = [
        AnswerSpec(
            p.answer, job.verifier, job.absolute_tolerance, job.relative_tolerance
        )
        for p in problems
    ]
    model, _, checkpoint_manifest = load_checkpoint_model(
        job.checkpoint, job.tokenizer.sha256
    )
    tokenizer, tokenizer_manifest = load_tokenizer_bundle(
        Path(job.tokenizer.path), job.tokenizer.sha256
    )
    engine = GenerationEngine(model, tokenizer)
    documents = []
    audits = []
    difficulties: dict[str, list[int]] = {}
    for pindex, (problem, spec) in enumerate(zip(problems, specs, strict=True)):
        kept: list[str] = []
        correct = 0
        question = problem.text + (
            "\nUse <think>...</think> optionally, then Final answer: ANSWER."
            if job.framed
            else ""
        )
        prefix = (1, 4, *tokenizer.encode(question), 6, 5)
        for sample in range(job.samples):
            seed = (job.seed + pindex * job.samples + sample) % 2**32
            options = GenerationOptions(
                max_new_tokens=job.max_tokens,
                strategy="sample",
                temperature=1.0,
                seed=seed,
                context_limit=min(model.config.context, 2048),
                max_seconds=300,
            )
            result = engine.generate((TokenRequest("candidate", prefix, options),))[0]
            answer = extract_answer(result.text, framed=job.framed)
            verified = (
                result.reason == "stop_token"
                and result.token_ids[-1] == 6
                and answer is not None
                and verify_answer(answer, spec)
            )
            correct += verified
            # Equivalent numeric spellings are not different reasoning traces.
            # Bare answers have one empty reasoning body, hence one kept row.
            trace = (
                result.text.strip()[7:].split("</think>", 1)[0]
                if job.framed and result.text.strip().startswith("<think>")
                else ""
            )
            selected = verified and diverse(trace, kept, job.diversity_jaccard)
            if selected:
                documents.append(
                    _document(problem, result.text, answer or "", job.framed, sample)
                )
                kept.append(trace)
            audits.append(
                {
                    "problem_id": problem.identifier,
                    "difficulty": problem.difficulty,
                    "sample": sample,
                    "seed": seed,
                    "text": result.text,
                    "tokens": list(result.token_ids),
                    "stop": result.reason,
                    "verified_correct": verified,
                    "selected": selected,
                    "reason": "kept"
                    if selected
                    else "duplicate"
                    if verified
                    else "incorrect_or_incomplete",
                }
            )
        difficulties.setdefault(problem.difficulty, []).append(correct)
    stats = {
        difficulty: {
            "problems": len(counts),
            "samples_per_problem": job.samples,
            "correct_candidates": sum(counts),
            "pass_at_k": {
                str(k): sum(pass_at_k(job.samples, c, k) for c in counts) / len(counts)
                for k in range(1, job.samples + 1)
            },
        }
        for difficulty, counts in difficulties.items()
    }
    recipe = asdict(job)
    for key in ("checkpoint", "tokenizer", "problems"):
        if isinstance(recipe[key], dict):
            recipe[key].pop("path")
    report = {
        "schema_version": 1,
        "protocol": "qvelt-rejection-v1",
        "source_sha256": source_digest(),
        "recipe": recipe,
        "selected": len(documents),
        "candidates": len(audits),
        "difficulty": stats,
        "verification_scope": "final_answer_only_not_reasoning_steps",
        "selection_unit": "reasoning_body_with_answer_only_singleton",
        "external_synthetic": False,
        "source_paths": [job.problems.path] if job.problems else [],
    }
    return _publish(
        directory,
        documents,
        audits,
        report,
        sources,
        (
            (
                "checkpoint.manifest.json",
                manifest_bytes(checkpoint_manifest),
                "source_checkpoint",
            ),
            (
                "tokenizer.manifest.json",
                manifest_bytes(tokenizer_manifest),
                "source_tokenizer",
            ),
        ),
    )


def ingest_external_traces(
    source: ArtifactRef,
    directory: Path,
    *,
    approved_for_training: bool = False,
    external_synthetic: bool = False,
    terms: ArtifactRef | None = None,
) -> Manifest:
    """Offline teacher-data switch, not an API client. Terms file identifies teacher and permitted uses."""
    if external_synthetic is not True or terms is None:
        raise ValueError(
            "External synthetic data is off by default; explicit switch and terms artifact required"
        )
    rows, manifest = licensed_records(source, approved_for_training)
    terms_path = Path(terms.path)
    with ArtifactStore(terms_path.parent).open_file(terms_path.name) as stream:
        terms_bytes = stream.read(8193)
    if (
        len(terms_bytes) > 8192
        or hashlib.sha256(terms_bytes).hexdigest() != terms.sha256
    ):
        raise ValueError("Teacher terms byte limit/hash mismatch")
    acknowledgement = parse_yaml_mapping(terms_bytes.decode())
    if (
        set(acknowledgement)
        != {
            "teacher",
            "revision",
            "license_id",
            "terms_url",
            "training_permitted",
            "reviewed",
        }
        or acknowledgement["training_permitted"] is not True
        or acknowledgement["reviewed"] is not True
        or any(
            not isinstance(value, str) or not 1 <= len(value) <= 1024
            for value in (
                acknowledgement[key]
                for key in ("teacher", "revision", "license_id", "terms_url")
            )
        )
    ):
        raise ValueError(
            "Terms require reviewed teacher/revision/permissive license/training permission"
        )
    if acknowledgement["license_id"] not in PERMISSIVE_LICENSES:
        raise ValueError("Teacher license must be explicitly reviewed and permissive")
    from urllib.parse import urlsplit

    url = urlsplit(cast(str, acknowledgement["terms_url"]))
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
    ):
        raise ValueError("Terms URL must be a public HTTPS reference")
    selected = []
    audits = []
    for row in rows:
        messages = parse_messages(row.text)
        if (
            len(messages) != 2
            or messages[0].role != "user"
            or messages[1].role != "assistant"
        ):
            raise ValueError(
                "External trace ingress requires complete single-turn chats"
            )
        assistant = messages[1]
        answer = extract_answer(assistant.content)
        correct = answer is not None and verify_answer(answer, AnswerSpec(row.answer))
        if correct:
            selected.append(row)
        audits.append(
            {
                "problem_id": row.identifier,
                "text": row.text,
                "verified_correct": correct,
                "selected": correct,
            }
        )
    return _publish(
        directory,
        selected,
        audits,
        {
            "schema_version": 1,
            "protocol": "qvelt-external-traces-v1",
            "external_synthetic": True,
            "source_sha256": source_digest(),
            "source_manifest_sha256": manifest_digest(manifest),
            "terms_sha256": terms.sha256,
            "verification_scope": "final_answer_only_not_reasoning_steps",
            "selected": len(selected),
            "source_paths": [source.path],
        },
        (manifest,),
        (("teacher-terms.yaml", terms_bytes, "teacher_terms"),),
    )


def dump_reasoning(job: ReasoningJob) -> str:
    return yaml.safe_dump(asdict(job), sort_keys=True)
