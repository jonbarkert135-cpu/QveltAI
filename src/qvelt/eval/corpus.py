"""Checksummed local task records and transparent bounded lexical overlap audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qvelt.data.decontamination import normalized_tokens
from qvelt.data.dedup import shingles
from qvelt.data.records import iter_records
from qvelt.eval.config import ArtifactRef, EvalJob
from qvelt.eval.scoring import parse_number
from qvelt.tokenizer.core import _validate_text
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifest_types import DataInfo, Manifest, identifier
from qvelt.utils.manifests import _reject_constant, _unique_mapping


def read_json(text: str | bytes) -> Any:
    return json.loads(
        text, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
    )


def licensed_raw(ref: ArtifactRef, uses: tuple[str, ...]) -> Manifest:
    store = ArtifactStore(Path(ref.path).parent)
    manifest = store.read_manifest(
        Path(ref.path).name, expected_sha256=ref.sha256, verify=False
    )
    if (
        not isinstance(manifest.metadata, DataInfo)
        or manifest.metadata.representation != "raw"
        or manifest.metadata.documents > 10000
        or sum(f.size_bytes for f in manifest.files) > 64 * 1024**2
    ):
        raise ValueError("Evaluation/audit sources must be bounded raw bundles")
    store.verify(manifest)
    policies = [f for f in manifest.files if f.role == "usage_policy"]
    if (
        not any(f.role == "license" for f in manifest.files)
        or len(policies) != 1
        or policies[0].size_bytes > 4096
    ):
        raise ValueError(
            "Evaluation/audit source needs license and explicit usage policy"
        )
    with store.open_file(policies[0].path) as stream:
        policy = read_json(stream.read(4097))
    if (
        not isinstance(policy, dict)
        or set(policy) != {"allowed_uses"}
        or not isinstance(policy["allowed_uses"], list)
        or not all(isinstance(use, str) for use in policy["allowed_uses"])
        or not set(uses) <= set(policy["allowed_uses"])
    ):
        raise ValueError("Source does not permit the requested evaluation/audit use")
    return manifest


@dataclass(frozen=True)
class EvalCase:
    identifier: str
    kind: str
    domain: str
    family: str
    partition: str
    prompt: str
    target: str
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for key in ("identifier", "domain", "family"):
            identifier(getattr(self, key), "evaluation." + key)
        if self.kind not in (
            "math",
            "numeric",
            "choice",
            "lm",
        ) or self.partition not in ("validation", "final", "control"):
            raise ValueError("Unknown evaluation kind/partition")
        for key in ("prompt", "target"):
            value = getattr(self, key)
            _validate_text(value)
            if len(value) > 16384 or key == "prompt" and not value.strip():
                raise ValueError("Evaluation text must be bounded and prompt nonempty")
        if type(self.choices) is not tuple or any(
            not isinstance(c, str) or not c.strip() or len(c) > 4096
            for c in self.choices
        ):
            raise ValueError("Invalid immutable choices")
        for choice in self.choices:
            _validate_text(choice)
        if self.kind in ("math", "numeric") and (
            self.choices or parse_number(self.target, self.kind) is None
        ):
            raise ValueError("Invalid exact/numeric gold answer")
        if self.kind == "choice" and (
            not 2 <= len(self.choices) <= 8
            or len({c.strip() for c in self.choices}) != len(self.choices)
            or self.target not in self.choices
        ):
            raise ValueError(
                "Choice task requires unique options and exact target option"
            )
        if self.kind == "lm" and (self.target or self.choices):
            raise ValueError("Plain LM task has no separate gold/choices")
        if self.kind != "choice" and self.choices:
            raise ValueError("Only choice tasks have choices")


def load_cases(job: EvalJob) -> tuple[Manifest, list[EvalCase], int]:
    if not job.approved_for_evaluation:
        raise ValueError("Evaluation source needs explicit recorded approval")
    manifest = licensed_raw(job.dataset, ("evaluation",))
    cases = []
    seen: set[str] = set()
    for document in iter_records(Path(job.dataset.path), job.dataset.sha256):
        row = read_json(document.text)
        if (
            not isinstance(row, dict)
            or set(row) != {"schema_version", *EvalCase.__dataclass_fields__}
            or type(row["schema_version"]) is not int
            or row.pop("schema_version") != 1
            or not isinstance(row["choices"], list)
        ):
            raise ValueError("Invalid version1 evaluation record")
        row["choices"] = tuple(row["choices"])
        case = EvalCase(**row)
        if (
            case.identifier in seen
            or document.identifier != case.identifier
            or document.family != case.family
        ):
            raise ValueError("Evaluation IDs/families must be unique and aligned")
        if case.partition == "final" and not job.allow_final:
            raise ValueError("Final holdout evaluation is disabled by this protocol")
        seen.add(case.identifier)
        cases.append(case)
    if (
        not cases
        or not isinstance(manifest.metadata, DataInfo)
        or len(cases) != manifest.metadata.documents
    ):
        raise ValueError("Evaluation case count disagrees with source")
    return manifest, cases[: job.max_cases], len(cases)


def _fragments(text: str) -> list[str]:
    try:
        row = read_json(text)
    except (ValueError, UnicodeError, RecursionError):
        return [text]
    if isinstance(row, dict):
        messages = row.get("messages")
        if isinstance(messages, list) and all(
            isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages
        ):
            return [m["content"] for m in messages]
        if isinstance(row.get("prompt"), str):
            return [row["prompt"]]
    return [text]


def audit_overlap(cases: list[EvalCase], job: EvalJob) -> dict[str, Any]:
    """Declared-source lexical checks only;never infer full/semantic cleanliness."""
    query = {c.identifier: " ".join(normalized_tokens(c.prompt)) for c in cases}
    lengths = {
        c.identifier: min(job.audit_ngram, len(normalized_tokens(c.prompt)))
        for c in cases
    }
    grams = {
        c.identifier: shingles(query[c.identifier], lengths[c.identifier])
        if lengths[c.identifier]
        else set()
        for c in cases
    }
    similar = {c.identifier: shingles(query[c.identifier], 5) for c in cases}
    findings: dict[str, dict[str, Any]] = {
        c.identifier: {
            "normalized_exact": False,
            "ngram": False,
            "near_jaccard": False,
            "maximum_jaccard": 0.0,
            "family_overlap": False,
            "source_sha256": [],
        }
        for c in cases
    }
    coverage = []
    for source in job.training_sources:
        manifest = licensed_raw(source, ("training", "decontamination"))
        documents = truncated = 0
        for document in iter_records(Path(source.path), source.sha256):
            documents += 1
            for fragment in _fragments(document.text):
                tokens = normalized_tokens(fragment)
                if len(tokens) > 8192:
                    truncated += 1
                    tokens = tokens[:8192]
                normalized = " ".join(tokens)
                sourcegrams = {
                    length: shingles(normalized, length)
                    for length in set(lengths.values())
                    if length
                }
                source_similar = shingles(normalized, 5)
                for case in cases:
                    result = findings[case.identifier]
                    exact = (
                        bool(query[case.identifier])
                        and " " + query[case.identifier] + " " in " " + normalized + " "
                    )
                    ngram = bool(
                        lengths[case.identifier]
                        and grams[case.identifier]
                        & sourcegrams[lengths[case.identifier]]
                    )
                    small = similar[case.identifier]
                    union = small | source_similar
                    score = len(small & source_similar) / len(union) if union else 0.0
                    family = bool(document.family and document.family == case.family)
                    result["normalized_exact"] |= exact
                    result["ngram"] |= ngram
                    result["near_jaccard"] |= score >= job.audit_jaccard
                    result["family_overlap"] |= family
                    result["maximum_jaccard"] = max(result["maximum_jaccard"], score)
                    if (
                        exact or ngram or family or score >= job.audit_jaccard
                    ) and source.sha256 not in result["source_sha256"]:
                        result["source_sha256"].append(source.sha256)
        coverage.append(
            {
                "manifest_sha256": source.sha256,
                "sources": [s.name for s in manifest.metadata.sources]
                if isinstance(manifest.metadata, DataInfo)
                else [],
                "documents": documents,
                "truncated_fragments": truncated,
            }
        )
    return {
        "scope": "declared lexical sources only;unknown lineage and semantic leakage remain unverified",
        "complete_training_history_verified": False,
        "cleanliness_claim": False,
        "declared_sources": coverage,
        "ngram_size": job.audit_ngram,
        "jaccard_threshold": job.audit_jaccard,
        "findings": findings,
    }
