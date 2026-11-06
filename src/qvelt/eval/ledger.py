"""Verify report-bound public receipts and generate the results ledger without raw corpora."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from qvelt.config import config_digest, parse_config
from qvelt.eval.config import ArtifactRef, EvalJob, eval_digest, parse_eval
from qvelt.eval.corpus import read_json
from qvelt.eval.runner import _json, _payload
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifest_types import CheckpointInfo, Manifest, RunInfo
from qvelt.utils.manifests import manifest_bytes, manifest_digest, parse_manifest


def _portable_job(job: EvalJob) -> dict[str, Any]:
    values = asdict(job)
    for name in ("checkpoint", "tokenizer", "dataset"):
        values[name]["path"] = (
            f"artifacts/{name}-{values[name]['sha256']}/manifest.json"
        )
    for ref in values["training_sources"]:
        ref["path"] = f"artifacts/training-{ref['sha256']}/manifest.json"
    return values


def export_receipt(
    reference: ArtifactRef, destination: Path, *, benchmark: str
) -> dict[str, Any]:
    """Export exact metrics/audit/config plus input manifests, excluding predictions/data/weights."""
    path = Path(reference.path)
    store = ArtifactStore(path.parent)
    report = store.read_manifest(path.name, expected_sha256=reference.sha256)
    if not isinstance(report.metadata, RunInfo):
        raise ValueError("Ledger requires a native evaluation report")
    policy = parse_eval(
        _payload(store, report, "evaluation_recipe", 128 * 1024).decode()
    )
    command = read_json(_payload(store, report, "reproduction_command", 64 * 1024))
    if any(Path(arg).is_absolute() for arg in command["argv"]):
        raise ValueError(
            "Public receipt needs a portable reproduction argv;record one when running evaluation"
        )
    receipt = {
        "schema_version": 1,
        "protocol": "qvelt-eval-receipt-v1",
        "benchmark": benchmark,
        "report_sha256": reference.sha256,
        "report_manifest": asdict(report),
        "evaluation": _portable_job(policy),
        "config_yaml": _payload(store, report, "config", 128 * 1024).decode(),
        "metrics": read_json(_payload(store, report, "metrics", 2 * 1024**2)),
        "audit": read_json(_payload(store, report, "overlap_audit", 8 * 1024**2)),
        "provenance": read_json(
            _payload(store, report, "evaluation_provenance", 128 * 1024)
        ),
        "command": command,
        "inputs": {
            name: read_json(_payload(store, report, name + "_manifest", 512 * 1024))
            for name in ("checkpoint", "tokenizer", "dataset")
        },
    }
    verify_receipt(receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_json(receipt))
    return receipt


def _manifest(value: Any) -> Manifest:
    return parse_manifest(_json(value))


def comparison_key(receipt: dict[str, Any]) -> str:
    config = dict(receipt["evaluation"])
    for key in (
        "checkpoint",
        "training_sources",
        "audit_ngram",
        "audit_jaccard",
    ):
        config.pop(key)
    config["evaluator_source_sha256"] = receipt["provenance"]["source_sha256"]
    return hashlib.sha256(_json(config)).hexdigest()


def verify_receipt(receipt: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "protocol",
        "benchmark",
        "report_sha256",
        "report_manifest",
        "evaluation",
        "config_yaml",
        "metrics",
        "audit",
        "provenance",
        "command",
        "inputs",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected
        or type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != "qvelt-eval-receipt-v1"
        or not isinstance(receipt["benchmark"], str)
        or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", receipt["benchmark"]) is None
    ):
        raise ValueError("Invalid ledger receipt schema")
    report = _manifest(receipt["report_manifest"])
    if (
        not isinstance(report.metadata, RunInfo)
        or manifest_digest(report) != receipt["report_sha256"]
    ):
        raise ValueError("Receipt report identity mismatch")
    roles: dict[str, Any] = {}
    for record in report.files:
        if record.role in roles:
            raise ValueError("Evaluation report roles must be unique")
        roles[record.role] = record
    payloads = {
        "metrics": _json(receipt["metrics"]),
        "overlap_audit": _json(receipt["audit"]),
        "evaluation_provenance": _json(receipt["provenance"]),
        "reproduction_command": _json(receipt["command"]),
        "config": receipt["config_yaml"].encode(),
    }
    if not isinstance(receipt["inputs"], dict) or set(receipt["inputs"]) != {
        "checkpoint",
        "tokenizer",
        "dataset",
    }:
        raise ValueError("Receipt requires all three input snapshots")
    for key, value in receipt["inputs"].items():
        payloads[key + "_manifest"] = manifest_bytes(_manifest(value))
    for role, data in payloads.items():
        if (
            role not in roles
            or roles[role].sha256 != hashlib.sha256(data).hexdigest()
            or roles[role].size_bytes != len(data)
        ):
            raise ValueError("Receipt payload is not bound to native report: " + role)
    policy = parse_eval(yaml.safe_dump(receipt["evaluation"]))
    if eval_digest(policy) != receipt["metrics"]["evaluation_sha256"]:
        raise ValueError("Receipt resolved evaluation config mismatch")
    for key in ("checkpoint", "tokenizer", "dataset"):
        if (
            manifest_digest(_manifest(receipt["inputs"][key]))
            != getattr(policy, key).sha256
        ):
            raise ValueError("Receipt input snapshot mismatch")
    cp = _manifest(receipt["inputs"]["checkpoint"])
    if (
        not isinstance(cp.metadata, CheckpointInfo)
        or cp.metadata.tokenizer_sha256 != policy.tokenizer.sha256
    ):
        raise ValueError("Receipt checkpoint/tokenizer lineage mismatch")
    if (
        config_digest(parse_config(receipt["config_yaml"]))
        != report.metadata.config_sha256
    ):
        raise ValueError("Receipt model/runtime config mismatch")
    command = receipt["command"]
    if (
        not isinstance(command, dict)
        or set(command) != {"argv", "kind"}
        or command["kind"] != "reproduction_command"
        or not isinstance(command["argv"], list)
        or not 1 <= len(command["argv"]) <= 128
        or any(
            not isinstance(x, str)
            or not 1 <= len(x) <= 2048
            or any(ord(c) < 32 for c in x)
            or Path(x).is_absolute()
            for x in command["argv"]
        )
    ):
        raise ValueError("Ledger reproduction command must be portable bounded argv")


def validate_release_coverage(
    receipts: list[dict[str, Any]], required: set[str]
) -> dict[str, Any]:
    """Artifact-completeness gate only;does not certify unknown history or model quality."""
    present = {receipt["benchmark"] for receipt in receipts}
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            "Release lacks benchmark evaluation/audit: " + ",".join(missing)
        )
    audits = []
    for receipt in receipts:
        verify_receipt(receipt)
        audit = receipt["audit"]
        sources = audit.get("declared_sources", [])
        tokenized = [s for s in sources if s.get("representation") == "token_ids"]
        cp = _manifest(receipt["inputs"]["checkpoint"])
        assert isinstance(cp.metadata, CheckpointInfo)
        if (
            audit.get("protocol") != "shard-aware-v2"
            or not tokenized
            or any(s.get("truncated_fragments") != 0 for s in sources)
            or any(
                s.get("document_boundaries_verified") is not True
                or s.get("split") != "train"
                for s in tokenized
            )
            or cp.metadata.data_sha256 not in {s["manifest_sha256"] for s in tokenized}
            or len(audit.get("findings", {})) != receipt["metrics"]["problems"]
        ):
            raise ValueError(
                "Release requires untruncated actual-training token audit for every evaluated benchmark"
            )
        audits.append(receipt["report_sha256"])
    return {
        "required": sorted(required),
        "reports": audits,
        "artifact_coverage_passed": True,
        "complete_training_history_verified": False,
        "semantic_cleanliness_verified": False,
        "quality_promotion": False,
    }


def build_results(
    paths: list[Path],
    destination: Path,
    *,
    release: bool = False,
    required: set[str] | None = None,
    check: bool = False,
) -> str:
    if len(paths) > 256 or len(set(paths)) != len(paths):
        raise ValueError("Ledger needs1..256 distinct receipts")
    receipts = []
    for path in paths:
        with path.open("rb") as stream:
            data = stream.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2:
            raise ValueError("Receipt exceeds size limit")
        receipt = read_json(data)
        verify_receipt(receipt)
        receipts.append((path, receipt))
    if len({r["report_sha256"] for _, r in receipts}) != len(receipts):
        raise ValueError("Duplicate evaluation report in ledger")
    if release:
        from qvelt.eval.benchmarks import BENCHMARKS

        validate_release_coverage(
            [r for _, r in receipts],
            required if required is not None else set(BENCHMARKS),
        )
    lines = [
        "# Qvelt results",
        "",
        "Generated from verified evaluation receipts;do not hand-edit numbers.",
        "",
        "All entries are Qvelt measurements. Limited original/development subsets are not official public benchmark scores.",
        "Wilson intervals are descriptive;template correlation/nonrandom prefixes limit population inference.",
        "Compare only rows with the same comparison key. No scores from external papers are imported.",
        "",
    ]
    for path, r in sorted(receipts, key=lambda pair: pair[1]["report_sha256"]):
        metrics = r["metrics"]
        checkpoint = _manifest(r["inputs"]["checkpoint"])
        model_files = [f for f in checkpoint.files if f.role == "model"]
        relative = Path(os_path_relative(path, destination.parent)).as_posix()
        lines.extend(
            [
                f"## {r['benchmark']} / {r['report_sha256'][:12]}",
                "",
                f"[Verified receipt]({relative})",
                "",
                f"- Checkpoint manifest: `{r['evaluation']['checkpoint']['sha256']}`",
                f"- Model payload SHA256: `{','.join(f.sha256 for f in model_files)}`",
                f"- Tokenizer: `{r['evaluation']['tokenizer']['sha256']}`",
                f"- Dataset: `{r['evaluation']['dataset']['sha256']}`",
                f"- Evaluator source: `{r['provenance']['source_sha256']}`",
                f"- Comparison key: `{comparison_key(r)}`",
                "",
                "Reproduction command:",
                "",
                "```bash",
                shlex.join(r["command"]["argv"]),
                "```",
                "",
                "Exact semantic evaluation config (artifact paths are relocatable content addresses):",
                "",
                "```yaml",
                yaml.safe_dump(r["evaluation"], sort_keys=True).rstrip(),
                "```",
                "",
                "Exact model/runtime config:",
                "",
                "```yaml",
                r["config_yaml"].rstrip(),
                "```",
                "",
                "Measured metrics,including failures/intervals:",
                "",
                "```json",
                json.dumps(metrics, sort_keys=True, indent=2),
                "```",
                "",
                "Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.",
                "",
            ]
        )
    text = "\n".join(lines)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if check:
        if not destination.exists() or destination.read_text() != text:
            raise ValueError("Generated RESULTS.md is stale")
    else:
        destination.write_text(text)
    return text


def os_path_relative(path: Path, parent: Path) -> str:
    from os.path import relpath

    return relpath(path, parent)


def check_readme(readme: Path) -> None:
    """README contains no numeric result claims;versions and command parameters are operational."""
    text = re.sub(r"```.*?```", "", readme.read_text(), flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    for line in text.splitlines():
        if re.search(r"\d", line) and re.search(
            r"(?i)accuracy|scores?|perplexity|parameters?|\bupdates?\b|\btokens?\b|\breward\b|\bloss\b|\bpassed\b|throughput",
            line,
        ):
            raise ValueError(
                "Move numeric README result claims into generated RESULTS.md: "
                + line[:160]
            )
