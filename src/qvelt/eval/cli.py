"""Bounded evaluation-source,audit,receipt,ledger and recovery command adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

EVAL_COMMANDS = {
    "eval-import",
    "eval-audit",
    "eval-receipt",
    "eval-registry",
    "eval-repair",
    "eval-harness",
    "results",
}


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    source = commands.add_parser(
        "eval-import", help="Import a reviewed immutable benchmark subset"
    )
    source.add_argument("benchmark")
    source.add_argument("path", type=Path)
    source.add_argument("--sha256", required=True)
    source.add_argument("--license-file", type=Path, required=True)
    source.add_argument("--license-sha256", required=True)
    source.add_argument("--count", type=int, default=16)
    source.add_argument("--directory", type=Path, required=True)
    for name in ("eval-audit", "eval-harness"):
        command = commands.add_parser(
            name, help="Inspect a pinned local evaluation recipe"
        )
        command.add_argument("path", type=Path)
        if name == "eval-audit":
            command.add_argument("--output", type=Path, required=True)
    receipt = commands.add_parser(
        "eval-receipt",
        help="Export a report-bound public receipt without corpus/model weights",
    )
    receipt.add_argument("path", type=Path)
    receipt.add_argument("--sha256", required=True)
    receipt.add_argument("--benchmark", required=True)
    receipt.add_argument("--output", type=Path, required=True)
    registry = commands.add_parser(
        "eval-registry", help="Generate/check the reviewed benchmark protocol inventory"
    )
    registry.add_argument(
        "--output", type=Path, default=Path("configs/eval/benchmarks.yaml")
    )
    registry.add_argument("--check", action="store_true")
    repair = commands.add_parser(
        "eval-repair", help="Repair missing committed-checkpoint regression reports"
    )
    repair.add_argument("path", type=Path, help="Training run directory")
    repair.add_argument("--repository", type=Path, default=Path("."))
    results = commands.add_parser(
        "results", help="Generate/check the results ledger from verified receipts"
    )
    results.add_argument("paths", type=Path, nargs="*")
    results.add_argument("--output", type=Path, default=Path("RESULTS.md"))
    results.add_argument("--readme", type=Path)
    results.add_argument("--check", action="store_true")
    results.add_argument("--release", action="store_true")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from qvelt.eval.config import ArtifactRef, load_eval
    from qvelt.utils.manifests import manifest_digest

    if args.command == "eval-import":
        from qvelt.eval.benchmarks import import_benchmark

        manifest = import_benchmark(
            args.benchmark,
            ArtifactRef(str(args.path), args.sha256),
            ArtifactRef(str(args.license_file), args.license_sha256),
            args.directory,
            count=args.count,
        )
        return {"manifest_sha256": manifest_digest(manifest)}
    if args.command == "eval-audit":
        from qvelt.eval.corpus import audit_overlap, load_cases
        from qvelt.eval.runner import _json
        from qvelt.utils.artifacts import ArtifactStore

        job = load_eval(args.path)
        _, cases, _ = load_cases(job)
        audit = audit_overlap(cases, job)
        record = ArtifactStore(args.output.parent).add_bytes(
            args.output.name, _json(audit), role="overlap_audit"
        )
        return {
            "audit_sha256": record.sha256,
            "cases": len(cases),
            "cleanliness_claim": False,
        }
    if args.command == "eval-receipt":
        from qvelt.eval.ledger import export_receipt

        receipt = export_receipt(
            ArtifactRef(str(args.path), args.sha256),
            args.output,
            benchmark=args.benchmark,
        )
        return {"report_sha256": receipt["report_sha256"]}
    if args.command == "eval-registry":
        import yaml

        from qvelt.eval.benchmarks import BENCHMARKS

        body = yaml.safe_dump(
            {"schema_version": 1, "benchmarks": BENCHMARKS}, sort_keys=True
        )
        if args.check:
            if not args.output.exists() or args.output.read_text() != body:
                raise ValueError("Benchmark inventory is stale")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(body)
        return {"entries": len(BENCHMARKS), "checked": args.check}
    if args.command == "eval-repair":
        from qvelt.config.training import load_training_job
        from qvelt.eval.regression import evaluate_checkpoint
        from qvelt.utils.checkpoints import committed_steps

        training_job = load_training_job(args.path / "job.yaml")
        steps = committed_steps(args.path)
        if not steps:
            raise ValueError("No committed checkpoints to evaluate")
        for step, _ in sorted(steps):
            evaluate_checkpoint(args.path, step, training_job, args.repository)
        return {"checkpoints_evaluated_or_verified": len(steps)}
    if args.command == "eval-harness":
        from qvelt.eval.harness import run_harness_parity

        return run_harness_parity(load_eval(args.path))
    if args.command == "results":
        from qvelt.eval.ledger import build_results, check_readme

        if args.readme:
            check_readme(args.readme)
        build_results(args.paths, args.output, release=args.release, check=args.check)
        return {
            "receipts": len(args.paths),
            "checked": args.check,
            "release_coverage_checked": args.release,
        }
    raise ValueError("Unknown evaluation command")
