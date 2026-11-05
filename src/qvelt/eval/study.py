"""Matched lineage,paired validation regressions and necessary further-study gates."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from qvelt.config import (
    MAX_CONFIG_BYTES,
    config_digest,
    dump_config,
    parse_yaml_mapping,
)
from qvelt.config.training import parse_training_job
from qvelt.eval.config import ArtifactRef, dump_eval, eval_digest, parse_eval
from qvelt.eval.corpus import load_cases, read_json
from qvelt.eval.runner import _json, _payload, run_evaluation
from qvelt.eval.scoring import parse_number, score_answer
from qvelt.eval.study_config import StudyEntry, StudyJob, dump_study
from qvelt.utils.artifacts import ArtifactStore, create_directory, hash_file
from qvelt.utils.checkpoints import source_digest
from qvelt.utils.manifest_types import ArtifactFile, CheckpointInfo, Manifest, RunInfo
from qvelt.utils.manifests import manifest_digest
from qvelt.utils.provenance import capture_code, capture_environment


def recipe_bytes(ref: ArtifactRef) -> bytes:
    path = Path(ref.path)
    with ArtifactStore(path.parent).open_file(path.name) as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES or hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ValueError("Study evaluation recipe bytes changed")
    return data


def checkpoint_ledger(entry: StudyEntry, tokenizer_sha: str) -> dict[str, Any]:
    path = Path(entry.checkpoint.path)
    store = ArtifactStore(path.parent)
    cp = store.read_manifest(
        path.name, expected_sha256=entry.checkpoint.sha256, verify=False
    )
    meta = cp.metadata
    if (
        not isinstance(meta, CheckpointInfo)
        or meta.world_size != 1
        or meta.precision != "fp32"
        or meta.tokenizer_sha256 != tokenizer_sha
        or sum(f.size_bytes for f in cp.files) > 512 * 1024**2
    ):
        raise ValueError("Study checkpoint/tokenizer/topology/capacity mismatch")
    store.verify(cp)
    training = parse_training_job(
        _payload(store, cp, "training_recipe", MAX_CONFIG_BYTES).decode()
    )
    if (
        training.experiment.runtime.seed != entry.seed
        or training.experiment.runtime.device != "cpu"
        or config_digest(training.experiment) != meta.config_sha256
        or training.experiment.model.parameter_count > 20_000_000
    ):
        raise ValueError("Study seed/config/device mismatch")
    run_store = ArtifactStore(path.parent.parents[1])
    run = run_store.read_manifest(expected_sha256=meta.run_sha256)
    if (
        not isinstance(run.metadata, RunInfo)
        or run.metadata.config_sha256 != meta.config_sha256
    ):
        raise ValueError("Study requires original full run/checkpoint lineage")
    app = read_json(_payload(store, cp, "application", 1024**2))
    expected_kind = {
        "base": "packed-pretraining-v2",
        "sft": "packed-sft-v2",
        "dpo": "qvelt-dpo-v1",
        "dpo-half-lr": "qvelt-dpo-v1",
        "grpo": "qvelt-grpo-v1",
    }[entry.role]
    if not isinstance(app, dict) or app.get("kind") != expected_kind:
        raise ValueError("Checkpoint application kind disagrees with study role")
    application = parse_yaml_mapping(
        _payload(run_store, run, "application_recipe", MAX_CONFIG_BYTES).decode()
    )
    parent = (
        application.get("initial_sha256")
        if entry.role == "sft"
        else application.get("reference_sha256")
    )
    if (
        entry.role == "base"
        and application.get("objective") != "pretraining"
        or entry.role == "sft"
        and application.get("objective") != "sft"
    ):
        raise ValueError("Study base/SFT objective mismatch")
    sampler = read_json(_payload(store, cp, "sampler", MAX_CONFIG_BYTES))
    weights = [f for f in cp.files if f.role == "model"]
    if len(weights) != 1:
        raise ValueError("Study requires one unsharded model payload")
    return {
        "role": entry.role,
        "seed": entry.seed,
        "checkpoint_sha256": entry.checkpoint.sha256,
        "run_sha256": meta.run_sha256,
        "parent_sha256": parent,
        "data_sha256": meta.data_sha256,
        "tokenizer_sha256": meta.tokenizer_sha256,
        "model": asdict(training.experiment.model),
        "training": asdict(training.training),
        "optimizer_boundary": meta.optimizer_step,
        "processed_targets_or_actions": meta.processed_tokens,
        "counters": app.get("counters"),
        "weights_sha256": weights[0].sha256,
        "weights_bytes": weights[0].size_bytes,
        "training_source_sha256": sampler["compatibility"]["source_sha256"],
        "code": asdict(run.metadata.code),
        "beta": application.get("beta"),
        "grpo": {
            k: application[k]
            for k in ("groups", "prompts", "max_tokens", "clip", "kl")
            if k in application
        },
    }


def validate_lineage(ledgers: list[dict[str, Any]]) -> None:
    if not ledgers:
        raise ValueError("Empty study lineage")
    if any(
        r["model"] != ledgers[0]["model"]
        or r["tokenizer_sha256"] != ledgers[0]["tokenizer_sha256"]
        or r["training_source_sha256"] != ledgers[0]["training_source_sha256"]
        for r in ledgers
    ):
        raise ValueError("Study models/tokenizers/executable training sources disagree")
    seeds = {r["seed"] for r in ledgers}
    bases = [r for r in ledgers if r["role"] == "base"]
    if len({r["weights_sha256"] for r in bases}) != len(seeds):
        raise ValueError("Initialization seeds cannot reuse identical base weights")
    for role in {r["role"] for r in ledgers}:
        if len({r["data_sha256"] for r in ledgers if r["role"] == role}) != 1:
            raise ValueError("Study data must agree across initialization seeds")
    for seed in seeds:
        rows = {r["role"]: r for r in ledgers if r["seed"] == seed}
        if rows["sft"]["parent_sha256"] != rows["base"]["checkpoint_sha256"] or any(
            r["parent_sha256"] != rows["sft"]["checkpoint_sha256"]
            for role, r in rows.items()
            if role not in ("base", "sft")
        ):
            raise ValueError("Study post-training parents do not match")
        if "dpo-half-lr" in rows:
            a, b = rows["dpo"], rows["dpo-half-lr"]
            left, right = dict(a["training"]), dict(b["training"])
            lr_a, lr_b = left.pop("learning_rate"), right.pop("learning_rate")
            if (
                left != right
                or lr_b != lr_a / 2
                or any(
                    a[k] != b[k]
                    for k in (
                        "data_sha256",
                        "beta",
                        "optimizer_boundary",
                        "processed_targets_or_actions",
                    )
                )
            ):
                raise ValueError(
                    "DPO half-LR ablation changes more than the declared factor"
                )


def verifier_controls() -> dict[str, Any]:
    cases = [
        ("2", "math", True, True),
        ("4/2", "math", True, True),
        ("2.0", "numeric", True, True),
        ("2.0000001", "numeric", True, True),
        ("2.01", "numeric", True, False),
        *[
            (s, "math", True, False)
            for s in (
                "1+1",
                "2 kg",
                "2\n3",
                "1/0",
                "NaN",
                "inf",
                "__import__('os').system('false')",
                "9" * 131,
                "2/0",
                "2 then 3",
                "Final answer: 2",
            )
        ],
        ("2", "math", False, False),
        ("2e0", "math", True, False),
        ("2e0", "numeric", True, True),
    ]
    failures = [
        i
        for i, (text, kind, complete, expected) in enumerate(cases)
        if score_answer(text, "2", kind, "whole-v1", complete=complete)["correct"]
        != expected
    ]
    return {
        "protocol": "bounded-verifier-adversarial-v1",
        "cases": len(cases),
        "failures": failures,
        "passed": not failures,
        "scope": "trusted bounded parser only;no arbitrary code/CAS execution",
    }


def paired_summary(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(before) != len(after) or not before:
        raise ValueError("Paired study rows must be nonempty and aligned")
    fixed = (
        "identifier",
        "kind",
        "domain",
        "family",
        "partition",
        "prompt",
        "target",
        "choices",
    )
    if len({r["identifier"] for r in before}) != len(before) or any(
        any(a[k] != b[k] for k in fixed) for a, b in zip(before, after, strict=True)
    ):
        raise ValueError("Paired study case identities/golds/prompts disagree")
    primary = [
        (a, b)
        for a, b in zip(before, after, strict=True)
        if a["partition"] == "validation"
        and a["domain"] == "math"
        and a["kind"] == "math"
    ]

    def correct(row: dict[str, Any]) -> bool:
        return bool(row["attempts"][0]["correct"])

    families = {}
    exact_deltas = []
    for family in sorted({a["family"] for a, _ in primary}):
        pairs = [(a, b) for a, b in primary if a["family"] == family]
        ca, cb = sum(correct(a) for a, _ in pairs), sum(correct(b) for _, b in pairs)
        delta = Fraction(cb - ca, len(pairs))
        exact_deltas.append(delta)
        families[family] = {
            "problems": len(pairs),
            "before_correct": ca,
            "after_correct": cb,
            "delta": float(delta),
        }

    def robust(rows: list[dict[str, Any]]) -> dict[str, int]:
        parents = {r["identifier"]: r for r in rows if r["partition"] == "validation"}
        selected = [r for r in rows if r["domain"] == "robustness"]
        invariant = 0
        for row in selected:
            parts = row["identifier"].split("-")
            if (
                len(parts) != 3
                or parts[0] != "robust"
                or parts[2] not in ("leading", "trailing")
                or parts[1] not in parents
            ):
                raise ValueError("Robustness variant lacks a declared parent")
            parent = parents[parts[1]]
            expected = (
                "  " + parent["prompt"]
                if parts[2] == "leading"
                else parent["prompt"] + "\n"
            )
            if (
                row["prompt"] != expected
                or row["target"] != parent["target"]
                or row["family"] != parent["family"]
            ):
                raise ValueError("Robustness transformation changes the parent task")
            x, y = row["attempts"][0], parent["attempts"][0]
            invariant += bool(
                x["complete"]
                and y["complete"]
                and x["format_valid"]
                and y["format_valid"]
                and parse_number(x["extracted"]) == parse_number(y["extracted"])
            )
        return {
            "problems": len(selected),
            "correct": sum(correct(r) for r in selected),
            "invariant": invariant,
        }

    def choices(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        result = {}
        for domain in ("web", "code"):
            selected = [
                r for r in rows if r["kind"] == "choice" and r["domain"] == domain
            ]
            result[domain] = {
                "problems": len(selected),
                "correct": sum(r["score"]["correct"] for r in selected),
                "normalized_correct": sum(
                    r["score"]["normalized_correct"] for r in selected
                ),
                "failed": sum(bool(r["score"]["failed"]) for r in selected),
            }
        return result

    return {
        "problems": len(primary),
        "before_correct": sum(correct(a) for a, _ in primary),
        "after_correct": sum(correct(b) for _, b in primary),
        "wins": sum(not correct(a) and correct(b) for a, b in primary),
        "losses": sum(correct(a) and not correct(b) for a, b in primary),
        "families": families,
        "macro_family_delta": float(sum(exact_deltas, Fraction()) / len(exact_deltas))
        if exact_deltas
        else None,
        "robust_before": robust(before),
        "robust_after": robust(after),
        "choice_before": choices(before),
        "choice_after": choices(after),
    }


def necessary_gates(
    paired: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    base_resources: dict[str, Any],
    resources: dict[str, Any],
    verifier: dict[str, Any],
) -> dict[str, bool | None]:
    families = paired["families"]
    primary = paired["problems"] >= 200 and len(families) >= 5
    robust = paired["robust_before"]["problems"] >= 40
    domains = ("web", "code", "math")
    lm = all(
        baseline["domains"].get(d, {}).get("lm", {}).get("targets", 0) > 0
        and candidate["domains"].get(d, {}).get("lm", {}).get("targets", 0)
        == baseline["domains"][d]["lm"]["targets"]
        for d in domains
    )
    choice = all(paired["choice_before"][d]["problems"] >= 1 for d in ("web", "code"))
    keys = (
        "protocol",
        "system",
        "machine",
        "torch",
        "source_sha256",
        "device",
        "precision",
        "threads",
        "prefix_tokens",
        "generated_steps",
        "warmups",
        "repetitions",
    )
    if any(resources[k] != base_resources[k] for k in keys):
        raise ValueError(
            "Resource measurements are not from a comparable protocol/host"
        )
    for item in (resources, base_resources):
        if (
            type(item["peak_rss_bytes"]) is not int
            or item["peak_rss_bytes"] <= 0
            or type(item["median_seconds"]) not in (float, int)
            or not math.isfinite(item["median_seconds"])
            or item["median_seconds"] <= 0
        ):
            raise ValueError("Invalid measured inference resources")
    return {
        "primary_coverage": primary,
        "robust_coverage": robust,
        "lm_coverage": lm,
        "choice_coverage": choice,
        "math_gain": paired["macro_family_delta"] >= 0.01 if primary else None,
        "no_family_regression": all(f["delta"] >= 0 for f in families.values())
        if primary
        else None,
        "format": all(
            candidate["generation"][k] <= baseline["generation"][k]
            for k in ("invalid", "truncated", "failed")
        ),
        "robust_correct": paired["robust_after"]["correct"]
        >= paired["robust_before"]["correct"]
        if robust
        else None,
        "robust_invariance": paired["robust_after"]["invariant"]
        >= paired["robust_before"]["invariant"]
        if robust
        else None,
        "choice": all(
            paired["choice_after"][d][k] >= paired["choice_before"][d][k]
            for d in ("web", "code")
            for k in ("correct", "normalized_correct")
        )
        and all(
            paired["choice_after"][d]["failed"] <= paired["choice_before"][d]["failed"]
            for d in ("web", "code")
        )
        if choice
        else None,
        "lm": all(
            candidate["domains"][d]["lm"]["perplexity"] is not None
            and baseline["domains"][d]["lm"]["perplexity"] is not None
            and candidate["domains"][d]["lm"]["perplexity"]
            <= 1.05 * baseline["domains"][d]["lm"]["perplexity"]
            for d in domains
        )
        if lm
        else None,
        "verifier": verifier["passed"] is True,
        "memory": resources["peak_rss_bytes"] <= 2 * 1024**3,
        "latency": resources["median_seconds"] <= 4 * base_resources["median_seconds"],
    }


def comparison_status(rows: list[dict[str, Any]]) -> str:
    required = {
        "primary_coverage",
        "robust_coverage",
        "lm_coverage",
        "choice_coverage",
        "math_gain",
        "no_family_regression",
        "format",
        "robust_correct",
        "robust_invariance",
        "choice",
        "lm",
        "verifier",
        "memory",
        "latency",
    }
    if any(
        set(r["gates"]) != required
        or any(v is not None and type(v) is not bool for v in r["gates"].values())
        for r in rows
    ):
        raise ValueError("Missing or invalid necessary gate evidence")
    if len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("Repeated seed is not independent study replication")
    coverage = {"primary_coverage", "robust_coverage", "lm_coverage", "choice_coverage"}
    if any(
        v is False for r in rows for k, v in r["gates"].items() if k not in coverage
    ):
        return "reject"
    if len(rows) >= 2 and all(v is True for r in rows for v in r["gates"].values()):
        return "eligible_for_extended_validation"
    return "inconclusive"


def run_study(study: StudyJob, directory: Path, repository: Path) -> dict[str, Any]:
    recipe = recipe_bytes(study.evaluation_recipe)
    protocol = parse_eval(recipe.decode("utf-8-sig"))
    if protocol.allow_final or protocol.sampling != "greedy":
        raise ValueError("Promotion study requires greedy nonfinal evaluation")
    _, cases, available = load_cases(protocol)
    if len(cases) != available:
        raise ValueError("Promotion study cannot silently cap its evaluation set")
    ledgers = [checkpoint_ledger(e, protocol.tokenizer.sha256) for e in study.entries]
    validate_lineage(ledgers)
    root = create_directory(directory)
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    predictions = {}
    resources = {}
    for entry in study.entries:
        name = f"{entry.role}-{entry.seed}"
        job = replace(protocol, checkpoint=entry.checkpoint)
        recipe_path = root / (name + ".yaml")
        recipe_path.write_text(dump_eval(job))
        reports[entry.role, entry.seed] = run_evaluation(job, root / name, repository)
        predictions[entry.role, entry.seed] = [
            read_json(line)
            for line in (root / name / "predictions.jsonl").read_text().splitlines()
        ]
        resource_path = root / (name + "-resources.json")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "qvelt.eval.resources",
                "--recipe",
                str(recipe_path),
                "--output",
                str(resource_path),
            ],
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            check=True,
            timeout=120,
        )
        measured = read_json(resource_path.read_bytes())
        if (
            measured["checkpoint_sha256"] != entry.checkpoint.sha256
            or measured["source_sha256"] != source_digest()
            or measured["evaluation_sha256"] != eval_digest(job)
        ):
            raise ValueError("Inference resource evidence identity mismatch")
        resources[entry.role, entry.seed] = measured
    verifier = verifier_controls()
    comparisons: dict[str, Any] = {}
    seeds = sorted({e.seed for e in study.entries})
    for role in sorted({e.role for e in study.entries} - {"base", "sft"}):
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            paired = paired_summary(predictions["sft", seed], predictions[role, seed])
            gates = necessary_gates(
                paired,
                reports["sft", seed]["metrics"],
                reports[role, seed]["metrics"],
                resources["sft", seed],
                resources[role, seed],
                verifier,
            )
            rows.append({"seed": seed, "paired": paired, "gates": gates})
        comparisons[role] = {
            "status": comparison_status(rows),
            "seeds": rows,
            "macro_delta_range": [
                min(r["paired"]["macro_family_delta"] for r in rows),
                max(r["paired"]["macro_family_delta"] for r in rows),
            ]
            if all(r["paired"]["macro_family_delta"] is not None for r in rows)
            else None,
        }
    result = {
        "protocol": study.protocol,
        "source_sha256": source_digest(),
        "scope": "original CPU necessary regression gates;not production promotion",
        "initialization_seeds": seeds,
        "models": [
            {
                "role": e.role,
                "seed": e.seed,
                "report": reports[e.role, e.seed],
                "training": ledgers[i],
                "resources": resources[e.role, e.seed],
            }
            for i, e in enumerate(study.entries)
        ],
        "comparisons_to_sft": comparisons,
        "sft_vs_base": {
            str(seed): paired_summary(
                predictions["base", seed], predictions["sft", seed]
            )
            for seed in seeds
        },
        "dpo_half_lr_vs_dpo": {
            str(seed): paired_summary(
                predictions["dpo", seed], predictions["dpo-half-lr", seed]
            )
            for seed in seeds
        }
        if any(e.role == "dpo-half-lr" for e in study.entries)
        else {},
        "verifier": verifier,
        "automatic_promotion": False,
        "production_missing": [
            "independent licensed quality/contamination evidence",
            "untouched preregistered final protocol",
            "actual user-run small/GPU resource acceptance",
        ],
        "uncertainty": "Paired family/initialization-seed ranges are descriptive;correlated templates/variants and tiny control counts do not establish population significance or public quality.",
        "final_cases_scored": 0,
    }
    checkout, code = capture_code(repository)
    environment = capture_environment()
    run_id = uuid.uuid4().hex
    # Child report configs already bind the shared model and evaluation runtime.
    first = root / f"{study.entries[0].role}-{study.entries[0].seed}"
    store = ArtifactStore(root)
    from qvelt.config import parse_config

    config = parse_config((first / "config.yaml").read_text())
    store.add_bytes("config.yaml", dump_config(config).encode(), role="config")
    store.add_file("uv.lock", checkout / "uv.lock", role="dependency_lock")
    store.add_bytes("study.yaml", dump_study(study).encode(), role="study_recipe")
    store.add_bytes("evaluation.yaml", recipe, role="evaluation_recipe")
    store.add_bytes("result.json", _json(result), role="study_result")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            info = hash_file(path)
            role = {"config.yaml": "config", "uv.lock": "dependency_lock"}.get(
                relative, "study_artifact"
            )
            files.append(ArtifactFile(relative, role, info.size_bytes, info.sha256))
    manifest = Manifest(
        kind="run",
        owner=run_id,
        created_at=datetime.now(UTC).isoformat(),
        metadata=RunInfo(
            run_id, config_digest(config), code, environment, config.runtime.seed, 1
        ),
        files=tuple(files),
    )
    store.publish(manifest)
    return {"study_sha256": manifest_digest(manifest), "result": result}
