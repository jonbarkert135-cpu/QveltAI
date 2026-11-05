"""Independent paired-count/macro-family expectations and strict study inputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from qvelt.config import ConfigError
from qvelt.eval.config import ArtifactRef, EvalJob, dump_eval
from qvelt.eval.corpus import EvalCase
from qvelt.eval.fixtures import protected_cases
from qvelt.eval.runner import aggregate
from qvelt.eval.smoke import control_cases
from qvelt.eval.study import (
    comparison_status,
    necessary_gates,
    paired_summary,
    recipe_bytes,
    validate_lineage,
    verifier_controls,
)
from qvelt.eval.study_config import StudyEntry, StudyJob, dump_study, load_study
from qvelt.eval.study_smoke import study_cases
from qvelt.utils.artifacts import hash_file


def job() -> EvalJob:
    ref = ArtifactRef("/local/manifest.json", "a" * 64)
    return EvalJob(ref, ref, ref)


def rows(improve: bool) -> list[dict[str, Any]]:
    primary = [
        EvalCase(
            f"{family * 40 + i:064x}",
            "math",
            "math",
            f"family-{family}",
            "validation",
            f"Original item {family},{i}",
            "1",
        )
        for family in range(5)
        for i in range(40)
    ]
    result = []
    for case in study_cases(primary):
        row = asdict(case)
        if case.kind in ("math", "numeric"):
            parent = (
                case.identifier.split("-")[1]
                if case.domain == "robustness"
                else case.identifier
            )
            good = improve and len(parent) == 64 and int(parent, 16) % 40 == 0
            row["attempts"] = [
                {
                    "correct": good,
                    "format_valid": True,
                    "extracted": "1" if good else "0",
                    "complete": True,
                    "stop": "complete",
                    "generated_tokens": 2,
                }
            ]
        elif case.kind == "choice":
            row["score"] = {"correct": True, "normalized_correct": True, "failed": None}
        else:
            row["score"] = {"targets": 10, "nll": 10.0, "failed": None}
        result.append(row)
    return result


def resource() -> dict[str, Any]:
    return {
        "protocol": "fixed",
        "system": "Linux",
        "machine": "x86_64",
        "torch": "2.9.1",
        "source_sha256": "a" * 64,
        "device": "cpu",
        "precision": "fp32",
        "threads": 1,
        "prefix_tokens": 8,
        "generated_steps": 16,
        "warmups": 2,
        "repetitions": 5,
        "peak_rss_bytes": 600_000_000,
        "median_seconds": 0.1,
    }


def test_paired_counts_match_known_five_family_contingency_table() -> None:
    before, after = rows(False), rows(True)
    result = paired_summary(before, after)
    assert result["problems"] == 200
    assert result["before_correct"] == 0 and result["after_correct"] == 5
    assert result["wins"] == 5 and result["losses"] == 0
    assert result["macro_family_delta"] == 0.025
    assert {r["delta"] for r in result["families"].values()} == {0.025}
    assert result["robust_before"] == {"problems": 40, "correct": 0, "invariant": 40}
    assert result["robust_after"] == {"problems": 40, "correct": 10, "invariant": 40}
    gates = necessary_gates(
        result,
        aggregate(before, job()),
        aggregate(after, job()),
        resource(),
        resource(),
        verifier_controls(),
    )
    assert all(v is True for v in gates.values())
    # Micro averaging would give a different answer when family sizes differ.
    shorter_before = before[:40] + before[40:60]
    shorter_after = after[:40] + after[40:60]
    assert paired_summary(shorter_before, shorter_after)[
        "macro_family_delta"
    ] == pytest.approx(0.0375)


@pytest.mark.parametrize("change", ["target", "prompt", "identifier", "family"])
def test_pairs_reject_changed_task_or_gold(change: str) -> None:
    before, after = rows(False), rows(False)
    after[0][change] += "changed"
    with pytest.raises(ValueError, match="identities"):
        paired_summary(before, after)


def test_robustness_declares_and_preserves_parent_task() -> None:
    before, after = rows(False), rows(False)
    for result in (before, after):
        result[200]["prompt"] = "a different mathematical question"
    with pytest.raises(ValueError, match="transformation"):
        paired_summary(before, after)
    primary = []
    import json

    for document in protected_cases("validation"):
        body = json.loads(document.text)
        body.pop("schema_version")
        body["choices"] = tuple(body["choices"])
        primary.append(EvalCase(**body))
    suite = study_cases(primary)
    assert len(suite) == 247 and len({c.identifier for c in suite}) == 247
    assert sum(c.domain == "robustness" for c in suite) == 40
    assert all(c.partition != "final" for c in suite)
    with pytest.raises(ValueError):
        study_cases(primary[:10])


@pytest.mark.parametrize(
    "regression", ["math", "format", "choice", "lm", "memory", "latency", "verifier"]
)
def test_each_measured_regression_blocks_necessary_gate(regression: str) -> None:
    before, after = rows(False), rows(True)
    paired = paired_summary(before, after)
    baseline, candidate = aggregate(before, job()), aggregate(after, job())
    measured, verifier = resource(), verifier_controls()
    if regression == "math":
        paired["macro_family_delta"] = 0.0
    elif regression == "format":
        candidate["generation"]["invalid"] += 1
    elif regression == "choice":
        paired["choice_after"]["web"]["correct"] = 0
    elif regression == "lm":
        candidate["domains"]["web"]["lm"]["perplexity"] *= 1.051
    elif regression == "memory":
        measured["peak_rss_bytes"] = 2 * 1024**3 + 1
    elif regression == "latency":
        measured["median_seconds"] = 0.40001
    else:
        verifier["passed"] = False
    gates = necessary_gates(paired, baseline, candidate, resource(), measured, verifier)
    assert gates["math_gain" if regression == "math" else regression] is False


def test_missing_coverage_is_not_a_success_and_resources_are_real_finite_values() -> (
    None
):
    before, after = rows(False), rows(True)
    paired = paired_summary(before, after)
    paired["problems"] = 10
    baseline, candidate = aggregate(before, job()), aggregate(after, job())
    gates = necessary_gates(
        paired, baseline, candidate, resource(), resource(), verifier_controls()
    )
    assert not gates["primary_coverage"] and gates["math_gain"] is None
    for bad in (float("nan"), 0, True):
        measured = resource()
        measured["median_seconds"] = bad
        with pytest.raises(ValueError, match="resources"):
            necessary_gates(
                paired, baseline, candidate, resource(), measured, verifier_controls()
            )
    measured = resource()
    measured["threads"] = 2
    with pytest.raises(ValueError, match="comparable"):
        necessary_gates(
            paired, baseline, candidate, resource(), measured, verifier_controls()
        )


def test_strict_study_yaml_and_recipe_integrity(tmp_path: Path) -> None:
    file = tmp_path / "evaluation.yaml"
    file.write_text(dump_eval(job()))
    ref = ArtifactRef(str(file), hash_file(file).sha256)
    entries = tuple(
        StudyEntry(role, 1, ArtifactRef("/cp/manifest.json", str(i) * 64))
        for i, role in enumerate(("base", "sft", "dpo", "grpo"), 1)
    )
    study = StudyJob(ref, entries)
    path = tmp_path / "study.yaml"
    path.write_text(dump_study(study))
    assert load_study(path) == study
    assert recipe_bytes(ref) == file.read_bytes()
    with pytest.raises(ConfigError):
        replace(study, entries=entries + (entries[0],))
    with pytest.raises(ConfigError):
        replace(entries[0], seed=True)
    path.write_text(dump_study(study) + "unknown: true\n")
    with pytest.raises(ConfigError):
        load_study(path)
    file.write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        recipe_bytes(ref)


def lineage() -> list[dict[str, Any]]:
    return [
        {
            "role": role,
            "seed": 1,
            "model": {"hidden": 32},
            "tokenizer_sha256": "token",
            "training_source_sha256": "source",
            "weights_sha256": role,
            "data_sha256": "pairs" if role == "dpo-half-lr" else role,
            "checkpoint_sha256": role,
            "parent_sha256": None
            if role == "base"
            else "base"
            if role == "sft"
            else "sft",
            "training": {
                "steps": 40,
                "learning_rate": 0.0005 if role == "dpo-half-lr" else 0.001,
            },
            "beta": 0.1,
            "optimizer_boundary": 40,
            "processed_targets_or_actions": 100,
        }
        for role in ("base", "sft", "dpo", "grpo", "dpo-half-lr")
    ]


def test_lineage_and_single_factor_ablation_are_enforced() -> None:
    records = lineage()
    records[-1]["data_sha256"] = "dpo"
    validate_lineage(records)
    for field, value in (
        ("parent_sha256", "different-sft"),
        ("data_sha256", "different-pairs"),
        ("beta", 0.2),
        ("processed_targets_or_actions", 99),
    ):
        changed = deepcopy(records)
        changed[-1][field] = value
        with pytest.raises(ValueError):
            validate_lineage(changed)
    changed = deepcopy(records)
    changed[-1]["training"]["steps"] = 41
    with pytest.raises(ValueError):
        validate_lineage(changed)
    extra = deepcopy(records)
    for r in extra:
        r["seed"] = 2
    with pytest.raises(ValueError, match="identical base"):
        validate_lineage(records + extra)


def test_verifier_positive_controls_and_exploitation_rejection() -> None:
    result = verifier_controls()
    assert result["cases"] >= 18 and result["passed"] and not result["failures"]
    assert len(control_cases()) == 7


def test_replication_missing_coverage_and_rejection_never_promote() -> None:
    before, after = rows(False), rows(True)
    gates = necessary_gates(
        paired_summary(before, after),
        aggregate(before, job()),
        aggregate(after, job()),
        resource(),
        resource(),
        verifier_controls(),
    )
    first: dict[str, Any] = {"seed": 1, "gates": gates}
    second: dict[str, Any] = {"seed": 2, "gates": dict(first["gates"])}
    assert comparison_status([]) == "inconclusive"
    assert comparison_status([first]) == "inconclusive"
    assert comparison_status([first, second]) == "eligible_for_extended_validation"
    second["gates"]["math_gain"] = None
    assert comparison_status([first, second]) == "inconclusive"
    second["gates"]["math_gain"] = False
    assert comparison_status([first, second]) == "reject"
    with pytest.raises(ValueError, match="Repeated seed"):
        comparison_status([first, first])
    with pytest.raises(ValueError, match="Missing"):
        comparison_status([{"seed": 1, "gates": {"math_gain": True}}])
