"""Actual shared-lineage training,CLI reports,CPU probes and preflight failures."""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

import pytest

from qvelt.cli import main
from qvelt.eval.config import ArtifactRef, dump_eval, load_eval
from qvelt.eval.study import checkpoint_ledger, run_study
from qvelt.eval.study_config import StudyJob
from qvelt.eval.study_smoke import prepare_study
from qvelt.utils.artifacts import ArtifactStore, hash_file

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> StudyJob:
    return prepare_study(
        tmp_path_factory.mktemp("promotion") / "training", REPO, seeds=(20261105,)
    )


def test_actual_study_cli_publication_resources_and_input_integrity(
    study: StudyJob, tmp_path: Path
) -> None:
    inputs = {
        e.checkpoint.path: hash_file(
            Path(e.checkpoint.path).parent / "model.safetensors"
        )
        for e in study.entries
    }
    recipe = Path(study.evaluation_recipe.path).parent / "study.yaml"
    directory = tmp_path / "analysis"
    assert (
        main(
            [
                "compare",
                str(recipe),
                "--directory",
                str(directory),
                "--repository",
                str(REPO),
            ]
        )
        == 0
    )
    ArtifactStore(directory).read_manifest()
    result = json.loads((directory / "result.json").read_text())
    assert result["automatic_promotion"] is False and result["final_cases_scored"] == 0
    assert result["initialization_seeds"] == [20261105] and len(result["models"]) == 5
    assert result["verifier"]["passed"]
    assert result["sft_vs_base"]["20261105"]["problems"] == 200
    assert result["dpo_half_lr_vs_dpo"]["20261105"]["problems"] == 200
    for model in result["models"]:
        measured = model["resources"]
        assert len(measured["seconds"]) == measured["repetitions"] == 5
        assert measured["median_seconds"] == statistics.median(measured["seconds"])
        assert measured["tokens_per_second"] == pytest.approx(
            16 / measured["median_seconds"]
        )
        assert measured["weights_bytes"] == model["training"]["weights_bytes"] > 0
        assert measured["peak_rss_bytes"] > measured["weights_bytes"]
        assert model["report"]["metrics"]["problems"] == 247
        assert model["training"]["seed"] == 20261105
    assert all(
        c["status"] != "eligible_for_extended_validation"
        for c in result["comparisons_to_sft"].values()
    )
    assert inputs == {
        e.checkpoint.path: hash_file(
            Path(e.checkpoint.path).parent / "model.safetensors"
        )
        for e in study.entries
    }
    with pytest.raises(FileExistsError):
        run_study(study, directory, REPO)


def test_actual_lineage_rejects_wrong_seed_role_and_tokenizer(study: StudyJob) -> None:
    entry = next(e for e in study.entries if e.role == "sft")
    token = load_eval(Path(study.evaluation_recipe.path)).tokenizer.sha256
    with pytest.raises(ValueError, match="seed"):
        checkpoint_ledger(replace(entry, seed=entry.seed + 1), token)
    with pytest.raises(ValueError, match="kind"):
        checkpoint_ledger(replace(entry, role="grpo"), token)
    with pytest.raises(ValueError, match="tokenizer"):
        checkpoint_ledger(entry, "f" * 64)


@pytest.mark.parametrize("change", ["final", "sampled", "capped"])
def test_disallowed_protocol_fails_before_creating_output(
    study: StudyJob, tmp_path: Path, change: str
) -> None:
    job = load_eval(Path(study.evaluation_recipe.path))
    if change == "final":
        job = replace(job, allow_final=True)
    elif change == "sampled":
        job = replace(job, sampling="sample", temperature=1)
    else:
        job = replace(job, max_cases=246)
    path = tmp_path / "eval.yaml"
    path.write_text(dump_eval(job))
    altered = replace(
        study, evaluation_recipe=ArtifactRef(str(path), hash_file(path).sha256)
    )
    directory = tmp_path / "failed"
    with pytest.raises(ValueError):
        run_study(altered, directory, REPO)
    assert not directory.exists()
