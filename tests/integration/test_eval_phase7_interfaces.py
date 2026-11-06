"""Real optional harness requests,native v2 and token-shard audit inputs."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qvelt.eval.config import ArtifactRef, EvalJob
from qvelt.eval.corpus import audit_overlap, load_cases
from qvelt.eval.harness import HarnessBackend, make_lm_adapter, run_harness_parity
from qvelt.eval.runner import load_model, run_evaluation
from qvelt.eval.shard_audit import shard_documents
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def phase7_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, EvalJob]:
    root = tmp_path_factory.mktemp("phase7")
    return root, create_evaluation_fixture(root / "fixture", ROOT)


def test_train_shard_documents_and_full_coverage(
    phase7_fixture: tuple[Path, EvalJob],
) -> None:
    root, job = phase7_fixture
    path = root / "fixture/model-data/prepared/manifest.json"
    manifest = ArtifactStore(path.parent).read_manifest()
    reference = ArtifactRef(str(path), manifest_digest(manifest))
    _, rows, coverage = shard_documents(reference)
    documents = list(rows)
    assert (
        documents and coverage["documents"] == len(documents) and coverage["tokens"] > 0
    )
    assert coverage["document_boundaries_verified"] and coverage["split"] == "train"
    _, cases, _ = load_cases(job)
    report = audit_overlap(
        cases, replace(job, protocol="qvelt-eval-v2", training_sources=(reference,))
    )
    assert report["protocol"] == "shard-aware-v2"
    assert report["declared_sources"][0]["representation"] == "token_ids"


def test_native_v2_actual_report(
    phase7_fixture: tuple[Path, EvalJob], tmp_path: Path
) -> None:
    _, job = phase7_fixture
    result = run_evaluation(
        replace(
            job,
            protocol="qvelt-eval-v2",
            sampling="sample",
            samples=3,
            pass_k=(1, 3),
            temperature=1.0,
        ),
        tmp_path / "eval",
        ROOT,
    )
    assert result["metrics"]["generation"]["attempts"] == 6
    assert "3" in result["metrics"]["generation"]["majority_vote_at_k"]


def test_real_harness_interface_and_independent_logp(
    phase7_fixture: tuple[Path, EvalJob],
) -> None:
    pytest.importorskip("lm_eval")
    from lm_eval.api.instance import Instance

    _, job = phase7_fixture
    tokenizer, _ = load_tokenizer_bundle(Path(job.tokenizer.path), job.tokenizer.sha256)
    model, _, _ = load_model(job)
    backend = HarnessBackend(model, tokenizer, max_length=256, max_tokens=4)
    adapter = make_lm_adapter(backend)
    instance = Instance(
        request_type="loglikelihood", doc={}, arguments=("Question:", " 2"), idx=0
    )
    actual = adapter.loglikelihood([instance])[0]
    prefix, targets = [1, *tokenizer.encode("Question:")], tokenizer.encode(" 2")
    with torch.no_grad():
        logits = model(torch.tensor([prefix + targets[:-1]]))[0]
        expected = sum(
            float(logits[len(prefix) - 1 + i].log_softmax(-1)[token])
            for i, token in enumerate(targets)
        )
    assert actual[0] == pytest.approx(expected, abs=1e-6)
    assert adapter.rank == 0 and adapter.world_size == 1
    assert (
        adapter.loglikelihood_rolling(
            [
                Instance(
                    request_type="loglikelihood_rolling",
                    doc={},
                    arguments=("2",),
                    idx=0,
                )
            ]
        )[0]
        < 0
    )
    assert isinstance(
        adapter.generate_until(
            [
                Instance(
                    request_type="generate_until",
                    doc={},
                    arguments=("Q", {"max_gen_toks": 4}),
                    idx=0,
                )
            ]
        )[0],
        str,
    )
    result = run_harness_parity(
        replace(job, prompt_format="plain-v1", choice_normalization="character")
    )
    assert result["parity"] and result["cases"] == 2


def test_delimiter_excluded_normalization_independent_reference(
    phase7_fixture: tuple[Path, EvalJob],
) -> None:
    from unittest.mock import patch

    from qvelt.eval.corpus import EvalCase
    from qvelt.eval.runner import score_choice

    _, job = phase7_fixture
    tokenizer, _ = load_tokenizer_bundle(Path(job.tokenizer.path), job.tokenizer.sha256)
    model, _, _ = load_model(job)
    case = EvalCase(
        "norm", "choice", "oracle", "oracle", "control", "Q:", " xyz", (" x", " xyz")
    )
    with patch("qvelt.eval.runner.continuation_logp", side_effect=[-1.4, -3.0]):
        result = score_choice(
            model,
            tokenizer,
            case,
            replace(job, choice_normalization="character-without-delimiter"),
        )
    assert not result["correct"] and result["normalized_correct"]
    assert result["selected_normalized"] == max(
        range(2), key=lambda i: [-1.4 / 1, -3 / 3][i]
    )
    pytest.importorskip("lm_eval")
    assert run_harness_parity(
        replace(
            job,
            prompt_format="plain-v1",
            choice_normalization="character-without-delimiter",
        )
    )["parity"]
