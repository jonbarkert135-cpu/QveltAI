"""Ledger tamper rejection, release coverage and hand-built source conversions."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from qvelt.eval.benchmarks import BENCHMARKS, benchmark_rows, import_benchmark
from qvelt.eval.config import ArtifactRef
from qvelt.eval.corpus import EvalCase, audit_overlap
from qvelt.eval.ledger import (
    build_results,
    check_readme,
    comparison_key,
    export_receipt,
    validate_release_coverage,
    verify_receipt,
)
from qvelt.eval.runner import run_evaluation
from qvelt.eval.shard_audit import shard_documents
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest


def test_multiple_declared_audit_sources_are_ordered_and_hash_bound(
    tmp_path: Path,
) -> None:
    from qvelt.utils.manifests import parse_manifest

    fixture = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    train_path = tmp_path / "fixture/model-data/prepared/manifest.json"
    training = ArtifactRef(
        str(train_path),
        manifest_digest(ArtifactStore(train_path.parent).read_manifest()),
    )
    job = replace(
        fixture,
        protocol="qvelt-eval-v2",
        training_sources=(training, *fixture.training_sources),
        max_cases=1,
        max_new_tokens=1,
    )
    report = run_evaluation(
        job, tmp_path / "evaluation", ROOT, command=("python", "scripts/first_model.py")
    )
    receipt = export_receipt(
        ArtifactRef(
            str(tmp_path / "evaluation/manifest.json"), report["report_sha256"]
        ),
        tmp_path / "receipt.json",
        benchmark="original-en-ru-code",
    )
    verify_receipt(receipt)
    for operation in ("remove", "reorder", "wrong_role"):
        bad = json.loads(json.dumps(receipt))
        files = bad["report_manifest"]["files"]
        indexes = [
            i
            for i, record in enumerate(files)
            if record["role"] == "audit_source_manifest"
        ]
        assert len(indexes) == 2
        if operation == "remove":
            files.pop(indexes[-1])
        elif operation == "reorder":
            files[indexes[0]]["sha256"], files[indexes[1]]["sha256"] = (
                files[indexes[1]]["sha256"],
                files[indexes[0]]["sha256"],
            )
        else:
            files[indexes[0]]["role"] = "metrics"
        bad["report_sha256"] = manifest_digest(
            parse_manifest(json.dumps(bad["report_manifest"]).encode())
        )
        with pytest.raises(ValueError, match="audit-source|unique"):
            verify_receipt(bad)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ledger_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("ledger")
    fixture = create_evaluation_fixture(root / "fixture", ROOT)
    train_path = root / "fixture/model-data/prepared/manifest.json"
    training = ArtifactRef(
        str(train_path),
        manifest_digest(ArtifactStore(train_path.parent).read_manifest()),
    )
    job = replace(fixture, protocol="qvelt-eval-v2", training_sources=(training,))
    run_evaluation(
        job,
        root / "eval",
        ROOT,
        command=(
            "python",
            "scripts/evaluation_suite_smoke.py",
            "--directory",
            "runs/reproduce",
        ),
    )
    report_path = root / "eval/manifest.json"
    receipt = export_receipt(
        ArtifactRef(
            str(report_path),
            manifest_digest(ArtifactStore(report_path.parent).read_manifest()),
        ),
        root / "receipt.json",
        benchmark="original-en-ru-code",
    )
    return root, receipt


def test_ledger_binds_metrics_and_provenance(
    ledger_case: tuple[Path, dict[str, Any]],
) -> None:
    root, receipt = ledger_case
    verify_receipt(receipt)
    for key in ("metrics", "audit", "provenance", "inputs", "command", "evaluation"):
        changed = copy.deepcopy(receipt)
        if key == "metrics":
            changed[key]["problems"] += 1
        elif key == "audit":
            changed[key]["cleanliness_claim"] = True
        elif key == "provenance":
            changed[key]["source_sha256"] = "0" * 64
        elif key == "inputs":
            changed[key]["checkpoint"]["metadata"]["optimizer_step"] += 1
        elif key == "command":
            changed[key]["argv"] = ["echo", "invented"]
        else:
            changed[key]["max_new_tokens"] += 1
        with pytest.raises(ValueError):
            verify_receipt(changed)
    result = build_results([root / "receipt.json"], root / "RESULTS.md")
    assert receipt["report_sha256"][:12] in result
    assert receipt["evaluation"]["checkpoint"]["sha256"] in result
    assert (
        build_results([root / "receipt.json"], root / "RESULTS.md", check=True)
        == result
    )
    (root / "RESULTS.md").write_text(result + "invented")
    with pytest.raises(ValueError, match="stale"):
        build_results([root / "receipt.json"], root / "RESULTS.md", check=True)


def test_release_requires_every_benchmark_and_actual_shards(
    ledger_case: tuple[Path, dict[str, Any]],
) -> None:
    _, receipt = ledger_case
    gate = validate_release_coverage([receipt], {"original-en-ru-code"})
    assert gate["artifact_coverage_passed"] and not gate["quality_promotion"]
    assert not gate["complete_training_history_verified"]
    with pytest.raises(ValueError, match="lacks benchmark"):
        validate_release_coverage([receipt], {"gsm8k"})
    with pytest.raises(ValueError, match="lacks benchmark"):
        build_results([], Path("/tmp/never-phase7-ledger.md"), release=True)
    changed = copy.deepcopy(receipt)
    changed["evaluation"]["max_new_tokens"] += 1
    assert comparison_key(receipt) != comparison_key(changed)
    changed = copy.deepcopy(receipt)
    changed["evaluation"]["checkpoint"]["sha256"] = "0" * 64
    assert comparison_key(receipt) == comparison_key(changed)


def test_readme_numeric_claim_guard(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text(
        "Python 3.12\nSee [RESULTS.md](RESULTS.md).\n```bash\ntrain --steps 10\n```\n"
    )
    check_readme(path)
    path.write_text("Our accuracy is 99 percent.\n")
    with pytest.raises(ValueError, match="numeric README"):
        check_readme(path)


def test_benchmark_hand_oracle_and_registry(tmp_path: Path) -> None:
    data = json.dumps(
        {"question": "What is two plus two?", "answer": "Two pairs.\n#### 4"}
    ).encode()
    rows = benchmark_rows("gsm8k", data, 1)
    assert (
        rows[0].target == "4"
        and rows[0].prompt == "Question: What is two plus two?\nAnswer:"
    )
    for gold in ("none", "#### NaN", "#### 1 #### 2"):
        with pytest.raises(ValueError):
            benchmark_rows(
                "gsm8k", json.dumps({"question": "Q", "answer": gold}).encode(), 1
            )
    for name in ("math-500", "amc-aime", "hellaswag", "humaneval"):
        with pytest.raises(ValueError, match="not approved"):
            import_benchmark(
                name,
                ArtifactRef("missing", "0" * 64),
                ArtifactRef("missing", "0" * 64),
                tmp_path / name,
            )
        assert not (tmp_path / name).exists()
    inventory = yaml.safe_load((ROOT / "configs/eval/benchmarks.yaml").read_text())
    assert inventory == {"schema_version": 1, "benchmarks": BENCHMARKS}


def test_actual_train_canary_without_document_join(
    ledger_case: tuple[Path, dict[str, Any]],
) -> None:
    root, _ = ledger_case
    fixture_path = root / "fixture/model-data/prepared/manifest.json"
    source = ArtifactRef(
        str(fixture_path),
        manifest_digest(ArtifactStore(fixture_path.parent).read_manifest()),
    )
    _, records, _ = shard_documents(source)
    documents = list(records)
    assert len(documents) >= 2
    query = EvalCase(
        "canary", "math", "audit", "unknown", "control", documents[0].text, "0"
    )
    # Use a saved job solely to supply bounded audit settings;no scoring/ingestion.
    from qvelt.eval.config import load_eval

    job = load_eval(root / "eval/evaluation.yaml")
    audit = audit_overlap([query], replace(job, training_sources=(source,)))
    assert audit["findings"]["canary"]["normalized_exact"]
    assert not audit["cleanliness_claim"]
    from qvelt.data.decontamination import normalized_tokens

    texts = [" ".join(normalized_tokens(doc.text)) for doc in documents]
    joined = [
        " ".join(left.split()[-3:] + right.split()[:3])
        for left, right in zip(texts, texts[1:], strict=False)
    ]
    crossing = next(
        query for query in joined if not any(query in text for text in texts)
    )
    boundary_case = replace(query, identifier="boundary", prompt=crossing)
    boundary_audit = audit_overlap(
        [boundary_case], replace(job, training_sources=(source,), audit_ngram=6)
    )
    assert not boundary_audit["findings"]["boundary"]["ngram"]
    assert not boundary_audit["findings"]["boundary"]["normalized_exact"]


def test_mmlu_hand_parquet_oracle() -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as parquet

    sink = pa.BufferOutputStream()
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "question": "2+2?",
                    "subject": "abstract_algebra",
                    "choices": ["1", "2", "3", "4"],
                    "answer": 3,
                }
            ]
        ),
        sink,
    )
    case = benchmark_rows("mmlu-abstract-algebra", sink.getvalue().to_pybytes(), 1)[0]
    assert case.target == " D" and case.choices == (" A", " B", " C", " D")
    assert case.prompt.endswith("A. 1\nB. 2\nC. 3\nD. 4\nAnswer:")


def test_arc_hand_parquet_oracle() -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as parquet

    sink = pa.BufferOutputStream()
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "control",
                    "question": "2+2?",
                    "choices": {"text": ["three", "four"], "label": ["1", "2"]},
                    "answerKey": "2",
                }
            ]
        ),
        sink,
    )
    case = benchmark_rows("arc-challenge", sink.getvalue().to_pybytes(), 1)[0]
    assert case.target == " four" and case.choices == (" three", " four")
    assert case.prompt == "Question: 2+2?\nAnswer:"
    assert BENCHMARKS["arc-challenge"]["conversion_license"].startswith("CC-BY-SA-4.0")


def test_encoded_preference_audit_uses_train_frames_only(
    ledger_case: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    from qvelt.data.demo import PERMISSION, authored_source
    from qvelt.data.records import write_raw_bundle
    from qvelt.posttrain.preference_smoke import math_preferences
    from qvelt.posttrain.preferences import load_preferences, prepare_preferences

    root, receipt = ledger_case
    raw = write_raw_bundle(
        math_preferences(), tmp_path / "raw", authored_source("audit-pairs"), PERMISSION
    )
    token_path = root / "fixture/model-data/prepared/tokenizer/manifest.json"
    token_sha = receipt["evaluation"]["tokenizer"]["sha256"]
    prepared = prepare_preferences(
        tmp_path / "raw/manifest.json",
        manifest_digest(raw),
        token_path,
        token_sha,
        tmp_path / "prepared",
        context=256,
        approved_for_training=True,
    )
    ref = ArtifactRef(
        str(tmp_path / "prepared/manifest.json"), manifest_digest(prepared)
    )
    from qvelt.tokenizer.workflow import load_tokenizer_bundle

    tokenizer, _ = load_tokenizer_bundle(token_path, token_sha)
    _, _, pairs = load_preferences(
        Path(ref.path), ref.sha256, tokenizer.vocab_size, 256
    )
    _, rows, coverage = shard_documents(ref)
    docs = list(rows)
    train = [p for p in pairs if p.split == "train"]
    assert len(docs) == 2 * len(train) and len(train) < len(pairs)
    assert (
        coverage["document_boundaries_verified"]
        and coverage["storage_format"] == "encoded_preferences"
    )
    assert coverage["tokens"] == sum(
        len(frame) for p in train for frame in (p.chosen, p.rejected)
    )
    assert docs[0].text == tokenizer.decode(train[0].chosen[1:-1])
