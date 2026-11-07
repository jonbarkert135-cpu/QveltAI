"""Reviewed benchmark protocol inventory and exact-snapshot development imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qvelt.data.records import write_raw_bundle
from qvelt.eval.config import ArtifactRef
from qvelt.eval.corpus import EvalCase, read_json
from qvelt.eval.scoring import parse_number
from qvelt.eval.smoke import case_document
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifest_types import Manifest, SourceIdentity

BENCHMARKS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "status": "approved_bounded_evaluation",
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "payload_sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "payload_bytes": 749738,
        "license_id": "MIT",
        "license_sha256": "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b",
        "license_url": "https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE",
        "metric": "strict first #### regex capture,literal normalized exact match;lm-eval0.4.13 task v3",
        "prompt": "Question: QUESTION\nAnswer:",
        "official_fewshot": 5,
        "development_fewshot": 0,
        "source_url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
    },
    "mmlu-abstract-algebra": {
        "status": "approved_bounded_evaluation",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "payload_sha256": "92338cdcdb43d2871691ef9a0e125e633ca57335ed635a635fe6dead94f224c2",
        "payload_bytes": 9964,
        "license_id": "MIT",
        "license_sha256": "5d841b9eee8bf0d721359b79fbae956a8b426d3139bfefee32149c7d4f2a1a9c",
        "license_url": "https://github.com/hendrycks/test/blob/4450500f923c49f1fb1dd3d99108a0bd9717b660/LICENSE",
        "metric": "argmax A-D continuation log likelihood;accuracy;single subject is not full MMLU",
        "prompt": "subject heading + question + A-D options + Answer:;leading-space answer letters",
        "official_fewshot": 5,
        "development_fewshot": 0,
        "source_url": "https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/abstract_algebra/test-00000-of-00001.parquet",
    },
    "math-500": {
        "status": "blocked_license",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "metric": "original MATH LaTeX/string equivalence,not the rational verifier",
        "reason": "reviewed MATH-500 card has no data license;MATH repo MIT covers loaders/eval;general LaTeX metric not implemented",
    },
    "amc-aime": {
        "status": "blocked_license",
        "metric": "contest exact final answer;AIME bounded integer answer",
        "reason": "competition problem reuse rights and exact snapshots not established;no data fetched",
    },
    "hellaswag": {
        "status": "blocked_access",
        "metric": "continuation acc and character-length-normalized acc_norm;lm-eval task v1",
        "reason": "publisher access HTTP451;no mirror bypass;prompt preprocessing read from installed harness only",
    },
    "arc-challenge": {
        "status": "approved_bounded_evaluation",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "payload_sha256": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
        "payload_bytes": 203808,
        "license_id": "CC-BY-SA-4.0",
        "license_sha256": "a0bd498de6b7991f563789715e3cf9f9a77860b6bb5b2c2b072fb3bb36361008",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "legal_text_url": "https://raw.githubusercontent.com/creativecommons/legalcode/4d2de083a6dcc5095ed1a30885bc48f868b9bb55/by-sa_4.0.txt",
        "source_url": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/test-00000-of-00001.parquet",
        "metric": "raw choice likelihood acc;acc_norm divides by choice text character length excluding the added delimiter;lm-eval taskv1",
        "prompt": "Question: QUESTION\nAnswer: + space + choice text",
        "official_fewshot": 0,
        "development_fewshot": 0,
        "attribution": "Allen Institute for AI;ARC-Challenge;Clark et al.2018;publisher card license declaration",
        "conversion_license": "CC-BY-SA-4.0;JSON/prompt formatting changes indicated;no raw text published",
    },
    "humaneval": {
        "status": "blocked_execution",
        "revision": "b9453e8f1865df0a0057a1383767caa400acfbcf",
        "license_id": "MIT",
        "metric": "execute official tests,unbiased pass@k;raw Python completion prefix",
        "reason": "publisher MIT/README verified;arbitrary Python sandbox not implemented;original bounded-expression controls are separate",
    },
    "original-en-ru-code": {
        "status": "approved_original_evaluation",
        "license_id": "Apache-2.0",
        "metric": "versioned original exact/numeric/choice/LM and bounded integer-function tests",
        "reason": "not a public benchmark;Russian original controls included;final800 untouched",
    },
    "original-needle": {
        "status": "approved_original_evaluation",
        "license_id": "Apache-2.0",
        "metric": "greedy whole-answer exact numeric retrieval by token budget and needle depth;context overflow is a failure",
        "reason": "D-INFERENCE-PHASE8-01;original diagnostic controls,not a standardized benchmark or a trained retrieval task",
    },
}


def benchmark_rows(name: str, data: bytes, count: int) -> list[EvalCase]:
    """Pure format converter; importer below separately enforces reviewed payload pins."""
    if type(count) is not int or not 1 <= count <= 64 or len(data) > 8 * 1024**2:
        raise ValueError("Benchmark development imports require1..64 cases/8MiB")
    result = []
    if name == "gsm8k":
        lines = data.splitlines()
        if len(lines) < count:
            raise ValueError("Insufficient source rows")
        for line in lines[:count]:
            row = read_json(line)
            if (
                not isinstance(row, dict)
                or set(row) != {"question", "answer"}
                or not all(isinstance(v, str) for v in row.values())
                or len(row["answer"].split("####")) != 2
            ):
                raise ValueError("Invalid original GSM8K fields")
            answer = row["answer"].split("####")[1].strip().replace(",", "")
            if parse_number(answer, "numeric") is None:
                raise ValueError("Invalid GSM8K numeric gold")
            identifier = hashlib.sha256(row["question"].encode()).hexdigest()
            result.append(
                EvalCase(
                    identifier,
                    "numeric",
                    "gsm8k-development-subset",
                    "gsm8k-original-test",
                    "validation",
                    "Question: " + row["question"] + "\nAnswer:",
                    answer,
                    language="en",
                )
            )
    elif name in ("mmlu-abstract-algebra", "arc-challenge"):
        import io

        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError(
                "Install eval-harness extra for the pinned Parquet converter;converted eval is standalone"
            ) from error
        table = parquet.read_table(io.BytesIO(data))
        if table.num_rows < count:
            raise ValueError("Insufficient MMLU rows")
        for row in table.slice(0, count).to_pylist():
            if name == "arc-challenge":
                choices = row.get("choices")
                if (
                    set(row) != {"id", "question", "choices", "answerKey"}
                    or not isinstance(row["id"], str)
                    or not isinstance(row["question"], str)
                    or not isinstance(choices, dict)
                    or set(choices) != {"text", "label"}
                    or not isinstance(choices["text"], list)
                    or not isinstance(choices["label"], list)
                    or not 2 <= len(choices["text"]) == len(choices["label"]) <= 8
                    or any(
                        not isinstance(v, str) or not v
                        for v in choices["text"] + choices["label"]
                    )
                    or len(set(choices["label"])) != len(choices["label"])
                    or row["answerKey"] not in choices["label"]
                ):
                    raise ValueError("Invalid ARC question/options/answer key")
                options = tuple(" " + choice for choice in choices["text"])
                result.append(
                    EvalCase(
                        row["id"],
                        "choice",
                        "arc-challenge-development-subset",
                        "arc-challenge-test",
                        "validation",
                        "Question: " + row["question"] + "\nAnswer:",
                        options[choices["label"].index(row["answerKey"])],
                        options,
                        language="en",
                    )
                )
                continue
            if (
                set(row) != {"question", "subject", "choices", "answer"}
                or row["subject"] != "abstract_algebra"
                or not isinstance(row["question"], str)
                or not isinstance(row["choices"], list)
                or len(row["choices"]) != 4
                or any(not isinstance(v, str) for v in row["choices"])
                or type(row["answer"]) is not int
                or not 0 <= row["answer"] < 4
            ):
                raise ValueError("Invalid MMLU subject/question/options/target")
            prompt = (
                "The following are multiple choice questions (with answers) about abstract algebra.\n\n"
                + row["question"].strip()
            )
            prompt += (
                "".join(
                    f"\n{letter}. {text}"
                    for letter, text in zip("ABCD", row["choices"], strict=True)
                )
                + "\nAnswer:"
            )
            identifier = hashlib.sha256(row["question"].encode()).hexdigest()
            options = tuple(" " + c for c in "ABCD")
            result.append(
                EvalCase(
                    identifier,
                    "choice",
                    "mmlu-abstract-algebra-development-subset",
                    "mmlu-abstract-algebra-test",
                    "validation",
                    prompt,
                    options[row["answer"]],
                    options,
                    language="en",
                )
            )
    else:
        raise ValueError("No approved implemented source adapter for " + name)
    if len({c.identifier for c in result}) != len(result):
        raise ValueError("Duplicate benchmark questions")
    return result


def import_benchmark(
    name: str,
    payload: ArtifactRef,
    license_file: ArtifactRef,
    directory: Path,
    *,
    count: int = 16,
) -> Manifest:
    entry = BENCHMARKS.get(name)
    if entry is None or entry["status"] != "approved_bounded_evaluation":
        raise ValueError("Benchmark is not approved: " + name)
    values = []
    for ref, expected, maximum in (
        (payload, entry["payload_sha256"], entry["payload_bytes"]),
        (license_file, entry["license_sha256"], 32768),
    ):
        path = Path(ref.path)
        with ArtifactStore(path.parent).open_file(path.name) as stream:
            data = stream.read(maximum + 1)
        if (
            ref.sha256 != expected
            or len(data) > maximum
            or hashlib.sha256(data).hexdigest() != expected
        ):
            raise ValueError(
                "Benchmark source/license bytes do not match reviewed snapshot"
            )
        values.append(data)
    cases = benchmark_rows(name, values[0], count)
    source = SourceIdentity(
        name + "-evaluation-only",
        entry["revision"],
        entry["license_id"],
        entry["license_url"],
    )
    report = {
        "protocol": "qvelt-benchmark-import-v1",
        "name": name,
        "entry": entry,
        "imported": len(cases),
        "official_full_result": False,
        "deviations": [
            "first-N development subset",
            *(
                ["zero-shot instead of five-shot"]
                if entry["official_fewshot"] == 5
                else []
            ),
            "model context/generation limits declared separately",
        ],
    }
    return write_raw_bundle(
        [case_document(c) for c in cases],
        directory,
        source,
        values[1],
        allowed_uses=("evaluation", "decontamination"),
        evidence=(
            (
                "benchmark.json",
                (json.dumps(report, sort_keys=True) + "\n").encode(),
                "benchmark_protocol",
            ),
        ),
    )
