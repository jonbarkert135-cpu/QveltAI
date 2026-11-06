"""Thin optional lm-eval0.4.13 LM bridge; no external models/services/datasets."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from qvelt.eval.config import EvalJob
from qvelt.eval.corpus import load_cases
from qvelt.eval.runner import load_model, score_choice
from qvelt.inference.config import GenerationOptions, TokenRequest
from qvelt.inference.generation import GenerationEngine
from qvelt.model.decoder import Decoder
from qvelt.tokenizer.core import QveltTokenizer, _validate_text
from qvelt.tokenizer.workflow import load_tokenizer_bundle


class HarnessBackend:
    """Single CPU FP32 plain-text backend with explicit continuation boundaries."""

    def __init__(
        self,
        model: Decoder,
        tokenizer: QveltTokenizer,
        *,
        max_length: int = 256,
        max_tokens: int = 32,
        seed: int = 20261106,
    ) -> None:
        if (
            type(max_length) is not int
            or not 4 <= max_length <= min(model.config.context, 2048)
            or type(max_tokens) is not int
            or not 1 <= max_tokens <= 128
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise ValueError("Invalid bounded harness policy")
        self.engine = GenerationEngine(model, tokenizer)
        self.model, self.tokenizer = model, tokenizer
        self.max_length, self.max_tokens, self.seed = max_length, max_tokens, seed

    @torch.no_grad()
    def _tokens(self, prefix: list[int], targets: list[int]) -> tuple[float, bool]:
        self.engine._check_model()
        if not targets:
            return 0.0, True
        if not prefix or len(prefix) + len(targets) - 1 > self.max_length:
            raise ValueError(
                "Harness likelihood exceeds context; silent truncation is disabled"
            )
        with torch.autocast("cpu", enabled=False):
            logits = self.model(
                torch.tensor([prefix + targets[:-1]], dtype=torch.int64)
            )[0, len(prefix) - 1 :].float()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite harness logits")
            labels = torch.tensor(targets, dtype=torch.int64)
            logps = logits.log_softmax(-1).gather(-1, labels[:, None]).sum()
        return float(logps), bool((logits.argmax(-1) == labels).all())

    def loglikelihood(self, pairs: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        results = []
        for context, continuation in pairs:
            _validate_text(context)
            _validate_text(continuation)
            if len(context) + len(continuation) > 32768:
                raise ValueError("Harness likelihood text exceeds capacity")
            spaces = len(context) - len(context.rstrip())
            if spaces:
                continuation, context = (
                    context[-spaces:] + continuation,
                    context[:-spaces],
                )
            prefix = self.tokenizer.encode(context)
            combined = self.tokenizer.encode(context + continuation)
            # Never silently assign probabilities across a retokenized prefix.
            if combined[: len(prefix)] != prefix:
                raise ValueError(
                    "Continuation retokenizes context; use an explicit word boundary"
                )
            results.append(self._tokens([1, *prefix], combined[len(prefix) :]))
        return results

    def loglikelihood_rolling(self, texts: list[str]) -> list[float]:
        from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

        result = []
        for text in texts:
            _validate_text(text)
            if len(text) > 32768:
                raise ValueError("Rolling text exceeds bounded capacity")
            tokens = self.tokenizer.encode(text)
            total = 0.0
            for window in get_rolling_token_windows(
                tokens, prefix_token=1, max_seq_len=self.max_length, context_len=1
            ):
                prefix, targets = make_disjoint_window(window)
                total += self._tokens(prefix, targets)[0]
            result.append(total)
        return result

    def generate_until(self, requests: list[tuple[str, dict[str, Any]]]) -> list[str]:
        results = []
        for index, (text, kwargs) in enumerate(requests):
            if set(kwargs) - {"until", "max_gen_toks", "do_sample", "temperature"}:
                raise ValueError("Unsupported harness generation setting")
            _validate_text(text)
            until = kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            if not isinstance(until, list) or any(
                not isinstance(x, str) for x in until
            ):
                raise ValueError("Harness until requires strings")
            sample = kwargs.get("do_sample", False)
            if type(sample) is not bool:
                raise ValueError("Harness do_sample must be bool")
            temperature = kwargs.get("temperature", 1.0 if sample else 0.0)
            if not sample and temperature != 0:
                raise ValueError("Nonzero temperature requires explicit sampling")
            options = GenerationOptions(
                max_new_tokens=kwargs.get("max_gen_toks", self.max_tokens),
                strategy="sample" if sample else "greedy",
                temperature=temperature,
                stop_tokens=(2,),
                stop_strings=tuple(until),
                context_limit=self.max_length,
                seed=(self.seed + index) % 2**32,
                max_seconds=300,
            )
            row = self.engine.generate(
                (TokenRequest("harness", (1, *self.tokenizer.encode(text)), options),)
            )[0]
            if row.reason in ("nonfinite", "timeout", "cancelled", "output_limit"):
                raise RuntimeError("Harness generation failed: " + row.reason)
            results.append(row.text)
        return results


def make_lm_adapter(backend: HarnessBackend) -> Any:
    """Return a real installed LM subclass; missing optional dependency is an error."""
    try:
        from lm_eval.api.model import LM
    except ImportError as error:
        raise RuntimeError(
            "Install the locked eval-harness extra to use lm-eval"
        ) from error

    class QveltLM(LM):  # type: ignore[misc]  # Optional upstream package has no typing marker.
        def __init__(self) -> None:
            super().__init__()
            self._device = torch.device("cpu")

        @property
        def tokenizer_name(self) -> str:
            return (
                "qvelt-byte-bpe-"
                + hashlib.sha256(backend.tokenizer.to_bytes()).hexdigest()
            )

        def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
            return backend.loglikelihood([r.args for r in requests])

        def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
            return backend.loglikelihood_rolling([r.args[0] for r in requests])

        def generate_until(self, requests: list[Any]) -> list[str]:
            return backend.generate_until([r.args for r in requests])

    return QveltLM()


def run_harness_parity(job: EvalJob) -> dict[str, Any]:
    """Execute actual harness task/aggregation on approved local choice records."""
    from datasets import Dataset, DatasetDict
    from lm_eval import evaluator
    from lm_eval.api.task import ConfigurableTask

    if job.prompt_format != "plain-v1":
        raise ValueError(
            "Harness parity uses explicit plain prompts,not chat-template text"
        )
    _, cases, _ = load_cases(job)
    choices = [case for case in cases if case.kind == "choice"]
    if not choices:
        raise ValueError("Harness local parity requires choice cases")
    tokenizer, _ = load_tokenizer_bundle(Path(job.tokenizer.path), job.tokenizer.sha256)
    model, _, _ = load_model(job)
    backend = HarnessBackend(
        model,
        tokenizer,
        max_length=min(job.context_limit, model.config.context),
        max_tokens=job.max_new_tokens,
        seed=job.seed,
    )
    exclude_delimiter = job.choice_normalization == "character-without-delimiter"
    if exclude_delimiter and any(
        not c.startswith(" ") for case in choices for c in case.choices
    ):
        raise ValueError("Delimiter-excluded task requires stored leading spaces")
    records = [
        {
            "prompt": case.prompt,
            "choices": [c[1:] if exclude_delimiter else c for c in case.choices],
            "answer": case.choices.index(case.target),
        }
        for case in choices
    ]

    class LocalChoices(ConfigurableTask):  # type: ignore[misc]  # Same untyped upstream boundary.
        def download(self, dataset_kwargs: dict[str, Any] | None = None) -> None:
            self.dataset = DatasetDict({"test": Dataset.from_list(records)})

    task = LocalChoices(
        config={
            "task": "qvelt_local_choice_v1",
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": "{{prompt}}",
            "doc_to_choice": "{{choices}}",
            "doc_to_target": "{{answer}}",
            "target_delimiter": " " if exclude_delimiter else "",
            "num_fewshot": 0,
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
            ],
        }
    )
    with torch.random.fork_rng(devices=[]):
        result = evaluator.evaluate(
            make_lm_adapter(backend),
            {"qvelt_local_choice_v1": task},
            bootstrap_iters=0,
            log_samples=True,
        )
    if result is None:
        raise RuntimeError("Harness returned no result")
    measured = result["results"]["qvelt_local_choice_v1"]
    expected = [
        score_choice(
            model,
            tokenizer,
            case,
            replace(
                job,
                choice_normalization="character-without-delimiter"
                if exclude_delimiter
                else "character",
            ),
        )
        for case in choices
    ]
    accuracy = sum(row["correct"] for row in expected) / len(expected)
    normalized = sum(row["normalized_correct"] for row in expected) / len(expected)
    if not math.isclose(
        measured["acc,none"], accuracy, abs_tol=1e-12
    ) or not math.isclose(measured["acc_norm,none"], normalized, abs_tol=1e-12):
        raise AssertionError("Actual harness and Qvelt local-choice metrics disagree")
    return {
        "protocol": "qvelt-lm-eval-0.4.13-local-parity",
        "cases": len(choices),
        "accuracy": accuracy,
        "character_normalized_accuracy": normalized,
        "parity": True,
        "scope": "original local tasks,not public benchmark accuracy;no remote dataset/model/service",
    }
