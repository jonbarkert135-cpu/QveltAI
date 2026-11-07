"""Request-local CPU generation with true padded batching and private KV/RNG state."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from qvelt.config import ModelConfig
from qvelt.inference.config import (
    EngineLimits,
    GenerationOptions,
    TextRequest,
    TokenRequest,
)
from qvelt.model.cache import KVCache, parameter_stamp
from qvelt.model.decoder import Decoder
from qvelt.tokenizer.core import QveltTokenizer

_DEFAULT_LIMITS = EngineLimits()


def cache_bytes(config: ModelConfig, batch: int, length: int) -> int:
    """FP32 K/V plus int64 position/document and bool validity logical payload."""
    if (
        type(batch) is not int
        or not 1 <= batch <= 8
        or type(length) is not int
        or not 1 <= length <= config.context
    ):
        raise ValueError("Invalid bounded cache dimensions")
    width = config.kv_heads * (config.hidden // config.heads)
    return batch * length * (8 * config.layers * width + 17)


@torch.no_grad()
def sampling_logits(
    logits: Tensor,
    options: GenerationOptions,
    *,
    history: Sequence[int] = (),
    generated: Sequence[int] = (),
) -> Tensor:
    """Penalty -> temperature -> top-k -> nucleus -> min-p, always keep a maximum.

    Repetition uses prompt+completion;frequency/presence count completion only.
    Top-k retains threshold ties;top-p retains the first token crossing the mass.
    Greedy applies penalties and ignores probabilistic filters/temperature.
    """
    if (
        logits.ndim != 1
        or logits.numel() == 0
        or logits.device.type != "cpu"
        or not bool(torch.isfinite(logits).all())
    ):
        raise FloatingPointError(
            "Sampler requires finite nonempty CPU vocabulary logits"
        )
    scores = logits.float().clone()
    for values in (history, generated):
        if len(values) > 4096 or any(
            type(t) is not int or not 0 <= t < scores.numel() for t in values
        ):
            raise ValueError("Sampler history is outside vocabulary/capacity")
    if options.repetition_penalty != 1 and history:
        ids = torch.tensor(sorted(set(history)), dtype=torch.int64)
        selected = scores[ids]
        scores[ids] = torch.where(
            selected < 0,
            selected * options.repetition_penalty,
            selected / options.repetition_penalty,
        )
    if generated and (options.frequency_penalty or options.presence_penalty):
        counts = torch.bincount(torch.tensor(generated), minlength=scores.numel())
        scores -= options.frequency_penalty * counts + options.presence_penalty * (
            counts > 0
        )
    if options.strategy == "greedy":
        return scores
    scores /= options.temperature
    if options.top_k:
        threshold = torch.topk(scores, min(options.top_k, scores.numel())).values[-1]
        scores.masked_fill_(scores < threshold, -torch.inf)
    if options.top_p < 1:
        ordered, indices = scores.sort(descending=True, stable=True)
        remove = ordered.softmax(-1).cumsum(-1) > options.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[indices[remove]] = -torch.inf
    if options.min_p:
        scores.masked_fill_(scores < scores.max() + math.log(options.min_p), -torch.inf)
    return scores


@torch.no_grad()
def sample_token(
    logits: Tensor,
    options: GenerationOptions,
    generator: torch.Generator,
    *,
    history: Sequence[int] = (),
    generated: Sequence[int] = (),
) -> int:
    scores = sampling_logits(logits, options, history=history, generated=generated)
    if options.strategy == "greedy":
        return int(scores.argmax())
    probabilities = torch.softmax(scores, dim=-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("Nonfinite categorical sampling distribution")
    return int(torch.multinomial(probabilities, 1, generator=generator))


def display_text(text: str, stops: tuple[str, ...], *, final: bool) -> tuple[str, bool]:
    found = [text.find(s) for s in stops if s in text]
    if found:
        return text[: min(found)], True
    if final:
        return text, False
    pending = max(
        (
            n
            for s in stops
            for n in range(1, min(len(s), len(text) + 1))
            if text.endswith(s[:n])
        ),
        default=0,
    )
    return text[: len(text) - pending] if pending else text, False


@dataclass(frozen=True)
class GenerationEvent:
    identifier: str
    token_id: int | None
    generated_tokens: int
    text: str
    done: bool
    reason: str | None
    cache_bytes: int


@dataclass(frozen=True)
class GenerationResult:
    identifier: str
    token_ids: tuple[int, ...]
    text: str
    reason: str
    prompt_tokens: int
    cache_bytes_at_stop: int


@dataclass
class _State:
    request: TokenRequest
    rng: torch.Generator
    actions: list[int] = field(default_factory=list)
    content: list[int] = field(default_factory=list)
    text: str = ""
    done: bool = False


class GenerationEngine:
    """Read-only model;streams own their cache and never expose cache tensors."""

    def __init__(
        self,
        model: Decoder,
        tokenizer: QveltTokenizer,
        limits: EngineLimits = _DEFAULT_LIMITS,
    ) -> None:
        if (
            not isinstance(model, Decoder)
            or not isinstance(tokenizer, QveltTokenizer)
            or not isinstance(limits, EngineLimits)
        ):
            raise ValueError("Engine requires validated decoder/tokenizer/limits")
        if (
            model.training
            or model.config.vocab != tokenizer.vocab_size
            or model.config.parameter_count > 20_000_000
            or any(
                p.device.type != "cpu"
                or (
                    p.dtype != torch.float32
                    and not (p.dtype == torch.uint8 and not p.requires_grad)
                )
                for p in model.parameters()
            )
        ):
            raise ValueError(
                "Engine requires bounded CPU FP32 eval model and matching tokenizer"
            )
        self.model, self.tokenizer, self.limits = model, tokenizer, limits
        self.stamp = parameter_stamp(model)

    def _check_model(self) -> None:
        if self.model.training or self.stamp != parameter_stamp(self.model):
            raise ValueError(
                "Generation model mode/weights changed;create a new engine"
            )

    def _token_batch(
        self, requests: Sequence[TokenRequest]
    ) -> tuple[TokenRequest, ...]:
        if (
            not isinstance(requests, Sequence)
            or not 1 <= len(requests) <= self.limits.max_batch
            or not all(isinstance(r, TokenRequest) for r in requests)
            or len({r.identifier for r in requests}) != len(requests)
        ):
            raise ValueError("Generation batch has invalid size,requests or IDs")
        return tuple(requests)

    def encode_requests(
        self, requests: Sequence[TextRequest], prompt_format: str = "chat-v1"
    ) -> tuple[TokenRequest, ...]:
        if prompt_format not in ("chat-v1", "plain-v1"):
            raise ValueError("Unknown generation prompt protocol")
        if (
            not isinstance(requests, Sequence)
            or not 1 <= len(requests) <= self.limits.max_batch
        ):
            raise ValueError("Text request batch exceeds capacity")
        encoded = []
        for request in requests:
            if not isinstance(request, TextRequest):
                raise ValueError("Expected validated text request")
            ids = self.tokenizer.encode(request.prompt)
            if any(t < 32 for t in ids):
                raise ValueError("Ordinary text unexpectedly encoded a role/control ID")
            prefix = [1, 4, *ids, 6, 5] if prompt_format == "chat-v1" else [1, *ids]
            encoded.append(
                TokenRequest(request.identifier, tuple(prefix), request.options)
            )
        return tuple(encoded)

    @torch.no_grad()
    def _forward(
        self,
        ids: Tensor,
        positions: Tensor,
        valid: Tensor,
        cache: KVCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, KVCache | None]:
        self._check_model()
        with torch.autocast("cpu", enabled=False):
            if use_cache:
                logits, cache = self.model.forward_cached(
                    ids,
                    cache,
                    positions=positions,
                    valid_mask=valid,
                    last_logits_only=True,
                )
            else:
                logits = self.model(ids, positions=positions, valid_mask=valid)
        return logits[:, -1].float(), cache

    def check_batch(
        self, requests: Sequence[TokenRequest], *, use_cache: bool = True
    ) -> None:
        if type(use_cache) is not bool:
            raise ValueError("Generation cache policy must be boolean")
        requests = self._token_batch(requests)
        self._check_model()
        batch = len(requests)
        prefix = max(len(r.tokens) for r in requests)
        total = prefix + max(r.options.max_new_tokens for r in requests)
        if (
            total > min(self.model.config.context, self.limits.max_context)
            or batch * total > self.limits.max_total_tokens
        ):
            raise ValueError("Padded generation batch exceeds context/token capacity")
        reserved = cache_bytes(self.model.config, batch, total) if use_cache else 0
        for request in requests:
            if (
                total > request.options.context_limit
                or any(
                    t >= self.model.config.vocab
                    for t in (*request.tokens, *request.options.stop_tokens)
                )
                or use_cache
                and reserved // batch > request.options.max_cache_bytes
            ):
                raise ValueError(
                    "Request exceeds context/vocabulary/cache capacity in this padded batch"
                )
        if reserved > self.limits.max_batch_cache_bytes:
            raise ValueError("Batch exceeds logical KV cache capacity")

    def stream(
        self,
        requests: Sequence[TokenRequest],
        *,
        use_cache: bool = True,
        cancelled: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> Generator[GenerationEvent, None, None]:
        self.check_batch(requests, use_cache=use_cache)
        requests = self._token_batch(requests)
        batch = len(requests)
        prefix = max(len(r.tokens) for r in requests)
        states = [
            _State(r, torch.Generator(device="cpu").manual_seed(r.options.seed))
            for r in requests
        ]
        ids = torch.zeros((batch, prefix), dtype=torch.int64, device="cpu")
        positions = torch.zeros_like(ids)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for i, r in enumerate(requests):
            n = len(r.tokens)
            ids[i, -n:] = torch.tensor(r.tokens, dtype=torch.int64, device="cpu")
            positions[i, -n:] = torch.arange(n, device="cpu")
            valid[i, -n:] = True
        started = clock()
        cache: KVCache | None = None
        logical_bytes = 0

        def terminal(state: _State, reason: str) -> GenerationEvent:
            state.done = True
            raw = self.tokenizer.decode(state.content)
            state.text = display_text(
                raw, state.request.options.stop_strings, final=True
            )[0][: state.request.options.max_output_chars]
            return GenerationEvent(
                state.request.identifier,
                None,
                len(state.actions),
                state.text,
                True,
                reason,
                logical_bytes,
            )

        try:
            while not all(s.done for s in states):
                for state in states:
                    if (
                        not state.done
                        and cancelled is not None
                        and cancelled(state.request.identifier)
                    ):
                        yield terminal(state, "cancelled")
                    elif (
                        not state.done
                        and clock() - started >= state.request.options.max_seconds
                    ):
                        yield terminal(state, "timeout")
                if all(s.done for s in states):
                    break
                active = torch.tensor(
                    [not s.done for s in states], dtype=torch.bool, device="cpu"
                )
                current_valid = valid.clone()
                current_valid &= active[:, None]
                logits, cache = self._forward(
                    ids[:, -1:] if use_cache and cache is not None else ids,
                    positions[:, -1:] if use_cache and cache is not None else positions,
                    current_valid[:, -1:]
                    if use_cache and cache is not None
                    else current_valid,
                    cache,
                    use_cache,
                )
                logical_bytes = (
                    cache_bytes(self.model.config, batch, cache.length) // batch
                    if cache is not None
                    else 0
                )
                next_ids = torch.zeros((batch, 1), dtype=torch.int64, device="cpu")
                next_positions = torch.zeros_like(next_ids)
                next_valid = torch.zeros_like(next_ids, dtype=torch.bool)
                for i, state in enumerate(states):
                    if state.done:
                        continue
                    if cancelled is not None and cancelled(state.request.identifier):
                        yield terminal(state, "cancelled")
                        continue
                    if clock() - started >= state.request.options.max_seconds:
                        yield terminal(state, "timeout")
                        continue
                    self._check_model()
                    try:
                        token = sample_token(
                            logits[i],
                            state.request.options,
                            state.rng,
                            history=(*state.request.tokens, *state.actions),
                            generated=state.actions,
                        )
                    except FloatingPointError:
                        yield terminal(state, "nonfinite")
                        continue
                    state.actions.append(token)
                    options = state.request.options
                    reason = None
                    if token in options.stop_tokens:
                        reason = "stop_token"
                    else:
                        state.content.append(token)
                    raw = self.tokenizer.decode(state.content)
                    text, stopped = display_text(raw, options.stop_strings, final=False)
                    if stopped:
                        reason = "stop_string"
                    if len(raw) > options.max_output_chars:
                        reason = "output_limit"
                    if reason is None and len(state.actions) >= options.max_new_tokens:
                        reason = "max_tokens"
                    state.done = reason is not None
                    if state.done:
                        text = display_text(raw, options.stop_strings, final=True)[0]
                    state.text = text[: options.max_output_chars]
                    next_ids[i, 0] = token
                    next_positions[i, 0] = (
                        len(state.request.tokens) + len(state.actions) - 1
                    )
                    next_valid[i, 0] = not state.done
                    # No no_grad/inference_mode context remains active across yield.
                    yield GenerationEvent(
                        state.request.identifier,
                        token,
                        len(state.actions),
                        state.text,
                        state.done,
                        reason,
                        logical_bytes,
                    )
                if not all(s.done for s in states):
                    ids = torch.cat((ids, next_ids), 1)
                    positions = torch.cat((positions, next_positions), 1)
                    valid = torch.cat((valid, next_valid), 1)
        finally:
            cache = None
            states.clear()

    def generate(
        self, requests: Sequence[TokenRequest], *, use_cache: bool = True
    ) -> list[GenerationResult]:
        requests = self._token_batch(requests)
        actions: dict[str, list[int]] = {r.identifier: [] for r in requests}
        final: dict[str, GenerationEvent] = {}
        for event in self.stream(requests, use_cache=use_cache):
            if event.token_id is not None:
                actions[event.identifier].append(event.token_id)
            if event.done:
                final[event.identifier] = event
        return [
            GenerationResult(
                r.identifier,
                tuple(actions[r.identifier]),
                final[r.identifier].text,
                str(final[r.identifier].reason),
                len(r.tokens),
                final[r.identifier].cache_bytes,
            )
            for r in requests
        ]
