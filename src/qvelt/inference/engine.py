"""Common native generation contract consumed by serving and own-policy rollouts."""

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

from qvelt.inference.config import EngineLimits, TokenRequest
from qvelt.inference.generation import GenerationEvent, GenerationResult
from qvelt.tokenizer.core import QveltTokenizer


class Engine(Protocol):
    tokenizer: QveltTokenizer
    limits: EngineLimits

    def check_batch(
        self, requests: Sequence[TokenRequest], *, use_cache: bool = True
    ) -> None: ...

    def stream(
        self,
        requests: Sequence[TokenRequest],
        *,
        use_cache: bool = True,
        cancelled: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = ...,
    ) -> Iterator[GenerationEvent]: ...

    def generate(
        self, requests: Sequence[TokenRequest], *, use_cache: bool = True
    ) -> list[GenerationResult]: ...
