"""Lossless UTF-8 byte-BPE baseline with explicit control-token boundaries.

Training consumes ordered Unicode documents. This module does not approve a
corpus, normalize text, add BOS/EOS implicitly, or interpret user text as roles.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from qvelt.artifacts import atomic_write

CONTROL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_turn|>",
    "<|tool|>",
) + tuple(f"<|reserved_{index:02d}|>" for index in range(24))
MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_TOKENIZER_BYTES = 64 * 1024 * 1024


class TokenizerError(ValueError):
    """Text, token IDs, training policy or tokenizer state is invalid."""


@dataclass(frozen=True)
class TokenizerSettings:
    """Requested vocabulary ceiling; a small corpus may produce fewer merges."""

    vocab_size: int = 32768
    min_frequency: int = 2
    isolate_digits: bool = True

    def __post_init__(self) -> None:
        if type(self.vocab_size) is not int or not 288 <= self.vocab_size <= 262144:
            raise TokenizerError("vocab_size must be an integer in [288, 262144]")
        if type(self.min_frequency) is not int or not 1 <= self.min_frequency < 2**32:
            raise TokenizerError("min_frequency must be an integer in [1, 2**32-1]")
        if type(self.isolate_digits) is not bool:
            raise TokenizerError("isolate_digits must be bool")


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TokenizerError("Expected Unicode text")
    try:
        size = len(text.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise TokenizerError("Lone Unicode surrogates are not valid UTF-8") from error
    if size > MAX_DOCUMENT_BYTES:
        raise TokenizerError("Document exceeds the 1 MiB UTF-8 limit")


def _backend(isolate_digits: bool) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.pre_tokenizer = (
        pre_tokenizers.Sequence(
            [pre_tokenizers.Split(Regex("[0-9]"), behavior="isolated"), byte_level]
        )
        if isolate_digits
        else byte_level
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.encode_special_tokens = True
    return tokenizer


class QveltTokenizer:
    """Small public interface; explicit controls cannot enter via plain text.

    Byte-level coverage guarantees whole-text round trips for valid UTF-8.
    Arbitrarily sliced byte-token sequences may decode with replacement characters.
    """

    def __init__(self, backend: Tokenizer) -> None:
        for index, token in enumerate(CONTROL_TOKENS):
            if backend.token_to_id(token) != index:
                raise TokenizerError("Control-token IDs do not match Qvelt schema 1")
        backend.encode_special_tokens = True
        self._backend = backend

    @property
    def vocab_size(self) -> int:
        """Actual vocabulary including the 32 controls and all 256 bytes."""
        return int(self._backend.get_vocab_size())

    def encode(self, text: str) -> list[int]:
        """Encode ordinary text without implicit control tokens or truncation."""
        _validate_text(text)
        return list(self._backend.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        """Preserve explicit controls; reject bool, negative and unknown IDs."""
        checked = list(ids)
        if any(type(i) is not int or not 0 <= i < self.vocab_size for i in checked):
            raise TokenizerError("Token IDs must be integers within the vocabulary")
        return str(self._backend.decode(checked, skip_special_tokens=False))

    def to_bytes(self) -> bytes:
        """Serialize complete backend state; not a dataset or run manifest."""
        return str(self._backend.to_str()).encode("utf-8")

    def save(self, path: Path) -> None:
        """Commit one local JSON file atomically without overwriting."""
        data = self.to_bytes()
        from typing import BinaryIO

        def write(stream: BinaryIO) -> None:
            stream.write(data)

        atomic_write(path, write)

    @classmethod
    def load(cls, path: Path) -> QveltTokenizer:
        """Load bounded local JSON; never download or execute tokenizer code."""
        with path.open("rb") as stream:
            data = stream.read(MAX_TOKENIZER_BYTES + 1)
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> QveltTokenizer:
        """Validate a bounded JSON snapshot, including its complete byte alphabet."""
        if len(data) > MAX_TOKENIZER_BYTES:
            raise TokenizerError("Tokenizer exceeds the 64 MiB limit")
        try:
            backend = Tokenizer.from_str(data.decode("utf-8"))
        except Exception as error:
            raise TokenizerError("Invalid tokenizer JSON") from error
        # Validate policy by comparing complete non-vocabulary state to a known
        # construction. Learned merges/vocab are data; arbitrary processors are not.
        import json

        raw = json.loads(backend.to_str())
        if raw["model"]["type"] != "BPE":
            raise TokenizerError("Only byte-BPE artifacts are supported")
        actual = dict(raw)
        actual["model"] = {**raw["model"], "vocab": {}, "merges": []}
        matched = False
        for digits in (False, True):
            expected = json.loads(_backend(digits).to_str())
            expected["added_tokens"] = raw["added_tokens"]
            if actual == expected:
                matched = True
        if not matched:
            raise TokenizerError("Unsupported normalization, padding or model policy")
        controls = raw["added_tokens"]
        if controls != [
            {
                "id": i,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i, token in enumerate(CONTROL_TOKENS)
        ]:
            raise TokenizerError("Exactly 32 special controls are required")
        vocab = backend.get_vocab()
        if set(vocab.values()) != set(range(len(vocab))):
            raise TokenizerError("Vocabulary IDs must be contiguous")
        if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(vocab):
            raise TokenizerError("Incomplete byte alphabet")
        return cls(backend)


def train_tokenizer(
    documents: Iterable[str], settings: TokenizerSettings | None = None
) -> QveltTokenizer:
    """Fit deterministic no-dropout BPE on the supplied document order.

    Memory for the trainer's pair statistics still scales with corpus diversity;
    an iterator and document limit do not bound total training memory.
    """
    settings = settings or TokenizerSettings()
    backend = _backend(settings.isolate_digits)
    count = 0

    def checked() -> Iterable[str]:
        nonlocal count
        for document in documents:
            _validate_text(document)
            # Control spellings are ordinary input, never role annotations.
            if any(token in document for token in CONTROL_TOKENS):
                raise TokenizerError(
                    "Training documents must not contain control spellings"
                )
            count += bool(document)
            yield document

    trainer = trainers.BpeTrainer(
        vocab_size=settings.vocab_size,
        min_frequency=settings.min_frequency,
        special_tokens=list(CONTROL_TOKENS),
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        show_progress=False,
    )
    backend.train_from_iterator(checked(), trainer=trainer)
    if not count:
        raise TokenizerError("At least one nonempty training document is required")
    return QveltTokenizer(backend)
