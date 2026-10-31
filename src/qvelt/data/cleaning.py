"""Seeded offline LID, explicit quality heuristics and conservative PII patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langdetect.detector_factory import PROFILES_DIRECTORY, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from qvelt.tokenizer.core import CONTROL_TOKENS, _validate_text


@dataclass(frozen=True)
class CleaningConfig:
    min_characters: int = 10
    max_characters: int = 100000
    max_repeated_line_fraction: float = 0.5
    max_tokens: int = 100000
    toxic_phrases: tuple[str, ...] = ("kill yourself", "убей себя")

    def __post_init__(self) -> None:
        if (
            any(
                type(v) is not int or v < 1
                for v in [self.min_characters, self.max_characters, self.max_tokens]
            )
            or self.min_characters > self.max_characters
        ):
            raise ValueError("Invalid cleaning size limits")
        if (
            type(self.max_repeated_line_fraction) not in (int, float)
            or not 0 <= self.max_repeated_line_fraction <= 1
        ):
            raise ValueError("Invalid repeated-line threshold")
        if not isinstance(self.toxic_phrases, tuple | list) or any(
            not isinstance(p, str) or not 1 <= len(p) <= 128 for p in self.toxic_phrases
        ):
            raise ValueError("Toxic phrases must be bounded explicit strings")
        object.__setattr__(self, "toxic_phrases", tuple(self.toxic_phrases))


@dataclass(frozen=True)
class CleanedDocument:
    text: str
    language: str
    language_probability: float
    heuristic_score: float
    rejection: str | None
    redactions: dict[str, int] = field(default_factory=dict)


def words(text: str, *, casefold: bool = False) -> list[str]:
    """Math-aware tokens preserve numbers, signs and operators in signatures."""
    return re.findall(
        r"\w+|[^\w\s]", text.casefold() if casefold else text, flags=re.UNICODE
    )


class Cleaner:
    """Heuristic decisions are auditable, not a learned quality/toxicity oracle."""

    def __init__(self, config: CleaningConfig, seed: int) -> None:
        self.config = config
        self.factory = DetectorFactory()
        self.factory.load_json_profile(
            [
                p.read_text()
                for p in sorted(Path(PROFILES_DIRECTORY).iterdir())
                if p.is_file()
            ]
        )
        self.factory.set_seed(seed)
        self.patterns = [
            ("email", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
            (
                "private_key",
                re.compile(
                    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                    re.S,
                ),
            ),
            (
                "api_key",
                re.compile(
                    r"\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,255}|sk-[A-Za-z0-9_-]{20,255})\b"
                ),
            ),
            (
                "key_assignment",
                re.compile(
                    r"(?i)\b(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[\x22\x27]?[A-Za-z0-9_/-]{16,255}[\x22\x27]?"
                ),
            ),
            (
                "phone",
                re.compile(
                    r"(?i)(?:\b(?:phone|tel|телефон)\s*[:=]\s*\+?[0-9][0-9 ().-]{6,24}[0-9]|\+[0-9]{1,3}[ .-](?:[0-9 ()-]{6,22}[0-9]))"
                ),
            ),
        ]

    def clean(self, text: str) -> CleanedDocument:
        _validate_text(text)
        counts = {}
        for name, pattern in self.patterns:
            text, n = pattern.subn("[REDACTED_" + name.upper() + "]", text)
            counts[name] = n
        reason = None
        if (
            not text.strip()
            or not self.config.min_characters <= len(text) <= self.config.max_characters
        ):
            reason = "character_limit"
        elif any(control in text for control in CONTROL_TOKENS):
            reason = "control_literal"
        elif len(words(text)) > self.config.max_tokens:
            reason = "token_resource_limit"
        elif any(
            re.search(r"(?<!\w)" + re.escape(p.casefold()) + r"(?!\w)", text.casefold())
            for p in self.config.toxic_phrases
        ):
            reason = "toxic_phrase_quarantine"
        lines = [line for line in text.splitlines() if line.strip()]
        repeated = 1 - len(set(lines)) / len(lines) if lines else 0.0
        if reason is None and repeated > self.config.max_repeated_line_fraction:
            reason = "repeated_lines"
        language = "unknown"
        probability = 0.0
        if sum(c.isalpha() for c in text) >= 10:
            try:
                detector = self.factory.create()
                detector.append(re.sub(r"\[REDACTED_[A-Z_]+\]", "", text)[:10000])
                values = detector.get_probabilities()
                if values:
                    language = values[0].lang
                    probability = float(values[0].prob)
            except LangDetectException:
                pass
        score = (1 - repeated) * min(
            1.0, sum(c.isalnum() for c in text) / max(1, len(text)) / 0.3
        )
        return CleanedDocument(text, language, probability, score, reason, counts)
