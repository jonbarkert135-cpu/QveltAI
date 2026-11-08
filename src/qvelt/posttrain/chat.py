"""Versioned explicit-role chat frames and shifted assistant-only target masks."""

from __future__ import annotations

import json
from dataclasses import dataclass

from qvelt.tokenizer.core import QveltTokenizer, _validate_text
from qvelt.utils.manifests import _reject_constant, _unique_mapping


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    reasoning: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("Unsupported chat role")
        _validate_text(self.content)
        if not self.content.strip():
            raise ValueError("Empty chat message")
        if self.reasoning is not None:
            _validate_text(self.reasoning)
            if self.role != "assistant" or not self.reasoning.strip():
                raise ValueError("Only assistants may have nonempty reasoning")


def parse_messages(text: str) -> tuple[Message, ...]:
    _validate_text(text)
    raw = json.loads(
        text, object_pairs_hook=_unique_mapping, parse_constant=_reject_constant
    )
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "messages"}:
        raise ValueError("Expected versioned chat document")
    if type(raw["schema_version"]) is not int or raw["schema_version"] not in (1, 2):
        raise ValueError("Unsupported chat schema")
    rows = raw["messages"]
    if not isinstance(rows, list) or not 2 <= len(rows) <= 128:
        raise ValueError("Invalid chat message count")
    result = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not {"role", "content"} <= set(row)
            or set(row) - {"role", "content", "reasoning"}
            or raw["schema_version"] == 1
            and "reasoning" in row
        ):
            raise ValueError("Invalid chat message fields")
        result.append(Message(**row))
    return tuple(result)


def assistant_mask(tokens: list[int], *, control_version: int = 2) -> list[bool]:
    """Return one flag per next-token pair; outer EOS never supplies supervision.

    Role IDs 3/4/5 and end-turn 6 are shared by tokenizer control versions 1/2.
    A frame may end in a partial assistant response followed by masked outer EOS.
    """
    if (
        type(control_version) is not int
        or control_version not in (1, 2)
        or len(tokens) < 4
        or tokens[0] != 1
        or tokens[-1] != 2
        or any(type(t) is not int or t < 0 for t in tokens)
    ):
        raise ValueError("Invalid chat frame")
    role = 0
    next_role = 4
    content = 0
    thinking = False
    thought = False
    flags = [False]
    for index, token in enumerate(tokens[1:-1], 1):
        if token in (3, 4, 5):
            if role or not (token == next_role or index == 1 and token == 3):
                raise ValueError("Invalid chat role order")
            role, content = token, 0
            thinking = thought = False
            flags.append(False)
        elif token == 6:
            if not role or not content or thinking:
                raise ValueError("Empty or unframed chat turn")
            flags.append(role == 5)
            next_role = 5 if role == 4 else 4
            role = 0
        elif token in (8, 9):
            if role != 5 or control_version != 2:
                raise ValueError("Reasoning controls require assistant role")
            if token == 8:
                if content or thinking or thought:
                    raise ValueError("Reasoning must be a single initial block")
                thinking = True
            else:
                if not thinking or not content:
                    raise ValueError("Reasoning closing control has no content")
                thinking, thought, content = False, True, 0
            flags.append(True)
        elif token >= 32:
            if not role:
                raise ValueError("Unframed chat content")
            flags.append(role == 5)
            content += 1
        else:
            raise ValueError("Unsupported control in chat content")
    if role and (role != 5 or content == 0) or not role and next_role != 4:
        raise ValueError("Chat must end in an assistant response")
    flags.append(False)
    if not any(flags):
        raise ValueError("Chat has no assistant targets")
    return flags[1:]


def encode_chat(
    messages: tuple[Message, ...], tokenizer: QveltTokenizer, max_tokens: int
) -> tuple[list[int], bool]:
    """Right truncate, retain masked outer EOS, and never synthesize a turn ending."""
    if tokenizer.control_version == 3:
        raise ValueError("Tokenizer v3 requires the unimplemented chat schema3")
    if type(max_tokens) is not int or max_tokens < 4 or not 2 <= len(messages) <= 128:
        raise ValueError("Invalid chat dimensions")
    roles = {"system": 3, "user": 4, "assistant": 5}
    tokens = [1]
    for message in messages:
        tokens.append(roles[message.role])
        if message.reasoning is not None:
            if tokenizer.control_version != 2:
                raise ValueError("Explicit reasoning requires tokenizer version2")
            tokens.extend([8, *tokenizer.encode(message.reasoning), 9])
        tokens.extend([*tokenizer.encode(message.content), 6])
    tokens.append(2)
    assistant_mask(tokens, control_version=tokenizer.control_version)
    truncated = len(tokens) > max_tokens
    if truncated:
        tokens = tokens[: max_tokens - 1] + [2]
    assistant_mask(tokens, control_version=tokenizer.control_version)
    return tokens, truncated
