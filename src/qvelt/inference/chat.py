"""Inference chat prefixes share the trained explicit role/reasoning frame."""

from qvelt.posttrain.chat import Message
from qvelt.tokenizer.core import QveltTokenizer


def chat_prefix(
    messages: tuple[Message, ...], tokenizer: QveltTokenizer
) -> tuple[int, ...]:
    if tokenizer.control_version == 3:
        raise ValueError("Tokenizer v3 requires the unimplemented chat schema3")
    if not 1 <= len(messages) <= 64 or not all(
        isinstance(m, Message) for m in messages
    ):
        raise ValueError("Chat requires1..64 validated messages")
    tokens = [1]
    expected = "user"
    roles = {"system": 3, "user": 4, "assistant": 5}
    for i, message in enumerate(messages):
        if not (i == 0 and message.role == "system"):
            if message.role != expected:
                raise ValueError("Chat roles must alternate user/assistant")
            expected = "assistant" if message.role == "user" else "user"
        if any(
            c in message.content or c in (message.reasoning or "")
            for c in tokenizer.control_tokens
        ):
            raise ValueError("Reserved control spelling in chat content")
        tokens.append(roles[message.role])
        if message.reasoning is not None:
            if tokenizer.control_version != 2:
                raise ValueError("Reasoning requires tokenizer v2")
            tokens.extend((8, *tokenizer.encode(message.reasoning), 9))
        tokens.extend((*tokenizer.encode(message.content), 6))
        if len(tokens) > 2048:
            raise ValueError("Chat prefix exceeds CPU context capacity")
    if messages[-1].role != "user":
        raise ValueError("Generation prompt must end with user")
    return (*tokens, 5)
