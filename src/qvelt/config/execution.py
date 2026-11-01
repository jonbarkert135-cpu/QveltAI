"""Runtime model policies, separate from learned tensor dimensions."""

from __future__ import annotations

from dataclasses import dataclass

from qvelt.config import ConfigError


@dataclass(frozen=True)
class ModelExecution:
    attention_backend: str = "reference"
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.attention_backend not in ("reference", "sdpa"):
            raise ConfigError("Unknown attention backend")
        if type(self.activation_checkpointing) is not bool:
            raise ConfigError("activation_checkpointing must be bool")
