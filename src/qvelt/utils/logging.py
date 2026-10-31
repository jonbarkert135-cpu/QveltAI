"""Small JSON console logger; no global/root logging configuration."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TextIO


class JsonFormatter(logging.Formatter):
    """Render UTC time, level, logger, message and optional exception as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def configure_logging(
    level: str = "INFO", *, stream: TextIO | None = None
) -> logging.Logger:
    """Replace only Qvelt-owned console handlers, leaving the root logger alone.

    This is not a secret-redaction layer: callers must not log credentials,
    raw configuration documents or sensitive training examples.
    """
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError("Unsupported log level")
    logger = logging.getLogger("qvelt")
    for handler in logger.handlers[:]:
        if handler.get_name() == "qvelt.console":
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name("qvelt.console")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
