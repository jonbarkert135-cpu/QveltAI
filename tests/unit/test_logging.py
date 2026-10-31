"""Structured logger behavior and isolation from application-owned handlers."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from datetime import datetime

import pytest

from qvelt.utils.logging import JsonFormatter, configure_logging


@pytest.fixture(autouse=True)
def restore_logger() -> Iterator[None]:
    logger = logging.getLogger("qvelt")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    for handler in logger.handlers[:]:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_json_fields_unicode_and_utc_timestamp() -> None:
    stream = io.StringIO()
    configure_logging(stream=stream).info("%s", "Привет\nworld")
    record = json.loads(stream.getvalue())
    assert record["message"] == "Привет\nworld"
    assert record["level"] == "INFO" and record["logger"] == "qvelt"
    offset = datetime.fromisoformat(record["time"]).utcoffset()
    assert offset is not None and offset.total_seconds() == 0
    assert len(stream.getvalue().splitlines()) == 1


def test_reconfiguration_does_not_duplicate_or_modify_root() -> None:
    root = logging.getLogger()
    old_level, old_handlers = root.level, root.handlers[:]
    stream = io.StringIO()
    configure_logging(stream=stream)
    logger = configure_logging("WARNING", stream=stream)
    logger.info("filtered")
    logger.warning("once")
    assert len(stream.getvalue().splitlines()) == 1
    assert root.level == old_level and root.handlers == old_handlers


def test_preserve_foreign_handler() -> None:
    logger = logging.getLogger("qvelt")
    foreign = logging.StreamHandler(io.StringIO())
    foreign.setFormatter(JsonFormatter())
    logger.addHandler(foreign)
    configure_logging(stream=io.StringIO())
    assert foreign in logger.handlers


def test_exception_is_structured() -> None:
    stream = io.StringIO()
    logger = configure_logging(stream=stream)
    try:
        raise ValueError("fixture failure")
    except ValueError:
        logger.exception("failed")
    record = json.loads(stream.getvalue())
    assert record["message"] == "failed"
    assert "ValueError: fixture failure" in record["exception"]
    assert isinstance(logger.handlers[-1].formatter, JsonFormatter)


def test_bad_level() -> None:
    with pytest.raises(ValueError):
        configure_logging("NOT_A_LEVEL")
