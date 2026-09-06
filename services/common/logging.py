"""Structured logging.

One configuration call at process start; every module then does
``log = get_logger(__name__)``. JSON in containers, human-readable locally.

A ``request_id`` is bound into the context by the API middleware so every log
line emitted while serving a request carries it, without threading a logger
through call signatures.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from services.common.config import get_settings

_CONFIGURED = False


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    global _CONFIGURED
    settings = get_settings()
    level = (level or settings.log_level).upper()
    fmt = fmt or settings.log_format

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level, logging.INFO),
        force=True,
    )
    # Uvicorn's own loggers would otherwise duplicate lines in a different shape.
    for noisy in ("uvicorn.access", "uvicorn.error", "neo4j", "httpx"):
        logging.getLogger(noisy).setLevel(
            logging.WARNING if noisy != "uvicorn.error" else logging.INFO
        )

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_request_context(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
