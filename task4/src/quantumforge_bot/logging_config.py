"""Structured JSON logging to rotating local files and standard output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .constants import LOG_FILE_NAME, LOG_STREAM_NAME, PACKAGE_LOGGER_NAME


_STANDARD_LOG_RECORD_FIELDS = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__
)
_YANDEX_LEVELS = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARN",
    "ERROR": "ERROR",
    "CRITICAL": "FATAL",
}


class JsonFormatter(logging.Formatter):
    """Render one JSON object per line for JSONL and Yandex Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": _YANDEX_LEVELS.get(record.levelname, record.levelname),
            "stream_name": LOG_STREAM_NAME,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(
    *,
    log_dir: Path,
    level: str,
    max_bytes: int,
    backup_count: int,
) -> Path:
    """Configure package logging and return the active JSONL file path."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILE_NAME
    formatter = JsonFormatter()
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    _close_handlers(package_logger)
    package_logger.setLevel(level)
    package_logger.propagate = False

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    package_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    package_logger.addHandler(console_handler)
    return log_path


def shutdown_logging() -> None:
    """Flush and close handlers, primarily for graceful shutdown and tests."""

    _close_handlers(logging.getLogger(PACKAGE_LOGGER_NAME))


def _close_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value

