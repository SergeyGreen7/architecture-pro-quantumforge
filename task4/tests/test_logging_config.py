from __future__ import annotations

import json
import logging

from quantumforge_bot.logging_config import configure_logging, shutdown_logging


def test_json_log_is_written_to_rotating_storage(tmp_path) -> None:
    log_path = configure_logging(
        log_dir=tmp_path,
        level="INFO",
        max_bytes=1024,
        backup_count=2,
    )
    logger = logging.getLogger("quantumforge_bot.test")

    logger.info(
        "rag_query_completed",
        extra={
            "request_id": "request-42",
            "duration_ms": 12.5,
            "answered_from_context": True,
        },
    )
    shutdown_logging()

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["level"] == "INFO"
    assert record["stream_name"] == "quantumforge-rag-bot"
    assert record["message"] == "rag_query_completed"
    assert record["request_id"] == "request-42"
    assert record["duration_ms"] == 12.5
    assert record["answered_from_context"] is True
    assert record["timestamp"].endswith("+00:00")


def test_log_file_rotates_at_configured_size(tmp_path) -> None:
    log_path = configure_logging(
        log_dir=tmp_path,
        level="INFO",
        max_bytes=300,
        backup_count=2,
    )
    logger = logging.getLogger("quantumforge_bot.rotation_test")

    for number in range(10):
        logger.info("rotation_record", extra={"record_number": number})
    shutdown_logging()

    assert log_path.exists()
    assert (tmp_path / "quantumforge-bot.jsonl.1").exists()
    for path in tmp_path.glob("quantumforge-bot.jsonl*"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert json.loads(line)["message"] == "rotation_record"
