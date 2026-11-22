"""Environment-based configuration with repository-aware defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    DEFAULT_SECURITY_MODE,
    DEFAULT_YANDEX_MODEL,
    ENV_CHROMA_COLLECTION,
    ENV_EMBEDDING_CACHE_DIR,
    ENV_EMBEDDING_DEVICE,
    ENV_EMBEDDING_MODEL,
    ENV_LOG_BACKUP_COUNT,
    ENV_LOG_DIR,
    ENV_LOG_LEVEL,
    ENV_LOG_MAX_BYTES,
    ENV_RAG_INDEX_DIR,
    ENV_RAG_MAX_CONTEXT_CHARS,
    ENV_RAG_MAX_QUERY_CHARS,
    ENV_RAG_MIN_SIMILARITY,
    ENV_RAG_TOP_K,
    ENV_RAG_SECURITY_MODE,
    ENV_TELEGRAM_BOT_TOKEN,
    ENV_TELEGRAM_WEBHOOK_SECRET,
    ENV_YANDEX_API_KEY,
    ENV_YANDEX_FOLDER_ID,
    ENV_YANDEX_MODEL,
    SUPPORTED_EMBEDDING_DEVICES,
    SUPPORTED_LOG_LEVELS,
    SUPPORTED_SECURITY_MODES,
    TASK4_DIR,
)


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


def _path_from_env(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute():
        path = TASK4_DIR / path
    return path.resolve()


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _similarity(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error
    if not -1.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be between -1 and 1")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    telegram_webhook_secret: str
    yandex_folder_id: str
    yandex_api_key: str
    yandex_model: str
    index_dir: Path
    embedding_model: str
    embedding_cache_dir: Path
    embedding_device: str
    collection_name: str
    top_k: int
    min_similarity: float
    max_context_chars: int
    max_query_chars: int
    log_dir: Path = DEFAULT_LOG_DIR
    log_level: str = DEFAULT_LOG_LEVEL
    log_max_bytes: int = DEFAULT_LOG_MAX_BYTES
    log_backup_count: int = DEFAULT_LOG_BACKUP_COUNT
    security_mode: str = DEFAULT_SECURITY_MODE

    @classmethod
    def from_env(cls, *, require_telegram: bool = True) -> "Settings":
        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value:
                raise ConfigurationError(f"Required environment variable {name} is missing")
            return value

        telegram_token = os.getenv(ENV_TELEGRAM_BOT_TOKEN, "").strip()
        webhook_secret = os.getenv(ENV_TELEGRAM_WEBHOOK_SECRET, "").strip()
        if require_telegram:
            telegram_token = required(ENV_TELEGRAM_BOT_TOKEN)
            webhook_secret = required(ENV_TELEGRAM_WEBHOOK_SECRET)
        if webhook_secret and (
            len(webhook_secret) > 256
            or any(
                not (
                    character.isascii()
                    and (character.isalnum() or character in "_-")
                )
                for character in webhook_secret
            )
        ):
            raise ConfigurationError(
                "TELEGRAM_WEBHOOK_SECRET may contain only letters, digits, '_' and '-'"
            )

        device = os.getenv(ENV_EMBEDDING_DEVICE, DEFAULT_EMBEDDING_DEVICE).strip().lower()
        if device not in SUPPORTED_EMBEDDING_DEVICES:
            raise ConfigurationError("EMBEDDING_DEVICE must be auto, cpu, mps, or cuda")

        log_level = os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL).strip().upper()
        if log_level not in SUPPORTED_LOG_LEVELS:
            raise ConfigurationError(
                "LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            )

        security_mode = os.getenv(
            ENV_RAG_SECURITY_MODE, DEFAULT_SECURITY_MODE
        ).strip().lower()
        if security_mode not in SUPPORTED_SECURITY_MODES:
            raise ConfigurationError(
                "RAG_SECURITY_MODE must be off, pre_prompt, post_filter, sanitize, or all"
            )

        return cls(
            telegram_bot_token=telegram_token,
            telegram_webhook_secret=webhook_secret,
            yandex_folder_id=required(ENV_YANDEX_FOLDER_ID),
            yandex_api_key=required(ENV_YANDEX_API_KEY),
            yandex_model=os.getenv(ENV_YANDEX_MODEL, DEFAULT_YANDEX_MODEL).strip(),
            index_dir=_path_from_env(ENV_RAG_INDEX_DIR, DEFAULT_INDEX_DIR),
            embedding_model=os.getenv(
                ENV_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL
            ).strip(),
            embedding_cache_dir=_path_from_env(
                ENV_EMBEDDING_CACHE_DIR,
                DEFAULT_EMBEDDING_CACHE_DIR,
            ),
            embedding_device=device,
            collection_name=os.getenv(
                ENV_CHROMA_COLLECTION, DEFAULT_CHROMA_COLLECTION
            ).strip(),
            top_k=_positive_int(ENV_RAG_TOP_K, DEFAULT_TOP_K),
            min_similarity=_similarity(
                ENV_RAG_MIN_SIMILARITY, DEFAULT_MIN_SIMILARITY
            ),
            max_context_chars=_positive_int(
                ENV_RAG_MAX_CONTEXT_CHARS, DEFAULT_MAX_CONTEXT_CHARS
            ),
            max_query_chars=_positive_int(
                ENV_RAG_MAX_QUERY_CHARS, DEFAULT_MAX_QUERY_CHARS
            ),
            log_dir=_path_from_env(ENV_LOG_DIR, DEFAULT_LOG_DIR),
            log_level=log_level,
            log_max_bytes=_positive_int(ENV_LOG_MAX_BYTES, DEFAULT_LOG_MAX_BYTES),
            log_backup_count=_positive_int(
                ENV_LOG_BACKUP_COUNT, DEFAULT_LOG_BACKUP_COUNT
            ),
            security_mode=security_mode,
        )
