"""FastAPI Telegram webhook for Yandex Serverless Containers."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request

from .bootstrap import AppServices, build_services
from .config import Settings
from .constants import (
    APP_TITLE,
    APP_VERSION,
    GENERIC_ERROR_MESSAGE,
    HEALTH_ENDPOINT,
    HELP_TEXT,
    TELEGRAM_HELP_COMMAND,
    TELEGRAM_START_COMMAND,
    TELEGRAM_WEBHOOK_ENDPOINT,
    TEXT_ONLY_MESSAGE,
    WELCOME_TEXT,
)
from .logging_config import configure_logging, shutdown_logging


logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    services_factory: Callable[[Settings], AppServices] = build_services,
) -> FastAPI:
    runtime_settings = settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal runtime_settings
        if runtime_settings is None:
            runtime_settings = Settings.from_env(require_telegram=True)
        log_path = configure_logging(
            log_dir=runtime_settings.log_dir,
            level=runtime_settings.log_level,
            max_bytes=runtime_settings.log_max_bytes,
            backup_count=runtime_settings.log_backup_count,
        )
        app.state.settings = runtime_settings
        try:
            logger.info(
                "application_starting",
                extra={"log_path": str(log_path)},
            )
            app.state.services = await asyncio.to_thread(
                services_factory, runtime_settings
            )
            logger.info(
                "application_started",
                extra={
                    "indexed_chunks": app.state.services.indexed_chunks,
                    "yandex_model": runtime_settings.yandex_model,
                    "embedding_model": runtime_settings.embedding_model,
                },
            )
            yield
        except Exception:
            logger.exception("application_startup_or_runtime_failed")
            raise
        finally:
            services = getattr(app.state, "services", None)
            if services is not None:
                await services.telegram.aclose()
            logger.info("application_stopped")
            shutdown_logging()

    application = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        lifespan=lifespan,
    )

    @application.get(HEALTH_ENDPOINT)
    async def health(request: Request) -> dict[str, object]:
        services: AppServices = request.app.state.services
        return {"status": "ok", "indexed_chunks": services.indexed_chunks}

    @application.post(TELEGRAM_WEBHOOK_ENDPOINT)
    async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ) -> dict[str, bool]:
        active_settings: Settings = request.app.state.settings
        provided_secret = x_telegram_bot_api_secret_token or ""
        yandex_request_id = request.headers.get("x-request-id")
        if not secrets.compare_digest(
            provided_secret, active_settings.telegram_webhook_secret
        ):
            logger.warning(
                "webhook_rejected",
                extra={"request_id": yandex_request_id, "reason": "invalid_secret"},
            )
            raise HTTPException(status_code=403, detail="Invalid webhook secret")

        update = await request.json()
        update_id = update.get("update_id")
        request_id = yandex_request_id or (
            str(update_id) if update_id is not None else "unknown"
        )
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            logger.debug(
                "webhook_update_ignored",
                extra={"request_id": request_id, "reason": "no_message"},
            )
            return {"ok": True}
        chat = message.get("chat")
        if not isinstance(chat, dict) or not isinstance(chat.get("id"), int):
            logger.debug(
                "webhook_update_ignored",
                extra={"request_id": request_id, "reason": "no_chat"},
            )
            return {"ok": True}
        chat_id = chat["id"]
        services: AppServices = request.app.state.services
        text = message.get("text")
        if not isinstance(text, str):
            logger.info(
                "non_text_message",
                extra={"request_id": request_id},
            )
            await services.telegram.send_message(chat_id, TEXT_ONLY_MESSAGE)
            return {"ok": True}

        command = text.strip().split(maxsplit=1)[0].split("@", maxsplit=1)[0].lower()
        if command == TELEGRAM_START_COMMAND:
            logger.info(
                "telegram_command",
                extra={"request_id": request_id, "command": TELEGRAM_START_COMMAND},
            )
            await services.telegram.send_message(chat_id, WELCOME_TEXT)
            return {"ok": True}
        if command == TELEGRAM_HELP_COMMAND:
            logger.info(
                "telegram_command",
                extra={"request_id": request_id, "command": TELEGRAM_HELP_COMMAND},
            )
            await services.telegram.send_message(chat_id, HELP_TEXT)
            return {"ok": True}

        started = time.perf_counter()
        logger.info(
            "rag_query_started",
            extra={"request_id": request_id, "query_length": len(text)},
        )
        try:
            await services.telegram.send_chat_action(chat_id)
            result = await asyncio.to_thread(services.pipeline.answer, text)
            await services.telegram.send_message(chat_id, result.answer)
            logger.info(
                "rag_query_completed",
                extra={
                    "request_id": request_id,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "answered_from_context": result.answered_from_context,
                    "retrieved_chunks": len(result.chunks),
                    "top_similarity": result.chunks[0].score if result.chunks else None,
                },
            )
        except ValueError as error:
            logger.warning(
                "rag_query_rejected",
                extra={
                    "request_id": request_id,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "reason": str(error),
                },
            )
            await services.telegram.send_message(chat_id, f"Некорректный вопрос: {error}")
        except Exception:
            logger.exception(
                "rag_query_failed",
                extra={
                    "request_id": request_id,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            await services.telegram.send_message(chat_id, GENERIC_ERROR_MESSAGE)
        return {"ok": True}

    return application


app = create_app()
